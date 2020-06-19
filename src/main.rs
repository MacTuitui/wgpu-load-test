use winit::{
    event::*,
    event_loop::{EventLoop, ControlFlow},
    window::{Window, WindowBuilder},
    dpi::*,
};
use futures::executor::block_on;
use cgmath::{Matrix4, Point3, Deg, Vector3};

mod icosphere;
use icosphere::IcoSphere;

#[repr(C)]
#[derive(Copy, Clone, Debug)]
struct Vertex {
    position: [f32; 3],
    tex_coords: [f32; 2],
    normal: [f32; 3],
}
unsafe impl bytemuck::Pod for Vertex {}
unsafe impl bytemuck::Zeroable for Vertex {}

//the lifetime here makes it static
impl Vertex {
    fn desc<'a>() -> wgpu::VertexBufferDescriptor<'a> {
        use std::mem;
        wgpu::VertexBufferDescriptor {
            stride: mem::size_of::<Vertex>() as wgpu::BufferAddress,
            step_mode: wgpu::InputStepMode::Vertex,
            attributes: &[
                //the position
                wgpu::VertexAttributeDescriptor {
                    offset: 0,
                    shader_location: 0,
                    format: wgpu::VertexFormat::Float3,
                },
                //the tex_coord after 3 floats
                wgpu::VertexAttributeDescriptor {
                    offset: mem::size_of::<[f32; 3]>() as wgpu::BufferAddress,
                    shader_location: 1,
                    format: wgpu::VertexFormat::Float2,
                },
                //the normal after 3+2 floats
                wgpu::VertexAttributeDescriptor {
                    offset: mem::size_of::<[f32; 5]>() as wgpu::BufferAddress,
                    shader_location: 2,
                    format: wgpu::VertexFormat::Float3,
                },
            ]
        }
    }
}

#[repr(C)] 
#[derive(Debug, Copy, Clone)] 
struct Uniforms {
    world: Matrix4<f32>,
    view: Matrix4<f32>,
    proj: Matrix4<f32>,
    light_pos: [f32;4],
    colors: [[f32;4];30],
    time:f32,
    costime:f32,
    sintime:f32,
}

impl Uniforms {
    fn new() -> Self {
        let colors: [[f32;4];30] = [[1.0,1.0,1.0,1.0]; 30];
        Self {
            world: Matrix4::from_scale(1.0),
            view:  Matrix4::from_scale(1.0),
            proj:  Matrix4::from_scale(1.0),
            light_pos: [0.0,0.0,0.0,1.0],
            colors,
            time:0.0,
            costime:1.0,
            sintime:0.0,
        }
    }
}
unsafe impl bytemuck::Pod for Uniforms {}
unsafe impl bytemuck::Zeroable for Uniforms {}


//texture for z-buffer
pub struct Texture {
    pub texture: wgpu::Texture,
    pub view: wgpu::TextureView,
    pub sampler: wgpu::Sampler,
}

impl Texture {
    pub const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;

    pub fn create_depth_texture(device: &wgpu::Device, sc_desc: &wgpu::SwapChainDescriptor, label: &str) -> Self {
        let size = wgpu::Extent3d {
            width: sc_desc.width,
            height: sc_desc.height,
            depth: 1,
        };
        let desc = wgpu::TextureDescriptor {
            label: Some(label),
            size,
            array_layer_count: 1,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: Self::DEPTH_FORMAT,
            usage: wgpu::TextureUsage::OUTPUT_ATTACHMENT
                | wgpu::TextureUsage::SAMPLED 
                | wgpu::TextureUsage::COPY_SRC,
        };
        let texture = device.create_texture(&desc);

        let view = texture.create_default_view();
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Nearest,
            lod_min_clamp: -100.0,
            lod_max_clamp: 100.0,
            compare: wgpu::CompareFunction::LessEqual,
        });

        Self { texture, view, sampler }
    }
}

// main.rs
struct State {
    surface: wgpu::Surface,
    device: wgpu::Device,
    queue: wgpu::Queue,
    sc_desc: wgpu::SwapChainDescriptor,
    swap_chain: wgpu::SwapChain,
    size: winit::dpi::PhysicalSize<u32>,
    render_pipeline: wgpu::RenderPipeline,
    depth_texture: Texture,
    main_bind_group: wgpu::BindGroup,
    clear_color: wgpu::Color,
    sphere: IcoSphere,
    uniforms: Uniforms,
    uniforms_buffer: wgpu::Buffer,
    frames:u32,
}

pub fn shader_from_spirv_bytes(device: &wgpu::Device, bytes: &[u8]) -> wgpu::ShaderModule {
    //load them so that we are not depending on shaderc
    let cursor = std::io::Cursor::new(bytes);
    let vs_spirv = wgpu::read_spirv(cursor).expect("failed to read hard-coded SPIRV");
    device.create_shader_module(&vs_spirv)
}
impl State {
    async fn new(window: &Window) -> Self {
        
        let size = window.inner_size();

        //needed for the swap_chain
        let surface = wgpu::Surface::create(window);

        //get the adapter -> the physical thing
        let adapter = wgpu::Adapter::request(
            &wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                //power_preference: wgpu::PowerPreference::LowPower,
                compatible_surface: Some(&surface),
            },
            wgpu::BackendBit::PRIMARY, 
            ).await.unwrap(); 

        //if we asked for a high performance, we should get the radeon
        //if we asked for a low  performance, we should get the intel
        println!("{:?}",adapter.get_info());

        //now we can get device and queue
        //the device is the abstraction of the actual physical device
        //the queue allows us to send commands to that device
        let (device, queue) = adapter.request_device(&wgpu::DeviceDescriptor {
            extensions: wgpu::Extensions {
                anisotropic_filtering: false,
            },
            limits: Default::default(),
        }).await;


        //describe the swapchain
        //a swapchain will be presented to the surface it's linked to
        //so we'll use a texture output in bgra8unorm??? see what nannou does
        //in fifo mode (v-sync, capped)
        let sc_desc = wgpu::SwapChainDescriptor {
            usage: wgpu::TextureUsage::OUTPUT_ATTACHMENT,
            //format: wgpu::TextureFormat::Bgra8Unorm,
            format: wgpu::TextureFormat::Bgra8UnormSrgb,
            width: size.width,
            height: size.height,
            present_mode: wgpu::PresentMode::Fifo,
        };
        //get the swap_chain!
        let swap_chain = device.create_swap_chain(&surface, &sc_desc);
        let clear_color = wgpu::Color{r:0.0,g:0.0,b:0.0,a:0.0};
        
        //load the shaders from spirv code
        let vs_module = shader_from_spirv_bytes(&device, include_bytes!("shaders/basic-vert.spv"));
        let fs_module = shader_from_spirv_bytes(&device, include_bytes!("shaders/basic-frag.spv"));

        let mut uniforms = Uniforms::new();
        let colors = [
            [0.97254914, 0.6196076, 0.59607846],
            [0.9783395, 0.6809013, 0.66096234],
            [0.98408467, 0.7357697, 0.7185086],
            [0.9890234, 0.7702383, 0.74490476],
            [0.99190575, 0.76168674, 0.69874924],
            [0.994777, 0.753008, 0.6482526],
            [0.9968645, 0.7771854, 0.63585895],
            [0.9983031, 0.8256165, 0.65769625],
            [0.9997391, 0.8705792, 0.6786348],
            [0.85759175, 0.8153513, 0.67769635],
            [0.6236557, 0.72829133, 0.6719481],
            [0.17853943, 0.63919467, 0.67023855],
            [0.34668234, 0.6802211, 0.70800763],
            [0.44948435, 0.7182693, 0.74332523],
            [0.53883755, 0.7505153, 0.77232283],
            [0.6266192, 0.77544653, 0.7927792],
            [0.70088154, 0.7993717, 0.8125669],
            [0.6438602, 0.74905026, 0.7662073],
            [0.47841805, 0.6421112, 0.67080325],
            [0.13841504, 0.5051259, 0.5534906],
            [0.011764375, 0.45019072, 0.50472844],
            [0.011764869, 0.40032014, 0.4603564],
            [0.026915757, 0.34360865, 0.40804705],
            [0.08994619, 0.28565487, 0.3355782],
            [0.12826383, 0.2089256, 0.2364954],
            [0.4130114, 0.38815737, 0.38927343],
            [0.6757641, 0.631422, 0.63176113],
            [0.84686, 0.79071134, 0.7907657],
            [0.9103428, 0.7790192, 0.7744527],
            [0.9421252, 0.70517576, 0.6927323],
            ];
            //put the palette in the unis, disregard, this is usually done from a picture
            for i in 0..uniforms.colors.len() {
                uniforms.colors[i][0]=colors[i][0];
                uniforms.colors[i][1]=colors[i][1];
                uniforms.colors[i][2]=colors[i][2];
                uniforms.colors[i][3]=1.0;
        }



        let uniforms_buffer = device.create_buffer_with_data(
            bytemuck::cast_slice(&[uniforms]),
            wgpu::BufferUsage::UNIFORM | wgpu::BufferUsage::COPY_DST,
        );

        let main_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            bindings: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStage::VERTEX,
                    ty: wgpu::BindingType::UniformBuffer {
                        dynamic: false,
                    },
                }
            ],
            label: Some("Bind group layout with uniform buffer")
        });

    let main_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        layout: &main_bind_group_layout,
        bindings: &[
            wgpu::Binding {
                binding: 0,
                resource: wgpu::BindingResource::Buffer {
                    buffer: &uniforms_buffer,
                    range: 0..std::mem::size_of_val(&uniforms) as wgpu::BufferAddress,
                }
            },
        ],
        label: Some("Bind group with uniform buffer set"),
    });
        //create the renderPipeline
        let render_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            bind_group_layouts: &[&main_bind_group_layout],
        });

        let depth_texture = Texture::create_depth_texture(&device, &sc_desc, "depth_texture");

        //the pipeline with the shaders
        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            layout: &render_pipeline_layout,
            //we have the vertex shader here
            vertex_stage: wgpu::ProgrammableStageDescriptor {
                module: &vs_module,
                entry_point: "main", // 1.
            },
            //the fragment shader
            fragment_stage: Some(wgpu::ProgrammableStageDescriptor { // 2.
                module: &fs_module,
                entry_point: "main",
            }),
            //raster details
            rasterization_state: Some(wgpu::RasterizationStateDescriptor {
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: wgpu::CullMode::Back,
                depth_bias: 0,
                depth_bias_slope_scale: 0.0,
                depth_bias_clamp: 0.0,
            }),
            color_states: &[
                wgpu::ColorStateDescriptor {
                    format: sc_desc.format,
                    color_blend: wgpu::BlendDescriptor::REPLACE,
                    alpha_blend: wgpu::BlendDescriptor::REPLACE,
                    write_mask: wgpu::ColorWrite::ALL,
                },
            ],
            primitive_topology: wgpu::PrimitiveTopology::TriangleList, // 1.
            depth_stencil_state: Some(wgpu::DepthStencilStateDescriptor {
                format: Texture::DEPTH_FORMAT,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Less,
                stencil_front: wgpu::StencilStateFaceDescriptor::IGNORE,
                stencil_back: wgpu::StencilStateFaceDescriptor::IGNORE,
                stencil_read_mask: 0,
                stencil_write_mask: 0,
            }),
            //what format the vertices are
            vertex_state: wgpu::VertexStateDescriptor {
                index_format: wgpu::IndexFormat::Uint32,
                vertex_buffers: &[
                    Vertex::desc(),
                ],
            },
            sample_count: 1, // 5.
            sample_mask: !0, // 6.
            alpha_to_coverage_enabled: false, // 7.
        });
 
        let mut sphere = IcoSphere::new(0.0,0.0);
        sphere.make_geometry(10,false);
        sphere.make_buffers(&device, 10.0);

        //return the State
        Self {
            surface,
            device,
            queue,
            sc_desc,
            swap_chain,
            size,
            clear_color,
            render_pipeline,
            depth_texture,
            main_bind_group,
            sphere,
            uniforms,
            uniforms_buffer,
            frames: 0,
        }
    }

    fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        self.size = new_size;
        self.sc_desc.width = new_size.width;
        self.sc_desc.height = new_size.height;
        self.swap_chain = self.device.create_swap_chain(&self.surface, &self.sc_desc);

        //TODO update depth texture
    }

    // input() won't deal with GPU code, so it can be synchronous
    fn input(&mut self, event: &WindowEvent) -> bool {
        match event {
            WindowEvent::KeyboardInput {
                input,
                ..
            } => {
                match input {
                    KeyboardInput {
                        state: ElementState::Pressed,
                        virtual_keycode: Some(VirtualKeyCode::Space),
                        ..
                    } => {
                        true
                    }
                    _ => {false}
                }
            }
            _ => {
                false
            }
        }
    }

    fn update(&mut self) {
        //do nothing
    }

    fn render(&mut self) {
        //where to draw
        let frame = self.swap_chain.get_next_texture()
            .expect("Timeout getting texture");

        //to be able to send commands to the gpu
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Render Encoder"),
        });


        //update the uniforms
        let angle = self.frames as f32*0.01;
        self.uniforms.time = angle;
        self.uniforms.costime = angle.cos();
        self.uniforms.sintime = angle.sin();

        let w = self.sc_desc.width;
        let h = self.sc_desc.height;
        //the matrices
        let aspect_ratio = w as f32 / h as f32;
        let proj = cgmath::perspective(Deg(90.0), aspect_ratio, 0.01, 100.0);

        //the view matrix
        let view = Matrix4::look_at(
            //eye
            Point3::new(10.0*angle.cos(), 10.0*angle.sin(),10.0),
            //center
            Point3::new(0.0,0.0,0.0),
            //up
            Vector3::new(0.0, 0.0, 1.0),
        );

        let world_scale = Matrix4::from_scale(1.0);

        //matrices, nothing actually happening
        self.uniforms.world = Matrix4::from_scale(1.0);
        self.uniforms.view = (view * world_scale).into();
        self.uniforms.proj = proj.into();

        //light position
        self.uniforms.light_pos[0] = -15.0*angle.cos();
        self.uniforms.light_pos[1] = 15.0*angle.sin();
        self.uniforms.light_pos[2] = 10.0;

        //uniforms
        let uniform_size = std::mem::size_of::<Uniforms>() as wgpu::BufferAddress;
        let uniforms_buffer = self.device.create_buffer_with_data(
            bytemuck::cast_slice(&[self.uniforms]),
            wgpu::BufferUsage::COPY_SRC,
        );

        //copy the new values to the real uniforms_buffer
        encoder.copy_buffer_to_buffer(
            &uniforms_buffer,
            0,
            &self.uniforms_buffer,
            0,
            uniform_size);


        //the render pass
        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                color_attachments: &[
                    wgpu::RenderPassColorAttachmentDescriptor {
                        attachment: &frame.view,
                        resolve_target: None,
                        load_op: wgpu::LoadOp::Clear,
                        store_op: wgpu::StoreOp::Store,
                        clear_color:self.clear_color,
                    }
                ],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachmentDescriptor {
                    attachment: &self.depth_texture.view,
                    depth_load_op: wgpu::LoadOp::Clear,
                    depth_store_op: wgpu::StoreOp::Store,
                    clear_depth: 1.0,
                    stencil_load_op: wgpu::LoadOp::Clear,
                    stencil_store_op: wgpu::StoreOp::Store,
                    clear_stencil: 0,
                }),
            });
            render_pass.set_pipeline(&self.render_pipeline); 
            //just the sphere once
            render_pass.set_vertex_buffer(0, &self.sphere.vertex_buffer.as_ref().unwrap(), 0, 0);
            render_pass.set_index_buffer(&self.sphere.index_buffer.as_ref().unwrap(),  0,0);
            render_pass.set_bind_group(0, &self.main_bind_group, &[]);
            render_pass.draw_indexed(0..self.sphere.n_indices,0, 0..1);
        }

        self.queue.submit(&[ encoder.finish() ]);
        //done
        self.frames+=1;
    }
}




fn main() {
    let event_loop = EventLoop::new();
    let window = WindowBuilder::new()
        .with_inner_size(LogicalSize::new(1024,1024))
        //.with_decorations(false)
        .build(&event_loop)
        .unwrap();

    let mut state = block_on(State::new(&window));

    event_loop.run(move |event, _, control_flow| {
        match event {
            Event::WindowEvent {
                ref event,
                window_id,
            } if window_id == window.id() => if !state.input(event) {
                match event {
                    WindowEvent::CloseRequested => *control_flow = ControlFlow::Exit,
                    WindowEvent::KeyboardInput {
                        input,
                        ..
                    } => {
                        match input {
                            KeyboardInput {
                                state: ElementState::Pressed,
                                virtual_keycode: Some(VirtualKeyCode::Escape),
                                ..
                            } => *control_flow = ControlFlow::Exit,
                            _ => {}
                        }
                    }
                    WindowEvent::Resized(physical_size) => {
                        state.resize(*physical_size);
                    }
                    WindowEvent::ScaleFactorChanged { new_inner_size, .. } => {
                        // new_inner_size is &mut so we have to dereference it twice
                        state.resize(**new_inner_size);
                    }
                    _ => {}
                }
            }
            Event::RedrawRequested(_) => {
                state.update();
                state.render();
            }
            Event::MainEventsCleared => {
                //decide when to request a redraw
                window.request_redraw();
            }
            _ => {}
        }
    });
}

