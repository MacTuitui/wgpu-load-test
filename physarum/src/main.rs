use winit::{
    event::*,
    event_loop::{EventLoop, ControlFlow},
    window::{Window, WindowBuilder},
    dpi::*,
};
use std::path::Path;
use futures::executor::block_on;
use rand::Rng;

const TAU:f32 = 6.283185307179586;

#[repr(C)]
#[derive(Copy, Clone, Debug)]
struct Vertex {
    position: [f32; 2],
    tex_coords: [f32; 2],
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
                    format: wgpu::VertexFormat::Float2, 
                },
                //the tex_coord after 2 floats
                wgpu::VertexAttributeDescriptor {
                    offset: mem::size_of::<[f32; 2]>() as wgpu::BufferAddress,
                    shader_location: 1,
                    format: wgpu::VertexFormat::Float2,
                },
            ]
        }
    }
}

const VERTICES: &[Vertex] = &[
    Vertex { position: [-1.0, -1.0], tex_coords: [0.0,0.0], }, // A
    Vertex { position: [ 1.0, -1.0], tex_coords: [1.0,0.0], }, // B
    Vertex { position: [ 1.0, 1.0], tex_coords: [1.0,1.0], }, // C
    Vertex { position: [-1.0, 1.0], tex_coords: [0.0,1.0], }, // D
];

const INDICES: &[u16] = &[
    0, 1, 2,
    0, 2, 3,
];

struct Data {
    pos: Vec<[f32;4]>,
}
impl Data {
    pub fn new() -> Self {
        Data{
            pos: Vec::new(),
        }
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
    clear_color: wgpu::Color,
    side: bool,
    frames:u32,
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    buffer_a: wgpu::Buffer,
    buffer_b: wgpu::Buffer,
    zero_buffer_size: wgpu::BufferAddress,
    particles_buffer: wgpu::Buffer,
    particles_size: u32,
    compute_bind_group_a: wgpu::BindGroup,
    compute_bind_group_b: wgpu::BindGroup,
    buffer_in_frag_bind_group_a: wgpu::BindGroup,
    buffer_in_frag_bind_group_b: wgpu::BindGroup,
    compute_pipeline: wgpu::ComputePipeline,
    compute_pipeline_layout: wgpu::PipelineLayout,
    compute_pipeline_blur: wgpu::ComputePipeline,
    render_pipeline_from_buff: wgpu::RenderPipeline,
}

pub fn shader_from_spirv(device: &wgpu::Device, path: &Path) -> wgpu::ShaderModule {
    //load them so that we are not depending on shaderc
    let file = std::fs::File::open(path).unwrap();
    let vs_spirv = wgpu::read_spirv(file).expect("failed to read hard-coded SPIRV");
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

    let vs_module_pass = shader_from_spirv(&device, Path::new("src/shaders/pass.vert.spv"));
    let frag_module_from_buf = shader_from_spirv(&device, Path::new("src/shaders/buffer.frag.spv"));
    //compute shaders
    let compute_module_main = shader_from_spirv(&device, Path::new("src/shaders/physarum.spv"));
    let compute_module_blur = shader_from_spirv(&device, Path::new("src/shaders/blur.spv"));


    let compute_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        bindings: &[
            //we want the buffers to be visible
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStage::COMPUTE,
                ty: wgpu::BindingType::StorageBuffer {
                    dynamic: false,
                    readonly: false,
                },
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStage::COMPUTE,
                ty: wgpu::BindingType::StorageBuffer {
                    dynamic: false,
                    readonly: false,
                },
            },
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStage::COMPUTE,
                ty: wgpu::BindingType::StorageBuffer {
                    dynamic: false,
                    readonly: false,
                },
            },
        ],
        label: Some("Bindgroup layout for compute, 3 buffers")
    });

    let compute_blur_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        bindings: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStage::COMPUTE,
                ty: wgpu::BindingType::StorageBuffer {
                    dynamic: false,
                    readonly: false,
                },
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStage::COMPUTE,
                ty: wgpu::BindingType::StorageBuffer {
                    dynamic: false,
                    readonly: false,
                },
            },
        ],
        label: Some("Bindgroup layout for compute, 2 buffers")
    });


    //for compute!!!!
    //
    //let's make a buffer full of floats
    
    let mut rand_data = Vec::new();
    for _i in 0..1024*1024 {
        rand_data.push(0.8);
    }
    let buffer_data = rand_data.iter().map(|p| *p).collect::<Vec<_>>();
    let buffer_size = rand_data.len() * std::mem::size_of::<f32>();

        let buffer_a = device.create_buffer_with_data(
            bytemuck::cast_slice(&buffer_data),
            wgpu::BufferUsage::STORAGE_READ | wgpu::BufferUsage::STORAGE | wgpu::BufferUsage::COPY_SRC,
        );
        let buffer_b = device.create_buffer_with_data(
            bytemuck::cast_slice(&buffer_data),
            wgpu::BufferUsage::STORAGE_READ | wgpu::BufferUsage::STORAGE | wgpu::BufferUsage::COPY_DST,
        );


    let zero_buffer_size = 1024*1024 * std::mem::size_of::<f32>();

    let mut particles = Data::new();

    let particles_size = 1024*1024*4;

    let mut rng = rand::thread_rng();
    for _i in 0..particles_size{
        let r = 0.25;
        let angle = rng.gen::<f32>()*TAU;
        let x = r*angle.cos()+0.5;
        let y = r*angle.sin()+0.5;
        particles.pos.push([x,y,angle/TAU+0.5,0.0]);
    }

    let particles_buffer_data = particles.pos.iter().map(|p| *p).collect::<Vec<_>>();
    let particles_buffer_size = particles_buffer_data.len() * std::mem::size_of::<[f32;4] >();

    let particles_buffer = device.create_buffer_with_data(
        bytemuck::cast_slice(&particles_buffer_data),
        wgpu::BufferUsage::STORAGE_READ | wgpu::BufferUsage::STORAGE
    );

    //the first compute pass will work with a,b and the particles
    let compute_bind_group_a = device.create_bind_group(&wgpu::BindGroupDescriptor {
        layout: &compute_bind_group_layout,
        bindings: &[
            wgpu::Binding {
                binding: 0,
                resource: wgpu::BindingResource::Buffer {
                    buffer: &buffer_a,
                    range: 0..buffer_size as wgpu::BufferAddress,
                }
            },
            wgpu::Binding {
                binding: 1,
                resource: wgpu::BindingResource::Buffer {
                    buffer: &buffer_b,
                    range: 0..buffer_size as wgpu::BufferAddress,
                }
            },
            wgpu::Binding {
                binding: 2,
                resource: wgpu::BindingResource::Buffer {
                    buffer: &particles_buffer,
                    range: 0..particles_buffer_size as wgpu::BufferAddress,
                }
            },
        ],
        label: Some("Compute bind group  (main)"),
    });

    //the second compute pass will work with b,a and no particles
    //blur/diffuse
    let compute_bind_group_b = device.create_bind_group(&wgpu::BindGroupDescriptor {
        layout: &compute_blur_bind_group_layout,
        bindings: &[
            wgpu::Binding {
                binding: 0,
                resource: wgpu::BindingResource::Buffer {
                    buffer: &buffer_b,
                    range: 0..buffer_size as wgpu::BufferAddress,
                }
            },
            wgpu::Binding {
                binding: 1,
                resource: wgpu::BindingResource::Buffer {
                    buffer: &buffer_a,
                    range: 0..buffer_size as wgpu::BufferAddress,
                }
            },
        ],
        label: Some("Compute bind group (blur)"),
    });

    //the first compute pipeline
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        bind_group_layouts: &[&compute_bind_group_layout],
    });

    let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        layout: &pipeline_layout,
        compute_stage: wgpu::ProgrammableStageDescriptor {
            module: &compute_module_main,
            entry_point: "main",
        },
    });

    //the second compute pipeline (blur/diffuse)
    let pipeline_blur_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        bind_group_layouts: &[&compute_blur_bind_group_layout],
    });
    let compute_pipeline_blur = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        layout: &pipeline_blur_layout,
        compute_stage: wgpu::ProgrammableStageDescriptor {
            module: &compute_module_blur,
            entry_point: "main",
        },
    });


    //use a buffer in the final frag
    let buffer_in_frag_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        bindings: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStage::FRAGMENT,
                ty: wgpu::BindingType::StorageBuffer {
                    dynamic: false,
                    readonly: true,
                },
            },
        ],
        label: Some("bindgroup layout for having a buffer in fragment")
    });


    //do we want to show buffer_a or buffer_b
    let buffer_in_frag_bind_group_a = device.create_bind_group(&wgpu::BindGroupDescriptor {
        layout: &buffer_in_frag_group_layout,
        bindings: &[
            wgpu::Binding {
                binding: 0,
                resource: wgpu::BindingResource::Buffer {
                    buffer: &buffer_a,
                    range: 0..buffer_size as wgpu::BufferAddress,
                }
            },
        ],
        label: Some("bindgroup for having buffer a in fragment")
    });
    let buffer_in_frag_bind_group_b = device.create_bind_group(&wgpu::BindGroupDescriptor {
        layout: &buffer_in_frag_group_layout,
        bindings: &[
            wgpu::Binding {
                binding: 0,
                resource: wgpu::BindingResource::Buffer {
                    buffer: &buffer_b,
                    range: 0..buffer_size as wgpu::BufferAddress,
                }
            },
        ],
        label: Some("bindgroup for having buffer b in fragment")
    });



    //render pipeline to show a buffer
        let render_pipeline_layout_from_buff = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            bind_group_layouts: &[&buffer_in_frag_group_layout],
        });

        /*
        let render_pipeline_from_buff = wgpu::RenderPipelineBuilder::from_layout(&render_pipeline_layout_from_buff, &vs_module_pass)
        .fragment_shader(&frag_module_from_buf)
        .color_format(format)
        .add_vertex_buffer::<Vertex>()
        .index_format(IndexFormat::Uint16)
        .sample_count(msaa_samples)
        .primitive_topology(wgpu::PrimitiveTopology::TriangleList)
        .build(device);
        */






        //create the vertex array
    let vertex_buffer = device.create_buffer_with_data(
        bytemuck::cast_slice(&VERTICES),
        wgpu::BufferUsage::VERTEX
    );
    let index_buffer = device.create_buffer_with_data(
        bytemuck::cast_slice(&INDICES),
        wgpu::BufferUsage::INDEX
    );

        
        let render_pipeline_from_buff = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            layout: &render_pipeline_layout_from_buff,
            //we have the vertex shader here
            vertex_stage: wgpu::ProgrammableStageDescriptor {
                module: &vs_module_pass,
                entry_point: "main", // 1.
            },
            //the fragment shader
            fragment_stage: Some(wgpu::ProgrammableStageDescriptor { // 2.
                module: &frag_module_from_buf,
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
            depth_stencil_state: None,
            //what format the vertices are
            vertex_state: wgpu::VertexStateDescriptor {
                index_format: wgpu::IndexFormat::Uint16,
                vertex_buffers: &[
                    Vertex::desc(),
                ],
            },
            sample_count: 1, // 5.
            sample_mask: !0, // 6.
            alpha_to_coverage_enabled: false, // 7.
        });
 

        //return the State
        Self {
            surface,
            device,
            queue,
            sc_desc,
            swap_chain,
            size,
            clear_color,
            side: false,
            frames: 0,
            vertex_buffer,
            index_buffer,
            buffer_a,
            buffer_b,
            zero_buffer_size: zero_buffer_size as u64,
            particles_buffer,
            particles_size,
            compute_bind_group_a,
            compute_bind_group_b,
            buffer_in_frag_bind_group_a,
            buffer_in_frag_bind_group_b,
            compute_pipeline,
            compute_pipeline_layout:pipeline_layout,
            compute_pipeline_blur,
            render_pipeline_from_buff,
        }
    }

    fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        self.size = new_size;
        self.sc_desc.width = new_size.width;
        self.sc_desc.height = new_size.height;
        self.swap_chain = self.device.create_swap_chain(&self.surface, &self.sc_desc);
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
                        self.side = !self.side;
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

        //compute pass
        {
            //clear b
            encoder.copy_buffer_to_buffer(
                &self.buffer_a,
                0,
                &self.buffer_b,
                0,
                self.zero_buffer_size);

            let mut cpass = encoder.begin_compute_pass();
            cpass.set_pipeline(&self.compute_pipeline);
            cpass.set_bind_group(0, &self.compute_bind_group_a, &[]);
            //should have a workgroup size > 64
            cpass.dispatch(self.particles_size/64, 1, 1);
        }
        {
            let mut cpass = encoder.begin_compute_pass();
            cpass.set_pipeline(&self.compute_pipeline_blur);
            cpass.set_bind_group(0, &self.compute_bind_group_b, &[]);
            //should have a workgroup size > 64
            cpass.dispatch(1024/8, 1024/8, 1);
        }
        //render buffer

          {
              let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                  color_attachments: &[
                      wgpu::RenderPassColorAttachmentDescriptor {
                          //the output destination
                          //could be a view on a texture
                          attachment: &frame.view,
                          resolve_target: None,
                          load_op: wgpu::LoadOp::Clear,
                          store_op: wgpu::StoreOp::Store,
                          clear_color:wgpu::Color {
                              r: 0.,
                              g: 0.,
                              b: 0.,
                              a: 0.,
                          },
                      }
                  ],
                  depth_stencil_attachment: None,
              });
              //the "pass" pipeline
              render_pass.set_pipeline(&self.render_pipeline_from_buff); 
              //with the right texture
              if self.side{
                  render_pass.set_bind_group(0, &self.buffer_in_frag_bind_group_b, &[]);
              } else {
                  render_pass.set_bind_group(0, &self.buffer_in_frag_bind_group_a, &[]);
              }
              render_pass.set_vertex_buffer(0, &self.vertex_buffer, 0, 0);
              render_pass.set_index_buffer(&self.index_buffer,  0,0);
              render_pass.draw_indexed(0..6,0, 0..1); 
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

