use std::collections::HashMap;
use cgmath::Vector3;


pub struct IcoSphere {
    pub points:Vec<Vector3<f32>>,
    pub faces:Vec<Face>,
    pub frac: f32,
    pub frac2: f32,
    pub vertex_buffer: Option<wgpu::Buffer>,
    pub index_buffer: Option<wgpu::Buffer>,
    pub n_indices: u32,
    pub vertex_map: HashMap<(usize, usize),usize>,
}

#[repr(C)]
#[derive(Copy, Clone, Debug)]
struct Vertex {
    position: [f32; 3],
    tex_coords: [f32; 2],
    normal: [f32; 3],
}
#[derive(Copy, Clone, Debug)]
pub struct Face{
    i0: usize,
    i1: usize,
    i2: usize,
}
unsafe impl bytemuck::Pod for Vertex {}
unsafe impl bytemuck::Zeroable for Vertex {}

impl Face {
    pub fn new(a: usize, b:usize, c:usize) -> Self {
        Face{
            i0:a,
            i1:b,
            i2:c,
        }
    }
}
impl IcoSphere {
    pub fn new(frac:f32, frac2:f32) -> Self {
        let points = Vec::new();
        let faces = Vec::new();
        let map = HashMap::new();
        IcoSphere{
            points,
            faces,
            frac, 
            frac2,
            vertex_buffer:None,
            index_buffer:None,
            n_indices:0,
            vertex_map: map,
        }
    }
    pub fn make_geometry(&mut self, resolution:usize, half: bool) {
        println!("Tesselating the sphere...");
        //load the first vertices
        let t = (1.0 + 5.0_f32.sqrt())*0.5;
        
        self.add_vertex(Vector3::new(-1.0,  t,  0.0)); //0
        self.add_vertex(Vector3::new( 1.0,  t,  0.0)); //1
        self.add_vertex(Vector3::new(-1.0, -t,  0.0)); //2
        self.add_vertex(Vector3::new( 1.0, -t,  0.0)); //3

        self.add_vertex(Vector3::new( 0.0, -1.0,  t)); //4 //erase this for 
        self.add_vertex(Vector3::new( 0.0,  1.0,  t)); //5
        self.add_vertex(Vector3::new( 0.0, -1.0, -t)); //6 //erase this for half
        self.add_vertex(Vector3::new( 0.0,  1.0, -t)); //7

        self.add_vertex(Vector3::new( t,  0.0, -1.0));
        self.add_vertex(Vector3::new( t,  0.0,  1.0));
        self.add_vertex(Vector3::new(-t,  0.0, -1.0));
        self.add_vertex(Vector3::new(-t,  0.0,  1.0));

        //faces
        // 5 faces around point 0
        self.faces.push(Face::new(0, 11, 5));
        self.faces.push(Face::new(0, 5, 1));
        self.faces.push(Face::new(0, 1, 7));
        self.faces.push(Face::new(0, 7, 10));
        self.faces.push(Face::new(0, 10, 11));

        // 5 adjacent faces 
        self.faces.push(Face::new(1, 5, 9));
        if !half { 
            self.faces.push(Face::new(5, 11, 4));
        }

        self.faces.push(Face::new(11, 10, 2));
        if !half {
            self.faces.push(Face::new(10, 7, 6));
        }
        self.faces.push(Face::new(7, 1, 8));

        // 5 faces around point 3
        if !half {
            self.faces.push(Face::new(3, 9, 4));
            self.faces.push(Face::new(3, 4, 2));
            self.faces.push(Face::new(3, 2, 6));
            self.faces.push(Face::new(3, 6, 8));
        }
        self.faces.push(Face::new(3, 8, 9));

        // 5 adjacent faces 
        if !half {
            self.faces.push(Face::new(4, 9, 5));
            self.faces.push(Face::new(2, 4, 11));
            self.faces.push(Face::new(6, 2, 10));
            self.faces.push(Face::new(8, 6, 7));
        }
        self.faces.push(Face::new(9, 8, 1));
        
        //refine

        // refine triangles
        for _j in 0..resolution {
            //make a new list of faces
            let mut faces2 = Vec::new();
            for i in 0..self.faces.len() {
                let i0 = self.faces[i].i0;
                let i1 = self.faces[i].i1;
                let i2 = self.faces[i].i2;
                // replace triangle by 4 triangles
                let a = self.add_middle_point(i0, i1);
                let b = self.add_middle_point(i1, i2);
                let c = self.add_middle_point(i2, i0);

                faces2.push(Face::new(i0, a, c));
                faces2.push(Face::new(i1, b, a));
                faces2.push(Face::new(i2, c, b));
                faces2.push(Face::new(a, b, c));
            }
            self.faces = faces2;
        }
        println!("done");
        println!("Number of vertices: {}", self.points.len());
        println!("Number of faces: {}", self.faces.len());

    }
    pub fn add_middle_point(&mut self, i0:usize, i1:usize) -> usize {
        //do we have it already
        let key = ( i0.min(i1), i0.max(i1));
        if self.vertex_map.contains_key(&key) {
            *self.vertex_map.get(&key).unwrap()
        } else {
            let pt = (self.points[i0]+self.points[i1])*0.5;
            let new_index = self.add_vertex(pt);
            self.vertex_map.insert(key,new_index);
            new_index
        }
    }
    pub fn add_vertex(&mut self, p:Vector3<f32>) -> usize {
        let mut pt = p.clone();
        let length = (p.x*p.x+ p.y*p.y+ p.z*p.z).sqrt();
        pt.x/=length;
        pt.y/=length;
        pt.z/=length;
        self.points.push(pt);
        self.points.len()-1
    }

    pub fn make_buffers(&mut self, device: &wgpu::Device, scale:f32) {

        //make the vertices
        let mut result = Vec::new();
        for i in 0..self.points.len() {
            let p = self.points[i].clone();
            let vertex = Vertex{
                position: [p.x*scale, p.y*scale, p.z*scale],
                tex_coords: [self.frac, self.frac],
                normal:[p.x, p.y, p.z], 
            };
            result.push(vertex);
        }
        
        let mut indices = Vec::new();
        for face in self.faces.iter() {
            if face.i2 > result.len() {
                println!("uhoh");
            }
            indices.push(face.i0 as u32);
            indices.push(face.i1 as u32);
            indices.push(face.i2 as u32);
        }
        self.n_indices = indices.len() as u32;

        //if you have enough memory to do the above
        let vertex_buffer = device.create_buffer_with_data(
            bytemuck::cast_slice(&result),
            wgpu::BufferUsage::VERTEX
        );
        self.vertex_buffer = Some(vertex_buffer);

        let index_buffer = device.create_buffer_with_data(
            bytemuck::cast_slice(&indices),
            wgpu::BufferUsage::INDEX
        );
        self.index_buffer = Some(index_buffer);


    }
}

