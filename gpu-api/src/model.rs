use log::warn;
use serde_derive::{Serialize, Deserialize};
use wgpu::{Device, Buffer, util::DeviceExt};
use crate::pipeline::model_pipeline;

pub const VIEW_MATRIX_SIZE: u64 = 64;
pub const VIEW_MATRIX_LEN: u64 = 16;
pub const MAX_MODEL_AMOUNT: u64 = 100000;

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ModelData {
	pub meshes: Vec<MeshData>
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct MeshData {
	pub primitives: Vec<PrimitiveData>
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct PrimitiveData {
	pub positions: Vec<[f32; 3]>,
	pub indices: Vec<u32>,
	pub normals: Vec<[f32; 3]>,
	pub tangents: Vec<[f32; 3]>,
	pub bitangents: Vec<[f32; 3]>
}

pub struct Mesh {
    pub name: String,
    pub vertex_buffer: wgpu::Buffer,
    pub index_buffer: wgpu::Buffer,
    pub num_elements: u32,
    pub material: usize,
}

pub struct Material {
    pub name: String,
    pub diffuse_texture: crate::texture::Texture,
    pub bind_group: wgpu::BindGroup,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ViewSource {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub angle_xz: f32,
    pub angle_y: f32,
    pub dist: f32,
    pub scale_x: f32,
    pub scale_y: f32,
    pub scale_z: f32
}

pub struct Object {
    pub name: String,
    pub meshes: Vec<Mesh>,
    pub instance_buffer: Buffer,
    pub view_source: ViewSource,
    pub view: glam::Mat4
}

pub fn generate_view_matrix(source: &ViewSource) -> glam::Mat4 {
    let cam_pos = glam::Vec3::new(
        source.angle_xz.cos() * source.angle_y.sin() * source.dist,
        source.angle_xz.sin() * source.dist + source.y,
        source.angle_xz.cos() * source.angle_y.cos() * source.dist
    );

    let view = glam::Mat4::look_at_rh(
        cam_pos,
        glam::Vec3::new(source.x, source.y, source.z),
        glam::Vec3::new(0.0, 1.0, 0.0)
    );

    let scale = glam::Mat4::from_scale(glam::Vec3::new(source.scale_x, source.scale_y, source.scale_z));

    view * scale
}

pub fn generate_view_matrix_orig() -> glam::Mat4 {
    let angle_xz = 0.2f32;
    let angle_y = 0.2f32;
    let dist = 20.0;
    let model_center_x = 5.0;
    let model_center_y = 7.0;
    
    let cam_pos = glam::Vec3::new(
        angle_xz.cos() * angle_y.sin() * dist,
        angle_xz.sin() * dist + model_center_y,
        angle_xz.cos() * angle_y.cos() * dist
    );

    let view = glam::Mat4::look_at_rh(
        cam_pos,
        glam::Vec3::new(model_center_x, model_center_y, 0.0),
        glam::Vec3::new(0.0, 1.0, 0.0)
    );

    let scale = glam::Mat4::from_scale(glam::Vec3::new(0.02, 0.02, 0.02));
    //let scale = glam::Mat4::from_scale(glam::Vec3::new(0.2, 0.2, 0.2));

    view * scale
}

pub fn create_object(device: &Device, name: &str, model_data: ModelData, view_source: ViewSource) -> Object {    
    let mut meshes = vec![];

    for mesh in model_data.meshes {
        let mut vertices = vec![];
	    let mut indices = vec![];

        for mut primitive in mesh.primitives {
            for position in primitive.positions {
                vertices.push(model_pipeline::Vertex {
                    position,
                    texture_coordinates: [0.0, 0.0],
                    normal: [1.0, 1.0, 1.0]                   
                });
            }

            indices.append(&mut primitive.indices);
        }

        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(&format!("{} Vertex Buffer", "dog")),
            contents: bytemuck::cast_slice(&vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });

        let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(&format!("{} Index Buffer", "dog")),
            contents: bytemuck::cast_slice(&indices),
            usage: wgpu::BufferUsages::INDEX,
        });

        meshes.push(Mesh {
            name: "".to_owned(),
            vertex_buffer,
            index_buffer,
            num_elements: indices.len() as u32,
            material: 0
        })
    }        

    let instance_buffer  = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Instance Buffer"),
        size: VIEW_MATRIX_LEN * MAX_MODEL_AMOUNT,
        usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false
    });

    let view = generate_view_matrix(&view_source);

    Object {
        name: name.to_owned(),
        meshes,        
        instance_buffer,
        view_source,
        view
    }
}