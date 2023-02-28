use serde_derive::{Serialize, Deserialize};
use wgpu::{Device, Buffer, util::DeviceExt};
use crate::{pipeline, instance::{Instance, InstanceRaw}};

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

pub struct Object {
    pub name: String,
    pub meshes: Vec<Mesh>,
    pub instances: Vec<Instance>,
    pub instance_data: Vec<InstanceRaw>,
    pub instance_buffer: Buffer,
    pub x: f32,
    pub y: f32,
    pub z: f32
}

impl Object {
    pub fn set_instance_buffer(&mut self, device: &Device) {
        let instances = crate::instance::create_one(self.x, self.y, self.z);
        let instance_data = instances.iter().map(crate::instance::Instance::to_raw).collect::<Vec<_>>();
        let instance_buffer = device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("Instance Buffer"),
                contents: bytemuck::cast_slice(&instance_data),
                usage: wgpu::BufferUsages::VERTEX,
            }
        );

        self.instance_buffer = instance_buffer;
    }
}

pub fn create_model(device: &Device, name: &str, model_data: ModelData, x: f32, y: f32, z: f32) -> Object {
    let mut meshes = vec![];

    for mesh in model_data.meshes {
        let mut vertices = vec![];
	    let mut indices = vec![];

        for mut primitive in mesh.primitives {
            for position in primitive.positions {
                vertices.push(pipeline::Vertex {
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

    let instances = crate::instance::create_one(x, y, z);
    let instance_data = instances.iter().map(crate::instance::Instance::to_raw).collect::<Vec<_>>();
    let instance_buffer = device.create_buffer_init(
        &wgpu::util::BufferInitDescriptor {
            label: Some("Instance Buffer"),
            contents: bytemuck::cast_slice(&instance_data),
            usage: wgpu::BufferUsages::VERTEX,
        }
    );

    Object {
        name: name.to_owned(),
        meshes,
        instances,
        instance_data,
        instance_buffer,
        x,
        y,
        z
    }
}
