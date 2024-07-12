use image::{ImageBuffer, DynamicImage};
use log::warn;
use serde_derive::{Serialize, Deserialize};
use wgpu::{Device, Buffer, util::DeviceExt, BindGroup, Queue, Sampler, BindGroupLayout};
use crate::{pipeline::model_pipeline, texture::TextureData};

pub const VIEW_MATRIX_ELEMENT_SIZE: u64 = 4;
pub const MAX_MODEL_AMOUNT: u64 = 100000;

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ModelData {
    pub name: String,
	pub meshes: Vec<MeshData>,
    pub textures: Vec<TextureData>
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
	pub bitangents: Vec<[f32; 3]>,
    pub texture_coordinates: Vec<[f32; 2]>,
    pub pbr_specular_glossiness_diffuse_texture_index: Option<usize>,
    pub pbr_specular_glossiness_texture_index: Option<usize>,
    pub base_color_texture_index: Option<usize>,
    pub metallic_roughness_texture_index: Option<usize>,
    pub normal_texture_index: Option<usize>,
    pub occlusion_texture_index: Option<usize>,
    pub emmisive_texture_index: Option<usize>
}

pub struct Mesh {
    pub name: String,
    pub primitives: Vec<Primitive>
}

pub struct Primitive {
    pub name: String,
    pub vertex_buffer: wgpu::Buffer,
    pub index_buffer: wgpu::Buffer,
    pub num_elements: u32,
    pub material: usize,
    pub pbr_specular_glossiness_diffuse_texture_index: Option<usize>,
    pub pbr_specular_glossiness_texture_index: Option<usize>,
    pub base_color_texture_index: Option<usize>,
    pub metallic_roughness_texture_index: Option<usize>,
    pub normal_texture_index: Option<usize>,
    pub occlusion_texture_index: Option<usize>,
    pub emmisive_texture_index: Option<usize>
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
    pub scale_x: f32,
    pub scale_y: f32,
    pub scale_z: f32
}

pub struct Object {
    pub name: String,
    pub meshes: Vec<Mesh>,
    pub instance_buffer: Buffer,
    pub view_sources: Vec<ViewSource>,
    pub texture_bind_groups: Vec<BindGroup>,
    pub views: Vec<f32>,
    pub views_size: u64,
    pub views_amount: u32
}

impl Object {
    pub fn update_view(&mut self) {
        self.views.clear();

        for view_source in &self.view_sources {
            let view_matrix = generate_view_matrix(view_source);
            self.views.extend_from_slice(&view_matrix.to_cols_array());
        }

        self.views_size = self.views.len() as u64 * VIEW_MATRIX_ELEMENT_SIZE;
        self.views_amount = self.view_sources.len() as u32;
    }
}

pub fn generate_view_matrix(source: &ViewSource) -> glam::Mat4 {    
    let translation = glam::Mat4::from_translation(glam::Vec3::new(source.x, source.y, source.z));
    let scale = glam::Mat4::from_scale(glam::Vec3::new(source.scale_x, source.scale_y, source.scale_z));

    translation * scale
}

pub fn create_object(device: &Device, queue: &Queue, texture_bind_group_layout: &BindGroupLayout, sampler: &Sampler, name: &str, model_data: ModelData, view_sources: Vec<ViewSource>) -> Object {
    let mut meshes = vec![];

    for mesh in model_data.meshes {
        let mut primitives = vec![];

        for primitive in mesh.primitives {
            let mut index = 0;
            let mut vertices = vec![];	        

            for position in primitive.positions {
                vertices.push(model_pipeline::Vertex {
                    position,
                    texture_coordinates: primitive.texture_coordinates[index],
                    normal: [1.0, 1.0, 1.0]                   
                });

                index = index + 1;
            }            

            let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(&format!("{} Vertex Buffer", "dog")),
                contents: bytemuck::cast_slice(&vertices),
                usage: wgpu::BufferUsages::VERTEX,
            });
    
            let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(&format!("{} Index Buffer", "dog")),
                contents: bytemuck::cast_slice(&primitive.indices),
                usage: wgpu::BufferUsages::INDEX,
            });
    
            primitives.push(Primitive {
                name: "".to_owned(),
                vertex_buffer,
                index_buffer,
                num_elements: primitive.indices.len() as u32,
                material: 0,
                pbr_specular_glossiness_diffuse_texture_index: primitive.pbr_specular_glossiness_diffuse_texture_index,
                pbr_specular_glossiness_texture_index: primitive.pbr_specular_glossiness_texture_index,
                base_color_texture_index: primitive.base_color_texture_index,
                metallic_roughness_texture_index: primitive.metallic_roughness_texture_index,
                normal_texture_index: primitive.normal_texture_index,
                occlusion_texture_index: primitive.occlusion_texture_index,
                emmisive_texture_index: primitive.emmisive_texture_index
            });
        }

        meshes.push(Mesh {
            name: "".to_owned(),
            primitives
        });
    }        

    let instance_buffer  = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Instance Buffer"),
        size: VIEW_MATRIX_ELEMENT_SIZE * MAX_MODEL_AMOUNT,
        usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false
    });

    let mut texture_bind_groups = vec![];    

    for texture_item in model_data.textures {
        let index_str = texture_item.index.to_string();        

        let texture_image = match texture_item.format.as_ref() {
            "R8G8B8A8" => {
                let image_buffer = ImageBuffer::from_raw(texture_item.width, texture_item.height, texture_item.pixels.expect("Texture pixels are empty")).expect("Failed to create image buffer");
                DynamicImage::ImageRgba8(image_buffer)
            }
            _ => {
                let image_buffer = ImageBuffer::from_raw(texture_item.width, texture_item.height, texture_item.pixels.expect("Texture pixels are empty")).expect("Failed to create image buffer");                
                DynamicImage::ImageRgb8(image_buffer)
            }
        };

        let texture = crate::texture::Texture::from_image(&device, &queue, &texture_image, Some(&("texture_".to_owned() + &index_str))).expect("Failed to create texture");        

        let texture_bind_group = device.create_bind_group(
            &wgpu::BindGroupDescriptor {
                layout: texture_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&texture.view),
                        //resource: wgpu::BindingResource::TextureViewArray(&textures.iter().map(|v| &v.view).collect::<Vec<&TextureView>>())
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(sampler)
                    }
                ],
                label: Some(&("texture_bind_group_".to_owned() + &index_str)),
            }
        );

        texture_bind_groups.push(texture_bind_group);        
    }

    let mut views = vec![];

    for view_source in &view_sources {
        let view_matrix = generate_view_matrix(view_source);
        views.extend_from_slice(&view_matrix.to_cols_array());
    }

    let views_size = views.len() as u64 * VIEW_MATRIX_ELEMENT_SIZE;

    let views_amount = view_sources.len() as u32;

    Object {
        name: name.to_owned(),
        meshes,        
        instance_buffer,
        texture_bind_groups,
        view_sources,
        views,
        views_size,
        views_amount
    }
}
