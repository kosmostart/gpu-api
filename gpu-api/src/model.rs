use glam::Mat4;
use image::{ImageBuffer, DynamicImage};
use wgpu::{Device, Buffer, util::DeviceExt, BindGroup, Queue, Sampler, BindGroupLayout};
use gpu_api_dto::{Animation, AnimationProperty, Interpolation, ModelData, Node, Skin, ViewSource};
use crate::{model_instance::ModelInstance, pipeline::model_pipeline};

pub const INSTANCE_SIZE: u64 = 64;
pub const MAX_MODEL_AMOUNT: u64 = 100000;

pub struct Object {
    pub name: String,
    pub nodes: Vec<ModelNode>,
    pub skins: Vec<Skin>,
    pub meshes: Vec<Mesh>,
    pub instance_buffer: Buffer,
    pub instances: Vec<ObjectInstance>,
    pub model_matrices: Vec<Mat4>,
    pub texture_bind_groups: Vec<BindGroup>,
    pub views: Vec<ModelInstance>,
    pub views_size: u64,
    pub instances_amount: u32    
}

pub struct ModelNode {
    pub index: usize,
    pub name: Option<String>,
    pub translation: glam::Vec3,
    pub rotation: glam::Quat,
    pub scale: glam::Vec3,
    pub local_transform: Mat4
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

pub struct ObjectGroup {
    pub active: bool,
    pub objects: Vec<Object>
}

pub struct ModelAnimationsGroup {
    pub active: bool,
    pub model_animations: Vec<ModelAnimations>
}

pub struct ObjectInstance {
    pub view_source: ViewSource,
    pub is_moving: bool,
    pub move_target: glam::Vec3,
    pub move_direction_normalized: glam::Vec3,
    pub x_move_done: bool,
    pub y_move_done: bool,
    pub z_move_done: bool,
    pub bounding_box: BoundingBox
}

pub struct BoundingBox {
    pub box0: glam::Vec3,
    pub box1: glam::Vec3
}

pub struct ModelAnimations {
    pub model_animations: Vec<ModelAnimation>
}

pub struct ModelAnimation {
    pub name: String,
    pub channels: Vec<ModelAnimationChannel>
}

pub struct ModelAnimationChannel {
    pub target_index: usize,
    pub property: AnimationProperty,
    pub interpolation: Interpolation,
    pub timestamps: Vec<f32>,
    pub translations: Vec<glam::Vec3>,
    pub rotations: Vec<glam::Quat>,
    pub scales: Vec<glam::Vec3>,
    pub weight_morphs: Vec<f32>,
    pub frame_index: usize,
    #[cfg(not(target_arch = "wasm32"))]
    pub start_instant: std::time::Instant,
    #[cfg(target_arch = "wasm32")]
    pub start_instant: web_time::Instant
}

impl Object {
    pub fn update_view(&mut self) {
        self.views.clear();
        self.model_matrices.clear();

        for instance in &self.instances {
            let model_matrix = generate_model_matrix(&instance.view_source);
            self.views.push(ModelInstance {
                model_matrix: model_matrix.to_cols_array()                
            });
            self.model_matrices.push(model_matrix);
        }

        self.views_size = self.views.len() as u64 * INSTANCE_SIZE;
        self.instances_amount = self.instances.len() as u32;
    }

    pub fn update_view_with_translation(&mut self, translation: &[f32; 3]) {
        self.views.clear();
        self.model_matrices.clear();

        let translation_matrix = glam::Mat4::from_translation(glam::Vec3::new(translation[0], translation[1], translation[2]));        

        for instance in &self.instances {
            let model_matrix = translation_matrix * generate_model_matrix(&instance.view_source);
            self.views.push(ModelInstance {
                model_matrix: model_matrix.to_cols_array()                
            });
            self.model_matrices.push(model_matrix);
        }

        self.views_size = self.views.len() as u64 * INSTANCE_SIZE;
        self.instances_amount = self.instances.len() as u32;
    }

    pub fn update_view_with_rotation(&mut self, rotation: &[f32; 4]) {
        self.views.clear();
        self.model_matrices.clear();

        let rotation_matrix = glam::Mat4::from_quat(glam::quat(rotation[0], rotation[1], rotation[2], rotation[3]));

        for instance in &self.instances {
            let model_matrix = rotation_matrix * generate_model_matrix(&instance.view_source);
            self.views.push(ModelInstance {
                model_matrix: model_matrix.to_cols_array()                
            });
            self.model_matrices.push(model_matrix);
        }

        self.views_size = self.views.len() as u64 * INSTANCE_SIZE;
        self.instances_amount = self.instances.len() as u32;
    }

    pub fn update_view_with_scale(&mut self, scale: &[f32; 3]) {
        self.views.clear();
        self.model_matrices.clear();

        let scale_matrix = glam::Mat4::from_scale(glam::Vec3::new(scale[0], scale[1], scale[2]));

        for instance in &self.instances {
            let model_matrix = scale_matrix * generate_model_matrix(&instance.view_source);
            self.views.push(ModelInstance {
                model_matrix: model_matrix.to_cols_array()                
            });
            self.model_matrices.push(model_matrix);
        }

        self.views_size = self.views.len() as u64 * INSTANCE_SIZE;
        self.instances_amount = self.instances.len() as u32;
    }
}

pub fn generate_model_matrix(source: &ViewSource) -> glam::Mat4 {
    let translation = glam::Mat4::from_translation(glam::Vec3::new(source.x, source.y, source.z));
    let scale = glam::Mat4::from_scale(glam::Vec3::new(source.scale_x, source.scale_y, source.scale_z));    

    translation * scale
}

pub fn create_object(device: &Device, queue: &Queue, texture_bind_group_layout: &BindGroupLayout, sampler: &Sampler, model_data: ModelData, loaded_images: Option<Vec<DynamicImage>>, view_sources: Vec<ViewSource>) -> (Object, ModelAnimations) {
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
                    normal: primitive.normals[index],
                    tangent: primitive.tangents[index],
                    joints: primitive.joints[index],
                    weights: primitive.weights[index]
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
        size: INSTANCE_SIZE * MAX_MODEL_AMOUNT,
        usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false
    });

    let mut texture_bind_groups = vec![];

    match loaded_images {
        Some(loaded_images) => {
            for texture_item in model_data.textures {
                let index_str = texture_item.index.to_string();                            
        
                let texture = crate::texture::Texture::from_image(&device, &queue, &loaded_images[texture_item.index], Some(&("texture_".to_owned() + &index_str))).expect("Failed to create texture");
        
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
        }
        None => {
            for texture_item in model_data.textures {
                let index_str = texture_item.index.to_string();

                log::warn!("{} {}", model_data.name, texture_item.image_encoded.as_ref().unwrap().len());

                let texture_image = image::load_from_memory_with_format(texture_item.image_encoded.as_ref().expect("Image encoded is empty"), image::ImageFormat::Jpeg).expect("Failed to load texture");
        
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
        }
    }    

    let mut views = vec![];
    let mut model_matrices = vec![];
    let mut instances = vec![];

    for view_source in view_sources {
        let model_matrix = generate_model_matrix(&view_source);        

        views.push(ModelInstance {
            model_matrix: model_matrix.to_cols_array()            
        });

        model_matrices.push(model_matrix);

        instances.push(ObjectInstance {
            view_source,
            is_moving: false,
            move_target: glam::vec3(0.0, 0.0, 0.0),
            move_direction_normalized: glam::vec3(0.0, 0.0, 0.0),
            x_move_done: false,
            y_move_done: false,
            z_move_done: false,
            bounding_box: BoundingBox { 
                box0: glam::vec3(0.0, 0.0, 0.0),
                box1: glam::vec3(0.0, 0.0, 0.0)
            }
        });
    }

    let views_size = views.len() as u64 * INSTANCE_SIZE;

    let instances_amount = instances.len() as u32;

    let mut model_animations = vec![];

    for animation in model_data.animations {
        let mut channels = vec![];

        for channel in animation.channels {
            let translations = channel.translations.into_iter().map(|v| glam::Vec3::from_array(v)).collect();
            let rotations = channel.rotations.into_iter().map(|v| glam::Quat::from_array(v)).collect();
            let scales = channel.scales.into_iter().map(|v| glam::Vec3::from_array(v)).collect();

            channels.push(ModelAnimationChannel {
                target_index: channel.target_index,
                property: channel.property,
                interpolation: channel.interpolation,
                timestamps: channel.timestamps,
                translations,
                rotations,
                scales,
                weight_morphs: channel.weight_morphs,
                frame_index: 0,
                #[cfg(not(target_arch = "wasm32"))]
                start_instant: std::time::Instant::now(),
                #[cfg(target_arch = "wasm32")]
                start_instant: web_time::Instant::now()
            });
        }

        model_animations.push(ModelAnimation {
            name: animation.name,
            channels            
        });
    }

    let mut nodes = vec![];

    for node in model_data.nodes {
        let local_transform = Mat4::from_cols_array_2d(&node.local_transform_matrix);
        
        nodes.push(ModelNode {
            index: node.index,
            name: node.name.map(|v| v.to_owned()),
            translation: glam::Vec3::from_array(node.translation),
            rotation: glam::Quat::from_array(node.rotation),
            scale: glam::Vec3::from_array(node.scale),
            local_transform: local_transform
        });
    }

    (Object {
        name: model_data.name,
        nodes,
        skins: model_data.skins,
        meshes,
        instance_buffer,
        texture_bind_groups,
        instances,
        model_matrices,
        views,
        views_size,
        instances_amount        
    },
    ModelAnimations {
        model_animations
    })
}

