use log::*;
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
    pub node_topological_sorting: Vec<usize>,
    pub node_map: std::collections::BTreeMap<usize, usize>,
    pub skins: Vec<ModelSkin>,
    pub meshes: Vec<Mesh>,
    pub animations: Vec<ModelAnimation>,
    pub instance_buffer: Buffer,
    pub instances: Vec<ObjectInstance>,
    pub model_matrices: Vec<Mat4>,
    pub texture_bind_groups: Vec<BindGroup>,
    pub joint_matrices_buffer: Buffer,    
    pub joint_matrices_bind_group: BindGroup,
    pub views: Vec<ModelInstance>,
    pub views_size: u64,
    pub instances_amount: u32    
}

pub struct ModelNode {
    pub index: usize,
    pub name: Option<String>,
    pub skin_index: Option<usize>,
    pub translation: glam::Vec3,
    pub rotation: glam::Quat,
    pub scale: glam::Vec3,
    pub global_transform_matrix: Mat4
}

pub struct ModelSkin {
    pub name: Option<String>,    
    pub joints: Vec<ModelJoint>
}

pub struct ModelJoint {
    pub node_index: usize,
    pub node_name: Option<String>,
    pub inverse_bind_matrix: Mat4
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

pub struct ModelAnimation {
    pub name: String,
    pub channels: Vec<ModelAnimationChannel>,
    pub frame_index: usize,
    pub frame_cycle_count: usize,
    pub joint_matrices: Vec<[[f32; 16]; model_pipeline::JOINT_MATRICES_AMOUNT]>
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
    pub channel_time: f32,
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

pub fn create_object(device: &Device, queue: &Queue, pipeline: &model_pipeline::Pipeline, model_data: ModelData, loaded_images: Option<Vec<DynamicImage>>, view_sources: Vec<ViewSource>) -> Object {
    let mut meshes = vec![];    

    for mesh in model_data.meshes {
        let mut primitives = vec![];

        for primitive in mesh.primitives {
            let mut index = 0;
            let mut vertices = vec![];

            for position in primitive.positions {
                vertices.push(model_pipeline::Vertex {
                    position,
                    texture_coordinates: if primitive.texture_coordinates.is_empty() {[0.0, 1.0]} else {primitive.texture_coordinates[index]},
                    normal: if primitive.normals.is_empty() {[1.0, 1.0, 1.0]} else {primitive.normals[index]},
                    tangent: if primitive.tangents.is_empty() {[1.0, 1.0, 1.0, 1.0]} else {primitive.tangents[index]},
                    joints: if primitive.joints.is_empty() {[0, 0, 0, 0]} else {primitive.joints[index]},
                    weights: if primitive.weights.is_empty() {[1.0, 1.0, 1.0, 1.0]} else {primitive.weights[index]}
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
                        layout: &pipeline.texture_bind_group_layout,
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: wgpu::BindingResource::TextureView(&texture.view),
                                //resource: wgpu::BindingResource::TextureViewArray(&textures.iter().map(|v| &v.view).collect::<Vec<&TextureView>>())
                            },
                            wgpu::BindGroupEntry {
                                binding: 1,
                                resource: wgpu::BindingResource::Sampler(&pipeline.sampler)
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
                        layout: &pipeline.texture_bind_group_layout,
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: wgpu::BindingResource::TextureView(&texture.view),
                                //resource: wgpu::BindingResource::TextureViewArray(&textures.iter().map(|v| &v.view).collect::<Vec<&TextureView>>())
                            },
                            wgpu::BindGroupEntry {
                                binding: 1,
                                resource: wgpu::BindingResource::Sampler(&pipeline.sampler)
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

    let mut animations = vec![];

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
                channel_time: 0.0,
                #[cfg(not(target_arch = "wasm32"))]
                start_instant: std::time::Instant::now(),
                #[cfg(target_arch = "wasm32")]
                start_instant: web_time::Instant::now()
            });
        }

        animations.push(ModelAnimation {
            name: animation.name,
            channels,
            frame_index: 0,
            frame_cycle_count: 0,
            joint_matrices: vec![]
        });
    }

    let joint_matrices: [[f32; 16]; model_pipeline::JOINT_MATRICES_AMOUNT] = [[1.0; 16]; model_pipeline::JOINT_MATRICES_AMOUNT];

    let joint_matrices_ref: &[[f32; 16]] = joint_matrices.as_ref();

    let joint_matrices_buffer = device.create_buffer_init(
        &wgpu::util::BufferInitDescriptor {
            label: Some("Joint matrices Buffer"),
            contents: bytemuck::cast_slice(joint_matrices_ref),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        }
    );

    let joint_matrices_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        layout: &pipeline.joint_matrices_bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: joint_matrices_buffer.as_entire_binding(),
            }
        ],
        label: Some("joint_matrices_bind_group")
    });

    let mut nodes = vec![];

    for node in model_data.nodes {
        let local_transform = Mat4::from_cols_array_2d(&node.local_transform_matrix);
        
        nodes.push(ModelNode {
            index: node.index,
            name: node.name,
            skin_index: node.skin_index,
            translation: glam::Vec3::from_array(node.translation),
            rotation: glam::Quat::from_array(node.rotation),
            scale: glam::Vec3::from_array(node.scale),
            global_transform_matrix: local_transform
        });
    }

    let mut skins = vec![];

    for skin in model_data.skins {
        let mut joints = vec![];
        let mut index = 0;

        for joint in skin.joints {
            joints.push(ModelJoint {
                node_index: joint.node_index,
                node_name: joint.node_name,
                inverse_bind_matrix: Mat4::from_cols_array_2d(&skin.inverse_bind_matrices[index])
            });

            index = index + 1;
        }

        skins.push(ModelSkin { 
            name: skin.name, 
            joints
        });        
    }

    let time_per_frame = 1.0 / 200.0;

    for animation in &mut animations {        
        info!("Animation {} started", animation.name);
        let mut animation_time = 0.0;

        let mut max_animation_time = 0.0;         

        for channel in &animation.channels {
            let max_channel_time = *channel.timestamps.last().expect("Empty animation timestamps");
            //info!("Channel max time value: {}", max_channel_time);            

            if max_channel_time > max_animation_time {
                max_animation_time = max_channel_time;
            }
        }
            
        while animation_time < max_animation_time {
            for channel in &mut animation.channels {
                let mut frame_index = channel.frame_index;            
    
                for timestamp in channel.timestamps.iter().skip(frame_index) {
                    if timestamp > &channel.channel_time {
                        break;
                    }
                    
                    frame_index = frame_index + 1;
                }
                
                if frame_index == channel.timestamps.len() {
                    frame_index = 0;
                    channel.frame_index = 0;
                    channel.channel_time = 0.0;
                }                
    
                /*
                if frame_index == channel.timestamps.len() {
                    frame_index = 0;
                    channel.frame_index = 0;
                    #[cfg(not(target_arch = "wasm32"))] {
                        channel.start_instant = std::time::Instant::now();
                    }                                                    
                    #[cfg(target_arch = "wasm32")] {
                        channel.start_instant = web_time::Instant::now();
                    }
                }
                */
    
                let previous_frame_index = match frame_index {
                    0 => 0,
                    _ => frame_index - 1
                };                
    
                let factor = (animation_time - channel.timestamps[previous_frame_index]) / (channel.timestamps[frame_index] - channel.timestamps[previous_frame_index]);
    
                match &channel.property {
                    AnimationProperty::Translation => {
                        let translation = channel.translations[previous_frame_index].lerp(channel.translations[frame_index], factor);
                        nodes[channel.target_index].translation = translation;
                    }
                    AnimationProperty::Rotation => {                                                        
                        let rotation = channel.rotations[previous_frame_index].lerp(channel.rotations[frame_index], factor).normalize();
                        nodes[channel.target_index].rotation = rotation;
                    }
                    AnimationProperty::Scale => {
                        let scale = channel.scales[previous_frame_index].lerp(channel.scales[frame_index], factor);
                        nodes[channel.target_index].scale = scale;
                    }
                    AnimationProperty::MorphTargetWeights => {
                        let weight_morph = channel.weight_morphs[frame_index];
                    }
                }

                channel.channel_time = channel.channel_time + time_per_frame;                
            }

            for node_index in model_data.node_topological_sorting.iter() {
                match model_data.node_map.get(node_index) {
                    Some(parent_index) => {
                        let parent_transform = nodes[*parent_index].global_transform_matrix;
                        let node = &mut nodes[*node_index];
        
                        let local_transform = glam::Mat4::from_scale_rotation_translation(node.scale, node.rotation, node.translation);
                        node.global_transform_matrix = parent_transform * local_transform;
                    }
                    None => {}
                }                                            
            }                                    

            let mut joint_matrices: [[f32; 16]; model_pipeline::JOINT_MATRICES_AMOUNT] = [[1.0; 16]; model_pipeline::JOINT_MATRICES_AMOUNT];
            
            let mut joint_matrix_index = 0;
            let skin_index = 0;
            
            for joint in &skins[skin_index].joints {
                //let joint_matrix = inverse_node_global_transform * object.nodes[joint.node_index].global_transform_matrix * joint.inverse_bind_matrix;
                let joint_matrix = nodes[joint.node_index].global_transform_matrix * joint.inverse_bind_matrix;

                joint_matrices[joint_matrix_index] = joint_matrix.to_cols_array();

                joint_matrix_index = joint_matrix_index + 1;
            }                                    
            
            /*
            // Inverse node global transform

            for node in &object.nodes {
                match node.skin_index {
                    Some(skin_index) => {
                        //let inverse_node_global_transform = node.global_transform_matrix.inverse();

                        let mut joint_matrix_index = 0;

                        for joint in &object.skins[skin_index].joints {
                            //let joint_matrix = inverse_node_global_transform * object.nodes[joint.node_index].global_transform_matrix * joint.inverse_bind_matrix;
                            let joint_matrix = object.nodes[joint.node_index].global_transform_matrix * joint.inverse_bind_matrix;

                            joint_matrices[joint_matrix_index] = joint_matrix.to_cols_array();

                            joint_matrix_index = joint_matrix_index + 1;
                        }
                    }
                    None => {}
                }
            }
            */

            animation.joint_matrices.push(joint_matrices);                                    

            animation_time = animation_time + time_per_frame;
        }
        
        animation.frame_cycle_count = animation.joint_matrices.len();        

        info!("Animation {} done, joint matrices total: {}", animation.name, animation.joint_matrices.len());
    }

    Object {
        name: model_data.name,
        nodes,
        node_topological_sorting: model_data.node_topological_sorting,
        node_map: model_data.node_map,
        skins,
        meshes,
        animations,
        instance_buffer,
        texture_bind_groups,
        joint_matrices_buffer,
        joint_matrices_bind_group,
        instances,
        model_matrices,
        views,
        views_size,
        instances_amount        
    }
}
