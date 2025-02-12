use std::num::NonZeroU32;
use std::borrow::Cow;
use log::*;
use wgpu::{Device, Surface, Adapter, Queue, RenderPipeline, Buffer, BindGroup, ShaderModule, BindGroupLayout, PipelineLayout, TextureFormat, RenderPass, Sampler, TextureView};
use wgpu::util::DeviceExt;
use crate::camera::{Camera, create_camera};
use crate::model::{Object, ObjectGroup};
use crate::texture::Texture;

pub const JOINT_MATRICES_AMOUNT: usize = 100;
pub const JOINT_MATRICES_UNIFORM_SIZE: u64 = 6400;

#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct Vertex {    
    pub position: [f32; 3],    
    pub texture_coordinates: [f32; 2],
    pub normal: [f32; 3],
    pub tangent: [f32; 4],
    pub joints: [u32; 4],
    pub weights: [f32; 4]
}

unsafe impl bytemuck::Pod for Vertex {}
unsafe impl bytemuck::Zeroable for Vertex {}

pub struct Pipeline {
    pub shader: ShaderModule,
    pub texture_bind_group_layout: BindGroupLayout,
    pub sampler: Sampler,
    pub depth_texture: Texture,
    pub depth_sampler: Sampler,
    pub camera_buffer: Buffer,
    pub camera_bind_group_layout: BindGroupLayout,
    pub camera_bind_group: BindGroup,    
    pub joint_matrices_bind_group_layout: BindGroupLayout,    
    pub pipeline_layout: PipelineLayout,
    pub swapchain_format: TextureFormat,
    pub render_pipeline: RenderPipeline
}

impl Pipeline {
    pub fn draw<'a>(&'a self, render_pass: &mut RenderPass<'a>, object_groups: &'a Vec<ObjectGroup>) {
        render_pass.set_pipeline(&self.render_pipeline);

        for object_group in object_groups {
            if object_group.active == false {
                continue;
            }
            
            for object in &object_group.objects {
                if object.instances_amount == 0 {
                    continue;
                }
                
                render_pass.set_vertex_buffer(1, object.instance_buffer.slice(..)); // Instances
                        
                let instances_range = 0..object.instances_amount;
                
                for mesh in &object.meshes {
                    for primitive in &mesh.primitives {
                        match primitive.base_color_texture_index {
                            Some(base_color_texture_index) => {
                                render_pass.set_bind_group(0, &object.texture_bind_groups[base_color_texture_index], &[]); // Texture
                            }
                            None => {
                                match primitive.pbr_specular_glossiness_diffuse_texture_index {
                                    Some(pbr_specular_glossiness_diffuse_texture_index) => {
                                        render_pass.set_bind_group(0, &object.texture_bind_groups[pbr_specular_glossiness_diffuse_texture_index], &[]); // Texture                            
                                    }
                                    None => {}
                                }
                            }
                        }
                        /*
                        match primitive.base_color_texture_index {
                            Some(base_color_texture_index) => {
                                render_pass.set_bind_group(0, &object.texture_bind_groups[base_color_texture_index], &[]); // Texture
                            }
                            None => {}
                        }
                        match primitive.metallic_roughness_texture_index {
                            Some(metallic_roughness_texture_index) => {
                                render_pass.set_bind_group(0, &object.texture_bind_groups[metallic_roughness_texture_index], &[]); // Texture
                            }
                            None => {}
                        }
                        match primitive.normal_texture_index {
                            Some(normal_texture_index) => {
                                render_pass.set_bind_group(0, &object.texture_bind_groups[normal_texture_index], &[]); // Texture                            
                            }
                            None => {}
                        }
                        match primitive.occlusion_texture_index {
                            Some(occlusion_texture_index) => {
                                render_pass.set_bind_group(0, &object.texture_bind_groups[occlusion_texture_index], &[]); // Texture
                            }
                            None => {}
                        }
                        match primitive.emmisive_texture_index {
                            Some(emmisive_texture_index) => {
                                render_pass.set_bind_group(0, &object.texture_bind_groups[emmisive_texture_index], &[]); // Texture
                            }
                            None => {}
                        }
                        */
                        render_pass.set_bind_group(1, &self.camera_bind_group, &[]); // Camera
                        render_pass.set_bind_group(2, &object.joint_matrices_bind_group, &[]); // Joint matrices
                        render_pass.set_vertex_buffer(0, primitive.vertex_buffer.slice(..));
                        render_pass.set_index_buffer(primitive.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
                        render_pass.draw_indexed(0..primitive.num_elements, 0, instances_range.clone());
                    }
                }                    
            }
        }            
    }
}

pub async fn new(device: &Device, config: &wgpu::SurfaceConfiguration, width: f32, height: f32, depth_stencil: Option<wgpu::DepthStencilState>) -> (Camera, Pipeline) {
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Shader2"),
        source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("model.wgsl")))
    });    

    let texture_bind_group_layout = device.create_bind_group_layout(        
        &wgpu::BindGroupLayoutDescriptor {
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            multisampled: false,
                            view_dimension: wgpu::TextureViewDimension::D2,
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        },
                        count: None
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,                        
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                ],
                label: Some("texture_bind_group_layout"),
            });

    let sampler = device.create_sampler(&wgpu::SamplerDescriptor {            
        address_mode_u: wgpu::AddressMode::Repeat,
        address_mode_v: wgpu::AddressMode::Repeat,
        address_mode_w: wgpu::AddressMode::Repeat,            
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        mipmap_filter: wgpu::FilterMode::Linear,
        ..Default::default()
    });    

    let camera = create_camera(width, height, 0.0, 0.0, 0.0);
    let camera_projection_matrix_ref: &[f32; 16] = camera.projection.as_ref();

    let camera_buffer = device.create_buffer_init(
        &wgpu::util::BufferInitDescriptor {
            label: Some("Camera Buffer"),
            contents: bytemuck::cast_slice(camera_projection_matrix_ref),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        }
    );

    let camera_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }
        ],
        label: Some("camera_bind_group_layout"),
    });

    let camera_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        layout: &camera_bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: camera_buffer.as_entire_binding(),
            }
        ],
        label: Some("camera_bind_group")
    });    

    let joint_matrices_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }
        ],
        label: Some("joint_matrices_group_layout"),
    });    

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("Pipeline Layout"),
        bind_group_layouts: &[
            &texture_bind_group_layout,
            &camera_bind_group_layout,
            &joint_matrices_bind_group_layout
        ],
        push_constant_ranges: &[]
    });    

    let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        multiview: None,
        label: Some("Model pipeline"),
        layout: Some(&pipeline_layout),        
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: Some("vs_main"),
            compilation_options: Default::default(),
            buffers: &[
                wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<Vertex>() as wgpu::BufferAddress,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &[                        
                        wgpu::VertexAttribute {
                            offset: 0,
                            shader_location: 0,
                            format: wgpu::VertexFormat::Float32x3,
                        },
                        wgpu::VertexAttribute {
                            offset: (std::mem::size_of::<[f32; 3]>()) as wgpu::BufferAddress,
                            shader_location: 1,
                            format: wgpu::VertexFormat::Float32x2
                        },
                        wgpu::VertexAttribute {
                            offset: (
                                std::mem::size_of::<[f32; 3]>() + 
                                std::mem::size_of::<[f32; 2]>()
                            ) as wgpu::BufferAddress,
                            shader_location: 2,
                            format: wgpu::VertexFormat::Float32x3
                        },
                        wgpu::VertexAttribute {
                            offset: (
                                std::mem::size_of::<[f32; 3]>() + 
                                std::mem::size_of::<[f32; 2]>() +
                                std::mem::size_of::<[f32; 3]>()
                            ) as wgpu::BufferAddress,
                            shader_location: 3,
                            format: wgpu::VertexFormat::Float32x4
                        },
                        wgpu::VertexAttribute {
                            offset: (
                                std::mem::size_of::<[f32; 3]>() + 
                                std::mem::size_of::<[f32; 2]>() +
                                std::mem::size_of::<[f32; 3]>() + 
                                std::mem::size_of::<[f32; 4]>()
                            ) as wgpu::BufferAddress,
                            shader_location: 4,
                            format: wgpu::VertexFormat::Uint32x4
                        },
                        wgpu::VertexAttribute {
                            offset: (
                                std::mem::size_of::<[f32; 3]>() + 
                                std::mem::size_of::<[f32; 2]>() +
                                std::mem::size_of::<[f32; 3]>() + 
                                std::mem::size_of::<[f32; 4]>() + 
                                std::mem::size_of::<[u32; 4]>()
                            ) as wgpu::BufferAddress,
                            shader_location: 5,
                            format: wgpu::VertexFormat::Float32x4
                        }
                    ]
                },
                crate::model_instance::ModelInstance::vertex_buffer_layout()
            ]
        },
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: Some("fs_main"),
            compilation_options: Default::default(),
            targets: &[
                Some(wgpu::ColorTargetState {
                    format: TextureFormat::Rgba8UnormSrgb,
                    blend: Some(wgpu::BlendState {
                        color: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::SrcAlpha,
                            dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                            operation: wgpu::BlendOperation::Add,
                        },
                        alpha: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::One,
                            dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                            operation: wgpu::BlendOperation::Add,
                        },
                    }),
                    write_mask: wgpu::ColorWrites::ALL,
                })
            ],
        }),
        primitive: wgpu::PrimitiveState::default(),
        depth_stencil,
        multisample: wgpu::MultisampleState::default(),
        cache: None
    });

    let depth_texture = Texture::create_depth_texture(device, config, "Depth texture");
    let depth_sampler = Texture::create_depth_samper(device);

    (camera, Pipeline {
        shader,
        texture_bind_group_layout,                
        sampler,
        depth_texture,
        depth_sampler,
        camera_buffer,
        camera_bind_group_layout,
        camera_bind_group,
        joint_matrices_bind_group_layout,        
        pipeline_layout,
        swapchain_format: TextureFormat::Rgba8UnormSrgb,
        render_pipeline      
    })
}
