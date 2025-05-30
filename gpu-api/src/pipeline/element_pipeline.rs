use std::{borrow::Cow, mem::size_of};
use wgpu::{Adapter, BindGroup, BindGroupLayout, DepthStencilState, Device, PipelineLayout, Queue, RenderPass, RenderPipeline, ShaderModule, Surface, TextureFormat};
use wgpu::util::DeviceExt;
use gpu_api_dto::image;
use crate::texture::Texture;

#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct Vertex {
    pub vertex_type: u32,
    pub position: [f32; 3],
    pub color: [f32; 4],
    pub element_coordinates: [f32; 4],
    pub has_element_border: u32,
    pub element_border_color: [f32; 4],
    pub component_coordinates: [f32; 4],
    pub texture_coordinates: [f32; 2],
    pub has_overlay: u32,    
    pub overlay_coordinates: [f32; 4]
}

unsafe impl bytemuck::Pod for Vertex {}
unsafe impl bytemuck::Zeroable for Vertex {}

pub struct Pipeline {
    pub shader: ShaderModule,
    pub vertex_buffer: wgpu::Buffer,
    pub index_buffer: wgpu::Buffer,
    pub texture: Texture,
    pub texture_bind_group_layout: BindGroupLayout,
    pub texture_bind_group: BindGroup,    
    pub pipeline_layout: PipelineLayout,    
    pub render_pipeline: RenderPipeline
}

impl Pipeline {
    pub fn draw<'a>(&'a self, render_pass: &mut RenderPass<'a>, indices_count: u32) {
        render_pass.set_pipeline(&self.render_pipeline);
    
        render_pass.set_bind_group(0, &self.texture_bind_group, &[]); // Texture
        render_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
        render_pass.set_index_buffer(self.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
        render_pass.draw_indexed(0..indices_count, 0, 0..1);
    }
}

pub fn new(device: &Device, queue: &Queue, vertices: &Vec<Vertex>, indices: &Vec<u32>, depth_stencil: Option<DepthStencilState>) -> Pipeline {
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Shader1"),
        source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("element.wgsl")))
    });

    let vertex_buffer = device.create_buffer_init(
        &wgpu::util::BufferInitDescriptor {
            label: Some("Vertex Buffer"),
            contents: bytemuck::cast_slice(&vertices),
            usage: wgpu::BufferUsages::VERTEX
        }
    );

    let index_buffer = device.create_buffer_init(
        &wgpu::util::BufferInitDescriptor {
            label: Some("Index Buffer"),
            contents: bytemuck::cast_slice(&indices),
            usage: wgpu::BufferUsages::INDEX
        }
    );    

    let diffuse_bytes = include_bytes!("../../../textures/happy-tree.png");

    let img = image::load_from_memory(diffuse_bytes).expect("Failed to load texture");
    let diffuse_texture = crate::texture::Texture::from_image(&device, &queue, &img, &gpu_api_dto::ImageFormat::R8G8B8A8, false, Some("happy-tree.png")).expect("Failed to create texture from image");

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
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        // This should match the filterable field of the
                        // corresponding Texture entry above.
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                ],
                label: Some("texture_bind_group_layout"),
            });

    let sampler = device.create_sampler(&wgpu::SamplerDescriptor {            
        address_mode_u: wgpu::AddressMode::ClampToEdge,
        address_mode_v: wgpu::AddressMode::ClampToEdge,
        address_mode_w: wgpu::AddressMode::ClampToEdge,
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        mipmap_filter: wgpu::FilterMode::Linear,
        ..Default::default()
    });
    
    let diffuse_bind_group = device.create_bind_group(
        &wgpu::BindGroupDescriptor {
            layout: &texture_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&diffuse_texture.view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                }
            ],
            label: Some("diffuse_bind_group"),
        }
    );        

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("Pipeline Layout"),
        bind_group_layouts: &[
            &texture_bind_group_layout            
        ],
        push_constant_ranges: &[]
    });    

    let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        multiview: None,
        label: Some("Element pipeline"),
        layout: Some(&pipeline_layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: Some("vs_main"),
            compilation_options: Default::default(),
            buffers: &[
                wgpu::VertexBufferLayout {
                    array_stride: size_of::<Vertex>() as wgpu::BufferAddress,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &[
                        wgpu::VertexAttribute {
                            offset: 0,
                            shader_location: 0,
                            format: wgpu::VertexFormat::Uint32
                        },
                        wgpu::VertexAttribute {
                            offset: size_of::<u32>() as wgpu::BufferAddress,
                            shader_location: 1,
                            format: wgpu::VertexFormat::Float32x3,
                        },
                        wgpu::VertexAttribute {
                            offset: (
                            size_of::<u32>() +
                            size_of::<[f32; 3]>()) as wgpu::BufferAddress,
                            shader_location: 2,
                            format: wgpu::VertexFormat::Float32x4
                        },
                        wgpu::VertexAttribute {
                            offset: (
                                size_of::<u32>() +
                                size_of::<[f32; 3]>() + 
                                size_of::<[f32; 4]>()) as wgpu::BufferAddress,
                            shader_location: 3,
                            format: wgpu::VertexFormat::Float32x4
                        },
                        wgpu::VertexAttribute {
                            offset: (
                                size_of::<u32>() +
                                size_of::<[f32; 3]>() + 
                                size_of::<[f32; 4]>() + 
                                size_of::<[f32; 4]>()) as wgpu::BufferAddress,
                            shader_location: 4,
                            format: wgpu::VertexFormat::Uint32
                        },
                        wgpu::VertexAttribute {
                            offset: (
                                size_of::<u32>() +
                                size_of::<[f32; 3]>() + 
                                size_of::<[f32; 4]>() + 
                                size_of::<[f32; 4]>() + 
                                size_of::<u32>()) as wgpu::BufferAddress,
                            shader_location: 5,
                            format: wgpu::VertexFormat::Float32x4
                        },
                        wgpu::VertexAttribute {
                            offset: (
                                size_of::<u32>() +
                                size_of::<[f32; 3]>() + 
                                size_of::<[f32; 4]>() + 
                                size_of::<[f32; 4]>() + 
                                size_of::<u32>() + 
                                size_of::<[f32; 4]>()) as wgpu::BufferAddress,
                            shader_location: 6,
                            format: wgpu::VertexFormat::Float32x4
                        },
                        wgpu::VertexAttribute {
                            offset: (
                                size_of::<u32>() +
                                size_of::<[f32; 3]>() + 
                                size_of::<[f32; 4]>() + 
                                size_of::<[f32; 4]>() + 
                                size_of::<u32>() + 
                                size_of::<[f32; 4]>()  + 
                                size_of::<[f32; 4]>()) as wgpu::BufferAddress,
                            shader_location: 7,
                            format: wgpu::VertexFormat::Float32x2
                        },
                        wgpu::VertexAttribute {
                            offset: (
                                size_of::<u32>() +
                                size_of::<[f32; 3]>() + 
                                size_of::<[f32; 4]>() + 
                                size_of::<[f32; 4]>() + 
                                size_of::<u32>() + 
                                size_of::<[f32; 4]>()  + 
                                size_of::<[f32; 4]>() +
                                size_of::<[f32; 2]>()) as wgpu::BufferAddress,
                            shader_location: 8,
                            format: wgpu::VertexFormat::Uint32
                        },
                        wgpu::VertexAttribute {
                            offset: (
                                size_of::<u32>() +
                                size_of::<[f32; 3]>() + 
                                size_of::<[f32; 4]>() + 
                                size_of::<[f32; 4]>() + 
                                size_of::<u32>() + 
                                size_of::<[f32; 4]>()  + 
                                size_of::<[f32; 4]>() +
                                size_of::<[f32; 2]>() +
                                size_of::<u32>()) as wgpu::BufferAddress,
                            shader_location: 9,
                            format: wgpu::VertexFormat::Float32x4
                        },
                        wgpu::VertexAttribute {
                            offset: (
                                size_of::<u32>() +
                                size_of::<[f32; 3]>() + 
                                size_of::<[f32; 4]>() + 
                                size_of::<[f32; 4]>() + 
                                size_of::<u32>() + 
                                size_of::<[f32; 4]>()  + 
                                size_of::<[f32; 4]>() +
                                size_of::<[f32; 2]>() +
                                size_of::<[f32; 4]>()) as wgpu::BufferAddress,
                            shader_location: 10,
                            format: wgpu::VertexFormat::Uint32
                        },
                        wgpu::VertexAttribute {
                            offset: (
                                size_of::<u32>() +
                                size_of::<[f32; 3]>() + 
                                size_of::<[f32; 4]>() + 
                                size_of::<[f32; 4]>() + 
                                size_of::<u32>() + 
                                size_of::<[f32; 4]>()  + 
                                size_of::<[f32; 4]>() +
                                size_of::<[f32; 2]>() +
                                size_of::<u32>() +
                                size_of::<u32>()) as wgpu::BufferAddress,
                            shader_location: 11,
                            format: wgpu::VertexFormat::Float32x4
                        }
                    ]
                }
            ]
        },
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: Some("fs_main"),
            compilation_options: Default::default(),
            targets: &[Some(wgpu::ColorTargetState {
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
            })],
        }),
        primitive: wgpu::PrimitiveState::default(),
        depth_stencil,
        multisample: wgpu::MultisampleState::default(),
        cache: None
    });

    Pipeline {
        shader,
        vertex_buffer,
        index_buffer,
        texture: diffuse_texture,
        texture_bind_group_layout,
        texture_bind_group: diffuse_bind_group,        
        pipeline_layout,        
        render_pipeline
    }
}
