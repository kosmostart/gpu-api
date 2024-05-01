use std::num::NonZeroU32;
use std::borrow::Cow;
use log::warn;
use wgpu::{Device, Surface, Adapter, Queue, RenderPipeline, Buffer, BindGroup, ShaderModule, BindGroupLayout, PipelineLayout, TextureFormat, RenderPass, Sampler, TextureView};
use wgpu::util::DeviceExt;
use crate::camera::{Camera, create_camera};
use crate::model::Object;
use crate::texture::TextureData;

#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct Vertex {    
    pub position: [f32; 3],    
    pub texture_coordinates: [f32; 2],
    pub normal: [f32; 3]
}

unsafe impl bytemuck::Pod for Vertex {}
unsafe impl bytemuck::Zeroable for Vertex {}

pub struct Pipeline {
    pub shader: ShaderModule,
    pub texture_bind_group_layout: BindGroupLayout,
    pub sampler: Sampler,
    pub camera_buffer: Buffer,
    pub camera_bind_group_layout: BindGroupLayout,
    pub camera_bind_group: BindGroup, 
    pub pipeline_layout: PipelineLayout,
    pub swapchain_format: TextureFormat,
    pub render_pipeline: RenderPipeline
}

impl Pipeline {
    pub fn draw<'a>(&'a self, render_pass: &mut RenderPass<'a>, objects: &'a Vec<Object>) {
        render_pass.set_pipeline(&self.render_pipeline);

        let mut index = 0;
    
        for object in objects {
            if object.views_amount == 0 {
                continue;
            }
            
            render_pass.set_vertex_buffer(1, object.instance_buffer.slice(..)); // Instances
                    
            let instances_range = 0..object.views_amount;
            
            for mesh in &object.meshes {                            
                render_pass.set_bind_group(0, &object.texture_bind_groups[0], &[]); // Texture
                render_pass.set_bind_group(1, &self.camera_bind_group, &[]); // Camera
                render_pass.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
                render_pass.set_index_buffer(mesh.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
                render_pass.draw_indexed(0..mesh.num_elements, 0, instances_range.clone());
            }

            index = index + 1;
        }
    }
}

pub async fn new(surface: &Surface<'_>, device: &Device, adapter: &Adapter, queue: &Queue, width: f32, height: f32) -> (Camera, Pipeline) {
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

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("Pipeline Layout"),
        bind_group_layouts: &[
            &texture_bind_group_layout,
            &camera_bind_group_layout
        ],
        push_constant_ranges: &[]
    });    

    let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        multiview: None,
        label: Some("Model pipeline"),
        layout: Some(&pipeline_layout),        
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: "vs_main",
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
                            offset: (std::mem::size_of::<[f32; 3]>() + std::mem::size_of::<[f32; 2]>()) as wgpu::BufferAddress,
                            shader_location: 2,
                            format: wgpu::VertexFormat::Float32x3
                        }                        
                    ]
                },
                crate::instance::InstanceRaw::vertex_buffer_layout()
            ]
        },
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: "fs_main",
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
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default()
    });

    (camera, Pipeline {
        shader,
        texture_bind_group_layout,        
        camera_buffer,
        camera_bind_group_layout,
        sampler,
        camera_bind_group,
        pipeline_layout,
        swapchain_format: TextureFormat::Rgba8UnormSrgb,
        render_pipeline      
    })
}
