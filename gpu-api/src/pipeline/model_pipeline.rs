use std::borrow::Cow;
use glam::Mat4;
use wgpu::{Device, Surface, Adapter, Queue, RenderPipeline, Buffer, BindGroup, ShaderModule, BindGroupLayout, PipelineLayout, TextureFormat};
use wgpu::util::DeviceExt;
use crate::camera::CameraUniform;

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
    pub texture_bind_group: BindGroup,
    pub camera_buffer: Buffer,
    pub camera_bind_group_layout: BindGroupLayout,
    pub camera_bind_group: BindGroup,    
    pub pipeline_layout: PipelineLayout,
    pub swapchain_format: TextureFormat,
    pub render_pipeline: RenderPipeline    
}

pub async fn new(surface: &Surface, device: &Device, adapter: &Adapter, queue: &Queue, width: f32, height: f32) -> (Mat4, Mat4, Pipeline) {
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

        let diffuse_bytes = include_bytes!("../../../textures/happy-tree.png");
        let diffuse_texture = crate::texture::Texture::from_bytes(&device, &queue, diffuse_bytes, "happy-tree.png").expect("Failed to create texture");

        let texture_bind_group = device.create_bind_group(
            &wgpu::BindGroupDescriptor {
                layout: &texture_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&diffuse_texture.view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(&diffuse_texture.sampler),
                    }
                ],
                label: Some("diffuse_bind_group"),
            }
        );    

    let (dog, dog2) = generate_matrix(width, height);

    let dog_ref: &[f32; 16] = dog.as_ref();

    let camera_buffer = device.create_buffer_init(
        &wgpu::util::BufferInitDescriptor {
            label: Some("Camera Buffer"),
            contents: bytemuck::cast_slice(dog_ref),
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
        label: Some("camera_bind_group"),
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
        label: None,
        layout: Some(&pipeline_layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: "vs_main",
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

    (dog, dog2, Pipeline {
        shader,
        texture_bind_group_layout,
        texture_bind_group,
        camera_buffer,
        camera_bind_group_layout,
        camera_bind_group,
        pipeline_layout,
        swapchain_format: TextureFormat::Rgba8UnormSrgb,
        render_pipeline      
    })
}

fn generate_matrix(width: f32, height: f32) -> (glam::Mat4, glam::Mat4) {
    let aspect_ratio = width / height;
    let angle_xz = 0.2f32;
    let angle_y = 0.2f32;
    let dist = 20.0;
    let model_center_x = 0.0;
    let model_center_y = 2.0;
    
    let projection = glam::Mat4::perspective_rh(std::f32::consts::FRAC_PI_4, aspect_ratio, 1.0, 50.0);
    
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

    let dog = glam::Mat4::from_scale(glam::Vec3::new(0.02, 0.02, 0.02));

    (projection, view * dog)
}
