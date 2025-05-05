use std::mem;
use std::ops::Mul;
use log::*;
use wgpu::{util::DeviceExt, DepthStencilState, RenderPass};
use glam::{Mat4, Vec3};

#[derive(Debug)]
pub struct Pipeline {
    pub pipeline: wgpu::RenderPipeline,
    pub uniform_bind_group: wgpu::BindGroup,
    pub uniform_buffer: wgpu::Buffer,
    pub vertex_buffer: wgpu::Buffer,
    pub index_buffer: wgpu::Buffer,
    pub instance_buffer: wgpu::Buffer
}

impl Pipeline {
    pub fn draw<'a>(&'a mut self, render_pass: &mut RenderPass<'a>, amount: u32) {
        render_pass.set_pipeline(&self.pipeline);
        render_pass.set_bind_group(0, &self.uniform_bind_group, &[]);
        render_pass.set_index_buffer(
            self.index_buffer.slice(..),
            wgpu::IndexFormat::Uint32
        );
        render_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
        render_pass.set_vertex_buffer(1, self.instance_buffer.slice(..));

        /*
        render_pass.set_scissor_rect(
            bounds.x,
            bounds.y,
            bounds.width,            
            bounds.height,
        );
        */

        render_pass.draw_indexed(0..QUAD_INDICES.len() as u32, 0, 0..amount);
    }

    pub fn new(device: &wgpu::Device, format: wgpu::TextureFormat, depth_stencil: Option<DepthStencilState>) -> Pipeline {
        let constant_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("iced_wgpu::quad uniforms layout"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: wgpu::BufferSize::new(
                            mem::size_of::<Uniforms>() as wgpu::BufferAddress,
                        ),
                    },
                    count: None,
                }],
            });

        let constants_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("iced_wgpu::quad uniforms buffer"),
            size: mem::size_of::<Uniforms>() as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let constants = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("iced_wgpu::quad uniforms bind group"),
            layout: &constant_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: constants_buffer.as_entire_binding(),
            }],
        });

        let layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("iced_wgpu::quad pipeline layout"),
                push_constant_ranges: &[],
                bind_group_layouts: &[&constant_layout],
            });

        let shader =
            device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("iced_wgpu::quad::shader"),
                source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(
                    include_str!("quad.wgsl"),
                )),
            });

        let pipeline =
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("Quad pipeline"),
                layout: Some(&layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs_main"),
                    compilation_options: Default::default(),
                    buffers: &[
                        wgpu::VertexBufferLayout {
                            array_stride: mem::size_of::<Vertex>() as u64,
                            step_mode: wgpu::VertexStepMode::Vertex,
                            attributes: &[wgpu::VertexAttribute {
                                shader_location: 0,
                                format: wgpu::VertexFormat::Float32x2,
                                offset: 0,
                            }],
                        },
                        wgpu::VertexBufferLayout {
                            array_stride: mem::size_of::<Quad>() as u64,
                            step_mode: wgpu::VertexStepMode::Instance,
                            attributes: &wgpu::vertex_attr_array!(
                                1 => Float32x2,
                                2 => Float32x2,
                                3 => Float32x4,
                                4 => Float32x4,
                                5 => Float32x4,
                                6 => Float32,
                                7 => Float32x4,
                                8 => Uint32,
                                9 => Float32x4,
                                10 => Float32x4,
                                11 => Float32x2,
                                12 => Float32
                            ),
                        },
                    ],
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some("fs_main"),
                    compilation_options: Default::default(),
                    targets: &[Some(wgpu::ColorTargetState {
                        format,
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
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    front_face: wgpu::FrontFace::Cw,
                    ..Default::default()
                },
                depth_stencil,
                multisample: wgpu::MultisampleState::default(),
                multiview: None,
                cache: None
            });

        let vertex_buffer =
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Quad vertex buffer"),
                contents: bytemuck::cast_slice(&QUAD_VERTICES),
                usage: wgpu::BufferUsages::VERTEX,
            });

        let index_buffer =
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Quad index buffer"),
                contents: bytemuck::cast_slice(&QUAD_INDICES),
                usage: wgpu::BufferUsages::INDEX,
            });

        let instance_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Quad instance buffer"),
            size: mem::size_of::<Quad>() as u64 * MAX_INSTANCES as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false
        });

        Pipeline {
            pipeline,
            uniform_bind_group: constants,
            uniform_buffer: constants_buffer,
            vertex_buffer,
            index_buffer,
            instance_buffer
        }
    }    
}

#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct Quad {    
    pub position: [f32; 2],
    pub size: [f32; 2],
    pub color: [f32; 4],
    pub border_color: [f32; 4],
    pub border_radius: [f32; 4],
    pub border_width: f32,
    pub component_coordinates: [f32; 4],
    pub has_overlay: u32,
    pub overlay_coordinates: [f32; 4],
    pub shadow_color: [f32; 4],
    pub shadow_offset: [f32; 2],
    pub shadow_blur_radius: f32
}

unsafe impl bytemuck::Zeroable for Quad {}
unsafe impl bytemuck::Pod for Quad {}


#[repr(C)]
#[derive(Clone, Copy)]
pub struct Vertex {
    _position: [f32; 2]
}

unsafe impl bytemuck::Zeroable for Vertex {}
unsafe impl bytemuck::Pod for Vertex {}

const QUAD_INDICES: [u32; 6] = [0, 1, 2, 0, 2, 3];

const QUAD_VERTICES: [Vertex; 4] = [
    Vertex {
        _position: [0.0, 0.0]
    },
    Vertex {
        _position: [1.0, 0.0]
    },
    Vertex {
        _position: [1.0, 1.0]
    },
    Vertex {
        _position: [0.0, 1.0]
    }
];

pub const MAX_INSTANCES: usize = 100_000;

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct Uniforms {
    transform: [f32; 16],
    scale: f32,
    // Uniforms must be aligned to their largest member,
    // this uses a mat4x4<f32> which aligns to 16, so align to that
    _padding: [f32; 3],
}

unsafe impl bytemuck::Pod for Uniforms {}
unsafe impl bytemuck::Zeroable for Uniforms {}

impl Uniforms {
    pub fn new(transformation: Transformation, scale: f32) -> Uniforms {
        Self {
            transform: *transformation.as_ref(),
            scale,
            _padding: [0.0; 3],
        }
    }
}
impl Default for Uniforms {
    fn default() -> Self {
        Self {
            transform: *Transformation::identity().as_ref(),
            scale: 1.0,
            _padding: [0.0; 3],
        }
    }
}

/// A 2D transformation matrix.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Transformation(Mat4);

impl Transformation {
    /// Get the identity transformation.
    pub fn identity() -> Transformation {
        Transformation(Mat4::IDENTITY)
    }

    /// Creates an orthographic projection.
    #[rustfmt::skip]
    pub fn orthographic(width: u32, height: u32) -> Transformation {
        Transformation(Mat4::orthographic_rh_gl(
            0.0, width as f32,
            height as f32, 0.0,
            -1.0, 1.0
        ))
    }

    /// Creates a translate transformation.
    pub fn translate(x: f32, y: f32) -> Transformation {
        Transformation(Mat4::from_translation(Vec3::new(x, y, 0.0)))
    }

    /// Creates a scale transformation.
    pub fn scale(x: f32, y: f32) -> Transformation {
        Transformation(Mat4::from_scale(Vec3::new(x, y, 1.0)))
    }
}

impl Mul for Transformation {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self {
        Transformation(self.0 * rhs.0)
    }
}

impl AsRef<[f32; 16]> for Transformation {
    fn as_ref(&self) -> &[f32; 16] {
        self.0.as_ref()
    }
}

impl From<Transformation> for [f32; 16] {
    fn from(t: Transformation) -> [f32; 16] {
        *t.as_ref()
    }
}

impl From<Transformation> for Mat4 {
    fn from(transformation: Transformation) -> Self {
        transformation.0
    }
}
