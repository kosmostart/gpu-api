use std::mem;

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct ModelInstance {
    pub model_matrix: [f32; 16]
}

unsafe impl bytemuck::Pod for ModelInstance {}
unsafe impl bytemuck::Zeroable for ModelInstance {}

impl ModelInstance {
    pub fn vertex_buffer_layout<'a>() -> wgpu::VertexBufferLayout<'a> {        
        wgpu::VertexBufferLayout {
            array_stride: mem::size_of::<ModelInstance>() as wgpu::BufferAddress,
            // We need to switch from using a step mode of Vertex to Instance
            // This means that our shaders will only change to use the next
            // instance when the shader starts processing a new instance
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &[                
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 3,
                    format: wgpu::VertexFormat::Float32x4
                },
                wgpu::VertexAttribute {
                    offset: mem::size_of::<[f32; 4]>() as wgpu::BufferAddress,
                    shader_location: 4,
                    format: wgpu::VertexFormat::Float32x4
                },
                wgpu::VertexAttribute {
                    offset: mem::size_of::<[f32; 8]>() as wgpu::BufferAddress,
                    shader_location: 5,
                    format: wgpu::VertexFormat::Float32x4
                },
                wgpu::VertexAttribute {
                    offset: mem::size_of::<[f32; 12]>() as wgpu::BufferAddress,
                    shader_location: 6,
                    format: wgpu::VertexFormat::Float32x4
                },
            ]
        }
    }
}
