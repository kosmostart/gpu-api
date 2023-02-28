use std::mem;
use log::*;
use cgmath::prelude::*;

pub const NUM_INSTANCES_PER_ROW: u32 = 1;
pub const SPACE_BETWEEN: f32 = 3.0;
pub const INSTANCE_DISPLACEMENT: cgmath::Vector3<f32> = cgmath::Vector3::new(NUM_INSTANCES_PER_ROW as f32 * 0.5, 0.0, NUM_INSTANCES_PER_ROW as f32 * 0.5);

pub struct Instance {
    position: cgmath::Vector3<f32>,
    rotation: cgmath::Quaternion<f32>,
}

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck_derive::Pod, bytemuck_derive::Zeroable)]
pub struct InstanceRaw {
    model: [[f32; 4]; 4],
}

impl Instance {
    pub fn to_raw(&self) -> InstanceRaw {
        /*
        Scale

        let m: cgmath::Matrix4<f32> = cgmath::Matrix4::new(
            0.5, 0.0, 0.0, 0.0,
            0.0, 0.5, 0.0, 0.0,
            0.0, 0.0, 0.5, 0.0,
            0.0, 0.0, 0.0, 1.0
        );
        */      

        let res = InstanceRaw {
            model: (cgmath::Matrix4::from_translation(self.position) * cgmath::Matrix4::from(self.rotation)).into()
            //model: (cgmath::Matrix4::from_translation(self.position)).into()
        };

        //info!("{:#?}", res);

        res
    }
}

impl InstanceRaw {
    pub fn vertex_buffer_layout<'a>() -> wgpu::VertexBufferLayout<'a> {        
        wgpu::VertexBufferLayout {
            array_stride: mem::size_of::<InstanceRaw>() as wgpu::BufferAddress,
            // We need to switch from using a step mode of Vertex to Instance
            // This means that our shaders will only change to use the next
            // instance when the shader starts processing a new instance
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &[
                wgpu::VertexAttribute {
                    offset: 0,
                    // While our vertex shader only uses locations 0, and 1 now, in later tutorials we'll
                    // be using 2, 3, and 4, for Vertex. We'll start at slot 5 not conflict with them later
                    shader_location: 3,
                    format: wgpu::VertexFormat::Float32x4
                },
                // A mat4 takes up 4 vertex slots as it is technically 4 vec4s. We need to define a slot
                // for each vec4. We'll have to reassemble the mat4 in
                // the shader.
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
                }
            ]
        }
    }
}

pub fn create_one(x: f32, y: f32, z: f32) -> Vec<Instance> {
    let position = cgmath::Vector3 { x, y, z };

    let rotation = if position.is_zero() {
        cgmath::Quaternion::from_axis_angle(cgmath::Vector3::unit_z(), cgmath::Deg(0.0))
    } else {
        cgmath::Quaternion::from_axis_angle(position.normalize(), cgmath::Deg(45.0))
    };

    //let rotation = cgmath::Quaternion::from_axis_angle(position.normalize(), cgmath::Deg(45.0));

    vec![
        Instance {
            position,
            rotation
        }
    ]
}
