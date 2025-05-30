use core::f32;

pub use bytemuck;
pub use bytemuck_derive;
use camera::Camera;
pub use glam;
use glam::vec3;
pub use gpu_api_dto;
use log::warn;
pub use wgpu;

pub mod frame_counter;
pub mod texture;
pub mod camera;
pub mod model_instance;
pub mod model;
pub mod pipeline {
    pub mod element_pipeline;
    pub mod model_pipeline;
    pub mod quad_pipeline;
}
pub mod object_picking;
