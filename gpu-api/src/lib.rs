pub use bytemuck;
pub use bytemuck_derive;
pub use glam;
pub use gpu_api_dto;
pub use image;
use log::warn;
pub use wgpu;

pub mod texture;
pub mod camera;
pub mod instance;
pub mod model;
pub mod pipeline {
    pub mod element_pipeline;
    pub mod model_pipeline;
    pub mod quad_pipeline;
}

pub fn screen_to_ndc(width: f32, height: f32, x: f32, y: f32) -> [f32; 3] {
    let x_ndc = (2.0 * x / width) - 1.0;
    let y_ndc = 1.0 - (2.0 * y / height);    
    let z_ndc = 1.0;

    [x_ndc, y_ndc, z_ndc]
}

pub fn screen_to_world(view: &glam::Mat4, projection_source: &glam::Mat4, width: f32, height: f32, x: f32, y: f32) -> glam::Vec4 {    
    let x_ndc = (2.0 * x / width) - 1.0;
    let y_ndc = 1.0 - (2.0 * y / height);    
    let z_ndc = 0.97;

    let w = 34.23465;

    //let clip_space = glam::vec4(x_ndc * w, y_ndc * w, z_ndc * w, w);
    //let clip_space = glam::vec4(-3.1921062, -1.9530865, 34.168816, 34.23465);
    let clip_space = glam::vec4(x_ndc * w, y_ndc * w, 34.168816, 34.23465);
    warn!("clip space {:?}", clip_space);

    

    let mut camera_space = projection_source.inverse() * clip_space;
    warn!("camera space {:?}", camera_space);

    //camera_space.z = -1.0;
    //camera_space.w = 0.0;
        
    let world_space = view.inverse() * camera_space;

    world_space
}
