pub use bytemuck;
pub use bytemuck_derive;
pub use glam;
use glam::Vec4Swizzles;
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


pub fn screen_to_world(projection: &glam::Mat4, width: f32, height: f32, x: f32, y: f32) -> glam::Vec4 {
    let x = 769.49225;
    let y = 412.0217;

    let x_ndc = (2.0 * x / width) - 1.0;
    let y_ndc = 1.0 - (2.0 * y / height);
    //let z_ndc = (2.0 * z) - 1.0;
    let z_ndc = 1.0;

    let ndc_space = glam::vec4(x_ndc, y_ndc, z_ndc, 1.0);

    warn!("ndc space is {:?}", ndc_space);    

    let w = 39.406364;

    let mut clip_space = ndc_space * w;

    clip_space.z = 39.345707;

    let q = projection.inverse();

    warn!("clip_space orig is {:?}", clip_space);
    warn!("world space orig is {:?}", q * clip_space);

    let clip_space2 = glam::vec4(1.0241574, 3.3257487, 39.345707, 39.406364);
    warn!("clip_space 2 is {:?}", clip_space2);
    let world_space = q * clip_space2;
    warn!("world space 2 is {:?}", world_space);

    world_space
}
