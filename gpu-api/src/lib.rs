pub use bytemuck;
pub use bytemuck_derive;
use camera::Camera;
pub use glam;
use glam::vec3;
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

pub fn screen_to_ray(camera: &Camera, width: f32, height: f32, x: f32, y: f32, point: glam::Vec3) -> f32 {
    let x_ndc = (2.0 * x / width) - 1.0;
    let y_ndc = 1.0 - (2.0 * y / height);    

    let ray_near = glam::vec4(x_ndc, y_ndc, 0.0, 1.0);
    let ray_far = glam::vec4(x_ndc, y_ndc, 1.0, 1.0);

    let ray_eye_near = camera.projection_source.inverse() * ray_near;
    let ray_eye_far = camera.projection_source.inverse() * ray_far;

    let ray_world_near = camera.view.inverse() * ray_eye_near;    
    let ray_world_far = camera.view.inverse() * ray_eye_far;
    
    let ray_world_near3 = glam::vec3(ray_world_near.x / ray_world_near.w, ray_world_near.y  / ray_world_near.w, ray_world_near.z  / ray_world_near.w);
    let ray_world_far3 = glam::vec3(ray_world_far.x / ray_world_far.w, ray_world_far.y  / ray_world_far.w, ray_world_far.z  / ray_world_far.w);    
    
    let point_near = point - ray_world_near3;
    let point_far = point - ray_world_far3;    
    let q1 = point_near.cross(point_far);
    let q2 = ray_world_far3 - ray_world_near3;

    //warn!("q1 {:?} {}", q1, q1.length());
    //warn!("q2 {:?} {}", q2, q2.length());

    q1.length() / q2.length()
}

pub fn screen_to_ray_2(camera: &Camera, width: f32, height: f32, x: f32, y: f32) -> (glam::Vec4, glam::Vec3) {
    let x_ndc = (2.0 * x / width) - 1.0;
    let y_ndc = 1.0 - (2.0 * y / height);    

    let ray_clip = glam::vec4(x_ndc, y_ndc, 0.0, 1.0);

    let ray_eye = camera.projection_source.inverse() * ray_clip;

    let ray_world = camera.view.inverse() * ray_eye;

    let ray_direction = vec3(ray_world.x, ray_world.y, ray_world.z) - camera.camera_position;

    (ray_world, ray_direction.normalize())
}

pub fn screen_to_world(camera: &Camera, width: f32, height: f32, x: f32, y: f32) -> glam::Vec4 {
    //let x = 769.0;
    //let y = 437.0;

    let x_ndc = (2.0 * x / width) - 1.0;
    let y_ndc = 1.0 - (2.0 * y / height);
    let z_ndc = 0.9983375;

    let w = camera.projection_source.row(2).w / (z_ndc + camera.projection_source.row(2).z);
    
    let clip_space = glam::vec4(x_ndc * w, y_ndc * w, z_ndc * w, w);    
    warn!("Got clip space {:?}", clip_space);    

    let camera_space = camera.projection_source.inverse() * clip_space;
    warn!("Got camera space {:?}", camera_space);    
        
    let world_space = camera.view.inverse() * camera_space;    

    world_space
}

