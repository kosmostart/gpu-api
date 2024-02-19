pub use bytemuck;
pub use bytemuck_derive;
pub use glam;
pub use image;
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


pub fn screen_to_world(q: &glam::Mat4, width: f32, height: f32, x: f32, y: f32, z: f32) {
    let inverted = q.inverse();

    let x_ndc = (2.0 * x / width) - 1.0;
    let y_ndc = (2.0 * y / height) - 1.0;
    let z_ndc = (2.0 * z) - 1.0;    

    let x_world = x_ndc * inverted.x_axis[0] + y_ndc * inverted.x_axis[1] + z_ndc * inverted.x_axis[2] + inverted.x_axis[3];
    let y_world = x_ndc * inverted.y_axis[0] + y_ndc * inverted.y_axis[1] + z_ndc * inverted.y_axis[2] + inverted.y_axis[3];
    let z_world = x_ndc * inverted.z_axis[0] + y_ndc * inverted.z_axis[1] + z_ndc * inverted.x_axis[2] + inverted.z_axis[3];
    

    /*
    local inv = vmath.inv(self.projection * self.view)

    x = (2 * x / render.get_width()) - 1
    y = (2 * y / render.get_height()) - 1
    z = (2 * z) - 1
    
    local x1 = x * inv.m00 + y * inv.m01 + z * inv.m02 + inv.m03
    local y1 = x * inv.m10 + y * inv.m11 + z * inv.m12 + inv.m13
    local z1 = x * inv.m20 + y * inv.m21 + z * inv.m22 + inv.m23
    
    return x1, y1, z1    
    */
}