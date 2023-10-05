pub const CAMERA_UNIFORM_SIZE: u64 = 64;

pub fn generate_projection_matrix(width: f32, height: f32) -> glam::Mat4 {
    let aspect_ratio = width / height;    
    
    let res = glam::Mat4::perspective_rh(std::f32::consts::FRAC_PI_4, aspect_ratio, 1.0, 50.0);        

    res
}