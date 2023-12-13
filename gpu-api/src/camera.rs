pub const CAMERA_UNIFORM_SIZE: u64 = 64;

pub fn generate_projection_matrix(width: f32, height: f32) -> glam::Mat4 {
    let aspect_ratio = width / height;    
    
    let res = glam::Mat4::perspective_rh(std::f32::consts::FRAC_PI_4, aspect_ratio, 0.1, 100.0);        

    let x = 0.0;
    let y = 0.0;
    let z = 0.0;
    let angle_xz = 0.4f32;
    let angle_y = 0.4f32;
    let dist = 30.0;

    let cam_pos = glam::Vec3::new(
        angle_xz.cos() * angle_y.sin() * dist,
        angle_xz.sin() * dist + y,
        angle_xz.cos() * angle_y.cos() * dist
    );

    let view = glam::Mat4::look_at_rh(
        cam_pos,
        glam::Vec3::new(x, y, z),
        glam::Vec3::new(0.0, 1.0, 0.0)
    );

    res * view
}