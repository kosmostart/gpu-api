use glam::Mat4;

pub const CAMERA_UNIFORM_SIZE: u64 = 64;

pub struct Camera {
    pub x: f32, 
    pub y: f32, 
    pub z: f32,
    pub projection_view: Mat4,
    pub projection: Mat4
}

impl Camera {
    pub fn update_projection(&mut self) {
        self.projection = generate_projection(&self.projection_view, self.x, self.y, self.z);
    }
}

pub fn create_camera(width: f32, height: f32, x: f32, y: f32, z: f32) -> Camera {
    let projection_view = generate_projection_view(width, height);
    let projection = generate_projection(&projection_view, x, y, z);

    Camera {
        x,
        y,
        z,
        projection_view,
        projection
    }
}

pub fn generate_projection_view(width: f32, height: f32) -> Mat4 {
    let aspect_ratio = width / height;    
    
    let projection = glam::Mat4::perspective_rh(std::f32::consts::FRAC_PI_4, aspect_ratio, 0.1, 100.0);        

    let center_x = 0.0;
    let center_y = 0.0;
    let center_z = 0.0;
    let angle_xz = 0.4f32;
    let angle_y = 0.8f32;
    let dist = 40.0;

    let cam_pos = glam::Vec3::new(
        angle_xz.cos() * angle_y.sin() * dist,
        angle_xz.sin() * dist + center_y,
        angle_xz.cos() * angle_y.cos() * dist
    );

    let view = glam::Mat4::look_at_rh(
        cam_pos,
        glam::Vec3::new(center_x, center_y, center_z),
        glam::Vec3::new(0.0, 1.0, 0.0)
    );    

    projection * view
}

pub fn generate_projection(projection_view: &Mat4, x: f32, y: f32, z: f32) -> Mat4 {
    let translation = glam::Mat4::from_translation(glam::Vec3::new(x, y, z));

    projection_view.clone() * translation
}
