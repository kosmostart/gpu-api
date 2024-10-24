use glam::{Mat4, Vec3};

pub const CAMERA_UNIFORM_SIZE: u64 = 64;

pub struct Camera {
    pub x: f32, 
    pub y: f32, 
    pub z: f32,
    pub camera_position: Vec3,
    pub projection_source: Mat4,
    pub view: Mat4,
    pub projection_view: Mat4,
    pub translation: Mat4,
    pub projection: Mat4
}

impl Camera {
    pub fn update_projection(&mut self) {
        let (translation, projection) = generate_projection(&self.projection_view, self.x, self.y, self.z);
        self.translation = translation;
        self.projection = projection;
    }
}

pub fn create_camera(width: f32, height: f32, x: f32, y: f32, z: f32) -> Camera {
    let (camera_position, projection_source, view, projection_view) = generate_projection_view(width, height);
    let (translation, projection) = generate_projection(&projection_view, x, y, z);

    Camera {
        x,
        y,
        z,
        camera_position,
        projection_source,
        view,
        projection_view,
        translation,
        projection
    }
}

pub fn generate_projection_view(width: f32, height: f32) -> (Vec3, Mat4, Mat4, Mat4) {
    let aspect_ratio = width / height;    
    
    let projection_source = glam::Mat4::perspective_rh(std::f32::consts::FRAC_PI_4, aspect_ratio, 0.1, 1000.0);        

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

    (cam_pos, projection_source, view, projection_source * view)
}

pub fn generate_projection(projection_view: &Mat4, x: f32, y: f32, z: f32) -> (Mat4, Mat4) {
    let translation = glam::Mat4::from_translation(glam::Vec3::new(x, y, z));

    (translation, projection_view.clone() * translation)
}
