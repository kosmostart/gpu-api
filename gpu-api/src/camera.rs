use glam::{Mat4, Vec3};

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct CameraUniform {
    pub camera_position: [f32; 3],
    pub padding: u32,
    pub view: [f32; 16],
    pub projection: [f32; 16]    
}

unsafe impl bytemuck::Pod for CameraUniform {}
unsafe impl bytemuck::Zeroable for CameraUniform {}

pub struct Camera {
    pub x: f32, 
    pub y: f32, 
    pub z: f32,
    pub angle_y: f32,
    pub angle_xz: f32,
    pub dist: f32,
    pub camera_position: Vec3,
    pub projection_source: Mat4,
    pub view: Mat4,
    pub projection_view: Mat4,    
    pub projection: Mat4
}

impl Camera {
    pub fn update(&mut self, width: f32, height: f32) {
        let (camera_position, projection_source, view, projection_view) = generate_projection_view(width, height, self.x, self.y, self.z, self.angle_xz, self.angle_y, self.dist);

        self.camera_position = camera_position;
        self.projection_source = projection_source;
        self.view = view;
        self.projection_view = projection_view;
        self.projection = projection_view;
    }    
}

pub fn create_camera(width: f32, height: f32, angle_xz: f32, angle_y: f32, dist: f32, x: f32, y: f32, z: f32) -> Camera {    
    let (camera_position, projection_source, view, projection_view) = generate_projection_view(width, height, x, y, z, angle_xz, angle_y, dist);    

    Camera {
        x,
        y,
        z,
        angle_xz,
        angle_y,        
        dist,
        camera_position,
        projection_source,
        view,
        projection_view,        
        projection: projection_view
    }
}

pub fn generate_projection_view(width: f32, height: f32, center_x: f32, center_y: f32, center_z: f32, angle_xz: f32, angle_y: f32, dist: f32) -> (Vec3, Mat4, Mat4, Mat4) {
    let aspect_ratio = width / height;        
    let projection_source = glam::Mat4::perspective_rh(std::f32::consts::FRAC_PI_4, aspect_ratio, 0.1, 1000.0);        

    let camera_position = glam::Vec3::new(
        angle_xz.cos() * angle_y.sin() * dist + center_x,
        angle_xz.sin() * dist + center_y,
        angle_xz.cos() * angle_y.cos() * dist + center_z
    );

    let view = glam::Mat4::look_at_rh(
        camera_position,
        glam::Vec3::new(center_x, center_y, center_z),
        glam::Vec3::new(0.0, 1.0, 0.0)
    );    

    (camera_position, projection_source, view, projection_source * view)
}
