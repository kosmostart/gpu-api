use gpu_api_dto::bytemuck;
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
    pub angle_y: f32,
    pub angle_xz: f32,
    pub dist: f32,    
    pub camera_position: Vec3,
    pub focus_point: Vec3,
    pub projection_source: Mat4,
    pub view: Mat4,    
    pub projection: Mat4
}

impl Camera {
    pub fn update(&mut self, width: f32, height: f32) {
        let (camera_position, focus_point, projection_source, view, projection) = generate_projection(width, height, self.focus_point.x, self.focus_point.y, self.focus_point.z, self.angle_xz, self.angle_y, self.dist);

        self.camera_position = camera_position;
        self.focus_point = focus_point;
        self.projection_source = projection_source;
        self.view = view;        
        self.projection = projection;
    }
}

pub fn create_camera(width: f32, height: f32, angle_xz: f32, angle_y: f32, dist: f32, focus_point_x: f32, focus_point_y: f32, focus_point_z: f32) -> Camera {    
    let (camera_position, focus_point, projection_source, view, projection) = generate_projection(width, height, focus_point_x, focus_point_y, focus_point_z, angle_xz, angle_y, dist);

    Camera {        
        angle_xz,
        angle_y,
        dist,
        camera_position,
        focus_point,
        projection_source,
        view,        
        projection
    }
}

pub fn generate_projection(width: f32, height: f32, focus_point_x: f32, focus_point_y: f32, focus_point_z: f32, angle_xz: f32, angle_y: f32, dist: f32) -> (Vec3, Vec3, Mat4, Mat4, Mat4) {
    let aspect_ratio = width / height;        
    let projection_source = glam::Mat4::perspective_rh(std::f32::consts::FRAC_PI_4, aspect_ratio, 0.1, 1000.0);        

    let camera_position = glam::Vec3::new(
        angle_xz.cos() * angle_y.sin() * dist + focus_point_x,
        angle_xz.sin() * dist + focus_point_y,
        angle_xz.cos() * angle_y.cos() * dist + focus_point_z
    );

    let focus_point = glam::Vec3::new(focus_point_x, focus_point_y, focus_point_z);

    let view = glam::Mat4::look_at_rh(
        camera_position,
        focus_point,
        glam::Vec3::new(0.0, 1.0, 0.0)
    );    

    (camera_position, focus_point, projection_source, view, projection_source * view)
}
