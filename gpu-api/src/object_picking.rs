use log::*;
use glam::vec3;
use crate::camera::Camera;

pub fn screen_to_ndc(width: f32, height: f32, x: f32, y: f32) -> [f32; 3] {
    let x_ndc = (2.0 * x / width) - 1.0;
    let y_ndc = 1.0 - (2.0 * y / height);    
    let z_ndc = 1.0;

    [x_ndc, y_ndc, z_ndc]
}

pub fn screen_to_distance(camera: &Camera, width: f32, height: f32, x: f32, y: f32, point: &glam::Vec3) -> f32 {
    let x_ndc = (2.0 * x / width) - 1.0;
    let y_ndc = 1.0 - (2.0 * y / height);    

    let ray_near_ndc = glam::vec4(x_ndc, y_ndc, 0.0, 1.0);
    let ray_far_ndc = glam::vec4(x_ndc, y_ndc, 1.0, 1.0);

    let ray_near_eye = camera.projection_source.inverse() * ray_near_ndc;
    let ray_far_eye = camera.projection_source.inverse() * ray_far_ndc;

    let ray_near_world = camera.view.inverse() * ray_near_eye;
    let ray_far_world = camera.view.inverse() * ray_far_eye;
    
    let ray_near_world3 = glam::vec3(ray_near_world.x / ray_near_world.w, ray_near_world.y  / ray_near_world.w, ray_near_world.z  / ray_near_world.w);
    let ray_far_world3 = glam::vec3(ray_far_world.x / ray_far_world.w, ray_far_world.y  / ray_far_world.w, ray_far_world.z  / ray_far_world.w);    
    
    let point_near = point - ray_near_world3;
    let point_far = point - ray_far_world3;    
    let q1 = point_near.cross(point_far);
    let q2 = ray_far_world3 - ray_near_world3;

    //warn!("q1 {:?} {}", q1, q1.length());
    //warn!("q2 {:?} {}", q2, q2.length());

    q1.length() / q2.length()
}

pub fn screen_to_ray_direction(camera: &Camera, width: f32, height: f32, x: f32, y: f32) -> glam::Vec3 {
    let x_ndc = (2.0 * x / width) - 1.0;
    let y_ndc = 1.0 - (2.0 * y / height);

    let ray_near_ndc = glam::vec4(x_ndc, y_ndc, 0.0, 1.0);

    let mut ray_near_eye = camera.projection_source.inverse() * ray_near_ndc;

    // !!! This is not obvious, but very important. !!!
    ray_near_eye.w = 0.0;

    let ray_near_world = camera.view.inverse() * ray_near_eye;
    let ray_near_world3 = glam::vec3(ray_near_world.x, ray_near_world.y, ray_near_world.z);
    
    //warn!("ray_near_world3 {:?}", ray_near_world3);

    ray_near_world3.normalize()
}

pub fn ray_plane_intersection(ray_origin: &glam::Vec3, ray_direction: &glam::Vec3, plane_origin: &glam::Vec3, plane_normal: &glam::Vec3) -> glam::Vec3 {    
    let d = plane_origin.dot(-plane_normal);

    let t = -(d + ray_origin.dot(*plane_normal)) / ray_direction.dot(*plane_normal);

    ray_origin + t * ray_direction

    /*
    float denom = dot(n, v);

    // Prevent divide by zero:
    if (abs(denom) <= 1e-4f)
        return std::nullopt;

    // If you want to ensure the ray reflects off only
    // the "top" half of the plane, use this instead:
    //
    // if (-denom <= 1e-4f)
    //     return std::nullopt;

    float t = -(dot(n, p) + d) / dot(n, v);

    // Use pointy end of the ray.
    // It is technically correct to compare t < 0,
    // but that may be undesirable in a raytracer.
    if (t <= 1e-4)
        return std::nullopt;

    return p + t * v;
    */
}

pub fn ray_aabb_intersection(ray_origin: &glam::Vec3, ray_direction_inv: &glam::Vec3, box0: &glam::Vec3, box1: &glam::Vec3) -> bool {    
    let mut t_min = 0.0;
    let mut t_max = f32::INFINITY;    

    fn min(x: f32, y: f32) -> f32 {
        if x < y {
            x 
        } else {
            y
        }
    }
    
    fn max(x: f32, y: f32) -> f32 {
        if x > y {
            x
        } else {
            y
        }
    }

    let t1 = (box0.x - ray_origin.x) * ray_direction_inv.x;
    let t2 = (box1.x - ray_origin.x) * ray_direction_inv.x;
    
    t_min = min(max(t1, t_min), max(t2, t_min));
    t_max = max(min(t1, t_max), min(t2, t_max));

    let t1 = (box0.y - ray_origin.y) * ray_direction_inv.y;
    let t2 = (box1.y - ray_origin.y) * ray_direction_inv.y;
    
    t_min = min(max(t1, t_min), max(t2, t_min));
    t_max = max(min(t1, t_max), min(t2, t_max));

    let t1 = (box0.z - ray_origin.z) * ray_direction_inv.z;
    let t2 = (box1.z - ray_origin.z) * ray_direction_inv.z;
    
    t_min = min(max(t1, t_min), max(t2, t_min));
    t_max = max(min(t1, t_max), min(t2, t_max));

    return t_min < t_max;
}

pub fn ray_aabb_intersection2(ray_origin: &glam::Vec3, ray_direction_inv: &glam::Vec3, box0: &glam::Vec3, box1: &glam::Vec3) -> bool {    
    // Absolute distances to lower and upper box coordinates
    let t_lower = (box0 - ray_origin) * ray_direction_inv;
    let t_upper = (box1 - ray_origin)* ray_direction_inv;

    //warn!("t_lower {}, t_upper {}", t_lower, t_upper);

    let ray_tmin = 0.0;
    let ray_tmax = f32::INFINITY;

    // The four t-intervals (for x -/ y- /z -slabs , and ray p(t))    

    let t_mins = t_lower.min(t_upper);
    let t_maxes = t_lower.max(t_upper);

    let t_mins = glam::vec4(t_mins.x, t_mins.y, t_mins.z, ray_tmin);
    let t_maxes = glam::vec4(t_maxes.x, t_maxes.y, t_maxes.z, ray_tmax);    

    let t_box_min = t_mins.max_element();
    let t_box_max = t_maxes.min_element();

    return t_box_min <= t_box_max;
}

pub fn screen_to_ray_orig(camera: &Camera, width: f32, height: f32, x: f32, y: f32) -> (glam::Vec4, glam::Vec3) {
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
