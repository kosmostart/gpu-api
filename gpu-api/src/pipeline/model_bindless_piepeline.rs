use glam::{Mat4, Vec3, Vec4};

unsafe impl bytemuck::Pod for InstanceData {}
unsafe impl bytemuck::Zeroable for InstanceData {}

#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct InstanceData {
    pub model_matrix: Mat4,
    pub is_animated: u32,
    pub node_index: u32,
    pub joints_offset: u32,
    pub material_index: u32,
}
