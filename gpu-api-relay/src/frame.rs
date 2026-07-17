use glam::Mat4;

#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct NodeUniform {    
    pub info: [u32; 4],    
    pub transform: Mat4,
}

unsafe impl bytemuck::Pod for NodeUniform {}
unsafe impl bytemuck::Zeroable for NodeUniform {}

#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct InstanceData {
    pub model_matrix: Mat4,
    pub is_animated: u32,
    pub node_index: u32,
    pub joints_offset: u32,
    pub material_index: u32,
}

unsafe impl bytemuck::Pod for InstanceData {}
unsafe impl bytemuck::Zeroable for InstanceData {}

pub struct FrameData {
    pub instances: Vec<InstanceData>,
    pub nodes: Vec<NodeUniform>,
    pub joints: Vec<Mat4>,
}

impl FrameData {
    pub fn clear(&mut self)     {
        self.instances.clear();
        self.nodes.clear();
        self.joints.clear();
    }
}
