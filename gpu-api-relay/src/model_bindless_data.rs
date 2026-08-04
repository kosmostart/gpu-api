use glam::{Mat4, Vec4};

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct CameraUniform {
    pub camera_position: [f32; 3],
    pub padding: u32,
    pub view: Mat4,
    pub projection: Mat4,
    pub frustum: [Vec4; 6],
}

unsafe impl bytemuck::Pod for CameraUniform {}
unsafe impl bytemuck::Zeroable for CameraUniform {}

#[repr(C)]
#[derive(Copy, Clone, Debug, Default)]
pub struct Vertex {    
    pub position: [f32; 3],    
    pub uv: [f32; 2],
    pub normal: [f32; 3],
    pub tangent: [f32; 3],
    pub bitangent: [f32; 3],
    pub joints: [u32; 4],
    pub weights: [f32; 4],
}

unsafe impl bytemuck::Pod for Vertex {}
unsafe impl bytemuck::Zeroable for Vertex {}

#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct NodeData {    
    pub info: [u32; 4],    
    pub transform: Mat4,
}

unsafe impl bytemuck::Pod for NodeData {}
unsafe impl bytemuck::Zeroable for NodeData {}

#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct InstanceData {
    pub model_matrix: glam::Mat4, // 64 байта
    
    // Блок метаданных
    pub is_animated: u32,         // 4 байта
    pub node_index: u32,          // 4 байта
    pub joints_offset: u32,       // 4 байта
    pub material_index: u32,      // 4 байта
    pub primitive_index: u32,     // 4 байта
    
    // Добиваем блок метаданных до 16-байтовой границы (20 + 12 = 32 байта)
    pub _pad0: u32,               // 4 байта
    pub _pad1: u32,               // 4 байта
    pub _pad2: u32,               // 4 байта
    
    // Безопасные AABB без скрытых SIMD-компонентов glam
    pub aabb_min: [f32; 3],       // 12 байт
    pub _pad_aabb1: u32,          // 4 байта (округляем до 16 байт)
    
    pub aabb_max: [f32; 3],       // 12 байт
    pub _pad_aabb2: u32,          // 4 байта (округляем до 16 байт)
}

unsafe impl bytemuck::Pod for InstanceData {}
unsafe impl bytemuck::Zeroable for InstanceData {}

#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct CullingTask {
    pub start_object_index: u32,
    pub object_count: u32,
    pub lod_level: u32,
    pub _padding: u32,
}

unsafe impl bytemuck::Pod for CullingTask {}
unsafe impl bytemuck::Zeroable for CullingTask {}

#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct VisibleInstanceData {
    pub instance_id: u32,
    pub material_index: u32,
}

unsafe impl bytemuck::Pod for VisibleInstanceData {}
unsafe impl bytemuck::Zeroable for VisibleInstanceData {}

pub struct PrimitiveMeta {    
    pub id: u32,    
    /// Vertex Buffer base index    
    pub base_vertex: i32,    
    pub lod_index_counts: [u32; 3],    
    pub lod_first_indices: [u32; 3],    
    /// Fixed VRAM budget for primitive
    pub max_global_instances: u32,
}


#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct MaterialFactors {
    pub base_color_factor: [f32; 4],
    pub emissive_factor: [f32; 3],
    pub metallic_factor: f32,    
    pub roughness_factor: f32,
    pub padding: [u32; 3],
}

unsafe impl bytemuck::Pod for MaterialFactors {}
unsafe impl bytemuck::Zeroable for MaterialFactors {}

#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct DrawIndexedIndirectCommand {
    /// Number of indices per mesh
    pub index_count: u32,
    // How many off these objects will be drawn
    pub instance_count: u32,
    /// Geometry offset in mega_index_buffer
    pub first_index: u32,
    /// Geometry offset in mega_vertex_buffer
    pub base_vertex: i32,
    /// Offset in instances_buffer per frame
    pub first_instance: u32,
}

unsafe impl bytemuck::Pod for DrawIndexedIndirectCommand {}
unsafe impl bytemuck::Zeroable for DrawIndexedIndirectCommand {}

pub struct FrameData {
    pub instances: Vec<InstanceData>,
    pub nodes: Vec<NodeData>,
    pub joints: Vec<Mat4>,
}

impl FrameData {
    pub fn clear(&mut self)     {
        self.instances.clear();
        self.nodes.clear();
        self.joints.clear();
    }
}
