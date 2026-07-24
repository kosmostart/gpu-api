use glam::Mat4;

#[repr(C)]
#[derive(Copy, Clone, Debug)]
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
    pub model_matrix: Mat4,
    pub is_animated: u32,
    pub node_index: u32,
    pub joints_offset: u32,
    pub material_index: u32,
}

unsafe impl bytemuck::Pod for InstanceData {}
unsafe impl bytemuck::Zeroable for InstanceData {}

/// Структура задачи для Compute-шейдера (Выравнивание WebGPU по 16 байт)
#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct CullingTask {
    pub start_object_index: u32,
    pub object_count: u32,
    pub material_index: u32,
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

/// Метаданные геометрии расширяются (Вы знаете смещение инстансов модели во VRAM изначально)
pub struct ModelGeometryMeta {
    pub id: u32,
    pub index_count: u32,
    pub first_index: u32,
    pub base_vertex: i32,
    /// С какого индекса в глобальном GPU-буфере начинаются инстансы этой модели в мире
    pub global_instance_buffer_offset: u32, 
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
    pub index_count: u32,    // Количество индексов меша модели
    pub instance_count: u32, // СКОЛЬКО таких объектов нашел куллинг октодерева
    pub first_index: u32,    // Смещение геометрии в mega_index_buffer
    pub base_vertex: i32,    // Смещение геометрии в mega_vertex_buffer
    pub first_instance: u32, // Смещение в instances_buffer кадра
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
