struct CameraUniform {
    camera_position: vec3<f32>,
    padding: u32,
    view: mat4x4<f32>,
    projection: mat4x4<f32>,
    frustum_planes: array<vec4<f32>, 6>,
};

struct InstanceData {
    model_matrix: mat4x4<f32>,
    is_animated: u32,
    node_index: u32,
    joints_offset: u32,
    material_index: u32,
    aabb_min: vec3<f32>,
    aabb_max: vec3<f32>,
};

// Исправлено: Добавлено явное смещение вывода для конкретного чанка
struct CullingTask {
    start_object_index: u32, // Индекс первого объекта чанка в global_instances
    object_count: u32,       // Сколько объектов в чанке сейчас
    dst_output_offset: u32,  // Равно chunk.gpu_buffer_offset (куда писать ID видимых)
    padding: u32, 
};

struct DrawIndexedIndirectCmd {
    index_count: u32,
    instance_count: atomic<u32>, // Атомарный счетчик для GPU
    first_index: u32,
    base_vertex: i32,
    first_instance: u32,         // Для DrawIndirect должен быть равен dst_output_offset
};

@group(0) @binding(0) var<uniform> camera: CameraUniform;

@group(1) @binding(0) var<storage, read> culling_tasks: array<CullingTask>;
@group(1) @binding(1) var<storage, read> global_instances: array<InstanceData>;
@group(1) @binding(2) var<storage, read_write> visible_instance_indices: array<u32>;
@group(1) @binding(3) var<storage, read_write> indirect_commands: array<DrawIndexedIndirectCmd>;

fn is_aabb_visible(aabb_min: vec3<f32>, aabb_max: vec3<f32>) -> bool {
    for (var i = 0u; i < 6u; i = i + 1u) {
        let plane = camera.frustum_planes[i];
        var p = aabb_min;
        if (plane.x >= 0.0) { p.x = aabb_max.x; }
        if (plane.y >= 0.0) { p.y = aabb_max.y; }
        if (plane.z >= 0.0) { p.z = aabb_max.z; }
        if (dot(plane.xyz, p) + plane.w < 0.0) { return false; }
    }
    return true;
}

@compute @workgroup_size(64)
fn culling_main(
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>
) {
    let task_index = workgroup_id.x;
    let total_tasks = arrayLength(&culling_tasks);
    if (task_index >= total_tasks) { return; }

    let task = culling_tasks[task_index];
    let cmd_id = task_index; 
    
    for (var i = local_id.x; i < task.object_count; i = i + 64u) {
        let global_instance_id = task.start_object_index + i;
        let instance = global_instances[global_instance_id];
        
        // ОПТИМИЗАЦИЯ: Трансформация AABB методом Эйли (без развертки в 8 вершин)
        let matrix = instance.model_matrix;
        
        // Извлекаем позицию (трансляцию) из матрицы
        let m_col3 = vec3<f32>(matrix[3].xyz);
        var world_min = m_col3;
        var world_max = m_col3;
        
        // Матричное умножение интервалов
        for (var col = 0u; col < 3u; col = col + 1u) {
            let m_col = matrix[col].xyz;
            let a = m_col * instance.aabb_min[col];
            let b = m_col * instance.aabb_max[col];
            
            world_min = world_min + min(a, b);
            world_max = world_max + max(a, b);
        }
        
        // Тест видимости
        if (is_aabb_visible(world_min, world_max)) {
            // Атомарно инкрементируем счетчик инстансов в команде отрисовки
            let local_slot = atomicAdd(&indirect_commands[cmd_id].instance_count, 1u);
            
            // Безопасная запись на основе смещения из Task, минуя чтение из командного буфера
            let write_index = task.dst_output_offset + local_slot;
            visible_instance_indices[write_index] = global_instance_id;
        }
    }
}
