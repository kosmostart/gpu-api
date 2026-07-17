// ============================================================================
// ГРУППА 0: ЧИСТЫЙ BINDLESS МАТЕРИАЛОВ И ТЕКСТУР (Шаг 3)
// ============================================================================
// Массивы текстур (binding_array). Сюда при старте уровня загружаются ВСЕ текстуры игры.
// Шейдер выбирает нужную текстуру динамически прямо во фрагментном проходе!
@group(0) @binding(0) var base_color_textures: binding_array<texture_2d<f32>>;
@group(0) @binding(1) var base_color_samplers: binding_array<sampler>;

@group(0) @binding(2) var metallic_roughness_textures: binding_array<texture_2d<f32>>;
@group(0) @binding(3) var metallic_roughness_samplers: binding_array<sampler>;

@group(0) @binding(4) var normal_textures: binding_array<texture_2d<f32>>;
@group(0) @binding(5) var normal_samplers: binding_array<sampler>;

@group(0) @binding(6) var emissive_textures: binding_array<texture_2d<f32>>;
@group(0) @binding(7) var emissive_samplers: binding_array<sampler>;

// Структура факторов материалов для всех объектов (тоже упакована в Storage Buffer)
struct MaterialFactors {
    base_color_factor: vec4<f32>,
    emissive_factor: vec3<f32>,
    metallic_factor: f32,
    roughness_factor: f32,
    padding: vec3<f32>,
}
@group(0) @binding(8) var<storage, read> global_materials: array<MaterialFactors>;

// ============================================================================
// ГРУППА 1: СТАТИЧНЫЕ / ОБЩИЕ ДАННЫЕ КАДРА
// ============================================================================
struct CameraUniform {
    camera_position: vec3<f32>,
    padding: u32,
    view: mat4x4<f32>,
    projection: mat4x4<f32>
};
@group(1) @binding(0) var<uniform> camera: CameraUniform;

// ============================================================================
// ГРУППА 2: ГЛОБАЛЬНЫЕ ДАННЫЕ GPU-DRIVEN (Трансформы, Кости, Инстансы)
// ============================================================================
// Шаг 1: Глобальный буфер трансформаций ВСЕХ нод всех объектов кадра
struct GpuNodeData {
    info: vec4<u32>,
    transform: mat4x4<f32>
};
@group(2) @binding(0) var<storage, read> global_nodes: array<GpuNodeData>;

// Шаг 2: Глобальный буфер матриц суставов ВСЕХ анимированных объектов кадра
@group(2) @binding(1) var<storage, read> global_joint_matrices: array<mat4x4<f32>>;

// Финальная структура инстанса, объединяющая геометрию, анимацию и материал
struct GpuInstanceData {
    model_matrix: mat4x4<f32>,
    is_animated: u32,
    node_index: u32,       // Индекс в массиве global_nodes (Трансформы)
    joints_offset: u32,    // Смещение начала скелета в global_joint_matrices (Анимация)
    material_index: u32,   // Индекс материала и текстур (Материалы/Bindless)
};
@group(2) @binding(2) var<storage, read> global_instances: array<GpuInstanceData>;

// ============================================================================
// ВХОДНЫЕ И ВЫХОДНЫЕ СТРУКТУРЫ
// ============================================================================
struct VertexInput {    
    @location(0) position: vec3<f32>,    
    @location(1) texture_coordinates: vec2<f32>,
    @location(2) normal: vec3<f32>,
    @location(3) tangent: vec3<f32>,
    @location(4) bitangent: vec3<f32>,
    @location(5) joints: vec4<u32>,
    @location(6) weights: vec4<f32>
};

struct FragmentInput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) texture_coordinates: vec2<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) tangent: vec3<f32>,
    @location(3) bitangent: vec3<f32>,
    @location(4) world_position: vec3<f32>,
    @location(5) @interpolate(flat) material_index: u32, // Пробрасываем ID материала во фрагментный шейдер
};

// ============================================================================
// ВЕРШИННЫЙ ШЕЙДЕР (VERTEX SHADER)
// ============================================================================
@vertex
fn vs_main(vertex_input: VertexInput, @builtin(instance_index) instance_idx: u32) -> FragmentInput {    
    let instance = global_instances[instance_idx];
    var model_matrix = instance.model_matrix;    
    let node = global_nodes[instance.node_index];

    // Рассчитываем скиннинг (Шаг 2)
    if (instance.is_animated == 1u) {
        if (node.info[0] == 1u) {
            model_matrix = model_matrix * node.transform;
        } else {
            let j0 = instance.joints_offset + vertex_input.joints[0];
            let j1 = instance.joints_offset + vertex_input.joints[1];
            let j2 = instance.joints_offset + vertex_input.joints[2];
            let j3 = instance.joints_offset + vertex_input.joints[3];

            var skin_matrix: mat4x4<f32> = 
                vertex_input.weights[0] * global_joint_matrices[j0] +
                vertex_input.weights[1] * global_joint_matrices[j1] +
                vertex_input.weights[2] * global_joint_matrices[j2] +
                vertex_input.weights[3] * global_joint_matrices[j3];

            model_matrix = model_matrix * skin_matrix * node.transform;            
        }        
    } else {
        model_matrix = model_matrix * node.transform;        
    }

    let model_position = model_matrix * vec4<f32>(vertex_input.position, 1.0);
    
    var out: FragmentInput;
    out.clip_position = camera.projection * model_position; 
    out.world_position = model_position.xyz;
    out.texture_coordinates = vertex_input.texture_coordinates;
    out.material_index = instance.material_index; // Передаем индекс дальше для текстурирования
    
    let normal_matrix = mat3x3<f32>(model_matrix[0].xyz, model_matrix[1].xyz, model_matrix[2].xyz);
    out.normal = normalize(normal_matrix * vertex_input.normal);
    out.tangent = normalize(normal_matrix * vertex_input.tangent);
    out.bitangent = normalize(normal_matrix * vertex_input.bitangent);
    
    return out;
}

// ============================================================================
// ФРАГМЕНТНЫЙ ШЕЙДЕР (FRAGMENT SHADER) - Демонстрация Bindless
// ============================================================================
@fragment
fn fs_main(in: FragmentInput) -> @location(0) vec4<f32> {
    // Получаем индекс материала этого конкретного инстанса персонажа/ландшафта
    let mat_idx = in.material_index;
    
    // Считываем математические факторы цвета и шероховатости из глобального буфера
    let factors = global_materials[mat_idx];

    // Выполняем Bindless-выборку текстур, используя динамический индекс!
    let base_color = textureSample(
        base_color_textures[mat_idx], 
        base_color_samplers[mat_idx], 
        in.texture_coordinates
    ) * factors.base_color_factor;

    let normal_map = textureSample(
        normal_textures[mat_idx], 
        normal_samplers[mat_idx], 
        in.texture_coordinates
    );

    let metallic_roughness = textureSample(
        metallic_roughness_textures[mat_idx], 
        metallic_roughness_samplers[mat_idx], 
        in.texture_coordinates
    );

    // ... далее идет ваша стандартная логика расчета PBR освещения (Lighting) ...

    return base_color;
}
