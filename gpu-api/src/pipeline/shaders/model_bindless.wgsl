enable wgpu_binding_array;

@group(0) @binding(0) var base_color_textures: binding_array<texture_2d<f32>>;
@group(0) @binding(1) var base_color_samplers: binding_array<sampler>;
@group(0) @binding(2) var metallic_roughness_textures: binding_array<texture_2d<f32>>;
@group(0) @binding(3) var metallic_roughness_samplers: binding_array<sampler>;
@group(0) @binding(4) var normal_textures: binding_array<texture_2d<f32>>;
@group(0) @binding(5) var normal_samplers: binding_array<sampler>;
@group(0) @binding(6) var emissive_textures: binding_array<texture_2d<f32>>;
@group(0) @binding(7) var emissive_samplers: binding_array<sampler>;

struct MaterialFactors {
    base_color_factor: vec4<f32>,
    emissive_factor: vec3<f32>,
    metallic_factor: f32,
    roughness_factor: f32,
    padding: vec3<f32>,
}
@group(0) @binding(8) var<storage, read> global_materials: array<MaterialFactors>;

struct CameraUniform {
    camera_position: vec3<f32>,
    padding: u32,
    view: mat4x4<f32>,
    projection: mat4x4<f32>,
    frustum_planes: array<vec4<f32>, 6>,
};
@group(1) @binding(0) var<uniform> camera: CameraUniform;

struct NodeData {
    info: vec4<u32>,
    transform: mat4x4<f32>,
};
@group(2) @binding(0) var<storage, read> global_nodes: array<NodeData>;
@group(2) @binding(1) var<storage, read> global_joint_matrices: array<mat4x4<f32>>;

struct InstanceData {
    model_matrix: mat4x4<f32>,
    is_animated: u32,
    node_index: u32,
    joints_offset: u32,
    material_index: u32,
};
@group(2) @binding(2) var<storage, read> global_instances: array<InstanceData>;

@group(2) @binding(3) var<storage, read> visible_instance_indices: array<u32>;

struct VertexInput {    
    @location(0) position: vec3<f32>,    
    @location(1) uv: vec2<f32>,
    @location(2) normal: vec3<f32>,
    @location(3) tangent: vec3<f32>,
    @location(4) bitangent: vec3<f32>,
    @location(5) joints: vec4<u32>,
    @location(6) weights: vec4<f32>
};

struct FragmentInput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) tangent: vec3<f32>,
    @location(3) bitangent: vec3<f32>,
    @location(4) world_position: vec3<f32>,
    @location(5) @interpolate(flat) material_index: u32, 
};

@vertex
fn vs_main(
    vertex_input: VertexInput, 
    @builtin(instance_index) draw_instance_idx: u32
) -> FragmentInput {
    let global_object_id = visible_instance_indices[draw_instance_idx];
        
    let instance = global_instances[global_object_id];
    var model_matrix = instance.model_matrix;
    let node = global_nodes[instance.node_index];
    
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
    out.uv = vertex_input.uv;
    out.material_index = instance.material_index; 
        
    let normal_matrix = mat3x3<f32>(model_matrix[0].xyz, model_matrix[1].xyz, model_matrix[2].xyz);
    out.normal = normalize(normal_matrix * vertex_input.normal);
    out.tangent = normalize(normal_matrix * vertex_input.tangent);
    out.bitangent = normalize(normal_matrix * vertex_input.bitangent);
    
    return out;    
}

@fragment
fn fs_main(in: FragmentInput) -> @location(0) vec4<f32> {    
    //return vec4<f32>(1.0, 0.0, 0.0, 1.0);
    
    let mat_idx = in.material_index;
    let factors = global_materials[mat_idx];
    
    let base_color = textureSample(
        base_color_textures[mat_idx], 
        base_color_samplers[mat_idx], 
        in.uv
    ) * factors.base_color_factor;

    let normal_map = textureSample(
        normal_textures[mat_idx], 
        normal_samplers[mat_idx], 
        in.uv
    );

    let metallic_roughness = textureSample(
        metallic_roughness_textures[mat_idx], 
        metallic_roughness_samplers[mat_idx], 
        in.uv
    );    

    return base_color;
}
