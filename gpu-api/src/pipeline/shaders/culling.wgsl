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

struct CullingTask {
    start_object_index: u32,
    object_count: u32,
    padding: vec2<u32>, 
};

struct DrawIndexedIndirectCmd {
    index_count: u32,
    instance_count: atomic<u32>,
    first_index: u32,
    base_vertex: i32,
    first_instance: u32,
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
    @builtin(global_invocation_id) global_id: vec3<u32>,    
    @builtin(workgroup_id) workgroup_id: vec3<u32>,    
    @builtin(local_invocation_id) local_id: vec3<u32>
) {    
    let task_index = workgroup_id.x;
    let total_tasks = arrayLength(&culling_tasks);
    if (task_index >= total_tasks) { return; }

    let task = culling_tasks[task_index];
        
    for (var i = local_id.x; i < task.object_count; i = i + 64u) {        
        let global_instance_id = task.start_object_index + i;
        let instance = global_instances[global_instance_id];
                
        let corners = array<vec3<f32>, 8>(
            vec3<f32>(instance.aabb_min.x, instance.aabb_min.y, instance.aabb_min.z),
            vec3<f32>(instance.aabb_max.x, instance.aabb_min.y, instance.aabb_min.z),
            vec3<f32>(instance.aabb_min.x, instance.aabb_max.y, instance.aabb_min.z),
            vec3<f32>(instance.aabb_max.x, instance.aabb_max.y, instance.aabb_min.z),
            vec3<f32>(instance.aabb_min.x, instance.aabb_min.y, instance.aabb_max.z),
            vec3<f32>(instance.aabb_max.x, instance.aabb_min.y, instance.aabb_max.z),
            vec3<f32>(instance.aabb_min.x, instance.aabb_max.y, instance.aabb_max.z),
            vec3<f32>(instance.aabb_max.x, instance.aabb_max.y, instance.aabb_max.z)
        );
        
        var world_min = vec3<f32>(1e30);
        var world_max = vec3<f32>(-1e30);
        for (var j = 0u; j < 8u; j = j + 1u) {
            let world_corner = (instance.model_matrix * vec4<f32>(corners[j], 1.0)).xyz;
            world_min = min(world_min, world_corner);
            world_max = max(world_max, world_corner);
        }
                
        if (is_aabb_visible(world_min, world_max)) {            
            let cmd_id = task_index; 
            let local_slot = atomicAdd(&indirect_commands[cmd_id].instance_count, 1u);
            let write_index = indirect_commands[cmd_id].first_instance + local_slot;
            visible_instance_indices[write_index] = global_instance_id;
        }
    }
}
