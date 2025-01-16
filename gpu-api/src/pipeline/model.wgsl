// Texture
@group(0) @binding(0)
var texture_data: texture_2d<f32>;
@group(0) @binding(1)
var texture_sampler: sampler;

// Camera
struct CameraUniform {
    projection: mat4x4<f32>
};

@group(1) @binding(0)
var<uniform> camera: CameraUniform;

struct VertexInput {    
    @location(0) position: vec3<f32>,    
    @location(1) texture_coordinates: vec2<f32>,
    @location(2) normal: vec3<f32>,
    @location(3) tangent: vec4<f32>,
    @location(4) joints: vec4<u32>,
    @location(5) weights: vec4<f32>
};

struct FragmentInput {
    @builtin(position) clip_position: vec4<f32>,    
    @location(0) texture_coordinates: vec2<f32>,
    @location(1) normal: vec3<f32>
};

struct InstanceInput {
    @location(6) model_matrix_0: vec4<f32>,
    @location(7) model_matrix_1: vec4<f32>,
    @location(8) model_matrix_2: vec4<f32>,
    @location(9) model_matrix_3: vec4<f32>
};

// Vertex shader
@vertex
//@stage(vertex)
fn vs_main(vertex_input: VertexInput, instance: InstanceInput) -> FragmentInput {    
    var fragment_input: FragmentInput;

    let model_matrix = mat4x4<f32>(
        instance.model_matrix_0,
        instance.model_matrix_1,
        instance.model_matrix_2,
        instance.model_matrix_3
    );

    fragment_input.clip_position = camera.projection * model_matrix * vec4<f32>(vertex_input.position, 1.0);
    fragment_input.texture_coordinates = vertex_input.texture_coordinates;
    fragment_input.normal = vertex_input.normal;
    
    return fragment_input;
}

// Fragment shader
@fragment
//@stage(fragment)
fn fs_main(fragment_input: FragmentInput) -> @location(0) vec4<f32> {
    return textureSample(texture_data, texture_sampler, fragment_input.texture_coordinates); // Texture
}
