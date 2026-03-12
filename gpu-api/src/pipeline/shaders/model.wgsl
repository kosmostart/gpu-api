// Material
@group(0) @binding(0)
var base_color_texture: texture_2d<f32>;
@group(0) @binding(1)
var base_color_sampler: sampler;
@group(0) @binding(2)
var metallic_roughness_texture: texture_2d<f32>;
@group(0) @binding(3)
var metallic_roughness_sampler: sampler;
@group(0) @binding(4)
var normal_texture: texture_2d<f32>;
@group(0) @binding(5)
var normal_sampler: sampler;
@group(0) @binding(6)
var<uniform> material_factors: MaterialFactorsUniform;
@group(0) @binding(7)
var emissive_texture: texture_2d<f32>;
@group(0) @binding(8)
var emissive_sampler: sampler;

// Camera
struct CameraUniform {
    camera_position: vec3<f32>,
    padding: u32,
    view: mat4x4<f32>,
    projection: mat4x4<f32>    
};

// Joint matrices
struct JointUniform {
    joint_matrices: array<mat4x4<f32>, 100>
};

// Node
struct NodeUniform {
    info: vec4<u32>,
    transform: mat4x4<f32>
};

// Material factors
struct MaterialFactorsUniform {
    base_color_factor: vec4<f32>,
    emissive_factor: vec3<f32>,
    metallic_factor: f32,
    padding: vec3<u32>,
    roughness_factor: f32
}

@group(1) @binding(0)
var<uniform> camera: CameraUniform;
@group(2) @binding(0)
var<uniform> joint_uniform: JointUniform;
@group(3) @binding(0)
var<uniform> node_uniform: NodeUniform;

struct VertexInput {    
    @location(0) position: vec3<f32>,    
    @location(1) texture_coordinates: vec2<f32>,
    @location(2) normal: vec3<f32>,
    @location(3) tangent: vec3<f32>,
    @location(4) bitangent: vec3<f32>,
    @location(5) joints: vec4<u32>,
    @location(6) weights: vec4<f32>
};

struct InstanceInput {
    @location(7) model_matrix_0: vec4<f32>,
    @location(8) model_matrix_1: vec4<f32>,
    @location(9) model_matrix_2: vec4<f32>,
    @location(10) model_matrix_3: vec4<f32>,
    @location(11) @interpolate(flat) is_animated: u32    
};

struct FragmentInput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) texture_coordinates: vec2<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) tangent: vec3<f32>,
    @location(3) bitangent: vec3<f32>    
};

@vertex
fn vs_main(vertex_input: VertexInput, instance: InstanceInput) -> FragmentInput {    
    var fragment_input: FragmentInput;

    var model_matrix = mat4x4<f32>(
        instance.model_matrix_0,
        instance.model_matrix_1,
        instance.model_matrix_2,
        instance.model_matrix_3
    );    

    if (instance.is_animated == 1) {
        if (node_uniform.info[0] == 1) {
            model_matrix = model_matrix * node_uniform.transform;
        } else {
            var skin_matrix: mat4x4<f32> = 
                vertex_input.weights[0] * joint_uniform.joint_matrices[vertex_input.joints[0]] +
                vertex_input.weights[1] * joint_uniform.joint_matrices[vertex_input.joints[1]] +
                vertex_input.weights[2] * joint_uniform.joint_matrices[vertex_input.joints[2]] +
                vertex_input.weights[3] * joint_uniform.joint_matrices[vertex_input.joints[3]];

            model_matrix = model_matrix * skin_matrix * node_uniform.transform;            
        }        
    } else {
        model_matrix = model_matrix * node_uniform.transform;        
    }

    let model_position = model_matrix * vec4<f32>(vertex_input.position, 1.0);

    fragment_input.clip_position = camera.projection * model_position;
    fragment_input.texture_coordinates = vertex_input.texture_coordinates;              
    fragment_input.normal = vertex_input.normal;
    fragment_input.tangent = vertex_input.tangent;
    fragment_input.bitangent = vertex_input.bitangent;
    
    return fragment_input;
}

@fragment
fn fs_main(fragment_input: FragmentInput) -> @location(0) vec4<f32> { 
    //return textureSample(base_color_texture, base_color_sampler, fragment_input.texture_coordinates);

    let light_position = vec3<f32>(4.0, 2.0, 6.0);
    let light_color = vec3<f32>(1.0, 1.0, 1.0);
    let light_intensity = 15.0;
    let world_pos = fragment_input.clip_position.xyz;

    let N_tangent = textureSample(normal_texture, normal_sampler, fragment_input.texture_coordinates).rgb * 2.0 - 1.0;
    let TBN = mat3x3<f32>(fragment_input.tangent, fragment_input.bitangent, fragment_input.normal);
    let N = normalize(TBN * N_tangent);

    let V = normalize(camera.camera_position - world_pos);
    let L = normalize(light_position - world_pos);
    let H = normalize(V + L);

    let base_color_tex = textureSample(base_color_texture, base_color_sampler, fragment_input.texture_coordinates).rgb;
    let metallic_roughness_tex = textureSample(metallic_roughness_texture, metallic_roughness_sampler, fragment_input.texture_coordinates).rg;
    let metallic = metallic_roughness_tex.r;
    let roughness = metallic_roughness_tex.g;

    let albedo = base_color_tex;
    let F0 = mix(vec3<f32>(0.04), albedo, metallic);

    // Basic BRDF approximation (Cook-Torrance)
    let NdotL = max(dot(N, L), 0.0);
    let NdotV = max(dot(N, V), 0.0);
    let NdotH = max(dot(N, H), 0.0);
    let HdotV = max(dot(H, V), 0.0);

    // D (Distribution) - Trowbridge-Reitz GGX
    let alpha = roughness * roughness;
    let alpha2 = alpha * alpha;
    let denom_d = NdotH * NdotH * (alpha2 - 1.0) + 1.0;
    let D = alpha2 / (PI * denom_d * denom_d);

    // G (Geometry) - Schlick-GGX
    let k = (roughness + 1.0) * (roughness + 1.0) / 8.0;
    let G_v = NdotV / (NdotV * (1.0 - k) + k);
    let G_l = NdotL / (NdotL * (1.0 - k) + k);
    let G = G_v * G_l;

    // F (Fresnel) - Schlick approximation
    let F = F0 + (vec3<f32>(1.0) - F0) * pow(clamp(1.0 - HdotV, 0.0, 1.0), 5.0);

    let specular = (D * G * F) / (4.0 * NdotL * NdotV + 0.001); // Add epsilon to avoid division by zero

    let k_s = F;
    let k_d = (vec3<f32>(1.0) - k_s) * (1.0 - metallic);

    let radiance = light_color * light_intensity;
    let color = (k_d * albedo / PI + specular) * radiance * NdotL;

    return vec4<f32>(color, 1.0);
}

const PI: f32 = 3.1415926535;
