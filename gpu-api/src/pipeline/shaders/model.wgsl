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

struct DirLightInfo {
    dir: vec4<f32>,
    color: vec4<f32>,
}

struct PointLightInfo {
    position: vec4<f32>,
    color: vec4<f32>,
}

struct SpotLightInfo {
    position: vec4<f32>,
    color: vec4<f32>,
    // x: inner angle, y: outter angle, z: angle decay, w: distance decay
    angel_decay: vec4<f32>,
}

struct LightsInfo {
    // x,y,z: dir light, point light, spotlight
    light_count: vec4<u32>,
    lights_info: array<vec4<f32>>,
}

struct SurfaceProps {
    world_pos: vec3<f32>,
    world_normal: vec3<f32>,
    reflection_dir: vec3<f32>,
    uv0: vec2f,
    f0: vec3<f32>,
    metallic: f32,
    roughness: f32,
    albedo: vec3<f32>,
    surface_ao: f32,
}

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
    @location(1) tangent_position: vec3<f32>,
    @location(2) tangent_light_position: vec3<f32>,
    @location(3) tangent_view_position: vec3<f32>,
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

    let world_normal = normalize((model_matrix * vec4<f32>(vertex_input.normal, 0.0)).xyz);
    let world_tangent = normalize((model_matrix * vec4<f32>(vertex_input.tangent, 0.0)).xyz);
    let world_bitangent = normalize((model_matrix * vec4<f32>(vertex_input.bitangent, 0.0)).xyz);

    let tangent_matrix = transpose(mat3x3<f32>(
        world_tangent,
        world_bitangent,
        world_normal,
    ));    

    let light_position = vec3<f32>(4.0, 2.0, 6.0);
        
    fragment_input.tangent_position = tangent_matrix * model_position.xyz;
    fragment_input.tangent_view_position = tangent_matrix * camera.camera_position;
    fragment_input.tangent_light_position = tangent_matrix * light_position;
    
    return fragment_input;
}

@fragment
fn fs_main(fragment_input: FragmentInput) -> @location(0) vec4<f32> { 
    let object_color = textureSample(base_color_texture, base_color_sampler, fragment_input.texture_coordinates);

    //return textureSample(base_color_texture, base_color_sampler, fragment_input.texture_coordinates);

    let object_normal = textureSample(normal_texture, normal_sampler, fragment_input.texture_coordinates);

    let ambient_strength = 0.1;
    let light_position = vec3<f32>(4.0, 2.0, 6.0);
    let light_color = vec3<f32>(1.0, 1.0, 1.0);
    let ambient_color = light_color * ambient_strength;

    // Create the lighting vectors
    let tangent_normal = object_normal.xyz * 2.0 - 1.0;
    let light_dir = normalize(fragment_input.tangent_light_position - fragment_input.tangent_position);
    let view_direction = normalize(fragment_input.tangent_view_position - fragment_input.tangent_position);
    let half_dir = normalize(view_direction + light_dir);

    let diffuse_strength = max(dot(tangent_normal, light_dir), 0.0);
    let diffuse_color = light_color * diffuse_strength;    

    let specular_strength = pow(max(dot(tangent_normal, half_dir), 0.0), 32.0);
    let specular_color = specular_strength * light_color;

    let result = (ambient_color + diffuse_color + specular_color) * object_color.xyz;

    return vec4<f32>(result, object_color.a);    
}

/*
struct MaterialUniform {
    albedo: vec3<f32>,
    metallic: f32,
    roughness: f32,
    ao: f32, // Ambient Occlusion
};

struct LightUniform {
    position: vec3<f32>,
    color: vec3<f32>,
    intensity: f32,
};

@group(2) @binding(0) var<uniform> material: MaterialUniform;
@group(3) @binding(0) var<uniform> light: LightUniform;

@group(4) @binding(0) var albedo_texture: texture_2d<f32>;
@group(4) @binding(1) var albedo_sampler: sampler;

@group(5) @binding(0) var normal_texture: texture_2d<f32>;
@group(5) @binding(1) var normal_sampler: sampler;

@fragment
fn fs_main(
    in: VertexOutput,
) -> @location(0) vec4<f32> {
    let N = normalize(in.world_normal);
    let V = normalize(camera.position - in.world_position); // Assuming camera position is available

    let albedo_map_color = textureSample(albedo_texture, albedo_sampler, in.uv).rgb;
    let base_color = albedo_map_color * material.albedo;

    // Basic PBR calculation (simplified for brevity)
    let L = normalize(light.position - in.world_position);
    let H = normalize(V + L);

    let NdotL = max(dot(N, L), 0.0);

    // Diffuse component (Lambertian)
    let diffuse = base_color / PI;

    // Specular component (simplified Cook-Torrance BRDF)
    let F0 = mix(vec3<f32>(0.04), base_color, material.metallic); // Fresnel reflectance at normal incidence
    let F = F0 + (1.0 - F0) * pow(1.0 - max(dot(H, V), 0.0), 5.0); // Fresnel equation (Schlick's approximation)

    let alpha = material.roughness * material.roughness;
    let D = pow(alpha / (PI * pow(pow(dot(N, H), 2.0) * (alpha * alpha - 1.0) + 1.0, 2.0)), 2.0); // Normal Distribution Function (GGX)

    let k = pow(material.roughness + 1.0, 2.0) / 8.0; // Geometric Shadowing (Schlick-GGX)
    let G = (dot(N, L) / (dot(N, L) * (1.0 - k) + k)) * (dot(N, V) / (dot(N, V) * (1.0 - k) + k));

    let specular = (D * G * F) / (4.0 * dot(N, L) * dot(N, V) + 0.001); // Adding epsilon to prevent division by zero

    let radiance = light.color * light.intensity;

    let Lo = (diffuse * (1.0 - F) + specular) * radiance * NdotL;

    let final_color = Lo * material.ao; // Apply ambient occlusion

    return vec4<f32>(final_color, 1.0);
}
*/
