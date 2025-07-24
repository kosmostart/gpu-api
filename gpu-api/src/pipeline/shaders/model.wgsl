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

    //return textureSample(base_color_texture, base_color_sampler, fragment_input.texture_coordinates);
    /*    
    // let world_normal = normalize(fs_in.world_normal);
    let world_normal = get_normal_from_map(fragment_input.world_normal, fragment_input.world_position, fragment_input.texture_coordinates);
    let view_direction = normalize(camera.camera_position - fragment_input.world_position);
    let reflection_dir = reflect(-view_direction, world_normal);
            
    

    let metallic_roughness = textureSample(metallic_roughness_texture, metallic_roughness_sampler, fragment_input.texture_coordinates);

    var surface_metallic = material_factors.metallic_factor * metallic_roughness.y;
    var surface_roughness = material_factors.roughness_factor * metallic_roughness.z;
    var surface_ao = 1.0;
    
    /*
  
    if is_ao_map_enabled() {
        surface_ao *= textureSample(t_ao, s_sampler_0, fs_in.uv0).r;
    }
    */
    
    var surface_albedo = textureSample(base_color_texture, base_color_sampler, fragment_input.texture_coordinates).xyz;    

    var f0 = vec3f(0.04);
    f0 = mix(f0, surface_albedo, surface_metallic);

    let surface_props = SurfaceProps(
        fragment_input.world_position,
        world_normal,
        reflection_dir,
        fragment_input.texture_coordinates,
        f0,
        surface_metallic,
        surface_roughness,
        surface_albedo,
        surface_ao,
    );

    var lo = vec3f(0.0);
    var light_info_index: u32 = 0;
    
    // compute dir lighting
    // compute point lighting

    let p1 = vec4<f32>(5.0, -5.0, -5.0, 1.0);
    let c1 = vec4<f32>(800.0, 800.0, 800.0, 1.0);
    let p2 = vec4<f32>(5.0, 5.0, 5.0, 1.0);
    let c2 = vec4<f32>(800.0, 800.0, 800.0, 1.0);
    let p3 = vec4<f32>(-5.0, -5.0, 5.0, 1.0);
    let c3 = vec4<f32>(800.0, 800.0, 800.0, 1.0);
    let p4 = vec4<f32>(5.0, -5.0, 5.0, 1.0);
    let c4 = vec4<f32>(800.0, 800.0, 800.0, 1.0);

    lo += lighting_point(p1.xyz, c1.rgb, surface_props, view_direction);
    //lo += lighting_point(p2.xyz, c2.rgb, surface_props, view_direction);
    //lo += lighting_point(p3.xyz, c3.rgb, surface_props, view_direction);
    //lo += lighting_point(p4.xyz, c4.rgb, surface_props, view_direction);

    /*
    let point_light_count = lighting_infos.light_count.y * 2;

    for (; light_info_index < point_light_count; light_info_index = light_info_index + 2) {
        let cur_point_light_pos = lighting_infos.lights_info[light_info_index];
        let cur_point_light_color = lighting_infos.lights_info[light_info_index+1];
        lo += lighting_point(cur_point_light_pos.xyz, cur_point_light_color.rgb, surface_props, camera_props);
    }
    */

    // compute spot lighting

    var color = lo;

    /*
    if is_ibl_enabled() {
        let ambient = ambient_lighting(surface_props, camera_props);
        color += ambient;
    }
    */

    // HDR tonemapping
    color = color / (color + vec3f(1.0));
    // the gamma correction below is not necessary, because the swapchain format is Bgra8UnormSrgb, 
    // which will covert linear space color to sRGB.
    // gamma correction
    // color = pow(color, vec3f(1.0/2.2));

    return vec4f(color, 1.0);
*/
    /*
    var alpha = f32(1.0);
    let light_dir: vec3<f32> = normalize(vec3<f32>(0.0, 0.0, 1.0));
    
    var use_directional_light = true;
    let texture_output_1 = textureSample(base_color_texture, base_color_sampler, fragment_input.texture_coordinates);
    let xyz_output_2 = texture_output_1.xyz;
    let multiply_output_3 = material_factors.base_color_factor.xyz * xyz_output_2;
    let texture_output_6 = textureSample(metallic_roughness_texture, metallic_roughness_sampler, fragment_input.texture_coordinates);
    let z_output_8 = texture_output_6.z;
    let multiply_output_9 = material_factors.roughness_factor * z_output_8;
    let texture_output_11 = textureSample(normal_texture, normal_sampler, fragment_input.texture_coordinates);
    let xyz_output_12 = texture_output_11.xyz;
    let multiply_output_15 = xyz_output_12 * 2.0000000000000000;
    let sub_output_16 = multiply_output_15 - 1.0000000000000000;
    let tangent_to_object_normal_output_17 = perturb_normal_to_arb(-fragment_input.view_position, fragment_input.normal, sub_output_16, fragment_input.texture_coordinates);
    let y_output_7 = texture_output_6.y;
    let multiply_output_10 = material_factors.metallic_factor * y_output_7;
    let brdf_v_18 = normalize(fragment_input.view_position);
    let brdf_l_18 = normalize(light_dir);
    let brdf_n_18 = normalize(tangent_to_object_normal_output_17);
    let brdf_h_18 = normalize(brdf_l_18 + brdf_v_18);
    let brdf_output_18 = brdf(brdf_v_18, brdf_n_18, brdf_h_18, brdf_l_18, multiply_output_3, multiply_output_9, multiply_output_10);
    use_directional_light = false;
    let texture_output_20 = textureSample(emissive_texture, emissive_sampler, fragment_input.texture_coordinates);
    let xyz_output_21 = texture_output_20.xyz;
    let multiply_output_22 = material_factors.emissive_factor * xyz_output_21;
    let add_output_23 = brdf_output_18 + multiply_output_22;
    var color: vec3<f32> = add_output_23;

    if (use_directional_light) {
        let light_color: vec3<f32> = vec3<f32>(1.0, 1.0, 1.0);
        let light_factor = clamp(dot(normalize(fragment_input.normal), light_dir), 0.0, 1.0) * light_color;
        color = color * light_factor.rgb;
    }
    
    //return vec4<f32>(color, alpha);
    return linear_to_srgb(vec4<f32>(color, alpha));
    */

    //return textureSample(base_color_texture, base_color_sampler, fragment_input.texture_coordinates);

    //return textureSample(normal_texture, normal_sampler, fragment_input.texture_coordinates);

    /*    
    let base_color = material_factors.base_color_factor * textureSample(base_color_texture, base_color_sampler, fragment_input.texture_coordinates);

    let metallic_roughness = textureSample(metallic_roughness_texture, metallic_roughness_sampler, fragment_input.texture_coordinates);
    let metallic = material_factors.metallic_factor * metallic_roughness.b;
    let roughness = material_factors.roughness_factor * metallic_roughness.g;

    var normal = textureSample(normal_texture, normal_sampler, fragment_input.texture_coordinates).xyz;
    normal = normal * 2.0 - 1.0;
    normal = normalize(fragment_input.world_tangent * normal.x + fragment_input.world_bitangent * normal.y + fragment_input.world_normal * normal.z);

    let light_direction = vec3<f32>(-0.25, 0.5, -0.5);

    let light = normalize(light_direction);
    let normal_dot_light = max(dot(normal, light), 0.0);
    let surface_color = base_color.rgb * (0.1 + normal_dot_light);

    return vec4(surface_color, base_color.a);    
    */
}

const PI: f32 = 3.1415926535;

fn get_normal_from_map(v_world_normal: vec3<f32>, world_pos: vec3<f32>, uv: vec2f) -> vec3<f32> {
    let N   = normalize(v_world_normal);

    /*
    if !is_normal_map_enabled() {
        return N;
    }
    */

    let tangent_normal = textureSample(normal_texture, normal_sampler, uv).xyz * 2.0 - 1.0;

    // https://www.w3.org/TR/WGSL/#dpdx-builtin
    let Q1  = dpdx(world_pos);
    let Q2  = dpdy(world_pos);
    let st1 = dpdx(uv);
    let st2 = dpdy(uv);
    
    let T  = normalize(Q1*st2.y - Q2*st1.y);
    let B  = -normalize(cross(N, T));
    let TBN = mat3x3f(T, B, N);

    return normalize(TBN * tangent_normal);
}

fn distribution_ggx(normal: vec3<f32>, half_vec: vec3<f32>, roughness: f32) -> f32 {
    let a = roughness * roughness;
    let a2 = a * a;
    let n_dot_h = max(dot(normal, half_vec), 0.0);
    let n_dot_h_2 = n_dot_h * n_dot_h;
    let nom = a2;
    var denom = n_dot_h_2 * (a2 - 1.0) + 1.0;
    denom = PI * denom * denom;
    return nom / denom;
}

fn geometry_schlick_ggx(n_dot_v: f32, roughness: f32) -> f32 {
    let r = roughness + 1.0;
    let k = (r * r) / 8.0;

    let nom = n_dot_v;
    let denom = n_dot_v * (1.0 - k) + k;

    return nom / denom;
}

fn geometry_smith(normal: vec3<f32>, view_dir: vec3<f32>, light_dir: vec3<f32>, roughness: f32) -> f32 {
    let n_dot_v = max(dot(normal, view_dir), 0.0);
    let n_dot_l = max(dot(normal, light_dir), 0.0);
    let ggx2 = geometry_schlick_ggx(n_dot_v, roughness);
    let ggx1 = geometry_schlick_ggx(n_dot_l, roughness);

    return ggx1 * ggx2;
}

fn fresnel_schlick(cos_theta: f32, f0: vec3<f32>) -> vec3<f32> {
    return f0 + (1.0 - f0) * pow(clamp(1.0 - cos_theta, 0.0, 1.0), 5.0);
}

fn fresnel_schlick_roughness(cos_theta: f32, F0: vec3<f32>, roughness: f32) -> vec3<f32>
{
    return F0 + (max(vec3<f32>(1.0 - roughness), F0) - F0) * pow(clamp(1.0 - cos_theta, 0.0, 1.0), 5.0);
}  

fn lighting_point(light_pos: vec3<f32>, light_color: vec3<f32>, surface_props: SurfaceProps, view_direction: vec3<f32>) -> vec3<f32> {
    let light_dir = normalize(light_pos - surface_props.world_pos);
    let half_dir = normalize(light_dir + view_direction);

    let distance = length(light_pos - surface_props.world_pos);
    let attenuation = 1.0 / (distance * distance);
    let radiance = light_color * attenuation;

    // Cook-Torrance BRDF
    let NDF = distribution_ggx(surface_props.world_normal, half_dir, surface_props.roughness);   
    let G   = geometry_smith(surface_props.world_normal, view_direction, light_dir, surface_props.roughness);      
    let F   = fresnel_schlick(clamp(dot(half_dir, view_direction), 0.0, 1.0), surface_props.f0);
        
    let numerator    = NDF * G * F; 
    let denominator = 4.0 * max(dot(surface_props.world_normal, view_direction), 0.0) * max(dot(surface_props.world_normal, light_dir), 0.0) + 0.0001;
    let specular = numerator / denominator;
    
    // kS is equal to Fresnel
    let kS = F;
    // for energy conservation, the diffuse and specular light can't
    // be above 1.0 (unless the surface emits light); to preserve this
    // relationship the diffuse component (kD) should equal 1.0 - kS.
    var kD = vec3<f32>(1.0) - kS;
    // multiply kD by the inverse metalness such that only non-metals 
    // have diffuse lighting, or a linear blend if partly metal (pure metals
    // have no diffuse light).
    kD = kD * (1.0 - surface_props.metallic);

    // scale light by NdotL
    let NdotL = max(dot(surface_props.world_normal, light_dir), 0.0);        

    // add to outgoing radiance Lo
    return (kD * surface_props.albedo / PI + specular) * radiance * NdotL;
}

/*
// compute ambient lighting
fn ambient_lighting(surface_props: SurfaceProps, view_direction: vec3<f32>) -> vec3<f32> {
    // let ambient = vec3<f32>(0.03) * albedo * surface_ao;
    let F = fresnel_schlick_roughness(max(dot(surface_props.world_normal, view_direction), 0.0), surface_props.f0, surface_props.roughness); 
    let kS = F;
    var kD = vec3<f32>(1.0) - kS;
    kD *= 1.0 - surface_props.metallic;
    let irradiance = textureSample(t_irradiance_cube_map, s_cube_sampler, surface_props.world_normal).rgb;
    let diffuse    = irradiance * surface_props.albedo;
    // sample both the pre-filter map and the BRDF lut and combine them together as per the Split-Sum approximation to get the IBL specular part.
    const MAX_REFLECTION_LOD: f32 = 4.0;
    let prefiltered_color = textureSampleLevel(t_prefiltered_reflection_map, s_cube_sampler, surface_props.reflection_dir,  surface_props.roughness * MAX_REFLECTION_LOD).rgb;    
    let brdf  = textureSample(t_brdf_lut, s_sampler_0, vec2(max(dot(surface_props.world_normal, view_direction), 0.0), surface_props.roughness)).rg;
    let specular = prefiltered_color * (F * brdf.x + brdf.y);

    let ambient = (kD * diffuse + specular) * surface_props.surface_ao;
    return ambient;
}
*/

//////////////////////////////////
/*
fn less_than_equal_f32(value1: f32, value2: f32) -> f32 {
  if (value1 <= value2) {
    return 1.0;
  }
  return 0.0;
}

fn less_than_equal_vec3_f32(value1: vec3<f32>, value2: vec3<f32>) -> vec3<f32> {
  return vec3<f32>(
    less_than_equal_f32(value1.x, value2.x),
    less_than_equal_f32(value1.y, value2.y),
    less_than_equal_f32(value1.z, value2.z)
  );
}

fn srgb_to_linear(value: vec4<f32>) -> vec4<f32> {
  return vec4<f32>(
    mix(
      pow(value.rgb * 0.9478672986 + vec3<f32>(0.0521327014), vec3<f32>(2.4)),
      value.rgb * 0.0773993808,
      less_than_equal_vec3_f32(value.rgb, vec3<f32>(0.04045))
    ),
    value.a
  );
}

fn linear_to_srgb(value: vec4<f32>) -> vec4<f32> {
  return vec4<f32>(
    mix(
      pow(value.rgb, vec3<f32>(0.41666)) * 1.055 - vec3<f32>(0.055),
      value.rgb * 12.92,
      vec3<f32>(less_than_equal_vec3_f32(value.rgb, vec3<f32>(0.0031308)))
    ),
    value.a
  );
}

fn perturb_normal_to_arb(
  eye_pos: vec3<f32>,
  surf_norm: vec3<f32>,
  map_n: vec3<f32>,
  uv: vec2<f32>
) -> vec3<f32> {
  let q0: vec3<f32> = vec3<f32>(dpdx(eye_pos.x), dpdx(eye_pos.y), dpdx(eye_pos.z));
  let q1: vec3<f32> = vec3<f32>(dpdy(eye_pos.x), dpdy(eye_pos.y), dpdy(eye_pos.z));
  let st0: vec2<f32> = dpdx(uv);
  let st1: vec2<f32> = dpdy(uv);
  let n: vec3<f32> = surf_norm; // normalized
  let q1perp: vec3<f32> = cross(q1, n);
  let q0perp: vec3<f32> = cross(n, q0);
  let t = q1perp * st0.x + q0perp * st1.x;
  let b = q1perp * st0.y + q0perp * st1.y;
  let det: f32 = max(dot(t, t), dot(b, b));
  var scale: f32;
  if (det == 0.0) {
    scale = 0.0;
  } else {
    scale = inverseSqrt(det);
  }
  return normalize(t * (map_n.x * scale) + b * (map_n.y * scale) + n * map_n.z);
}

fn d_ggx(n_dot_h: f32, roughness: f32) -> f32 {
  let a: f32 = n_dot_h * roughness;
  let k: f32 = roughness / (1.0 - pow(n_dot_h, 2.0) + pow(a, 2.0));
  return pow(k, 2.0) * (1.0 / PI);
}

fn v_smith_ggx_correlated_fast(n_dot_v: f32, n_dot_l: f32, roughness: f32) -> f32 {
  let a: f32 = roughness;
  let ggxv: f32 = n_dot_l * (n_dot_v * (1.0 - a) + a);
  let ggxl: f32 = n_dot_v * (n_dot_l * (1.0 - a) + a);
  return 0.5 / (ggxv + ggxl);
}

fn brdf(
  v: vec3<f32>,
  n: vec3<f32>,
  h: vec3<f32>,
  l: vec3<f32>,
  base_color: vec3<f32>,
  metallic: f32,
  roughness: f32
) -> vec3<f32> {
  let black = vec3<f32>(0.0);
  let v_dot_h = dot(v, h);
  let n_dot_v = dot(v, n);
  let n_dot_l = dot(l, n);
  let n_dot_h = dot(n, h);

  let c_diff = mix(base_color, black, metallic);
  let f0 = mix(vec3<f32>(0.04), base_color, metallic);
  let alpha = pow(roughness, 2.0);

  let f = f0 + (1.0 - f0) * pow(1.0 - abs(v_dot_h), 5.0);

  let f_diffuse = (1.0 - f) * (1.0 / PI) * c_diff;
  let f_specular = f * d_ggx(n_dot_h, alpha)
    * v_smith_ggx_correlated_fast(n_dot_v, n_dot_l, alpha)
    / (4.0 * abs(n_dot_v) * abs(n_dot_l));

  return f_diffuse + f_specular;
}
*/
