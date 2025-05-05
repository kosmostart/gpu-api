struct VertexInput {
    @location(0) @interpolate(flat) vertex_type: u32,
    @location(1) position: vec3<f32>,
    @location(2) color: vec4<f32>,
    @location(3) element_coordinates: vec4<f32>,
    @location(4) @interpolate(flat) has_element_border: u32,
    @location(5) element_border_color: vec4<f32>,
    @location(6) component_coordinates: vec4<f32>,
    @location(7) texture_coordinates: vec2<f32>,
    @location(8) @interpolate(flat) has_overlay: u32,
    @location(9) overlay_coordinates: vec4<f32>
};

struct FragmentInput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) @interpolate(flat) vertex_type: u32,
    @location(1) color: vec4<f32>,
    @location(2) element_coordinates: vec4<f32>,
    @location(3) @interpolate(flat) has_element_border: u32,
    @location(4) element_border_color: vec4<f32>,
    @location(5) component_coordinates: vec4<f32>,
    @location(6) texture_coordinates: vec2<f32>,
    @location(7) @interpolate(flat) has_overlay: u32,
    @location(8) overlay_coordinates: vec4<f32>
};

// Vertex shader
@vertex
//@stage(vertex)
fn vs_main(vertex_input: VertexInput) -> FragmentInput {
    var fragment_input: FragmentInput;
    
    fragment_input.clip_position = vec4<f32>(vertex_input.position, 1.0);
    fragment_input.vertex_type = vertex_input.vertex_type;
    fragment_input.color = vertex_input.color;
    fragment_input.element_coordinates = vertex_input.element_coordinates;
    fragment_input.has_element_border = vertex_input.has_element_border;
    fragment_input.element_border_color = vertex_input.element_border_color;
    fragment_input.component_coordinates = vertex_input.component_coordinates;
    fragment_input.texture_coordinates = vertex_input.texture_coordinates;
    fragment_input.has_overlay = vertex_input.has_overlay;
    fragment_input.overlay_coordinates = vertex_input.overlay_coordinates;

    return fragment_input;
}

@group(0) @binding(0)
var texture_data: texture_2d<f32>;
@group(0) @binding(1)
var texture_sampler: sampler;

// Fragment shader
@fragment
//@stage(fragment)
fn fs_main(fragment_input: FragmentInput) -> @location(0) vec4<f32> {
    switch fragment_input.vertex_type {
        case 1u: {            
            return textureSample(texture_data, texture_sampler, fragment_input.texture_coordinates); // Texture
        }
        default: {
            if (
                fragment_input.clip_position[0] < fragment_input.component_coordinates[0] ||
                fragment_input.clip_position[0] > fragment_input.component_coordinates[2] ||
                fragment_input.clip_position[1] < fragment_input.component_coordinates[1] ||
                fragment_input.clip_position[1] > fragment_input.component_coordinates[3]
            ) {
                discard;
            }

            if (
                fragment_input.has_overlay == 1u && (
                    fragment_input.clip_position[0] >= fragment_input.overlay_coordinates[0] &&
                    fragment_input.clip_position[0] <= fragment_input.overlay_coordinates[2] &&
                    fragment_input.clip_position[1] >= fragment_input.overlay_coordinates[1] &&
                    fragment_input.clip_position[1] <= fragment_input.overlay_coordinates[3]
                )
            ) {
                discard;
            }
            
            if (fragment_input.has_element_border == 1u) {
                let element_border_width = 1.0;
                let element_border_radius = 10.0;

                let center_x1 = fragment_input.element_coordinates[0] + element_border_radius;
                let center_y1 = fragment_input.element_coordinates[1] + element_border_radius;

                let dist_x1 = center_x1 - fragment_input.clip_position[0];
                let dist_y1 = center_y1 - fragment_input.clip_position[1];

                let r1 = sqrt(dist_x1 * dist_x1 + dist_y1 * dist_y1);            

                if (fragment_input.clip_position[0] <= center_x1 && fragment_input.clip_position[1] <= center_y1) {
                    if (r1 <= element_border_radius && r1 >= (element_border_radius - element_border_width)) {
                        //let dist: f32 = distance(vec2<f32>(fragment_input.element_coordinates[0], fragment_input.element_coordinates[1]), vec2<f32>(center_x1, center_y1));
                        
                        //let a = 1.0 - smoothstep(element_border_radius - element_border_width, element_border_radius, r1);
                        
                        //return mix(fragment_input.color, fragment_input.element_border_color, a);
                                                
                        //return vec4<f32>(fragment_input.element_border_color[0], fragment_input.element_border_color[1], fragment_input.element_border_color[2], a);

                        return fragment_input.element_border_color;
                    } else {
                        discard;
                    }
                }

                let center_x2 = fragment_input.element_coordinates[2] - element_border_radius;
                let center_y2 = fragment_input.element_coordinates[1] + element_border_radius;

                let dist_x2 = center_x2 - fragment_input.clip_position[0];
                let dist_y2 = center_y2 - fragment_input.clip_position[1];

                let r2 = sqrt(dist_x2 * dist_x2 + dist_y2 * dist_y2);

                if (fragment_input.clip_position[0] >= center_x2 && fragment_input.clip_position[1] <= center_y2) {
                    if (r2 <= element_border_radius && r2 >= (element_border_radius - element_border_width)) {           
                        //return vec4<f32>(fragment_input.element_border_color[0], fragment_input.element_border_color[1], fragment_input.element_border_color[2], 1.0 - smoothstep(-0.5, 0.5, r2));
                        return fragment_input.element_border_color;
                    } else {
                        discard;
                    }
                }
            
                let center_x3 = fragment_input.element_coordinates[2] - element_border_radius;
                let center_y3 = fragment_input.element_coordinates[3] - element_border_radius;

                let dist_x3 = center_x3 - fragment_input.clip_position[0];
                let dist_y3 = center_y3 - fragment_input.clip_position[1];

                let r3 = sqrt(dist_x3 * dist_x3 + dist_y3 * dist_y3);

                if (fragment_input.clip_position[0] >= center_x3 && fragment_input.clip_position[1] >= center_y3) {
                    if (r3 <= element_border_radius && r3 >= (element_border_radius - element_border_width)) {                
                        return fragment_input.element_border_color;
                    } else {
                        discard;
                    }
                }            

                let center_x4 = fragment_input.element_coordinates[0] + element_border_radius;
                let center_y4 = fragment_input.element_coordinates[3] - element_border_radius;

                let dist_x4 = center_x4 - fragment_input.clip_position[0];
                let dist_y4 = center_y4 - fragment_input.clip_position[1];

                let r4 = sqrt(dist_x4 * dist_x4 + dist_y4 * dist_y4);

                if (fragment_input.clip_position[0] <= center_x4 && fragment_input.clip_position[1] >= center_y4) {
                    if (r4 <= element_border_radius && r4 >= (element_border_radius - element_border_width)) {                
                        return fragment_input.element_border_color;
                    } else {
                        discard;
                    }
                }            
                
                if (                 
                    (fragment_input.clip_position[0] >= fragment_input.element_coordinates[0] && fragment_input.clip_position[0] <= fragment_input.element_coordinates[0] + element_border_width) ||
                    (fragment_input.clip_position[0] >= fragment_input.element_coordinates[2] - element_border_width && fragment_input.clip_position[0] <= fragment_input.element_coordinates[2]) ||
                    (fragment_input.clip_position[1] >= fragment_input.element_coordinates[1] && fragment_input.clip_position[1] <= fragment_input.element_coordinates[1] + element_border_width) ||
                    (fragment_input.clip_position[1] >= fragment_input.element_coordinates[3] - element_border_width && fragment_input.clip_position[1] <= fragment_input.element_coordinates[3])
                ) {
                    return fragment_input.element_border_color;
                }
            }

            return fragment_input.color;
            //return fragment_input.element_border_color;
            //return vec4<f32>(fragment_input.element_border_color[0], fragment_input.element_border_color[1], fragment_input.element_border_color[2], a);
        }
    }            
}

/*
@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> @builtin(position) vec4<f32> {
    let tc = vec2<f32>(
        f32(vertex_index >> 1u),
        f32(vertex_index & 1u),
    ) * 2.0;

    return vec4<f32>(tc * 2.0 - 1.0, 0.0, 1.0);
}

@fragment
fn fs_main(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
    let screen = vec2<f32>(800.0, 600.0);
    let radius = 200.0;

    let dist = length((pos.xy - screen / 2.0) / radius);
    let color = smoothstep(0.49, 0.51, dist);

    return vec4<f32>(vec3<f32>(color), 1.0);
}
*/
