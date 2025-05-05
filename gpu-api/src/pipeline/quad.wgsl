struct Globals {
    transform: mat4x4<f32>,
    scale: f32,
}

@group(0) @binding(0) var<uniform> globals: Globals;

struct VertexInput {
    @location(0) v_pos: vec2<f32>,
    @location(1) pos: vec2<f32>,
    @location(2) scale: vec2<f32>,
    @location(3) color: vec4<f32>,
    @location(4) border_color: vec4<f32>,
    @location(5) border_radius: vec4<f32>,
    @location(6) border_width: f32,
    @location(7) component_coordinates: vec4<f32>,    
    @location(8) @interpolate(flat) has_overlay: u32,
    @location(9) overlay_coordinates: vec4<f32>,
    @location(10) shadow_color: vec4<f32>,
    @location(11) shadow_offset: vec2<f32>,
    @location(12) shadow_blur_radius: f32
}

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) color: vec4<f32>,
    @location(1) border_color: vec4<f32>,
    @location(2) pos: vec2<f32>,
    @location(3) scale: vec2<f32>,
    @location(4) border_radius: vec4<f32>,
    @location(5) border_width: f32,
    @location(6) component_coordinates: vec4<f32>,    
    @location(7) @interpolate(flat) has_overlay: u32,
    @location(8) overlay_coordinates: vec4<f32>,
    @location(9) shadow_color: vec4<f32>,
    @location(10) shadow_offset: vec2<f32>,
    @location(11) shadow_blur_radius: f32
}

// Compute the normalized quad coordinates based on the vertex index.
fn vertex_position(vertex_index: u32) -> vec2<f32> {
    // #: 0 1 2 3 4 5
    // x: 1 1 0 0 0 1
    // y: 1 0 0 0 1 1
    return vec2<f32>((vec2(1u, 2u) + vertex_index) % vec2(6u) < vec2(3u));
}

fn distance_alg(
    frag_coord: vec2<f32>,
    position: vec2<f32>,
    size: vec2<f32>,
    radius: f32
) -> f32 {
    var inner_half_size: vec2<f32> = (size - vec2<f32>(radius, radius) * 2.0) / 2.0;
    var top_left: vec2<f32> = position + vec2<f32>(radius, radius);
    return rounded_box_sdf(frag_coord - top_left - inner_half_size, inner_half_size, 0.0);
}

// Given a vector from a point to the center of a rounded rectangle of the given `size` and
// border `radius`, determines the point's distance from the nearest edge of the rounded rectangle
fn rounded_box_sdf(to_center: vec2<f32>, size: vec2<f32>, radius: f32) -> f32 {
    return length(max(abs(to_center) - size + vec2<f32>(radius, radius), vec2<f32>(0.0, 0.0))) - radius;
}

// Based on the fragement position and the center of the quad, select one of the 4 radi.
// Order matches CSS border radius attribute:
// radi.x = top-left, radi.y = top-right, radi.z = bottom-right, radi.w = bottom-left
fn select_border_radius(radi: vec4<f32>, position: vec2<f32>, center: vec2<f32>) -> f32 {
    var rx = radi.x;
    var ry = radi.y;
    rx = select(radi.x, radi.y, position.x > center.x);
    ry = select(radi.w, radi.z, position.x > center.x);
    rx = select(rx, ry, position.y > center.y);
    return rx;
}

@vertex
fn vs_main(input: VertexInput) -> VertexOutput {
    var out: VertexOutput;

    var pos: vec2<f32> = (input.pos + min(input.shadow_offset, vec2<f32>(0.0, 0.0)) - input.shadow_blur_radius) * globals.scale;
    var scale: vec2<f32> = (input.scale + vec2<f32>(abs(input.shadow_offset.x), abs(input.shadow_offset.y)) + input.shadow_blur_radius * 2.0) * globals.scale;

    var min_border_radius = min(input.scale.x, input.scale.y) * 0.5;
    var border_radius: vec4<f32> = vec4<f32>(
        min(input.border_radius.x, min_border_radius),
        min(input.border_radius.y, min_border_radius),
        min(input.border_radius.z, min_border_radius),
        min(input.border_radius.w, min_border_radius)
    );

    var transform: mat4x4<f32> = mat4x4<f32>(
        vec4<f32>(scale.x + 1.0, 0.0, 0.0, 0.0),
        vec4<f32>(0.0, scale.y + 1.0, 0.0, 0.0),
        vec4<f32>(0.0, 0.0, 1.0, 0.0),
        vec4<f32>(pos - vec2<f32>(0.5, 0.5), 0.0, 1.0)
    );

    out.color = input.color;
    out.border_color = input.border_color;
    out.pos = pos;
    out.scale = scale;
    out.border_radius = border_radius * globals.scale;
    out.border_width = input.border_width * globals.scale;
    out.position = globals.transform * transform * vec4<f32>(input.v_pos, 0.0, 1.0);
    out.component_coordinates = input.component_coordinates;    
    out.has_overlay = input.has_overlay;
    out.overlay_coordinates = input.overlay_coordinates;
    out.shadow_color = input.shadow_color;
    out.shadow_offset = input.shadow_offset * globals.scale;
    out.shadow_blur_radius = input.shadow_blur_radius * globals.scale;

    return out;
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    if (
        input.position[0] < input.component_coordinates[0] ||
        input.position[0] > input.component_coordinates[2] ||
        input.position[1] < input.component_coordinates[1] ||
        input.position[1] > input.component_coordinates[3]
    ) {
        discard;
    }

    if (
        input.has_overlay == 1u && (
            input.position[0] >= input.overlay_coordinates[0] &&
            input.position[0] <= input.overlay_coordinates[2] &&
            input.position[1] >= input.overlay_coordinates[1] &&
            input.position[1] <= input.overlay_coordinates[3]
        )
    ) {
        discard;
    }

    var mixed_color: vec4<f32> = input.color;

    var border_radius = select_border_radius(
        input.border_radius,
        input.position.xy,
        (input.pos + input.scale * 0.5).xy
    );

    if (input.border_width > 0.0) {
        var internal_border: f32 = max(border_radius - input.border_width, 0.0);

        var internal_distance: f32 = distance_alg(
            input.position.xy,
            input.pos + vec2<f32>(input.border_width, input.border_width),
            input.scale - vec2<f32>(input.border_width * 2.0, input.border_width * 2.0),
            internal_border
        );

        var border_mix: f32 = smoothstep(
            max(internal_border - 0.5, 0.0),
            internal_border + 0.5,
            internal_distance
        );

        mixed_color = mix(input.color, input.border_color, vec4<f32>(border_mix, border_mix, border_mix, border_mix));
    }

    var dist: f32 = distance_alg(
        vec2<f32>(input.position.x, input.position.y),
        input.pos,
        input.scale,
        border_radius
    );

    var radius_alpha: f32 = 1.0 - smoothstep(
        max(border_radius - 0.5, 0.0),
        border_radius + 0.5,
        dist
    );

    let quad_color = vec4<f32>(mixed_color.x, mixed_color.y, mixed_color.z, mixed_color.w * radius_alpha);

    if input.shadow_color.a > 0.0 {
        let shadow_distance = rounded_box_sdf(input.position.xy - input.pos - input.shadow_offset - (input.scale / 2.0), input.scale / 2.0, border_radius);
        let shadow_alpha = 1.0 - smoothstep(-input.shadow_blur_radius, input.shadow_blur_radius, shadow_distance);
        let shadow_color = input.shadow_color;
        let base_color = select(
            vec4<f32>(shadow_color.x, shadow_color.y, shadow_color.z, 0.0),
            quad_color,
            quad_color.a > 0.0
        );

        return mix(base_color, shadow_color, (1.0 - radius_alpha) * shadow_alpha);
    } else {
        return quad_color;
    }
}

struct GradientVertexInput {
    @builtin(vertex_index) vertex_index: u32,
    @location(0) @interpolate(flat) colors_1: vec4<u32>,
    @location(1) @interpolate(flat) colors_2: vec4<u32>,
    @location(2) @interpolate(flat) colors_3: vec4<u32>,
    @location(3) @interpolate(flat) colors_4: vec4<u32>,
    @location(4) @interpolate(flat) offsets: vec4<u32>,
    @location(5) direction: vec4<f32>,
    @location(6) position_and_scale: vec4<f32>,
    @location(7) border_color: vec4<f32>,
    @location(8) border_radius: vec4<f32>,
    @location(9) border_width: f32,
}

struct GradientVertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(1) @interpolate(flat) colors_1: vec4<u32>,
    @location(2) @interpolate(flat) colors_2: vec4<u32>,
    @location(3) @interpolate(flat) colors_3: vec4<u32>,
    @location(4) @interpolate(flat) colors_4: vec4<u32>,
    @location(5) @interpolate(flat) offsets: vec4<u32>,
    @location(6) direction: vec4<f32>,
    @location(7) position_and_scale: vec4<f32>,
    @location(8) border_color: vec4<f32>,
    @location(9) border_radius: vec4<f32>,
    @location(10) border_width: f32,
}

fn interpolate_color(from_: vec4<f32>, to_: vec4<f32>, factor: f32) -> vec4<f32> {
    return mix(from_, to_, factor);
}

@vertex
fn gradient_vs_main(input: GradientVertexInput) -> GradientVertexOutput {
    var out: GradientVertexOutput;

    var pos: vec2<f32> = input.position_and_scale.xy * globals.scale;
    var scale: vec2<f32> = input.position_and_scale.zw * globals.scale;

    var min_border_radius = min(input.position_and_scale.z, input.position_and_scale.w) * 0.5;
    var border_radius: vec4<f32> = vec4<f32>(
        min(input.border_radius.x, min_border_radius),
        min(input.border_radius.y, min_border_radius),
        min(input.border_radius.z, min_border_radius),
        min(input.border_radius.w, min_border_radius)
    );

    var transform: mat4x4<f32> = mat4x4<f32>(
        vec4<f32>(scale.x + 1.0, 0.0, 0.0, 0.0),
        vec4<f32>(0.0, scale.y + 1.0, 0.0, 0.0),
        vec4<f32>(0.0, 0.0, 1.0, 0.0),
        vec4<f32>(pos - vec2<f32>(0.5, 0.5), 0.0, 1.0)
    );

    out.position = globals.transform * transform * vec4<f32>(vertex_position(input.vertex_index), 0.0, 1.0);
    out.colors_1 = input.colors_1;
    out.colors_2 = input.colors_2;
    out.colors_3 = input.colors_3;
    out.colors_4 = input.colors_4;
    out.offsets = input.offsets;
    out.direction = input.direction * globals.scale;
    out.position_and_scale = vec4<f32>(pos, scale);
    out.border_color = input.border_color;
    out.border_radius = border_radius * globals.scale;
    out.border_width = input.border_width * globals.scale;

    return out;
}

fn random(coords: vec2<f32>) -> f32 {
    return fract(sin(dot(coords, vec2(12.9898,78.233))) * 43758.5453);
}

/// Returns the current interpolated color with a max 8-stop gradient
fn gradient(
    raw_position: vec2<f32>,
    direction: vec4<f32>,
    colors: array<vec4<f32>, 8>,
    offsets: array<f32, 8>,
    last_index: i32
) -> vec4<f32> {
    let start = direction.xy;
    let end = direction.zw;

    let v1 = end - start;
    let v2 = raw_position - start;
    let unit = normalize(v1);
    let coord_offset = dot(unit, v2) / length(v1);

    //need to store these as a var to use dynamic indexing in a loop
    //this is already added to wgsl spec but not in wgpu yet
    var colors_arr = colors;
    var offsets_arr = offsets;

    var color: vec4<f32>;

    let noise_granularity: f32 = 0.3/255.0;

    for (var i: i32 = 0; i < last_index; i++) {
        let curr_offset = offsets_arr[i];
        let next_offset = offsets_arr[i+1];

        if (coord_offset <= offsets_arr[0]) {
            color = colors_arr[0];
        }

        if (curr_offset <= coord_offset && coord_offset <= next_offset) {
            let from_ = colors_arr[i];
            let to_ = colors_arr[i+1];
            let factor = smoothstep(curr_offset, next_offset, coord_offset);

            color = interpolate_color(from_, to_, factor);
        }

        if (coord_offset >= offsets_arr[last_index]) {
            color = colors_arr[last_index];
        }
    }

    return color + mix(-noise_granularity, noise_granularity, random(raw_position));
}

@fragment
fn gradient_fs_main(input: GradientVertexOutput) -> @location(0) vec4<f32> {
    let colors = array<vec4<f32>, 8>(
        unpack_u32(input.colors_1.xy),
        unpack_u32(input.colors_1.zw),
        unpack_u32(input.colors_2.xy),
        unpack_u32(input.colors_2.zw),
        unpack_u32(input.colors_3.xy),
        unpack_u32(input.colors_3.zw),
        unpack_u32(input.colors_4.xy),
        unpack_u32(input.colors_4.zw),
    );

    let offsets_1: vec4<f32> = unpack_u32(input.offsets.xy);
    let offsets_2: vec4<f32> = unpack_u32(input.offsets.zw);

    var offsets = array<f32, 8>(
        offsets_1.x,
        offsets_1.y,
        offsets_1.z,
        offsets_1.w,
        offsets_2.x,
        offsets_2.y,
        offsets_2.z,
        offsets_2.w,
    );

    //TODO could just pass this in to the shader but is probably more performant to just check it here
    var last_index = 7;
    for (var i: i32 = 0; i <= 7; i++) {
        if (offsets[i] > 1.0) {
            last_index = i - 1;
            break;
        }
    }

    var mixed_color: vec4<f32> = gradient(input.position.xy, input.direction, colors, offsets, last_index);

    let pos = input.position_and_scale.xy;
    let scale = input.position_and_scale.zw;

    var border_radius = select_border_radius(
        input.border_radius,
        input.position.xy,
        (pos + scale * 0.5).xy
    );

    if (input.border_width > 0.0) {
        var internal_border: f32 = max(border_radius - input.border_width, 0.0);

        var internal_distance: f32 = distance_alg(
            input.position.xy,
            pos + vec2<f32>(input.border_width, input.border_width),
            scale - vec2<f32>(input.border_width * 2.0, input.border_width * 2.0),
            internal_border
        );

        var border_mix: f32 = smoothstep(
            max(internal_border - 0.5, 0.0),
            internal_border + 0.5,
            internal_distance
        );

        mixed_color = mix(mixed_color, input.border_color, vec4<f32>(border_mix, border_mix, border_mix, border_mix));
    }

    var dist: f32 = distance_alg(
        input.position.xy,
        pos,
        scale,
        border_radius
    );

    var radius_alpha: f32 = 1.0 - smoothstep(
        max(border_radius - 0.5, 0.0),
        border_radius + 0.5,
        dist);

    return vec4<f32>(mixed_color.x, mixed_color.y, mixed_color.z, mixed_color.w * radius_alpha);
}

fn unpack_u32(color: vec2<u32>) -> vec4<f32> {
    let rg: vec2<f32> = unpack2x16float(color.x);
    let ba: vec2<f32> = unpack2x16float(color.y);

    return vec4<f32>(rg.y, rg.x, ba.y, ba.x);
}
