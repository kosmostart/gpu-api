struct Globals {
    transform: mat4x4<f32>,
    scale: f32
}

@group(0) @binding(0) var<uniform> globals: Globals;

struct VertexInput {
    @location(0) color: vec4<f32>,
    @location(1) pos: vec3<f32>,
};

struct VertexOutput {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) @interpolate(flat) color: vec4<f32>,
};

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    
    // 1. Scaling local cooridnates of vertex
    let scaled_pos = in.pos * globals.scale;    
    // 2. Aplying transform matrix, for WGSL order is matrix * vec    
    out.clip_pos = globals.transform * vec4<f32>(scaled_pos, 1.0);
        
    out.color = in.color; 
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return in.color;
}
