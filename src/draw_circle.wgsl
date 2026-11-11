const CORNERS = array<vec2<f32>, 6>(
        vec2(-1.0, -1.0),
        vec2(1.0, -1.0),
        vec2(1.0, 1.0),
        vec2(-1.0, -1.0),
        vec2(1.0, 1.0),
        vec2(-1.0, 1.0),
    );

struct Globals {
    size: vec2<f32>,
    radius: f32,
}

struct Out {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

var<immediate> globals: Globals;

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32, @location(0) center: vec2<f32>) -> Out {
    let corner = CORNERS[vertex_index];
    let normalized_coordinates = (center + corner * globals.radius) / globals.size * 2.0
        - vec2<f32>(1.0, 1.0);
    let out = Out(vec4<f32>(normalized_coordinates, 0.0, 1.0), corner);
    return out;
}

@fragment
fn fs_main(in: Out) -> @location(0) vec4<f32> {
    if dot(in.uv, in.uv) > 1.0 {
        discard;
    }
    return vec4<f32>(0.0, 0.0, 1.0, 1.0);
}
