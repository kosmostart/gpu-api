<h1 align="center">
  gpu-api
</h1>

A collection of [`wgpu`](https://github.com/gfx-rs/wgpu/) pipelines, which can be used for single render pass.

It supports:

- basic shapes
- images
- 3D models with animation, skinning, node attachment, normal mapping, PBR
- GLTF loading
- SVG (not yet)
- ray casting for object picking: ray-AABB intersection, ray-plane intersection

For working example navigate to `test-app` and do `cargo run`.

Quad shader code and gradient infrastructure is based on `iced` crate quad implementation with minor additions.
