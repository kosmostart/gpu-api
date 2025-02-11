use bitcode::{Encode, Decode};
pub use bitcode;

#[derive(Encode, Decode, Debug, Clone)]
pub struct ModelData {
    pub name: String,
    pub nodes: Vec<Node>,
    pub node_topological_sorting: Vec<usize>,
    pub node_map: std::collections::BTreeMap<usize, usize>,
    pub skins: Vec<Skin>,
	pub meshes: Vec<MeshData>,
    pub textures: Vec<TextureData>,
    pub animations: Vec<Animation>    
}

#[derive(Encode, Decode, Debug, Clone)]
pub struct Node {
    pub index: usize,
    pub children_nodes: Vec<usize>,
    pub name: Option<String>,
    pub skin_index: Option<usize>,
    pub translation: [f32; 3], 
    pub rotation: [f32; 4], 
    pub scale: [f32; 3],
    pub local_transform_matrix: [[f32; 4]; 4]
}

#[derive(Encode, Decode, Debug, Clone)]
pub struct Skin {
    pub name: Option<String>,
    pub inverse_bind_matrices: Vec<[[f32; 4]; 4]>,
    pub joints: Vec<Joint>
}

#[derive(Encode, Decode, Debug, Clone)]
pub struct Joint {
    pub node_index: usize,
    pub node_name: Option<String>
}

#[derive(Encode, Decode, Debug, Clone)]
pub struct MeshData {
	pub primitives: Vec<PrimitiveData>
}

#[derive(Encode, Decode, Debug, Clone)]
pub struct PrimitiveData {
	pub positions: Vec<[f32; 3]>,
	pub indices: Vec<u32>,
	pub normals: Vec<[f32; 3]>,
	pub tangents: Vec<[f32; 4]>,	
    pub joints: Vec<[u32; 4]>,
    pub weights: Vec<[f32; 4]>,
    pub texture_coordinates: Vec<[f32; 2]>,
    pub pbr_specular_glossiness_diffuse_texture_index: Option<usize>,
    pub pbr_specular_glossiness_texture_index: Option<usize>,
    pub base_color_texture_index: Option<usize>,
    pub metallic_roughness_texture_index: Option<usize>,
    pub normal_texture_index: Option<usize>,
    pub occlusion_texture_index: Option<usize>,
    pub emmisive_texture_index: Option<usize>
}

#[derive(Encode, Decode, Debug, Clone)]
pub struct Animation {
    pub name: String,
    pub channels: Vec<AnimationChannel>
}

#[derive(Encode, Decode, Debug, Clone)]
pub struct AnimationChannel {
    pub target_index: usize,
    pub property: AnimationProperty,
    pub interpolation: Interpolation,
    pub timestamps: Vec<f32>,
    pub translations: Vec<[f32; 3]>,
    pub rotations: Vec<[f32; 4]>,
    pub scales: Vec<[f32; 3]>,
    pub weight_morphs: Vec<f32>
}

#[derive(Encode, Decode, Debug, Clone)]
pub enum Interpolation {
    Linear,
    Step,
    CubicSpline
}

#[derive(Encode, Decode, Debug, Clone)]
pub enum AnimationProperty {    
    Translation,    
    Rotation,    
    Scale,    
    MorphTargetWeights
}

#[derive(Encode, Decode, Debug, Clone)]
pub struct TextureData {
    pub index: usize,
    pub format: String,
    pub width: u32,
    pub height: u32,
    pub image_encoded: Option<Vec<u8>>
}

#[derive(Encode, Decode, Debug, Clone, serde_derive::Serialize, serde_derive::Deserialize)]
pub struct ViewSource {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub scale_x: f32,
    pub scale_y: f32,
    pub scale_z: f32
}

pub fn deserialize_model_data(buf: &[u8]) -> ModelData {    
    bitcode::decode(&buf).expect("Failed to deserialize model data")
}
