use rkyv::{archived_root, Archive, Serialize, Deserialize, Infallible};
pub use rkyv;

#[derive(Archive, Serialize, Deserialize, Debug, Clone)]
pub struct ModelData {
    pub name: String,
	pub meshes: Vec<MeshData>,
    pub textures: Vec<TextureData>
}

#[derive(Archive, Serialize, Deserialize, Debug, Clone)]
pub struct MeshData {
	pub primitives: Vec<PrimitiveData>
}

#[derive(Archive, Serialize, Deserialize, Debug, Clone)]
pub struct PrimitiveData {
	pub positions: Vec<[f32; 3]>,
	pub indices: Vec<u32>,
	pub normals: Vec<[f32; 3]>,
	pub tangents: Vec<[f32; 3]>,
	pub bitangents: Vec<[f32; 3]>,
    pub texture_coordinates: Vec<[f32; 2]>,
    pub pbr_specular_glossiness_diffuse_texture_index: Option<usize>,
    pub pbr_specular_glossiness_texture_index: Option<usize>,
    pub base_color_texture_index: Option<usize>,
    pub metallic_roughness_texture_index: Option<usize>,
    pub normal_texture_index: Option<usize>,
    pub occlusion_texture_index: Option<usize>,
    pub emmisive_texture_index: Option<usize>
}

#[derive(Archive, Serialize, Deserialize, Debug, Clone)]
pub struct TextureData {
    pub index: usize,
    pub format: String,
    pub width: u32,
    pub height: u32,
    pub pixels: Option<Vec<u8>>    
}

#[derive(Archive, Serialize, Deserialize, Debug, Clone, serde_derive::Serialize, serde_derive::Deserialize)]
pub struct ViewSource {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub scale_x: f32,
    pub scale_y: f32,
    pub scale_z: f32
}

pub fn deserialize_model_data(buf: &[u8]) -> ModelData {
    let archived = unsafe { archived_root::<ModelData>(buf) };
    archived.deserialize(&mut Infallible).expect("Failed to deserialize model data")
}
