use std::io::Write;
use log::*;
use model_load::gpu_api_dto::rkyv;

fn main() {
    env_logger::init();

    let model_name = "plane";
    let mut model_data = model_load::load(model_name, &format!("../models/{0}/{0}.gltf", model_name));
    let mut texture_index = 0;

    for texture in &mut model_data.textures {
        match texture.pixels.take() {
            Some(pixels) => {                
                let mut file = std::fs::File::create(&format!("{}_{}.pixels", model_name, &texture_index.to_string())).expect("Failed to create texture file");
                file.write_all(&pixels).expect("Failed to write texture pixes to file");                
            }
            None => {
                info!("Pixels are empty");
            }
        }

        texture_index = texture_index + 1;
    }
    
    let mut file = std::fs::File::create(&format!("{}.model", model_name)).expect("Failed to create file");    

    let bytes = rkyv::to_bytes::<rkyv::rancor::Error>(&model_data).expect("Failed to serialize model data");

    let res = file.write_all(&bytes).expect("Failed to write model data to file");
    info!("{:?}", res);
}
