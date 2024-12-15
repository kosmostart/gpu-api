use std::io::Write;
use log::*;
use model_load::gpu_api_dto::{self, rkyv};

fn main() {
    env_logger::init();

    process_model("box", "glb");    
}

fn process_model(model_name: &str, extension: &str) {    
    let mut model_data = model_load::load(model_name, &format!("../models/{0}/{0}.{1}", model_name, extension));
    let mut texture_index = 0;

    for texture in &mut model_data.textures {
        match texture.pixels.take() {
            Some(pixels) => {                
                let mut file = std::fs::File::create(&format!("{}_{}.texture", model_name, &texture_index.to_string())).expect("Failed to create texture file");
                file.write_all(&pixels).expect("Failed to write texture pixes to file");                
            }
            None => {
                info!("Pixels are empty");
            }
        }

        texture_index = texture_index + 1;
    }
    
    let mut file = std::fs::File::create(&format!("{}.model", model_name)).expect("Failed to create file");    

    //let bytes = rkyv::to_bytes::<rkyv::rancor::Error>(&model_data).expect("Failed to serialize model data");
    let bytes = gpu_api_dto::bitcode::encode(&model_data);

    let res = file.write_all(&bytes).expect("Failed to write model data to file");
    info!("{:?}", res);
}