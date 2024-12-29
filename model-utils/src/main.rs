use std::io::Write;
use log::*;
use image::ImageFormat;
use model_load::{load, gpu_api_dto};

fn main() {
    env_logger::init();

    process_model("box", "glb");    
}

fn process_model(model_name: &str, extension: &str) {    
    let (model_data, loaded_images) = load(model_name, &format!("../models/{0}/{0}.{1}", model_name, extension));
    let mut texture_index = 0;    

    for loaded_image in loaded_images {
        loaded_image.save_with_format(&format!("{}_{}.png", model_name, &texture_index.to_string()), ImageFormat::Png).expect("Failed to save texture image");
        texture_index = texture_index + 1;
    }
    
    let mut file = std::fs::File::create(&format!("{}.model", model_name)).expect("Failed to create file");    
    
    let bytes = gpu_api_dto::bitcode::encode(&model_data);

    let res = file.write_all(&bytes).expect("Failed to write model data to file");
    info!("{:?}", res);
}
