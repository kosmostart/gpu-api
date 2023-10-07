use std::io::Write;
use log::*;

mod model_load;

fn main() {
    env_logger::init();

    let model_data = model_load::load("../models/plane/plane.gltf");

    let mut file = std::fs::File::create("plane.json").expect("Failed to create file");
    let res = file.write_all(&serde_json::to_vec(&model_data).expect("Failed to serialize model data"));

    info!("{:?}", res);
}