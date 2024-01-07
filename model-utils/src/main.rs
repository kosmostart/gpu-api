use std::io::Write;
use log::*;

fn main() {
    env_logger::init();

    let model_name = "plane";

    let mut model_data = model_load::load(model_name, &format!("../models/{0}/{0}.gltf", model_name));

    let mut texture_index = 0;

    for texture in &mut model_data.textures {
        match texture.pixels.take() {
            Some(pixels) => {                
                let mut file = std::fs::File::create(&format!("{}_{}", model_name, &texture_index.to_string())).expect("Failed to create texture file");

                file.write_all(&pixels).expect("Failed to write texture pixes to file");

                //image::ImageBuffer::from_raw(texture_item.width, texture_item.height, texture_item.pixels.expect("Texture pixels are empty")).expect("Failed to create image buffer")                                
            }
            None => {
                info!("Pixels are empty");
            }
        }

        texture_index = texture_index + 1;
    }

    /*let mut file = std::fs::File::create(&image_index.to_string()).expect("Failed to image create file");

    use base64::{Engine, engine::general_purpose};

    file.write_all(general_purpose::STANDARD_NO_PAD.encode(&compress(&image.pixels)).as_bytes()).expect("Failed to write image pixes to file");

    let mut file = std::fs::File::open(&image_index.to_string()).expect("Failed to open image file");
    let mut q = "".to_owned();

    file.read_to_string(&mut q).unwrap();

    let dog = decompress(&general_purpose::STANDARD_NO_PAD.decode(&q).unwrap(), image.pixels.len()).unwrap();        
    */

    let mut file = std::fs::File::create(&format!("{}.json", model_name)).expect("Failed to create file");
    let res = file.write_all(&serde_json::to_vec(&model_data).expect("Failed to serialize model data"));

    info!("{:?}", res);
}