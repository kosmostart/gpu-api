use std::io::{Write, Read};
use log::*;
use gltf::{mesh::{util::{ReadIndices, ReadTexCoords}, Mode}, Node};
use gpu_api::{model::{ModelData, MeshData, PrimitiveData}, texture::TextureData};

fn nodes(node: Node, node_level: usize) {
    info!("Node level: {}, node name: {:?}, node index: {}", node_level, node.name(), node.index());
    //info!("{:?}", node.mesh());
    //info!("{:?}", node.camera());
    info!("{:?}", node.transform());


    for child_node in node.children() {        
        nodes(child_node, node_level + 1);
    }
}

pub fn load(model_name: &str, model_path: &str) -> ModelData {
    info!("Loading model {} from path {}", model_name, model_path);
    let (gltf_data, buffers, images) = gltf::import(model_path).expect("Model import failed");    

    //let mut node_level = 0;

    //for node in gltf_data.nodes() {
    //    nodes(node, node_level);
    //}

    //return;

    let mut meshes = vec![];

    for mesh in gltf_data.meshes() {
        info!("Mesh {:?}, index {}", mesh.name(), mesh.index());

        let mut primitives = vec![];

        for primitive in mesh.primitives() {
            
            //info!("Primitive {}, mode {:?}", primitive.index(), primitive.mode());
            //info!("{:#?}", primitive.attributes());

            let reader = primitive.reader(|buffer| Some(&buffers[buffer.index()]));            

            let mut positions = vec![];
            
            match reader.read_positions() {
                Some(iter) => {
                    for vertex_position in iter {
                        //info!("{:?}", vertex_position);
                        positions.push(vertex_position);
                    }
                }
                None => {}
            }

            let mut indices = vec![];

            match reader.read_indices() {
                Some(iter) => {
                    match iter {
                        ReadIndices::U8(dog) => {
                            for q in dog {
                                indices.push(q as u32);
                            }
                        }
                        ReadIndices::U16(dog) => {
                            for q in dog {
                                indices.push(q as u32);
                            }
                        }
                        ReadIndices::U32(dog) => {
                            for q in dog {
                                indices.push(q);
                            }
                        }
                    }
                }
                None => {}
            }

            let mut normals = vec![];

            match reader.read_normals() {
                Some(iter) => {
                    for q in iter {                        
                        normals.push(q);
                    }
                }
                None => {}
            }

            let mut tangents = vec![];
            let mut bitangents = vec![];
            
            let mut index = 0;

            //bitangent = cross(normal.xyz, tangent.xyz) * tangent.w.

            match reader.read_tangents() {
                Some(iter) => {
                    for q in iter {
                        tangents.push([q[0], q[1], q[2]]);
                        let normal = normals[index];
                        bitangents.push([normal[0] * q[0] * q[3], normal[1] * q[1] * q[3], normal[2] * q[2] * q[3]]);
                        index = index + 1;
                    }
                }
                None => {}
            }            

            /*
            match reader.read_joints() {
                Some(iter) => {}
                None => {}
            }
            */

            //let iter = reader.read_morph_targets();

            /*
            match reader.read_weights() {
                Some(iter) => {}
                None => {}
            }
            */

            let mut texture_coordinates = vec![];
            
            match reader.read_tex_coords(0) {
                Some(coords) => {
                    match coords {
                        ReadTexCoords::F32(iter) => {
                            for q in iter {                                
                                texture_coordinates.push(q);
                            }
                        }
                        ReadTexCoords::U8(iter) => {
                            for q in iter {
                                warn!("set 0 u8 {:?}", q);
                            }
                        }
                        ReadTexCoords::U16(iter) => {
                            for q in iter {
                                warn!("set 0 u16 {:?}", q);
                            }
                        }
                    }
                    
                }
                None => {}
            }

            match reader.read_tex_coords(1) {
                Some(coords) => {
                    match coords {
                        ReadTexCoords::F32(iter) => {
                            for q in iter {
                                warn!("set 1 f32 {:?}", q);
                            }
                        }
                        ReadTexCoords::U8(iter) => {
                            for q in iter {
                                warn!("set 1 u8 {:?}", q);
                            }
                        }
                        ReadTexCoords::U16(iter) => {
                            for q in iter {
                                warn!("set 1 u16 {:?}", q);
                            }
                        }
                    }
                    
                }
                None => {}
            }

            /*
            match reader.read_colors(0) {
                Some(iter) => {}
                None => {}
            } 
            */           

            info!("{} positions total: {}", model_name, positions.len());
            info!("{} indices total: {}", model_name, indices.len());
            info!("{} normals total: {}", model_name, normals.len());
            info!("{} tangents total: {}", model_name, tangents.len());
            info!("{} bitangents total: {}", model_name, bitangents.len());
            info!("{} texture coordinates total: {}", model_name, texture_coordinates.len());

            primitives.push(PrimitiveData {
                positions,
                indices,
                normals,
                tangents,
                bitangents,
                texture_coordinates
            });
        }

        meshes.push(MeshData {            
            primitives
        });        
    }

    info!("{} meshes total: {}", model_name, meshes.len());
    info!("{} images total: {}", model_name, images.len());

    let mut textures = vec![];
    //let mut image_index = 0;

    for image in images {
        info!("{} image format {:?}, width {}, height {}", model_name, image.format, image.width, image.height);            

        //let mut file = std::fs::File::create(&image_index.to_string()).expect("Failed to image create file");

        //file.write_all(&image.pixels).expect("Failed to write image pixes to file");

        //image::ImageBuffer::from_raw(texture_item.width, texture_item.height, texture_item.pixels.expect("Texture pixels are empty")).expect("Failed to create image buffer")
        
        //let mut file = std::fs::File::open(&image_index.to_string()).expect("Failed to open image file");
        //let mut q = vec![];

        //file.read_to_end(&mut q).unwrap();
        
        textures.push(TextureData {
            format: format!("{:?}", image.format),
            width: image.width,
            height: image.height,
            pixels: Some(image.pixels)
        });
        
        //image_index = image_index + 1;
    }

    ModelData {
        name: model_name.to_owned(),
        meshes,
        textures
    }
}