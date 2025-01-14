use std::io::{Cursor, Read, Write};
use image::{DynamicImage, Rgb, Rgba};
use log::*;
use gltf::{image::Format, mesh::{util::{ReadIndices, ReadJoints, ReadTexCoords, ReadWeights}, Mode}, Node};
use gpu_api_dto::{Animation, AnimationChannel, AnimationChannelPayload, Interpolation, MeshData, ModelData, PrimitiveData, TextureData};
pub use gpu_api_dto;

pub fn load(model_name: &str, model_path: &str) -> (ModelData, Vec<DynamicImage>) {
    info!("Loading model {} from path {}", model_name, model_path);    
    let (document, buffers, images) = gltf::import(model_path).expect("Model import failed");
    
    info!("Found {} scenes", document.scenes().count());

    for scene in document.scenes() {
        info!("Found scene {:?}, index {}", scene.name(), scene.index());        
    }

    info!("Found {} nodes", document.nodes().count());

    for node in document.nodes() {
        info!("Found node {:?}, index {}, mesh {:?}, skin {:?}, weights {:?}", 
            node.name(), 
            node.index(), 
            node.mesh().map(|v| v.name()),
            node.skin().map(|v| v.name()),
            node.weights(),
            //node.transform()
        );
    }

    info!("Found {} skins", document.skins().count());
/*
    let mut res_skins = vec![];

    for skin in document.skins() {
        info!("Found skin {:?}, index {}", skin.name(), skin.index());

        let num_joints = skin.joints().count();
        let reader = skin.reader(|b| Some(&buffers[b.index()][..b.length()]));

        let inv_b_mats = if let Some(inv_b_mats) = reader.read_inverse_bind_matrices() {
            inv_b_mats.map(|mat| Mat4::from_cols_array_2d(&mat)).collect()
        } else {
            // The inverse bind matrices are sometimes not provided. This has to
            // be interpreted as all of them being the identity transform.
            vec![Mat4::IDENTITY; num_joints]
        };

        let joints = skin.joints().map(|node| Labeled::new(Joint { node_idx: node.index() }, node.name())).collect();

        res_skins.push(Labeled::new(Skin { inverse_bind_matrices: inv_b_mats, joints }, skin.name()))
    }
*/
    let mut meshes = vec![];    

    for mesh in document.meshes() {
        info!("Found Mesh {:?}, index {}", mesh.name(), mesh.index());

        if mesh.name() == Some("Body_low.003") {
            //Cloth_low.003
            //"Belt1_low.003"
            //"Body_low.002"
            //"Body_low.003"
            continue;
        }

        let mut primitives = vec![];

        for primitive in mesh.primitives() {
            
            info!("Found primitive {}, mode {:?}", primitive.index(), primitive.mode());
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
                    // Different type indexes should not intersect.
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
                                indices.push(q as u32);
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

            match reader.read_colors(0) {
                Some(iter) => {}
                None => {}
            }

            match reader.read_joints(0) {
                Some(joints) => {
                    match joints {
                        ReadJoints::U8(iter) => {
                            info!("Joints count is {}", iter.count());
                            //for q in iter {                                
                            //}
                        }
                        ReadJoints::U16(iter) => {
                            for q in iter {
                                info!("Found u16 joint");
                            }
                        }
                    }
                }
                None => {}
            }

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
            
            match reader.read_weights(0) {
                Some(weights) => {
                    match weights {
                        ReadWeights::U8(iter) => {
                            for q in iter {
                                info!("Found u8 weight");
                            }
                        }
                        ReadWeights::U16(iter) => {
                            for q in iter {
                                info!("Found u16 weight");
                            }
                        }
                        ReadWeights::F32(iter) => {
                            info!("Weights count is {}", iter.count());
                            //for q in iter {                                
                            //}
                        }
                    }
                }
                None => {}
                None => {}
            }

            reader.read_morph_targets();

            let mut pbr_specular_glossiness_diffuse_texture_index = None;
            let mut pbr_specular_glossiness_texture_index = None;
            let mut base_color_texture_index = None;
            let mut metallic_roughness_texture_index = None;
            let mut normal_texture_index = None;
            let mut occlusion_texture_index = None;
            let mut emmisive_texture_index = None;

            match primitive.material().index() {
                Some(material_index) => {
                    let material = document.materials().nth(material_index).expect("Failed to get material by index");                    
                    info!("Found material");
                    
                    match material.pbr_specular_glossiness() {
                        Some(pbr_specular_glossiness) => {
                            match pbr_specular_glossiness.diffuse_texture() {
                                Some(diffuse_texture) => {
                                    info!("Found pbr specular glossiness diffuse texture, index {}", diffuse_texture.texture().index());
                                    pbr_specular_glossiness_diffuse_texture_index = Some(diffuse_texture.texture().index());
                                }
                                None => {}
                            }
                            match pbr_specular_glossiness.specular_glossiness_texture() {
                                Some(specular_glossiness_texture) => {
                                    info!("Found pbr specular glossiness specular glossiness texture, index {}", specular_glossiness_texture.texture().index());
                                    pbr_specular_glossiness_texture_index = Some(specular_glossiness_texture.texture().index());
                                }
                                None => {}
                            }
                        }
                        None => {}
                    }

                    let pbr_metallic_roughness = material.pbr_metallic_roughness();
                    match pbr_metallic_roughness.base_color_texture() {
                        Some(base_color_texture) => {
                            info!("Found base color texture, index {}", base_color_texture.texture().index());
                            base_color_texture_index = Some(base_color_texture.texture().index());
                        }
                        None => {}
                    }
                    match pbr_metallic_roughness.metallic_roughness_texture() {
                        Some(metallic_roughness_texture) => {
                            info!("Found metallic roughness texture, index {}", metallic_roughness_texture.texture().index());
                            metallic_roughness_texture_index = Some(metallic_roughness_texture.texture().index());
                        }
                        None => {}
                    }
                    match material.normal_texture() {
                        Some(normal_texture) => {
                            info!("Found normal texture, index {}", normal_texture.texture().index());
                            normal_texture_index = Some(normal_texture.texture().index());
                        }
                        None => {}
                    }
                    match material.occlusion_texture() {
                        Some(occlusion_texture) => {
                            info!("Found occlusion texture, index occlusion_texture.texture().index()");
                            occlusion_texture_index = Some(occlusion_texture.texture().index());
                        }
                        None => {}
                    }
                    match material.emissive_texture() {
                        Some(emmisive_texture) => {
                            info!("Found emmisive texture, index {}", emmisive_texture.texture().index());
                            emmisive_texture_index = Some(emmisive_texture.texture().index());
                        }
                        None => {}
                    }                    
                }
                None => {
                    info!("Primitive material index is empty");
                }
            }

            info!("{}, mesh {:?} positions total: {}", model_name, mesh.name(), positions.len());
            info!("{}, mesh {:?} indices total: {}", model_name, mesh.name(), indices.len());
            info!("{}, mesh {:?} normals total: {}", model_name, mesh.name(), normals.len());
            info!("{}, mesh {:?} tangents total: {}", model_name, mesh.name(), tangents.len());
            info!("{}, mesh {:?} bitangents total: {}", model_name, mesh.name(), bitangents.len());
            info!("{}, mesh {:?} texture coordinates total: {}", model_name, mesh.name(), texture_coordinates.len());

            primitives.push(PrimitiveData {
                positions,
                indices,
                normals,
                tangents,
                bitangents,
                texture_coordinates,
                pbr_specular_glossiness_diffuse_texture_index,
                pbr_specular_glossiness_texture_index,
                base_color_texture_index,
                metallic_roughness_texture_index,
                normal_texture_index,
                occlusion_texture_index,
                emmisive_texture_index
            });
        }

        meshes.push(MeshData {            
            primitives
        });        
    }

    info!("{} meshes total: {}", model_name, meshes.len());
    info!("{} images total: {}", model_name, images.len());    

    let mut textures = vec![];
    let mut texture_index = 0;
    let mut loaded_images = vec![];

    for texture in document.textures() {
        info!("Found texture");
        let image = texture.source();
        info!("Image name {:?}, index {}", image.name(), image.index());
        let image_gltf_data = &images[image.index()];

        info!("Model {} image, format {:?}, width {}, height {}", model_name, image_gltf_data.format, image_gltf_data.width, image_gltf_data.height);

        //let bytes_byffer = vec![];
        //let mut cursor = Cursor::new(bytes_byffer);        

        match image_gltf_data.format {
            Format::R8G8B8A8 => {
                let image_buffer = image::ImageBuffer::from_raw(image_gltf_data.width, image_gltf_data.height, image_gltf_data.pixels.clone()).expect("Failed to create image buffer");
                //image_buffer.write_to(&mut cursor, image::ImageFormat::Png).expect("Failed to write image to buffer");        
                loaded_images.push(DynamicImage::ImageRgba8(image_buffer));
            }
            _ => {
                let image_buffer = image::ImageBuffer::from_raw(image_gltf_data.width, image_gltf_data.height, image_gltf_data.pixels.clone()).expect("Failed to create image buffer");
                //image_buffer.write_to(&mut cursor, image::ImageFormat::Png).expect("Failed to write image to buffer");        
                loaded_images.push(DynamicImage::ImageRgb8(image_buffer));
            }
        }        

        textures.push(TextureData {
            index: texture_index,
            format: format!("{:?}", image_gltf_data.format),
            width: image_gltf_data.width,
            height: image_gltf_data.height,
            image_encoded: None
        });

        texture_index = texture_index + 1;
    }

    let mut animations = vec![];

    for animation in document.animations() {
        info!("Found animation, channels {}, samplers {}", animation.channels().count(), animation.samplers().count());

        let mut channels = vec![];
        
        for channel in animation.channels() {
            info!("Found animation channel for node {:?} with {:?}, index {}", channel.target().node().name(), channel.target().property(), channel.target().node().index());

            let interpolation = match channel.sampler().interpolation() {
                gltf::animation::Interpolation::Linear => Interpolation::Linear,
                gltf::animation::Interpolation::Step => Interpolation::Step,
                gltf::animation::Interpolation::CubicSpline => Interpolation::CubicSpline
            };

            let reader = channel.reader(|buffer| Some(&buffers[buffer.index()]));

            let timestamps = match reader.read_inputs() {
                Some(inputs) => {
                    match inputs {
                        gltf::accessor::Iter::Standard(times) => {
                            let times: Vec<f32> = times.collect();                            
                            times
                        }
                        gltf::accessor::Iter::Sparse(_) => {
                            info!("Sparse keyframes not supported");
                            vec![]
                        }
                    }
                }
                None => {
                    info!("Empty animations timestamps");
                    vec![]
                }
            };        

            let payload = match reader.read_outputs() {
                Some(outputs) => {
                    match outputs {
                        gltf::animation::util::ReadOutputs::Translations(translation_iterator) => {
                            let mut translations = vec![];

                            for value in translation_iterator {
                                translations.push(value);                                
                            }

                            //info!("Translations: {}", translations.len());

                            AnimationChannelPayload::Translations(translations)
                        }
                        gltf::animation::util::ReadOutputs::Rotations(rotation) => {
                            let mut rotations = vec![];

                            match rotation {
                                gltf::animation::util::Rotations::I8(_rotation_iterator) => {
                                    info!("I8 rotations not supported");
                                }
                                gltf::animation::util::Rotations::U8(_rotation_iterator) => {
                                    info!("U8 rotations not supported");
                                }
                                gltf::animation::util::Rotations::I16(_rotation_iterator) => {
                                    info!("I16 rotations not supported");
                                }
                                gltf::animation::util::Rotations::U16(_rotation_iterator) => {
                                    info!("U16 rotations not supported");
                                }
                                gltf::animation::util::Rotations::F32(rotation_iterator) => {
                                    for value in rotation_iterator {
                                        rotations.push(value);                                
                                    }
                                }
                            }

                            //info!("Rotations: {}", rotations.len());

                            AnimationChannelPayload::Rotations(rotations)
                        }
                        gltf::animation::util::ReadOutputs::Scales(scale_iterator) => {
                            let mut scales = vec![];            

                            for value in scale_iterator {
                                scales.push(value);                                
                            }

                            //info!("Scales: {}", scales.len());

                            AnimationChannelPayload::Scales(scales)
                        }
                        gltf::animation::util::ReadOutputs::MorphTargetWeights(morph_target_weights) => {
                            let mut weight_morphs = vec![];

                            match morph_target_weights {
                                gltf::animation::util::MorphTargetWeights::I8(_morph_iterator) => {
                                    info!("I8 rotations not supported");
                                }
                                gltf::animation::util::MorphTargetWeights::U8(_morph_iterator) => {
                                    info!("U8 rotations not supported");
                                }
                                gltf::animation::util::MorphTargetWeights::I16(_morph_iterator) => {
                                    info!("I16 rotations not supported");
                                }
                                gltf::animation::util::MorphTargetWeights::U16(_morph_iterator) => {
                                    info!("U16 rotations not supported");
                                }
                                gltf::animation::util::MorphTargetWeights::F32(morph_iterator) => {
                                    for value in morph_iterator {
                                        weight_morphs.push(value);
                                    }
                                }
                            }

                            //info!("Weight morphs: {}", weight_morphs.len());

                            AnimationChannelPayload::WeightMorphs(weight_morphs)
                        }
                    }
                }
                None => {
                    panic!("Empty animation outputs");
                }
            };                                                    

            channels.push(AnimationChannel {        
                interpolation,        
                timestamps,
                payload
            })
        }        

        animations.push(Animation {
            name: animation.name().unwrap_or("Default").to_owned(),
            channels
        });
    }

    (ModelData {
        name: model_name.to_owned(),
        meshes,
        textures,
        animations
    }, loaded_images)
}
