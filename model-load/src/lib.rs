use std::io::{Cursor, Read, Write};
use image::{DynamicImage, Rgb, Rgba};
use log::*;
use gltf::{image::Format, iter, mesh::util::{ReadIndices, ReadJoints, ReadTexCoords, ReadWeights}};
use gpu_api_dto::{Animation, AnimationChannel, AnimationProperty, Interpolation, MeshData, ModelData, Node, Skin, Joint, PrimitiveData, TextureData};
pub use gpu_api_dto;

pub fn load(model_name: &str, model_path: &str) -> (ModelData, Vec<DynamicImage>) {
    info!("Loading model {} from path {}", model_name, model_path);    
    let (document, buffers, images) = gltf::import(model_path).expect("Model import failed");
    
    info!("Found {} scenes", document.scenes().count());

    for scene in document.scenes() {
        info!("Found scene {:?}, index {}", scene.name(), scene.index());        
    }

    info!("Found {} nodes", document.nodes().count());    

    let mut nodes = vec![];

    /// Computes topological ordering and children->parent map.
    fn node_indices_topological_sort(nodes: iter::Nodes) -> (Vec<usize>, std::collections::BTreeMap<usize, usize>) {
        // NOTE: The algorithm uses BTreeMaps to guarantee consistent ordering.

        // Maps parent to list of children
        let mut children = std::collections::BTreeMap::<usize, Vec<usize>>::new();
        for node in nodes {
            children.insert(node.index(), node.children().map(|n| n.index()).collect());
        }        

        // Maps child to parent
        let parents: std::collections::BTreeMap<usize, usize> =
            children.iter().flat_map(|(parent, children)| children.iter().map(|ch| (*ch, *parent))).collect();

        // Initialize the BFS queue with nodes that don't have any parent (i.e. roots)
        let mut queue: std::collections::VecDeque<usize> = children.keys().filter(|n| parents.get(n).is_none()).cloned().collect();

        let mut topological_sort = Vec::<usize>::new();

        while let Some(n) = queue.pop_front() {
            topological_sort.push(n);
            for ch in &children[&n] {
                queue.push_back(*ch);
            }
        }

        (topological_sort, parents)
    }

    let (node_topological_sorting, node_map) = node_indices_topological_sort(document.nodes());
    info!("{:?}", node_topological_sorting);
    info!("{:?}", node_map);

    for node in document.nodes() {
        /*
        info!("Found node {:?}, index {}, mesh {:?}, skin {:?}, weights {:?}", 
            node.name(), 
            node.index(), 
            node.mesh().map(|v| v.name()),
            node.skin().map(|v| v.name()),
            node.weights()            
        );
        */

        let mut child_list = node.index().to_string();        
        child_list.push_str(" node :");

        for q in node.children() {            
            //info!("Found child node, index {}", q.index());
            child_list.push_str(" ");
            child_list.push_str(&q.index().to_string());
        }
        
        info!("{}", child_list);

        let local_transform_matrix = node.transform().matrix();

        let (translation, rotation, scale) = node.transform().decomposed();        

        nodes.push(Node {
            index: node.index(),
            name: node.name().map(|v| v.to_owned()),
            translation, 
            rotation, 
            scale,
            local_transform_matrix
        });
    }    

    let mut skins = vec![];

    for skin in document.skins() {
        info!("Found skin {:?}, index {}", skin.name(), skin.index());

        let num_joints = skin.joints().count();
        let reader = skin.reader(|b| Some(&buffers[b.index()][..b.length()]));   
        
        let inv_b_mats: Vec<[[f32; 4]; 4]> = match reader.read_inverse_bind_matrices() {            
            Some(inv_b_mats) => {
                inv_b_mats.collect()
            }
            None => {
                let mut inv_b_mats = vec![];

                let q = [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0]
                ];
                
                for _ in 0..num_joints {
                    inv_b_mats.push(q);
                }

                inv_b_mats
            }            
        };        

        let mut joints = vec![];

        let mut index = 0;        

        for joint in skin.joints() {            
            joints.push(Joint { 
                node_index: joint.index(),
                node_name: joint.name().map(|v| v.to_owned())
            });

            index = index + 1;
        }

        info!("Joints total: {}", joints.len());

        skins.push(Skin {
                name: skin.name().map(|v| v.to_owned()),
                inverse_bind_matrices: inv_b_mats, 
                joints
            }
        );
    }

    info!("Skins total: {}", skins.len());

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

            //bitangent = cross(normal.xyz, tangent.xyz) * tangent.w.

            match reader.read_tangents() {
                Some(iter) => {                    
                    for tangent in iter {
                        tangents.push(tangent);                        
                    }                    
                }
                None => {}
            }

            match reader.read_colors(0) {
                Some(iter) => {}
                None => {}
            }

            let mut joints = vec![];

            match reader.read_joints(0) {
                Some(read_joints) => {
                    match read_joints {
                        ReadJoints::U8(iter) => {                            
                            for joint in iter {
                                joints.push([joint[0] as u32, joint[1] as u32, joint[2] as u32, joint[3] as u32]);
                            }
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

            let mut weights = vec![];
            
            match reader.read_weights(0) {
                Some(read_weights) => {
                    match read_weights {
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
                            for weight in iter {
                                weights.push(weight);
                            }
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
            info!("{}, mesh {:?} joints total: {}", model_name, mesh.name(), joints.len());
            info!("{}, mesh {:?} weights total: {}", model_name, mesh.name(), weights.len());
            info!("{}, mesh {:?} texture coordinates total: {}", model_name, mesh.name(), texture_coordinates.len());

            if tangents.is_empty() {
                for _ in 0..positions.len() {
                    tangents.push([0.0, 0.0, 0.0, 0.0]);
                }
            }

            primitives.push(PrimitiveData {
                positions,
                indices,
                normals,
                tangents,                
                joints,
                weights,
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

            let animation_property = match channel.target().property() {
                gltf::animation::Property::Translation => AnimationProperty::Translation,
                gltf::animation::Property::Rotation => AnimationProperty::Rotation,
                gltf::animation::Property::Scale => AnimationProperty::Scale,
                gltf::animation::Property::MorphTargetWeights => AnimationProperty::MorphTargetWeights
            };


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

            let mut translations = vec![];
            let mut rotations = vec![];
            let mut scales = vec![];
            let mut weight_morphs = vec![];

            match reader.read_outputs() {
                Some(outputs) => {
                    match outputs {
                        gltf::animation::util::ReadOutputs::Translations(translation_iterator) => {
                            for value in translation_iterator {
                                translations.push(value);                                
                            }
                        }
                        gltf::animation::util::ReadOutputs::Rotations(rotation) => {
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
                        }
                        gltf::animation::util::ReadOutputs::Scales(scale_iterator) => {                            
                            for value in scale_iterator {
                                scales.push(value);                                
                            }
                        }
                        gltf::animation::util::ReadOutputs::MorphTargetWeights(morph_target_weights) => {
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
                        }
                    }
                }
                None => {
                    panic!("Empty animation outputs");
                }
            };                                                    

            channels.push(AnimationChannel {
                target_index: channel.target().node().index(),
                property: animation_property,    
                interpolation,        
                timestamps,
                translations,
                rotations,
                scales,
                weight_morphs
            })
        }        

        animations.push(Animation {
            name: animation.name().unwrap_or("Default").to_owned(),
            channels
        });
    }

    (ModelData {        
        name: model_name.to_owned(),
        nodes,
        node_topological_sorting,
        node_map,
        skins,
        meshes,
        textures,
        animations
    }, loaded_images)
}
