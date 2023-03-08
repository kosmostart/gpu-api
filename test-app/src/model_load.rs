use log::*;
use gltf::{mesh::{util::ReadIndices, Mode}, Node};
use gpu_api::model::{ModelData, MeshData, PrimitiveData};

fn nodes(node: Node, node_level: usize) {
    info!("Node level: {}, node name: {:?}, node index: {}", node_level, node.name(), node.index());
    //info!("{:?}", node.mesh());
    //info!("{:?}", node.camera());
    info!("{:?}", node.transform());


    for child_node in node.children() {        
        nodes(child_node, node_level + 1);
    }
}

pub fn load(model_path: &str) -> ModelData {        
    let (gltf_data, buffers, _) = gltf::import(model_path).unwrap();

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
            
            info!("Primitive {}, mode {:?}", primitive.index(), primitive.mode());
            info!("{:#?}", primitive.attributes());

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

            /*
            match reader.read_tex_coords() {
                Some(iter) => {}
                None => {}
            }
            */

            /*
            match reader.read_colors() {
                Some(iter) => {}
                None => {}
            }
            */

            info!("Positions total: {}", positions.len());
            info!("Indices total: {}", indices.len());
            info!("Normals total: {}", normals.len());
            info!("Tangents total: {}", tangents.len());
            info!("Bitangents total: {}", bitangents.len());

            primitives.push(PrimitiveData {
                positions,
                indices,
                normals,
                tangents,
                bitangents
            });
        }

        meshes.push(MeshData {            
            primitives
        });
    }    

    ModelData {
        meshes: meshes
    }
}
