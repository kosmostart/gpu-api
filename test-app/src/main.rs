use std::sync::Arc;
use log::*;
use winit::{dpi::{PhysicalPosition, PhysicalSize}, event::{ElementState, Event, MouseScrollDelta, WindowEvent}, event_loop::{ControlFlow, EventLoop}, window::Window};
use wgpu::{util::DeviceExt, MemoryHints, RequestAdapterOptions, DeviceDescriptor, StoreOp};
#[cfg(target_arch = "wasm32")]
use winit::{event_loop::EventLoopProxy, platform::web::{WindowExtWebSys, EventLoopExtWebSys}};
#[cfg(not(target_arch = "wasm32"))]
use tokio::runtime::Runtime;
use gpu_api::{bytemuck, camera::CameraUniform, frame_counter::FrameCounter, glam::Mat4, gpu_api_dto::{AnimationComputation, AnimationProperty}, model::{create_object, ModelAnimationChannel, ObjectGroup}, pipeline::{self, model_pipeline::CAMERA_UNIFORM_SIZE, quad_pipeline}};
use gpu_api::gpu_api_dto::ViewSource;
use element::{Color, ElementCfg, create_element};

mod element;

#[derive(Debug)]
pub enum AppEvent {
}

#[derive(Debug)]
pub struct Halfes {
    pub x: f32,
    pub y: f32
}

pub struct Layout {    
    pub size: PhysicalSize<u32>,
    pub halfes: Halfes,    
    pub cursor_physical_position: Option<PhysicalPosition<f64>>    
}

pub struct Scene {
    vertices: Vec<pipeline::element_pipeline::Vertex>,
    indices: Vec<u32>,
    element_index: u32
}

async fn run() {    
    let mut window_attributes = Window::default_attributes();

    window_attributes = window_attributes
        .with_title("Test application")
        .with_inner_size(winit::dpi::LogicalSize::new(800.0, 600.0));
        
    #[cfg(target_arch = "wasm32")]
    {
        use wasm_bindgen::JsCast;        
        use winit::platform::web::WindowAttributesExtWebSys;

        let canvas = web_sys::window()
            .expect("Failed to get window")
            .document()
            .expect("Failed to get window")
            .get_element_by_id("canvas")
            .expect("Failed to get canvas element")
            .dyn_into::<web_sys::HtmlCanvasElement>()
            .expect("Failed to cast canvas");

        let _ = canvas.set_attribute("style", "width: 800px;height: 600px;outline: none;");

        window_attributes = window_attributes.with_canvas(Some(canvas));
    }

    let event_loop: EventLoop<AppEvent> = EventLoop::with_user_event().build().expect("Failed to create event loop");    
    let window = Arc::new(event_loop.create_window(window_attributes).expect("Failed to create window"));
    let instance = wgpu::Instance::default();
    let surface = instance.create_surface(window.clone()).expect("Failed to create surface");
    let adapter = instance    
        .request_adapter(&RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::default(),
            force_fallback_adapter: false,            
            compatible_surface: Some(&surface),
        })
        .await
        .expect("Failed to find an appropriate adapter");
    
    warn!("{:#?}", adapter.get_info());
    
    let (device, queue) = adapter
        .request_device(        
            &DeviceDescriptor {
                label: None,
                //required_features: wgpu::Features::empty(),
                required_features: wgpu::Features::TEXTURE_FORMAT_16BIT_NORM,
                required_limits: if cfg!(target_arch = "wasm32") {
                    wgpu::Limits::downlevel_webgl2_defaults()
                } else {
                    wgpu::Limits::default()
                },
                memory_hints: MemoryHints::Performance,
                trace: wgpu::Trace::Off
            })
        .await
        .expect("Failed to create device");    

    let size = window.inner_size();
    let scale_factor = window.scale_factor();

    let halfes = Halfes {
        x: (size.width / 2) as f32,
        y: (size.height / 2)  as f32
    };    

    info!("{:?}, {}", size, scale_factor);
    info!("{:?}", halfes);    

    let mut layout = Layout {        
        size,
        halfes,        
        cursor_physical_position: None        
    };        

    let mut config = surface
    .get_default_config(&adapter, layout.size.width, layout.size.height)
    .expect("Surface isn't supported by the adapter.");

    config.format = wgpu::TextureFormat::Rgba8Unorm;
    config.view_formats.push(wgpu::TextureFormat::Rgba8UnormSrgb);

    surface.configure(&device, &config);

    let background_color = Color {
        r: 0.10196,
        g: 0.10196,
        b: 0.10196,
        a: 1.0
    };

    let border_color = Color {
        r: 0.0,
        g: 1.0,
        b: 0.0,
        a: 1.0
    };

    let mut scene1 = Scene {
        vertices: vec![],
        indices: vec![],
        element_index: 0
    };

    let mut scene2 = Scene {
        vertices: vec![],
        indices: vec![],
        element_index: 0
    };

    while scene1.element_index < 100 {
        let element_cfg = ElementCfg { 
            x: scene1.element_index as i32 * 10 + 30,
            y: 30, 
            width: 30, 
            height: 30,
            background_color, 
            border_color: Some(border_color)
        };
    
        create_element(&layout, element_cfg, &mut scene1);

        scene1.element_index = scene1.element_index + 1;
    }

    while scene2.element_index < 100 {
        let element_cfg = ElementCfg { 
            x: scene2.element_index as i32 * 10 + 30,
            y: 130,
            width: 30, 
            height: 30,
            background_color, 
            border_color: Some(border_color)
        };
    
        create_element(&layout, element_cfg, &mut scene2);

        scene2.element_index = scene2.element_index + 1;
    }
    
    let mut current_scene = "scene1".to_owned();

    let mut indices_count = scene1.indices.len() as u32;

    let depth_stencil_state = Some(wgpu::DepthStencilState {
        format: wgpu::TextureFormat::Depth32Float,
        depth_write_enabled: true,
        depth_compare: wgpu::CompareFunction::Always,
        stencil: wgpu::StencilState::default(),
        bias: wgpu::DepthBiasState::default()
    });

    let model_depth_stencil_state = Some(wgpu::DepthStencilState {
        format: wgpu::TextureFormat::Depth32Float,
        depth_write_enabled: true,
        depth_compare: wgpu::CompareFunction::Less,
        stencil: wgpu::StencilState::default(),
        bias: wgpu::DepthBiasState::default()
    });

    //let mut element_pipeline = pipeline::element_pipeline::new(&device, &queue, &scene1.vertices, &scene1.indices, depth_stencil_state.clone());
    
    let (mut camera, model_pipeline) = pipeline::model_pipeline::new(&device, &config, layout.size.width as f32, layout.size.height as f32, model_depth_stencil_state).await;

    let mut object_group = ObjectGroup {
        active: true,
        objects: vec![]
    };

    let frame_cycle_length = 200;

    /*
    let (model_data, loaded_images) = model_load::load("overlord", "../models/overlord/overlord.gltf");
    
    let view_source = ViewSource {
        x: -25.0,
        y: -20.0,
        z: 10.0,        
        scale_x: 0.1,
        scale_y: 0.1,
        scale_z: 0.1
    };
    
    let object = create_object(&device, &queue, &model_pipeline, model_data, Some(loaded_images), vec![view_source]);
    object_group.objects.push(object);    
    */
    
    /*
    let (model_data, loaded_images) = model_load::load("helm", "../models/helm/DamagedHelmet.gltf");
    
    let view_source = ViewSource {
        x: 10.0,
        y: 8.0,
        z: 0.0,        
        scale_x: 1.0,
        scale_y: 1.0,
        scale_z: 1.0
    };
    
    let object = create_object(&device, &queue, &model_pipeline, model_data, Some(loaded_images), vec![view_source]);
    objects.push(object);
*/

    let (model_data, loaded_images) = model_load::load("damaged-helmet", "../models/damaged-helmet/DamagedHelmet.gltf", false, true, vec![], false);
    //let (model_data, loaded_images) = model_load::load("test", "../models/test/xgimega_tier_2.gltf", false, true, vec![]);
    //let (model_data, loaded_images) = model_load::load("test", "../models/test/Warrior.glb", false, true, vec![71], true);
    
    let view_source = ViewSource {
        x: 0.0,
        y: 0.0,
        z: 0.0,        
        scale_x: 5.0,
        scale_y: 5.0,
        scale_z: 5.0
    };
    
    let object = create_object(&device, &queue, &model_pipeline, model_data, vec![view_source], loaded_images, frame_cycle_length);
    
    //object.update_view_with_rotation(gpu_api::glam::Quat::from_rotation_y(-1.0));

    object_group.objects.push(object);

/*
    let (model_data, loaded_images) = model_load::load("skin-test", "../models/skin-test/skin-test.glb");
    
    let view_source = ViewSource {
        x: -2.0,
        y: -5.0,
        z: 0.0,        
        scale_x: 5.0,
        scale_y: 5.0,
        scale_z: 5.0
    };
    
    let object = create_object(&device, &queue, &model_pipeline, model_data, Some(loaded_images), vec![view_source]);
    object_group.objects.push(object);
 */
    /*    
    let (model_data, loaded_images) = model_load::load("box", "../models/box/box.glb");
    
    let view_source = ViewSource {
        x: -10.0,
        y: 0.0,
        z: 0.0,        
        scale_x: 1.0,
        scale_y: 1.0,
        scale_z: 1.0
    };
    
    let object = create_object(&device, &queue, &model_pipeline, model_data, Some(loaded_images), vec![view_source]);
    object_group.objects.push(object);
    */

    /*    

    let (model_data, loaded_images) = model_load::load("animated-cube", "../models/animated-cube/animated-cube.gltf");
    
    let view_source = ViewSource {
        x: 5.0,
        y: 7.0,
        z: 0.0,        
        scale_x: 1.0,
        scale_y: 1.0,
        scale_z: 1.0
    };    

    let object = create_object(&device, &queue, &model_pipeline, model_data, Some(loaded_images), vec![view_source]);
    object_group.objects.push(object);
    */

    let mut object_groups = vec![];
    object_groups.push(object_group);
    
    let mut quad_pipeline = pipeline::quad_pipeline::Pipeline::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb, depth_stencil_state);

    let transformation = quad_pipeline::Transformation::orthographic(layout.size.width, layout.size.height);
    let mut quad_uniforms = quad_pipeline::Uniforms::new(transformation, scale_factor as f32);

    let component_coordinates = [0.0, 0.0, 950.0, 950.0];
    let has_overlay = 0;
    let overlay_coordinates = [0.0, 0.0, 0.0, 0.0];

    let shadow_color = [0.0, 0.0, 0.0, 0.0];
    let shadow_offset = [0.0, 0.0];
    let shadow_blur_radius = 0.0;

    let quads = vec![        
        quad_pipeline::Quad {
            border_color: [0.0, 0.5, 0.0, 1.0],
            border_radius: [10.0, 10.0, 10.0, 10.0],
            color: [1.0, 0.0, 0.0, 1.0],
            border_width: 0.0,
            position: [100.0, 100.0],
            size: [100.0, 100.0],
            component_coordinates,
            has_overlay,
            overlay_coordinates,
            shadow_color,
            shadow_offset,
            shadow_blur_radius
        },
        quad_pipeline::Quad {
            border_color: [0.0, 0.5, 0.0, 1.0],
            border_radius: [15.0, 15.0, 15.0, 15.0],
            color: [1.0, 0.0, 0.0, 1.0],
            border_width: 0.0,
            position: [300.0, 100.0],
            size: [30.0, 30.0],
            component_coordinates,
            has_overlay,
            overlay_coordinates,
            shadow_color,
            shadow_offset,
            shadow_blur_radius
        },
        quad_pipeline::Quad {
            border_color: [0.0, 0.5, 0.0, 1.0],
            border_radius: [10.0, 10.0, 10.0, 10.0],
            color: [1.0, 1.0, 1.0, 1.0],
            border_width: 1.0,
            position: [500.0, 500.0],
            size: [100.0, 100.0],
            component_coordinates,
            has_overlay,
            overlay_coordinates,
            shadow_color,
            shadow_offset,
            shadow_blur_radius
        }        
    ];

    let mut staging_belt = wgpu::util::StagingBelt::new(5 * 1024);

    let mut frame_counter = FrameCounter::new(200);

    event_loop.run(move |event, target| {        
        target.set_control_flow(ControlFlow::Wait);

        match event {
            Event::WindowEvent { event: window_event, window_id } => {
                match window_event {
                    WindowEvent::CloseRequested => {                        
                        info!("Event loop close requested");
                        target.exit();
                    }
                    WindowEvent::Resized(new_size) => {                        
                        info!("Resized");

                        let scale_factor = window.scale_factor();
                        info!("{:?}, {}", new_size, scale_factor);

                        layout.size = new_size;

                        layout.halfes = Halfes {
                            x: layout.size.width as f32 / 2.0,
                            y: layout.size.height as f32  / 2.0
                        };
                        
                        info!("{:?}", layout.halfes);

                        quad_uniforms = quad_pipeline::Uniforms::new(transformation, scale_factor as f32);

                        surface.configure(&device, &config);
                    }
                    WindowEvent::CursorMoved { device_id: _, position, .. } => {                        
                        //warn!("{:?}", position);                        

                        let norm_x = position.x as f32 / layout.size.width as f32 - 0.5;
                        let norm_y = position.y as f32 / layout.size.height as f32 - 0.5;
                        camera.angle_y = norm_x * 5.0;
                        camera.angle_xz = norm_y;
                    }
                    WindowEvent::MouseInput { device_id: _, state, button, .. } => {
                        match state {
                            ElementState::Pressed => {                                
                            }
                            ElementState::Released => {
                                match current_scene.as_ref() {
                                    "scene1" => {
                                        /*
                                        element_pipeline.vertex_buffer = device.create_buffer_init(
                                            &wgpu::util::BufferInitDescriptor {
                                                label: Some("Vertex Buffer"),
                                                contents: bytemuck::cast_slice(&scene1.vertices),
                                                usage: wgpu::BufferUsages::VERTEX
                                            }
                                        );
                                    
                                        element_pipeline.index_buffer = device.create_buffer_init(
                                            &wgpu::util::BufferInitDescriptor {
                                                label: Some("Index Buffer"),
                                                contents: bytemuck::cast_slice(&scene1.indices),
                                                usage: wgpu::BufferUsages::INDEX
                                            }
                                        );
                                        
                                        indices_count = scene1.indices.len() as u32;
    
                                        current_scene = "scene2".to_owned();
                                        */
                                    }
                                    "scene2" => {
                                        /*
                                        element_pipeline.vertex_buffer = device.create_buffer_init(
                                            &wgpu::util::BufferInitDescriptor {
                                                label: Some("Vertex Buffer"),
                                                contents: bytemuck::cast_slice(&scene2.vertices),
                                                usage: wgpu::BufferUsages::VERTEX
                                            }
                                        );
                                    
                                        element_pipeline.index_buffer = device.create_buffer_init(
                                            &wgpu::util::BufferInitDescriptor {
                                                label: Some("Index Buffer"),
                                                contents: bytemuck::cast_slice(&scene2.indices),
                                                usage: wgpu::BufferUsages::INDEX
                                            }
                                        );
                                        */
                                        
                                        indices_count = scene2.indices.len() as u32;
    
                                        current_scene = "scene1".to_owned();
                                    }
                                    _ => {}
                                }
                                 

                                window.request_redraw();
                            }                            
                        }
                    }
                    WindowEvent::MouseWheel { device_id: _, delta, phase: _ } => {                        
                        match delta {
                            MouseScrollDelta::LineDelta(_, vertical_delta) => {
                                if vertical_delta > 0.0 {
                                    camera.dist = camera.dist - 10.0;
                                } else {
                                    camera.dist = camera.dist + 10.0;
                                }
                            }
                            MouseScrollDelta::PixelDelta(position) => {
                                if position.y > 0.0 {
                                    camera.dist = camera.dist - 10.0;
                                    info!("1");
                                } else {
                                    camera.dist = camera.dist + 10.0;
                                    info!("2");
                                }
                            }
                        }        
                    }
                    WindowEvent::ModifiersChanged(state) => {
                        info!("Modifiers changed");                        
                    }
                    WindowEvent::KeyboardInput { device_id: _, event, is_synthetic } => {
                        warn!("{:?}", event);                    
                    }
                    WindowEvent::RedrawRequested => {                        
                        //info!("Redraw requested");

                        if frame_counter.update() {
                            // Get a command encoder for the current frame
                                let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                                    label: Some("Redraw")
                                }
                            );
            
                            // Get the next frame
                            let frame = surface.get_current_texture().expect("Get next frame");                        
                            let mut texture_view_descriptor = wgpu::TextureViewDescriptor::default();
                            texture_view_descriptor.format = Some(wgpu::TextureFormat::Rgba8UnormSrgb);
                            let view = &frame.texture.create_view(&texture_view_descriptor);                        
            
                            {
                                let mut uniform_buffer = staging_belt.write_buffer(
                                    &mut encoder,
                                    &quad_pipeline.uniform_buffer,
                                    0,
                                    wgpu::BufferSize::new(std::mem::size_of::<quad_pipeline::Uniforms>() as u64)
                                        .unwrap(),
                                    &device
                                );
            
                                uniform_buffer.copy_from_slice(bytemuck::bytes_of(&quad_uniforms));
                            }
                            
                            let amount = {
                                let i = 0;
                                let total = quads.len();
                                let end = (i + quad_pipeline::MAX_INSTANCES).min(total);
                                let res = end - i;
            
                                let instance_bytes = bytemuck::cast_slice(&quads[i..end]);
            
                                let mut instance_buffer = staging_belt.write_buffer(
                                    &mut encoder,
                                    &quad_pipeline.instance_buffer,
                                    0,
                                    wgpu::BufferSize::new(instance_bytes.len() as u64).unwrap(),
                                    &device,
                                );
            
                                instance_buffer.copy_from_slice(instance_bytes);
            
                                res
                            };

                            camera.update(layout.size.width as f32, layout.size.height as f32);
            
                            {                                                            
                                let mut camera_slice = staging_belt.write_buffer(
                                    &mut encoder,
                                    &model_pipeline.camera_buffer,
                                    0,
                                    wgpu::BufferSize::new(CAMERA_UNIFORM_SIZE).expect("Failed to allocate camera slice"),
                                    &device
                                );

                                let camera_uniform = CameraUniform {
                                    camera_position: camera.camera_position.to_array(),
                                    padding: 0,
                                    view: camera.view.to_cols_array(),
                                    projection: camera.projection.to_cols_array()
                                };
            
                                camera_slice.copy_from_slice(bytemuck::bytes_of(&camera_uniform));
                            }
                            
                            {                            
                                for object_group in &mut object_groups {
                                    for object in &mut object_group.objects {
                                        if object_group.active == false {
                                            continue;
                                        }
                                        
                                        if object.instances_amount == 0 {
                                            continue;
                                        }                                        

                                        let animation_index = 1;

                                        match object.animation_computation {
                                            AnimationComputation::NotAnimated => {
                                                for mesh in &object.meshes {
                                                    match &mesh.node_transform {
                                                        Some(node_transform) => {
                                                            let mut node_transform_slice = staging_belt.write_buffer(
                                                                &mut encoder,
                                                                &mesh.node_transform_buffer,
                                                                0,
                                                                wgpu::BufferSize::new(gpu_api::pipeline::model_pipeline::NODE_TRANSFORM_UNIFORM_SIZE).expect("Failed to allocate node transform slice"),
                                                                &device
                                                            );
                                        
                                                            node_transform_slice.copy_from_slice(bytemuck::bytes_of(node_transform));
                                                        }
                                                        None => {}
                                                    }
                                                }
                                            }
                                            AnimationComputation::ComputeInRealTime => {
                                                for channel in &mut object.animations[animation_index].channels {
                                                    let current_time = channel.start_instant.elapsed().as_secs_f32();

                                                    let mut frame_index = channel.frame_index;
                    
                                                    for timestamp in channel.timestamps.iter().skip(frame_index) {
                                                        if timestamp > &current_time {
                                                            break;
                                                        }
                                                        
                                                        frame_index = frame_index + 1;
                                                    }

                                                    if frame_index == channel.timestamps.len() {
                                                        frame_index = 0;
                                                        channel.frame_index = 0;
                                                        #[cfg(not(target_arch = "wasm32"))] {
                                                            channel.start_instant = std::time::Instant::now();
                                                        }                                                    
                                                        #[cfg(target_arch = "wasm32")] {
                                                            channel.start_instant = web_time::Instant::now();
                                                        }
                                                    }

                                                    let previous_frame_index = match frame_index {
                                                        0 => 0,
                                                        _ => frame_index - 1
                                                    };

                                                    let factor = (current_time - channel.timestamps[previous_frame_index]) / (channel.timestamps[frame_index] - channel.timestamps[previous_frame_index]);

                                                    match &channel.property {
                                                        AnimationProperty::Translation => {
                                                            let translation = channel.translations[previous_frame_index].lerp(channel.translations[frame_index], factor);
                                                            object.nodes[channel.target_index].translation = translation;
                                                        }
                                                        AnimationProperty::Rotation => {                                                        
                                                            let rotation = channel.rotations[previous_frame_index].lerp(channel.rotations[frame_index], factor).normalize();
                                                            object.nodes[channel.target_index].rotation = rotation;
                                                        }
                                                        AnimationProperty::Scale => {
                                                            let scale = channel.scales[previous_frame_index].lerp(channel.scales[frame_index], factor);
                                                            object.nodes[channel.target_index].scale = scale;
                                                        }
                                                        AnimationProperty::MorphTargetWeights => {
                                                            let weight_morph = channel.weight_morphs[frame_index];
                                                        }
                                                    }                                    
                                                }
                                                
                                                for node_index in object.node_topological_sorting.iter() {
                                                    match object.node_map.get(node_index) {
                                                        Some(parent_index) => {
                                                            let parent_transform = object.nodes[*parent_index].global_transform_matrix;
                                                            let node = &mut object.nodes[*node_index];
                                            
                                                            let local_transform = gpu_api::glam::Mat4::from_scale_rotation_translation(node.scale, node.rotation, node.translation);

                                                            node.global_transform_matrix = parent_transform * local_transform;
                                                        }
                                                        None => {}
                                                    }                                            
                                                }                                    

                                                let mut joint_matrices: [[f32; 16]; gpu_api::pipeline::model_pipeline::JOINT_MATRICES_AMOUNT] = [Mat4::IDENTITY.to_cols_array(); gpu_api::pipeline::model_pipeline::JOINT_MATRICES_AMOUNT];
                                                
                                                let mut joint_matrix_index = 0;
                                                let skin_index = 0;
                                                
                                                for joint in &object.skins[skin_index].joints {
                                                    //let joint_matrix = inverse_node_global_transform * object.nodes[joint.node_index].global_transform_matrix * joint.inverse_bind_matrix;
                                                    let joint_matrix = object.nodes[joint.node_index].global_transform_matrix * joint.inverse_bind_matrix;

                                                    joint_matrices[joint_matrix_index] = joint_matrix.to_cols_array();

                                                    joint_matrix_index = joint_matrix_index + 1;
                                                }
                                                
                                                for node in &object.nodes {
                                                    match node.mesh_index {
                                                        Some(mesh_index) => {                                                            
                                                            match &mut object.meshes[mesh_index].node_transform {
                                                                Some(node_transform) => {                                                                    
                                                                    node_transform.transform = node.global_transform_matrix.to_cols_array();
                                                                }
                                                                None => {}
                                                            }
                                                        }
                                                        None => {}
                                                    }
                                                }                                                
                                                                                                
                                                /*
                                                // Inverse node global transform            

                                                for node in &nodes {                
                                                    match node.skin_index {
                                                        Some(skin_index) => {
                                                            let inverse_node_global_transform = node.global_transform_matrix;
                                                            let mut joint_matrix_index = 0;

                                                            for joint in &skins[skin_index].joints {                
                                                                let joint_matrix = inverse_node_global_transform * nodes[joint.node_index].global_transform_matrix * joint.inverse_bind_matrix;            
                                                
                                                                //let joint_matrix = nodes[joint.node_index].global_transform_matrix * joint.inverse_bind_matrix;
                                                                
                                                                joint_matrices[joint_matrix_index] = joint_matrix.to_cols_array();
                                                
                                                                joint_matrix_index = joint_matrix_index + 1;
                                                            }
                                                        }
                                                        None => {}
                                                    }                
                                                }
                                                */                                    

                                                let joint_matrices_ref: &[[f32; 16]] = joint_matrices.as_ref();                                                

                                                {                                                                
                                                    let mut joint_matrices_slice = staging_belt.write_buffer(
                                                        &mut encoder,
                                                        &object.joint_matrices_buffer,
                                                        0,
                                                        wgpu::BufferSize::new(gpu_api::pipeline::model_pipeline::JOINT_MATRICES_UNIFORM_SIZE).expect("Failed to allocate joint matrices slice"),
                                                        &device
                                                    );
                                
                                                    joint_matrices_slice.copy_from_slice(bytemuck::cast_slice(joint_matrices_ref));
                                                }

                                                for mesh in &object.meshes {
                                                    match &mesh.node_transform {
                                                        Some(node_transform) => {
                                                            let mut node_transform_slice = staging_belt.write_buffer(
                                                                &mut encoder,
                                                                &mesh.node_transform_buffer,
                                                                0,
                                                                wgpu::BufferSize::new(gpu_api::pipeline::model_pipeline::NODE_TRANSFORM_UNIFORM_SIZE).expect("Failed to allocate node transform slice"),
                                                                &device
                                                            );
                                        
                                                            node_transform_slice.copy_from_slice(bytemuck::bytes_of(node_transform));
                                                        }
                                                        None => {}
                                                    }
                                                }
                                            }
                                            AnimationComputation::PreComputed => {                                                
                                                if object.animations[animation_index].frame_index == object.animations[animation_index].frame_cycle_count {
                                                    object.animations[animation_index].frame_index = 7;
                                                }
                                                
                                                let joint_matrices_ref: &[[f32; 16]] = object.animations[animation_index].joint_matrices[object.animations[animation_index].frame_index].as_ref();
                                                
                                                {
                                                    let mut joint_matrices_slice = staging_belt.write_buffer(
                                                        &mut encoder,
                                                        &object.joint_matrices_buffer,
                                                        0,
                                                        wgpu::BufferSize::new(gpu_api::pipeline::model_pipeline::JOINT_MATRICES_UNIFORM_SIZE).expect("Failed to allocate joint matrices slice"),
                                                        &device
                                                    );
                                
                                                    joint_matrices_slice.copy_from_slice(bytemuck::cast_slice(joint_matrices_ref));
                                                }

                                                for mesh in &object.meshes {
                                                    if mesh.node_transform.is_some() {
                                                        let mut node_transform_slice = staging_belt.write_buffer(
                                                            &mut encoder,
                                                            &mesh.node_transform_buffer,
                                                            0,
                                                            wgpu::BufferSize::new(gpu_api::pipeline::model_pipeline::NODE_TRANSFORM_UNIFORM_SIZE).expect("Failed to allocate node transform slice"),
                                                            &device
                                                        );

                                                        let mesh_node_transform = &object.animations[animation_index].mesh_node_transforms[mesh.index];
                                                        
                                                        let node_transform = &mesh_node_transform.node_transforms[object.animations[animation_index].frame_index];
                                    
                                                        node_transform_slice.copy_from_slice(bytemuck::bytes_of(node_transform));
                                                    }                                                    
                                                }

                                                object.animations[animation_index].frame_index = object.animations[animation_index].frame_index + 1;                                                
                                            }
                                        }
                                        
                                        let mut view_slice = staging_belt.write_buffer(
                                            &mut encoder,
                                            &object.instance_buffer,
                                            0,
                                            wgpu::BufferSize::new(object.views_size).expect("Failed to allocate view slice"),
                                            &device
                                        );
                    
                                        view_slice.copy_from_slice(bytemuck::cast_slice(&object.views));
                                    }
                                }
                            }
            
                            // Clear frame
                            {
                                let mut render_pass = encoder.begin_render_pass(
                                    &wgpu::RenderPassDescriptor {
                                        label: Some("Render pass"),
                                        color_attachments: &[
                                            Some(wgpu::RenderPassColorAttachment {
                                                view,
                                                resolve_target: None,
                                                ops: wgpu::Operations {
                                                    load: wgpu::LoadOp::Clear(
                                                        wgpu::Color {
                                                            r: 1.0,
                                                            g: 1.0,
                                                            b: 1.0,
                                                            a: 1.0
                                                        },
                                                    ),
                                                    store: StoreOp::Store
                                                }
                                            })
                                        ],
                                        depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                                            view: &model_pipeline.depth_texture.view,
                                            depth_ops: Some(wgpu::Operations {
                                                load: wgpu::LoadOp::Clear(1.0),
                                                store: wgpu::StoreOp::Store,
                                            }),
                                            stencil_ops: None,
                                        }),
                                        timestamp_writes: None,
                                        occlusion_query_set: None
                                    }
                                );                                                        
            
                                model_pipeline.draw(&mut render_pass, &object_groups);
                                //element_pipeline.draw(&mut render_pass, indices_count);
                                quad_pipeline.draw(&mut render_pass, amount as u32);
                            }
            
                            staging_belt.finish();
                            queue.submit(Some(encoder.finish()));
                            frame.present();
                            staging_belt.recall();                        
                        }

                        window.request_redraw();
                    }              
                    _ => {}
                }
            }
            Event::UserEvent(_) => {                
            }                        
            _ => {}
        }
    }).expect("Event loop failed");
}

fn main() {
    #[cfg(not(target_arch = "wasm32"))]
    {        
        env_logger::init();

        let rt = Runtime::new().expect("Failed to create runtime");
        
        rt.block_on(run());
    }
    #[cfg(target_arch = "wasm32")]
    {
        std::panic::set_hook(Box::new(console_error_panic_hook::hook));
        console_log::init_with_level(log::Level::Warn).expect("Could not initialize logger");
        wasm_bindgen_futures::spawn_local(run(event_loop, window));
    }
}
