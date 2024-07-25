use std::sync::Arc;
use std::collections::HashMap;
use log::*;
use winit::{dpi::{LogicalSize, PhysicalPosition, PhysicalSize}, event::{ElementState, Event, WindowEvent}, event_loop::{ActiveEventLoop, ControlFlow, EventLoop}, window::Window};
use wgpu::{util::DeviceExt, MemoryHints, RequestAdapterOptions, DeviceDescriptor, StoreOp};
#[cfg(target_arch = "wasm32")]
use winit::{event_loop::EventLoopProxy, platform::web::{WindowExtWebSys, EventLoopExtWebSys}};
#[cfg(not(target_arch = "wasm32"))]
use tokio::runtime::Runtime;
use gpu_api::{bytemuck, pipeline::{self, quad_pipeline}, model::{self, create_object, ViewSource}};
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
    
    let (device, queue) = adapter
        .request_device(        
            &DeviceDescriptor {
                label: None,
                required_features: wgpu::Features::empty(),
                required_limits: if cfg!(target_arch = "wasm32") {
                    wgpu::Limits::downlevel_webgl2_defaults()
                } else {
                    wgpu::Limits::default()
                },
                memory_hints: MemoryHints::Performance
            }, None)
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

    let mut element_pipeline = pipeline::element_pipeline::new(&surface, &device, &adapter, &queue, &scene1.vertices, &scene1.indices);
    
    let (camera, model_pipeline) = pipeline::model_pipeline::new(&surface, &device, &adapter, &queue, layout.size.width as f32, layout.size.height as f32).await;

    let mut objects = vec![];

    let model_data = model_load::load("overlord", "../models/overlord/overlord.gltf");
    
    let view_source = ViewSource {
        x: -5.0,
        y: 0.0,
        z: 0.0,        
        scale_x: 0.05,
        scale_y: 0.05,
        scale_z: 0.05
    };
    
    let object = create_object(&device, &queue, &model_pipeline.texture_bind_group_layout, &model_pipeline.sampler, "overlord", model_data, vec![view_source]);
    objects.push(object);
/*
    let model_data = model_load::load("helm", "../models/helm/DamagedHelmet.gltf");
    
    let view_source = ViewSource {
        x: 10.0,
        y: 8.0,
        z: 0.0,        
        scale_x: 1.0,
        scale_y: 1.0,
        scale_z: 1.0
    };
    
    let object = create_object(&device, &queue, &model_pipeline.texture_bind_group_layout, &model_pipeline.sampler, "helm", model_data, vec![view_source]);
    objects.push(object);
*/
    let model_data = model_load::load("duck", "../models/duck/duck.gltf");
    
    let view_source = ViewSource {
        x: 1.0,
        y: 4.0,
        z: 0.0,        
        scale_x: 0.02,
        scale_y: 0.02,
        scale_z: 0.02
    };
    
    let object = create_object(&device, &queue, &model_pipeline.texture_bind_group_layout, &model_pipeline.sampler, "duck", model_data, vec![view_source]);
    objects.push(object);

    let model_data = model_load::load("plane", "../models/plane/plane.gltf");
    
    let view_source = ViewSource {
        x: 5.0,
        y: 0.0,
        z: 0.0,        
        scale_x: 1.0,
        scale_y: 1.0,
        scale_z: 1.0
    };
    
    let object = create_object(&device, &queue, &model_pipeline.texture_bind_group_layout, &model_pipeline.sampler, "plane", model_data, vec![view_source]);
    objects.push(object);

    let model_data = model_load::load("box", "../models/box/box.glb");
    
    let view_source = ViewSource {
        x: -10.0,
        y: 0.0,
        z: 0.0,        
        scale_x: 1.0,
        scale_y: 1.0,
        scale_z: 1.0
    };
    
    let object = create_object(&device, &queue, &model_pipeline.texture_bind_group_layout, &model_pipeline.sampler, "box", model_data, vec![view_source]);
    objects.push(object);

    let model_data = model_load::load("animated-cube", "../models/animated-cube/animated-cube.gltf");
    
    let view_source = ViewSource {
        x: 8.0,
        y: 7.0,
        z: 0.0,        
        scale_x: 1.0,
        scale_y: 1.0,
        scale_z: 1.0
    };
    
    let object = create_object(&device, &queue, &model_pipeline.texture_bind_group_layout, &model_pipeline.sampler, "animated-cube", model_data, vec![view_source]);
    objects.push(object);
    
    let mut quad_pipeline = pipeline::quad_pipeline::Pipeline::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb);

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
                        //info!("{:?}", position);
                        layout.cursor_physical_position = Some(position);
                    }
                    WindowEvent::MouseInput { device_id: _, state, button, .. } => {
                        match state {
                            ElementState::Pressed => {                                
                            }
                            ElementState::Released => {
                                match current_scene.as_ref() {
                                    "scene1" => {
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
                                    }
                                    "scene2" => {
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
                        info!("{:?}", delta);
                    }
                    WindowEvent::ModifiersChanged(state) => {
                        info!("Modifiers changed");                        
                    }
                    WindowEvent::KeyboardInput { device_id: _, event, is_synthetic } => {
                        warn!("{:?}", event);                    
                    }
                    WindowEvent::RedrawRequested => {
                        info!("Redraw requested");                
        
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
        
                        {
                            let camera_projection_ref: &[f32; 16] = camera.projection.as_ref();
                        
                            let mut camera_slice = staging_belt.write_buffer(
                                &mut encoder,
                                &model_pipeline.camera_buffer,
                                0,
                                wgpu::BufferSize::new(gpu_api::camera::CAMERA_UNIFORM_SIZE).expect("Failed to allocate camera slice"),
                                &device
                            );
        
                            camera_slice.copy_from_slice(bytemuck::cast_slice(camera_projection_ref));
                        }
                        
                        {
                            for object in &objects {
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
                                    depth_stencil_attachment: None,
                                    timestamp_writes: None,
                                    occlusion_query_set: None
                                }
                            );                                                        
        
                            model_pipeline.draw(&mut render_pass, &objects);
                            element_pipeline.draw(&mut render_pass, indices_count);
                            quad_pipeline.draw(&mut render_pass, amount as u32);
                        }
        
                        staging_belt.finish();
                        queue.submit(Some(encoder.finish()));
                        frame.present();
                        staging_belt.recall();          
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
