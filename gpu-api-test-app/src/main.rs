use std::collections::HashMap;
use log::*;
use winit::{event_loop::{EventLoop, ControlFlow, EventLoopWindowTarget}, window::Window, event::{Event, WindowEvent, ElementState}, dpi::{PhysicalPosition,  PhysicalSize}};
#[cfg(target_arch = "wasm32")]
use winit::{event_loop::EventLoopProxy, platform::web::WindowExtWebSys};
#[cfg(not(target_arch = "wasm32"))]
use tokio::runtime::Runtime;
use gpu_api::{bytemuck, wgpu, wgpu::util::DeviceExt};
use gpu_api::{pipeline, model::create_model};

mod model_load;

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

async fn run(event_loop: EventLoop<AppEvent>, window: Window) {    
    let instance = wgpu::Instance::default();
    let surface = unsafe { instance.create_surface(&window) }.expect("Failed to create surface");
    let adapter = instance    
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::default(),
            force_fallback_adapter: false,            
            compatible_surface: Some(&surface),
        })
        .await
        .expect("Failed to find an appropriate adapter");
    
    let (device, queue) = adapter
        .request_device(
            &wgpu::DeviceDescriptor {
                features: wgpu::Features::empty(),
                //limits: wgpu::Limits::default(),
                // WebGL doesn't support all of wgpu's features, so if
                // we're building for the web we'll have to disable some.
                limits: if cfg!(target_arch = "wasm32") {
                    wgpu::Limits::downlevel_webgl2_defaults()
                } else {
                    wgpu::Limits::default()
                },                
                label: None
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
        
    let swapchain_capabilities = surface.get_capabilities(&adapter);
    let swapchain_format = swapchain_capabilities.formats[0];

    surface.configure(
        &device,
        &wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,            
            format: swapchain_format,
            width: layout.size.width,
            height: layout.size.height,
            present_mode: wgpu::PresentMode::AutoVsync,
            alpha_mode: wgpu::CompositeAlphaMode::Auto,
            view_formats: vec![]
        }
    );            
      
    let (mut camera, mut camera_controller, mut camera_uniform, mut pipeline) = pipeline::new(&surface, &device, &adapter, &queue, layout.size.width as f32, layout.size.height as f32).await;

    let mut objects = vec![];    

    let model_data = model_load::load("../models/box/box.gltf");
    
    let object = create_model(&device, "1", model_data, 0.0, 0.0, 0.0);
    objects.push(object);    

    run2(event_loop, move |event, _, control_flow| {        
        *control_flow = ControlFlow::Wait;
        
        match event {
            Event::WindowEvent { event: window_event, window_id } => {
                match window_event {
                    WindowEvent::CloseRequested => {                        
                        info!("Event loop close requested");                        
                        *control_flow = ControlFlow::Exit;
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
                            }                            
                        }
                    }
                    WindowEvent::MouseWheel { device_id: _, delta, phase: _, modifiers: _ } => {
                        info!("{:?}", delta);
                    }
                    WindowEvent::ModifiersChanged(state) => {
                        info!("Modifiers changed");                        
                    }
                    WindowEvent::KeyboardInput { device_id: _, input, is_synthetic } => {
                        //info!("old shift is: {}", input.modifiers.shift());                        
                        match input.state {
                            ElementState::Pressed => {
                                match input.virtual_keycode {
                                    Some(virtual_keycode) => {
                                        //info!("{:?}", virtual_keycode);

                                        
                                        match virtual_keycode {
                                            winit::event::VirtualKeyCode::Up => {
                                                camera_controller.is_forward_pressed = true;
                                                window.request_redraw();
                                            }
                                            winit::event::VirtualKeyCode::Down => {
                                                camera_controller.is_backward_pressed = true;
                                                window.request_redraw();
                                            }
                                            winit::event::VirtualKeyCode::Left => {
                                                camera_controller.is_left_pressed = true;
                                                window.request_redraw();
                                            }
                                            winit::event::VirtualKeyCode::Right => {
                                                camera_controller.is_right_pressed = true;
                                                window.request_redraw();
                                            }
                                            _ => {}
                                        }                                                                                
                                    }
                                    None => {}
                                }
                            }
                            ElementState::Released => {
                                /*
                                match input.virtual_keycode {
                                    Some(virtual_keycode) => {
                                        //info!("{:?}", virtual_keycode);
                                        match virtual_keycode {
                                            winit::event::VirtualKeyCode::Up => {
                                                camera_controller.is_forward_pressed = false;
                                                window.request_redraw();
                                            }
                                            winit::event::VirtualKeyCode::Down => {
                                                camera_controller.is_backward_pressed = false;
                                                window.request_redraw();
                                            }
                                            winit::event::VirtualKeyCode::Left => {
                                                camera_controller.is_left_pressed = false;
                                                window.request_redraw();
                                            }
                                            winit::event::VirtualKeyCode::Right => {
                                                camera_controller.is_right_pressed = false;
                                                window.request_redraw();
                                            }
                                            _ => {}
                                        }
                                    }
                                    None => {}
                                }
                                */
                            }
                        }                        
                    }                    
                    _ => {}
                }
            }
            Event::UserEvent(_) => {                
            }           
            Event::RedrawRequested { .. } => {
                info!("Redraw requested");

                camera_controller.update_camera(&mut camera);
                camera_uniform.update_view_proj(&camera);
                queue.write_buffer(
                    &pipeline.camera_buffer,
                    0,
                    bytemuck::cast_slice(&[camera_uniform])
                );

                // Get a command encoder for the current frame
                let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("Redraw")
                    }
                );

                // Get the next frame
                let frame = surface.get_current_texture().expect("Get next frame");
                let view = &frame.texture.create_view(&wgpu::TextureViewDescriptor::default());                

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
                                        store: true
                                    }
                                })
                            ],
                            depth_stencil_attachment: None
                        }
                    );
                                        
                    render_pass.set_pipeline(&pipeline.render_pipeline);

                    for object in &objects {
                        render_pass.set_vertex_buffer(1, object.instance_buffer.slice(..)); // Instances
                    
                        let instances_range = 0..object.instances.len() as u32;
                        
                        for mesh in &object.meshes {                            
                            render_pass.set_bind_group(0, &pipeline.texture_bind_group, &[]); // Texture
                            render_pass.set_bind_group(1, &pipeline.camera_bind_group, &[]); // Camera
                            render_pass.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
                            render_pass.set_index_buffer(mesh.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
                            render_pass.draw_indexed(0..mesh.num_elements, 0, instances_range.clone());
                        }
                    }                    
                }                
                
                queue.submit(Some(encoder.finish()));
                frame.present();                
            }
            Event::RedrawEventsCleared => {                                
            }        
            _ => {}
        }
    })
}

pub fn run2<F>(event_loop: EventLoop<AppEvent>, event_handler: F) where F: 'static + FnMut(Event<'_, AppEvent>, &EventLoopWindowTarget<AppEvent>, &mut ControlFlow) {    
    #[cfg(target_arch = "wasm32")]
    event_loop.spawn(event_handler);
    #[cfg(not(target_arch = "wasm32"))]
    event_loop.run(event_handler);
}

fn main() {
    let event_loop = EventLoop::with_user_event();
    let window = Window::new(&event_loop).unwrap();

    #[cfg(not(target_arch = "wasm32"))]
    {        
        env_logger::init();

        let rt = Runtime::new().expect("Failed to create runtime");
        
        rt.block_on(run(event_loop, window));
    }
    #[cfg(target_arch = "wasm32")]
    {
        std::panic::set_hook(Box::new(console_error_panic_hook::hook));
        console_log::init_with_level(log::Level::Warn).expect("Could not initialize logger");
                
        let canvas = web_sys::Element::from(window.canvas());        
        let _ = canvas.set_attribute("width", "1500px");
        let _ = canvas.set_attribute("height", "900px");
        let _ = canvas.set_attribute("style", "width: 1500px;height: 900px;outline: none;");

        web_sys::window()
            .and_then(|win| win.document())
            .and_then(|doc| doc.body())
            .and_then(|body| {
                body.append_child(&canvas)
                    .ok()
            })
            .expect("Couldn't append canvas to document body");

        wasm_bindgen_futures::spawn_local(run(event_loop, window));        
    }
}
