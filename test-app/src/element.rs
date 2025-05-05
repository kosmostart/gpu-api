use gpu_api::pipeline::element_pipeline::Vertex;
use crate::{Layout, Scene};

fn ndc_for_x(x_half: f32, x: f32) -> f32 {
    if x == x_half {
        return 0.0;
    }

    if x < x_half {
        return x / x_half - 1.0;
    }

    if x > x_half {
        return (x - x_half) / x_half;
    }
    
    panic!("Failed to calculate NDC for x");
}

fn ndc_for_y(y_half: f32, y: f32) -> f32 {
    if y == y_half {
        return 0.0;
    }

    if y < y_half {
        return 1.0 - y / y_half;
    }

    if y > y_half {
        return - (y - y_half) / y_half;
    }
    
    panic!("Failed to calculate NDC for y");
}

#[derive(Debug, Clone, Copy)]
pub struct Color {
    pub r: f32,
    pub g: f32,
    pub b: f32,
    pub a: f32
}

#[derive(Debug)]
pub struct ElementCfg {
    pub x: i32,
    pub y: i32,
    pub width: i32,
    pub height: i32,    
    pub background_color: Color,
    pub border_color: Option<Color>,
}

pub fn create_element(layout: &Layout, element_cfg: ElementCfg, scene: &mut Scene) {
    let width = element_cfg.width as f32;
    let height = element_cfg.height as f32;

    let x1 = element_cfg.x as f32;
    let y1 = element_cfg.y as f32;
    let x2 = x1 + width;
    let y2 = y1 + height;

    let x1 = x1 as f32;
    let y1 = y1 as f32;
    let x2 = x2 as f32;
    let y2 = y2 as f32;

    let x1_ndc = ndc_for_x(layout.halfes.x, x1);
    let y1_ndc = ndc_for_y(layout.halfes.y, y1);
    let x2_ndc = ndc_for_x(layout.halfes.x, x2);
    let y2_ndc = ndc_for_y(layout.halfes.y, y2);    

    let element_coordinates = [x1, y1, x2, y2];    
    
    let (has_element_border, element_border_color) = match &element_cfg.border_color {
        Some(color) => (1, [color.r, color.g, color.b, color.a]),
        None => (0, [1.0, 1.0, 1.0, 1.0])
    };    

    let vertex_type = 1;

    let component_coordinates = [0.0, 0.0, 800.0, 800.0];
    let has_overlay = 0;
    let overlay_coordinates = [0.0, 0.0, 0.0, 0.0];

    let element_vertices = vec![
        Vertex {
            vertex_type,
            position: [x1_ndc, y1_ndc, 0.0],
            color: [element_cfg.background_color.r, element_cfg.background_color.g, element_cfg.background_color.b, element_cfg.background_color.a],
            element_coordinates: element_coordinates.clone(),
            has_element_border,
            element_border_color,
            component_coordinates,
            texture_coordinates: [0.0, 0.0],
            has_overlay,
            overlay_coordinates
        },
        Vertex {
            vertex_type,
            position: [x1_ndc, y2_ndc, 0.0],
            color: [element_cfg.background_color.r, element_cfg.background_color.g, element_cfg.background_color.b, element_cfg.background_color.a],
            element_coordinates: element_coordinates.clone(),
            has_element_border,
            element_border_color,
            component_coordinates,
            texture_coordinates: [0.0, 1.0],
            has_overlay,
            overlay_coordinates
        },
        Vertex {
            vertex_type,
            position: [x2_ndc, y2_ndc, 0.0],
            color: [element_cfg.background_color.r, element_cfg.background_color.g, element_cfg.background_color.b, element_cfg.background_color.a],
            element_coordinates: element_coordinates.clone(),
            has_element_border,
            element_border_color,
            component_coordinates,
            texture_coordinates: [1.0, 1.0],
            has_overlay,
            overlay_coordinates
        },
        Vertex {
            vertex_type,
            position: [x2_ndc, y1_ndc, 0.0],
            color: [element_cfg.background_color.r, element_cfg.background_color.g, element_cfg.background_color.b, element_cfg.background_color.a],
            element_coordinates: element_coordinates.clone(),
            has_element_border,
            element_border_color,
            component_coordinates,
            texture_coordinates: [1.0, 0.0],
            has_overlay,
            overlay_coordinates
        }
    ];

    for vertex in &element_vertices {
        scene.vertices.push(*vertex);
    }    

    scene.indices.push(scene.element_index * 4);
    scene.indices.push(scene.element_index * 4 + 1);
    scene.indices.push(scene.element_index * 4 + 2);
    scene.indices.push(scene.element_index * 4);
    scene.indices.push(scene.element_index * 4 + 2);
    scene.indices.push(scene.element_index * 4 + 3);
}
