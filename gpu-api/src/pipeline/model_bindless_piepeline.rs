use gpu_api_relay::frame::{DrawIndexedIndirectCommand, FrameData, CullingTask};
use wgpu::RenderPass;

pub struct GpuWorldResources {
    // Uber-буферы геометрии
    pub mega_vertex_buffer: wgpu::Buffer,
    pub mega_index_buffer: wgpu::Buffer,

    // Перманентные Storage-буферы всех объектов мира (загружаются один раз)
    pub instances_buffer: wgpu::Buffer,
    pub nodes_buffer: wgpu::Buffer,
    pub joints_buffer: wgpu::Buffer,
    pub materials_buffer: wgpu::Buffer,

    // ДОБАВЛЕНО: Буфер задач куллинга (CPU -> GPU)
    pub culling_tasks_buffer: wgpu::Buffer,
    // ДОБАВЛЕНО: Буфер индексов видимых объектов, заполняемый Compute-шейдером (GPU -> GPU)
    pub visible_indices_buffer: wgpu::Buffer,

    // Буфер для Indirect команд отрисовки
    pub indirect_commands_buffer: wgpu::Buffer,

    // Pipelines
    pub culling_compute_pipeline: wgpu::ComputePipeline, // ДОБАВЛЕНО для куллинга
    pub gpu_driven_render_pipeline: wgpu::RenderPipeline,

    // Bind Groups
    pub materials_bind_group: wgpu::BindGroup,
    pub camera_bind_group: wgpu::BindGroup,
    pub culling_compute_bind_group: wgpu::BindGroup, // ДОБАВЛЕНО для Compute Pass
    pub gpu_driven_bind_group: wgpu::BindGroup,      // Должен включать visible_indices_buffer на binding: 3
}

pub fn load(
    resources: &GpuWorldResources,
    queue: &wgpu::Queue,
    culling_tasks: &[CullingTask], // Принимаем легкие задачи из октодерева) {
    initial_indirect_commands: &[DrawIndexedIndirectCommand] // Заготовки команд с instance_count = 0
) {
    // ============================================================================
    // ЭТАП 1: ОБНОВЛЕНИЕ БУФЕРОВ (CPU -> GPU)
    // ============================================================================
    // 1. Загружаем задачи куллинга от октодерева чанков
    if !culling_tasks.is_empty() {
        queue.write_buffer(&resources.culling_tasks_buffer, 0, bytemuck::cast_slice(culling_tasks));
    }
    
    // 2. Сбрасываем indirect-команды (записываем заготовки, где instance_count = 0)
    // GPU Compute-шейдер сам увеличит их атомарно для каждого видимого объекта!
    queue.write_buffer(&resources.indirect_commands_buffer, 0, bytemuck::cast_slice(initial_indirect_commands));
}

pub fn draw_gpu_driven_frame(
    resources: &GpuWorldResources,
    encoder: &mut wgpu::CommandEncoder,
    view: &wgpu::TextureView,
    depth_view: &wgpu::TextureView, // Рекомендуется добавить для Depth-теста
    culling_tasks: &[CullingTask], // Принимаем легкие задачи из октодерева
    initial_indirect_commands: &[DrawIndexedIndirectCommand], // Заготовки команд с instance_count = 0
) {    

    // ============================================================================
    // ЭТАП 2: COMPUTE PASS (Пообъектный куллинг на GPU)
    // ============================================================================
    {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("GPU-Driven Culling Pass"),
            timestamp_writes: None,
        });
        cpass.set_pipeline(&resources.culling_compute_pipeline);
        cpass.set_bind_group(0, &resources.culling_compute_bind_group, &[]);
        
        // Запускаем по одному потоку на каждую задачу (каждый видимый чанк)
        // Внутри шейдера dispatch_thread_id будет соответствовать индексу задачи.
        cpass.dispatch_workgroups(culling_tasks.len() as u32, 1, 1);
    } // cpass автоматически завершается здесь (drop)

    // ВАЖНО: Если WebGPU требует барьеров памяти, wgpu управляет ими автоматически 
    // между Compute и Render пассами, так как буферы используются в разных конвейерах.

    // ============================================================================
    // ЭТАП 3: RENDER PASS (Отрисовка Uber-буфера без CPU циклов)
    // ============================================================================
    {
        let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("GPU-Driven Dynamic Objects Pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK), // Или Load, если земля уже там
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: depth_view,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Clear(1.0),
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });

        render_pass.set_pipeline(&resources.gpu_driven_render_pipeline);

        // Привязываем три наши глобальные AAA-группы ресурсов
        render_pass.set_bind_group(0, &resources.materials_bind_group, &[]); // Массив текстур
        render_pass.set_bind_group(1, &resources.camera_bind_group, &[]);    // Общая камера
        // Включает: global_instances, global_nodes, global_joints И visible_indices_buffer (binding: 3)
        render_pass.set_bind_group(2, &resources.gpu_driven_bind_group, &[]); 

        // Привязываем единственный глобальный Uber-буфер геометрии
        render_pass.set_vertex_buffer(0, resources.mega_vertex_buffer.slice(..));
        render_pass.set_index_buffer(resources.mega_index_buffer.slice(..), wgpu::IndexFormat::Uint32);
       
        // Запускаем непрямую отрисовку для каждой зарегистрированной модели
        for i in 0..initial_indirect_commands.len() {
            let offset = (i * std::mem::size_of::<DrawIndexedIndirectCommand>()) as wgpu::BufferAddress;
            // Видеокарта берет instance_count, который только что посчитал Compute-шейдер!
            render_pass.draw_indexed_indirect(&resources.indirect_commands_buffer, offset);
        }
    }
}
