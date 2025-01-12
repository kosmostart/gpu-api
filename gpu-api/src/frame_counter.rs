pub struct FrameCounter {
    // Instant of the last time we printed the frame time.
    #[cfg(not(target_arch = "wasm32"))]
    last_printed_instant: std::time::Instant,
    #[cfg(target_arch = "wasm32")]
    last_printed_instant: web_time::Instant,
    // Number of frames since the last time we printed the frame time.
    frame_count: u32
}

impl FrameCounter {
    pub fn new() -> Self {
        Self {
            #[cfg(not(target_arch = "wasm32"))]
            last_printed_instant: std::time::Instant::now(),
            #[cfg(target_arch = "wasm32")]
            last_printed_instant: web_time::Instant::now(),
            frame_count: 0
        }
    }

    pub fn update(&mut self) {
        self.frame_count = self.frame_count + 1;

        #[cfg(not(target_arch = "wasm32"))]
        let new_instant = std::time::Instant::now();
        #[cfg(target_arch = "wasm32")]
        let new_instant = web_time::Instant::now();

        let elapsed_secs = (new_instant - self.last_printed_instant).as_secs_f32();

        if elapsed_secs > 1.0 {
            let elapsed_ms = elapsed_secs * 1000.0;
            let frame_time = elapsed_ms / self.frame_count as f32;
            let fps = self.frame_count as f32 / elapsed_secs;

            log::warn!("Frame time {:.2} ms ({:.1} FPS)", frame_time, fps);

            self.last_printed_instant = new_instant;
            self.frame_count = 0;
        }
    }
}
