use log::info;

pub struct FrameCounter {
    pub frame_cycle_index: usize,
    pub frame_cycle_length: usize,
    pub time_per_frame: f32,    
    #[cfg(not(target_arch = "wasm32"))]
    pub last_frame_instant: std::time::Instant,
    #[cfg(target_arch = "wasm32")]
    pub last_frame_instant: web_time::Instant,
    /// Instant of the last time we printed the frame time.
    #[cfg(not(target_arch = "wasm32"))]
    pub last_printed_instant: std::time::Instant,
    #[cfg(target_arch = "wasm32")]
    pub last_printed_instant: web_time::Instant,
    /// Number of frames since the last time we printed the frame time.
    pub frame_count: u32
}

impl FrameCounter {
    pub fn new(frame_cycle_length: usize) -> Self {
        Self {
            frame_cycle_index: 0,
            frame_cycle_length,
            time_per_frame: 1.0 / frame_cycle_length as f32,
            #[cfg(not(target_arch = "wasm32"))]
            last_frame_instant: std::time::Instant::now(),
            #[cfg(target_arch = "wasm32")]
            last_frame_instant: web_time::Instant::now(),
            #[cfg(not(target_arch = "wasm32"))]
            last_printed_instant: std::time::Instant::now(),
            #[cfg(target_arch = "wasm32")]
            last_printed_instant: web_time::Instant::now(),
            frame_count: 0
        }
    }

    pub fn simple_update(&mut self) {
        // 1. Get the current time based on the architecture
        #[cfg(not(target_arch = "wasm32"))]
        let now = std::time::Instant::now();
        #[cfg(target_arch = "wasm32")]
        let now = web_time::Instant::now();

        // 2. Calculate time passed since the last frame
        let delta_time = now.duration_since(self.last_frame_instant).as_secs_f32();
        self.time_per_frame = delta_time;
        self.last_frame_instant = now;

        // 3. Increment total frames tracked in this interval
        self.frame_count += 1;

        // 4. Calculate time passed since the last print log
        let elapsed_since_print = now.duration_since(self.last_printed_instant).as_secs_f32();

        // 5. Update and print FPS every 1.0 second
        if elapsed_since_print >= 1.0 {
            let fps = self.frame_count as f32 / elapsed_since_print;
            
            // Print to console (works on both native stdout and WASM with wasm-bindgen)
            info!("FPS: {:.1} (Frame time: {:.2}ms)", fps, self.time_per_frame * 1000.0);

            // Reset intervals for the next measurement cycle
            self.frame_count = 0;
            self.last_printed_instant = now;
        }
    }

    pub fn update(&mut self) -> bool {        
        #[cfg(not(target_arch = "wasm32"))]
        let new_instant = std::time::Instant::now();
        #[cfg(target_arch = "wasm32")]
        let new_instant = web_time::Instant::now();

        let frame_time = (new_instant - self.last_frame_instant).as_secs_f32();

        let res = frame_time > self.time_per_frame;

        if res {
            self.frame_count = self.frame_count + 1;
            self.frame_cycle_index = self.frame_cycle_index + 1;

            if self.frame_cycle_index == self.frame_cycle_length {
                self.frame_cycle_index = 0;        
            }
            self.last_frame_instant = new_instant;
        }



        let elapsed_secs = (new_instant - self.last_printed_instant).as_secs_f32();
        
        if elapsed_secs > 1.0 {
            let elapsed_ms = elapsed_secs * 1000.0;
            let frame_time = elapsed_ms / self.frame_count as f32;
            let fps = self.frame_count as f32 / elapsed_secs;

            info!("FPS: {:.1} (Frame time: {:.2}ms)", fps, frame_time);

            self.last_printed_instant = new_instant;
            self.frame_count = 0;
        }
        
        res
    }
}
