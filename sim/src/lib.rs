use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub struct Simulation {
    width: u32,
    height: u32,
    frame: Vec<u8>,
    time: f32,
}

#[wasm_bindgen]
impl Simulation {
    #[wasm_bindgen(constructor)]
    pub fn new(width: u32, height: u32) -> Simulation {
        console_error_panic_hook::set_once();

        let mut simulation = Simulation {
            width,
            height,
            frame: vec![0; (width as usize) * (height as usize) * 4],
            time: 0.0,
        };

        simulation.draw_field();
        simulation
    }

    pub fn tick(&mut self, dt_seconds: f32) {
        self.time += dt_seconds.max(0.0);
        self.draw_field();
    }

    pub fn reset(&mut self) {
        self.time = 0.0;
        self.draw_field();
    }

    pub fn width(&self) -> u32 {
        self.width
    }

    pub fn height(&self) -> u32 {
        self.height
    }

    pub fn frame_ptr(&self) -> *const u8 {
        self.frame.as_ptr()
    }

    pub fn frame_len(&self) -> usize {
        self.frame.len()
    }
}

impl Simulation {
    fn draw_field(&mut self) {
        let width = self.width as usize;
        let height = self.height as usize;
        let inv_width = 1.0 / self.width.max(1) as f32;
        let inv_height = 1.0 / self.height.max(1) as f32;
        let t = self.time;

        for y in 0..height {
            let ny = y as f32 * inv_height;

            for x in 0..width {
                let nx = x as f32 * inv_width;
                let dx = nx - 0.5;
                let dy = ny - 0.5;
                let radius = (dx * dx + dy * dy).sqrt();
                let angle = dy.atan2(dx);

                let wave_a = (nx * 24.0 + t * 2.2).sin();
                let wave_b = (ny * 18.0 - t * 1.7).cos();
                let swirl = (angle * 3.0 + radius * 38.0 - t * 4.4).sin();
                let pulse = ((1.0 - radius).max(0.0) * 3.0).min(1.0);
                let field = (wave_a + wave_b + swirl) / 3.0;

                let red = to_byte(0.45 + 0.35 * field + 0.20 * pulse);
                let green = to_byte(0.50 + 0.30 * wave_b + 0.18 * swirl);
                let blue = to_byte(0.58 + 0.32 * swirl - 0.18 * field);
                let alpha = 255;

                let index = (y * width + x) * 4;
                self.frame[index] = red;
                self.frame[index + 1] = green;
                self.frame[index + 2] = blue;
                self.frame[index + 3] = alpha;
            }
        }
    }
}

fn to_byte(value: f32) -> u8 {
    (value.clamp(0.0, 1.0) * 255.0).round() as u8
}
