use wasm_bindgen::prelude::*;

const DEFAULT_WORLD_SIZE: f32 = 4096.0;
const DEFAULT_POPULATION: usize = 10_000;
const MAX_POPULATION: usize = 100_000;

const FIXED_STEP_SECONDS: f32 = 1.0 / 60.0;
const MAX_STEPS_PER_TICK: u32 = 8;

const RENDER_STRIDE_FLOATS: usize = 8;
const INPUT_COUNT: usize = 37;
const RAY_COUNT: usize = 7;
const HIDDEN_COUNT: usize = 8;
const OUTPUT_COUNT: usize = 5;
const GENOME_LEN: usize = HIDDEN_COUNT * (INPUT_COUNT + 1) + OUTPUT_COUNT * (HIDDEN_COUNT + 1);

const MIN_SPEED: f32 = 15.0;
const MAX_SPEED: f32 = 80.0;
const ACCELERATION: f32 = 70.0;
const MAX_TURN_RATE: f32 = 4.0;
const COLOR_BLEND_RATE: f32 = 4.0;

const GRID_CELL_SIZE: f32 = 32.0;
const VISION_RANGE: f32 = 96.0;
const RAY_SAMPLE_STEP: f32 = GRID_CELL_SIZE * 0.5;
const RAY_HALF_WIDTH: f32 = 7.0;
const RAY_FORWARD_SLOP: f32 = 0.055;

const RAY_COS: [f32; RAY_COUNT] = [
    0.5,
    0.766_044_44,
    0.939_692_6,
    1.0,
    0.939_692_6,
    0.766_044_44,
    0.5,
];
const RAY_SIN: [f32; RAY_COUNT] = [
    -0.866_025_4,
    -0.642_787_64,
    -0.342_020_15,
    0.0,
    0.342_020_15,
    0.642_787_64,
    0.866_025_4,
];

const AGENT_LENGTH: f32 = 10.0;
const HEAD_OFFSET: f32 = AGENT_LENGTH * 0.55;
const BODY_BACK: f32 = -HEAD_OFFSET * 0.9;
const BODY_FRONT: f32 = HEAD_OFFSET * 0.3;
const BODY_HALF_WIDTH: f32 = 2.8;
const HEAD_ON_RADIUS: f32 = 4.5;
const HEAD_ON_RADIUS_SQUARED: f32 = HEAD_ON_RADIUS * HEAD_ON_RADIUS;
const HEAD_ON_DOT: f32 = -0.75;

const MATE_RADIUS: f32 = 10.0;
const MATE_RADIUS_SQUARED: f32 = MATE_RADIUS * MATE_RADIUS;
const MATE_ALIGNMENT_DOT: f32 = 0.95;
const MIN_MATE_AGE_SECONDS: f32 = 4.0;
const MATE_DURATION_SECONDS: f32 = 1.5;

const AGE_INPUT_CAP_SECONDS: f32 = 30.0;
const MUTATION_RATE: f32 = 0.03;
const MUTATION_MAGNITUDE: f32 = 0.25;
const GENE_LIMIT: f32 = 4.0;

#[wasm_bindgen]
pub struct Simulation {
    world_size: f32,
    population: usize,
    accumulator: f32,
    steps: u32,
    births: u32,
    deaths: u32,
    rng: SmallRng,

    pos_x: Vec<f32>,
    pos_y: Vec<f32>,
    dir_x: Vec<f32>,
    dir_y: Vec<f32>,
    speed: Vec<f32>,
    color_r: Vec<f32>,
    color_g: Vec<f32>,
    color_b: Vec<f32>,
    age: Vec<f32>,
    generation: Vec<u32>,

    genomes: Vec<f32>,
    child_genome: Vec<f32>,

    turn_command: Vec<f32>,
    accel_command: Vec<f32>,
    target_r: Vec<f32>,
    target_g: Vec<f32>,
    target_b: Vec<f32>,

    mate_partner: Vec<i32>,
    mate_timer: Vec<f32>,
    dead: Vec<bool>,
    bred: Vec<bool>,

    grid_cols: usize,
    grid_heads: Vec<i32>,
    grid_next: Vec<i32>,

    render_agents: Vec<f32>,
}

#[wasm_bindgen]
impl Simulation {
    #[wasm_bindgen(constructor)]
    pub fn new(world_size: f32, population: u32, seed: u32) -> Simulation {
        console_error_panic_hook::set_once();

        let world_size = sanitize_world_size(world_size);
        let population = sanitize_population(population);
        let grid_cols = ((world_size / GRID_CELL_SIZE).ceil() as usize).max(1);
        let grid_len = grid_cols * grid_cols;

        let mut simulation = Simulation {
            world_size,
            population,
            accumulator: 0.0,
            steps: 0,
            births: 0,
            deaths: 0,
            rng: SmallRng::new(seed as u64),

            pos_x: vec![0.0; population],
            pos_y: vec![0.0; population],
            dir_x: vec![1.0; population],
            dir_y: vec![0.0; population],
            speed: vec![MIN_SPEED; population],
            color_r: vec![1.0; population],
            color_g: vec![1.0; population],
            color_b: vec![1.0; population],
            age: vec![0.0; population],
            generation: vec![0; population],

            genomes: vec![0.0; population * GENOME_LEN],
            child_genome: vec![0.0; GENOME_LEN],

            turn_command: vec![0.0; population],
            accel_command: vec![0.0; population],
            target_r: vec![0.0; population],
            target_g: vec![0.0; population],
            target_b: vec![0.0; population],

            mate_partner: vec![-1; population],
            mate_timer: vec![0.0; population],
            dead: vec![false; population],
            bred: vec![false; population],

            grid_cols,
            grid_heads: vec![-1; grid_len],
            grid_next: vec![-1; population],

            render_agents: vec![0.0; population * RENDER_STRIDE_FLOATS],
        };

        simulation.randomize_all_agents();
        simulation.refresh_render_agents();
        simulation
    }

    pub fn tick(&mut self, dt_seconds: f32) {
        if !dt_seconds.is_finite() || dt_seconds <= 0.0 {
            return;
        }

        self.accumulator += dt_seconds.min(0.25);

        let mut steps_this_tick = 0;
        while self.accumulator >= FIXED_STEP_SECONDS && steps_this_tick < MAX_STEPS_PER_TICK {
            self.step(FIXED_STEP_SECONDS);
            self.accumulator -= FIXED_STEP_SECONDS;
            steps_this_tick += 1;
        }

        // Keep the sub-step remainder so the simulation rate stays wall-clock consistent. Only
        // when the catch-up cap is hit do we drop the excess backlog to avoid a spiral of death.
        if steps_this_tick == MAX_STEPS_PER_TICK {
            self.accumulator = self.accumulator.min(FIXED_STEP_SECONDS);
        }

        self.refresh_render_agents();
    }

    pub fn advance_steps(&mut self, step_count: u32) {
        for _ in 0..step_count {
            self.step(FIXED_STEP_SECONDS);
        }

        if step_count > 0 {
            self.refresh_render_agents();
        }
    }

    pub fn reset(&mut self, seed: u32) {
        self.accumulator = 0.0;
        self.steps = 0;
        self.births = 0;
        self.deaths = 0;
        self.rng = SmallRng::new(seed as u64);
        self.randomize_all_agents();
        self.refresh_render_agents();
    }

    pub fn world_size(&self) -> f32 {
        self.world_size
    }

    pub fn fixed_step_seconds(&self) -> f32 {
        FIXED_STEP_SECONDS
    }

    pub fn population(&self) -> u32 {
        self.population as u32
    }

    pub fn births(&self) -> u32 {
        self.births
    }

    pub fn deaths(&self) -> u32 {
        self.deaths
    }

    pub fn sim_steps(&self) -> u32 {
        self.steps
    }

    pub fn generation(&self) -> u32 {
        self.generation.iter().copied().max().unwrap_or(0)
    }

    pub fn agent_ptr(&self) -> *const f32 {
        self.render_agents.as_ptr()
    }

    pub fn agent_f32_len(&self) -> usize {
        self.render_agents.len()
    }

    pub fn agent_stride_f32(&self) -> usize {
        RENDER_STRIDE_FLOATS
    }
}

impl Simulation {
    fn step(&mut self, dt: f32) {
        self.rebuild_grid();
        self.update_decisions();
        self.apply_decisions(dt);
        self.rebuild_grid();
        self.resolve_collisions();
        self.rebuild_grid();
        self.update_breeding(dt);
        self.steps = self.steps.wrapping_add(1);
    }

    fn randomize_all_agents(&mut self) {
        for index in 0..self.population {
            self.reset_random_agent(index);
        }
    }

    fn reset_random_agent(&mut self, index: usize) {
        self.randomize_genome(index);
        self.reset_random_pose(index, 0);
    }

    fn reset_random_pose(&mut self, index: usize, generation: u32) {
        self.pos_x[index] = self.rng.next_f32() * self.world_size;
        self.pos_y[index] = self.rng.next_f32() * self.world_size;

        let angle = self.rng.next_f32() * std::f32::consts::TAU;
        self.dir_x[index] = angle.cos();
        self.dir_y[index] = angle.sin();
        self.speed[index] = MIN_SPEED + self.rng.next_f32() * (MAX_SPEED - MIN_SPEED);

        self.color_r[index] = 0.25 + self.rng.next_f32() * 0.75;
        self.color_g[index] = 0.25 + self.rng.next_f32() * 0.75;
        self.color_b[index] = 0.25 + self.rng.next_f32() * 0.75;
        self.age[index] = 0.0;
        self.generation[index] = generation;
        self.mate_partner[index] = -1;
        self.mate_timer[index] = 0.0;
        self.turn_command[index] = 0.0;
        self.accel_command[index] = 0.0;
        self.target_r[index] = self.color_r[index];
        self.target_g[index] = self.color_g[index];
        self.target_b[index] = self.color_b[index];
    }

    fn reset_child_pose(
        &mut self,
        victim: usize,
        parent_a: usize,
        parent_b: usize,
        generation: u32,
    ) {
        let jitter_x = self.rng.next_signed_f32() * 1.5;
        let jitter_y = self.rng.next_signed_f32() * 1.5;
        self.pos_x[victim] = wrap(self.pos_x[victim] + jitter_x, self.world_size);
        self.pos_y[victim] = wrap(self.pos_y[victim] + jitter_y, self.world_size);

        let mut dx = self.dir_x[parent_a] + self.dir_x[parent_b];
        let mut dy = self.dir_y[parent_a] + self.dir_y[parent_b];
        let len = (dx * dx + dy * dy).sqrt();
        if len > 0.000_1 {
            dx /= len;
            dy /= len;
        } else {
            let angle = self.rng.next_f32() * std::f32::consts::TAU;
            dx = angle.cos();
            dy = angle.sin();
        }

        self.dir_x[victim] = dx;
        self.dir_y[victim] = dy;
        self.speed[victim] =
            ((self.speed[parent_a] + self.speed[parent_b]) * 0.5).clamp(MIN_SPEED, MAX_SPEED);
        self.color_r[victim] =
            ((self.color_r[parent_a] + self.color_r[parent_b]) * 0.5).clamp(0.0, 1.0);
        self.color_g[victim] =
            ((self.color_g[parent_a] + self.color_g[parent_b]) * 0.5).clamp(0.0, 1.0);
        self.color_b[victim] =
            ((self.color_b[parent_a] + self.color_b[parent_b]) * 0.5).clamp(0.0, 1.0);
        self.age[victim] = 0.0;
        self.generation[victim] = generation;
        self.mate_partner[victim] = -1;
        self.mate_timer[victim] = 0.0;
        self.turn_command[victim] = 0.0;
        self.accel_command[victim] = 0.0;
        self.target_r[victim] = self.color_r[victim];
        self.target_g[victim] = self.color_g[victim];
        self.target_b[victim] = self.color_b[victim];
    }

    fn randomize_genome(&mut self, index: usize) {
        let start = index * GENOME_LEN;
        let end = start + GENOME_LEN;

        for gene in &mut self.genomes[start..end] {
            *gene = self.rng.next_signed_f32() * 0.75;
        }
    }

    fn update_decisions(&mut self) {
        let mut inputs = [0.0; INPUT_COUNT];
        let mut hidden = [0.0; HIDDEN_COUNT];
        let mut outputs = [0.0; OUTPUT_COUNT];

        for index in 0..self.population {
            self.write_vision_inputs(index, &mut inputs);
            self.evaluate_network(index, &inputs, &mut hidden, &mut outputs);

            self.turn_command[index] = outputs[0];
            self.accel_command[index] = outputs[1];
            self.target_r[index] = output_to_color(outputs[2]);
            self.target_g[index] = output_to_color(outputs[3]);
            self.target_b[index] = output_to_color(outputs[4]);
        }
    }

    fn apply_decisions(&mut self, dt: f32) {
        let color_blend = (COLOR_BLEND_RATE * dt).min(1.0);

        for index in 0..self.population {
            let turn = self.turn_command[index].clamp(-1.0, 1.0) * MAX_TURN_RATE * dt;
            let cos_turn = turn.cos();
            let sin_turn = turn.sin();
            let old_dx = self.dir_x[index];
            let old_dy = self.dir_y[index];
            let new_dx = old_dx * cos_turn - old_dy * sin_turn;
            let new_dy = old_dx * sin_turn + old_dy * cos_turn;
            let dir_len = (new_dx * new_dx + new_dy * new_dy).sqrt().max(0.000_1);

            self.dir_x[index] = new_dx / dir_len;
            self.dir_y[index] = new_dy / dir_len;
            self.speed[index] = (self.speed[index]
                + self.accel_command[index].clamp(-1.0, 1.0) * ACCELERATION * dt)
                .clamp(MIN_SPEED, MAX_SPEED);

            self.pos_x[index] = wrap(
                self.pos_x[index] + self.dir_x[index] * self.speed[index] * dt,
                self.world_size,
            );
            self.pos_y[index] = wrap(
                self.pos_y[index] + self.dir_y[index] * self.speed[index] * dt,
                self.world_size,
            );

            self.color_r[index] += (self.target_r[index] - self.color_r[index]) * color_blend;
            self.color_g[index] += (self.target_g[index] - self.color_g[index]) * color_blend;
            self.color_b[index] += (self.target_b[index] - self.color_b[index]) * color_blend;
            self.age[index] += dt;
        }
    }

    fn write_vision_inputs(&self, index: usize, inputs: &mut [f32; INPUT_COUNT]) {
        inputs.fill(0.0);

        let origin_x = self.pos_x[index];
        let origin_y = self.pos_y[index];
        let dir_x = self.dir_x[index];
        let dir_y = self.dir_y[index];
        let mut ray_dir_x = [0.0; RAY_COUNT];
        let mut ray_dir_y = [0.0; RAY_COUNT];
        let mut ray_distance = [VISION_RANGE; RAY_COUNT];
        let mut ray_color = [[0.0; 3]; RAY_COUNT];

        for ray in 0..RAY_COUNT {
            ray_dir_x[ray] = dir_x * RAY_COS[ray] - dir_y * RAY_SIN[ray];
            ray_dir_y[ray] = dir_x * RAY_SIN[ray] + dir_y * RAY_COS[ray];
        }

        for ray in 0..RAY_COUNT {
            let mut sample_distance = RAY_SAMPLE_STEP;
            let mut previous_cell = usize::MAX;

            while sample_distance <= VISION_RANGE {
                let sample_x = wrap(origin_x + ray_dir_x[ray] * sample_distance, self.world_size);
                let sample_y = wrap(origin_y + ray_dir_y[ray] * sample_distance, self.world_size);
                let cell = self.cell_index(sample_x, sample_y);

                if cell == previous_cell {
                    sample_distance += RAY_SAMPLE_STEP;
                    continue;
                }

                previous_cell = cell;
                let mut cursor = self.grid_heads[cell];

                while cursor >= 0 {
                    let other = cursor as usize;
                    cursor = self.grid_next[other];

                    if other == index {
                        continue;
                    }

                    let dx = wrap_delta(self.pos_x[other] - origin_x, self.world_size);
                    let dy = wrap_delta(self.pos_y[other] - origin_y, self.world_size);
                    let forward = dx * ray_dir_x[ray] + dy * ray_dir_y[ray];
                    if forward <= 0.0 || forward > VISION_RANGE || forward >= ray_distance[ray] {
                        continue;
                    }

                    let lateral = (dx * -ray_dir_y[ray] + dy * ray_dir_x[ray]).abs();
                    let width = RAY_HALF_WIDTH + forward * RAY_FORWARD_SLOP;
                    if lateral <= width {
                        ray_distance[ray] = forward;
                        ray_color[ray] = [
                            self.color_r[other],
                            self.color_g[other],
                            self.color_b[other],
                        ];
                    }
                }

                if ray_distance[ray] < VISION_RANGE {
                    break;
                }

                sample_distance += RAY_SAMPLE_STEP;
            }
        }

        for ray in 0..RAY_COUNT {
            let base = ray * 5;
            if ray_distance[ray] < VISION_RANGE {
                inputs[base] = 1.0;
                inputs[base + 1] = 1.0 - ray_distance[ray] / VISION_RANGE;
                inputs[base + 2] = ray_color[ray][0];
                inputs[base + 3] = ray_color[ray][1];
                inputs[base + 4] = ray_color[ray][2];
            }
        }

        let self_base = RAY_COUNT * 5;
        inputs[self_base] =
            ((self.speed[index] - MIN_SPEED) / (MAX_SPEED - MIN_SPEED)).clamp(0.0, 1.0);
        inputs[self_base + 1] = (self.age[index] / AGE_INPUT_CAP_SECONDS).clamp(0.0, 1.0);
    }

    fn evaluate_network(
        &self,
        index: usize,
        inputs: &[f32; INPUT_COUNT],
        hidden: &mut [f32; HIDDEN_COUNT],
        outputs: &mut [f32; OUTPUT_COUNT],
    ) {
        let mut cursor = index * GENOME_LEN;

        for hidden_value in hidden.iter_mut() {
            let mut sum = 0.0;
            for input in inputs {
                sum += *input * self.genomes[cursor];
                cursor += 1;
            }
            sum += self.genomes[cursor];
            cursor += 1;
            *hidden_value = sum.tanh();
        }

        for output in outputs.iter_mut() {
            let mut sum = 0.0;
            for hidden_value in hidden.iter() {
                sum += *hidden_value * self.genomes[cursor];
                cursor += 1;
            }
            sum += self.genomes[cursor];
            cursor += 1;
            *output = sum.tanh();
        }
    }

    fn resolve_collisions(&mut self) {
        self.dead.fill(false);

        for index in 0..self.population {
            let head_x = wrap(
                self.pos_x[index] + self.dir_x[index] * HEAD_OFFSET,
                self.world_size,
            );
            let head_y = wrap(
                self.pos_y[index] + self.dir_y[index] * HEAD_OFFSET,
                self.world_size,
            );
            let center_cell_x = self.cell_coord(head_x);
            let center_cell_y = self.cell_coord(head_y);

            'search: for cell_y in center_cell_y - 1..=center_cell_y + 1 {
                for cell_x in center_cell_x - 1..=center_cell_x + 1 {
                    let cell = self.wrapped_cell_index(cell_x, cell_y);
                    let mut cursor = self.grid_heads[cell];

                    while cursor >= 0 {
                        let other = cursor as usize;
                        cursor = self.grid_next[other];

                        if other == index {
                            continue;
                        }

                        if self.hits_body_side(head_x, head_y, other)
                            && !self.is_head_on(index, other, head_x, head_y)
                        {
                            self.dead[index] = true;
                            break 'search;
                        }
                    }
                }
            }
        }

        let mut deaths_this_step = 0;
        for index in 0..self.population {
            if self.dead[index] {
                self.reset_random_agent(index);
                deaths_this_step += 1;
            }
        }

        if deaths_this_step > 0 {
            self.deaths = self.deaths.wrapping_add(deaths_this_step);
        }
    }

    fn hits_body_side(&self, head_x: f32, head_y: f32, target: usize) -> bool {
        let dx = wrap_delta(head_x - self.pos_x[target], self.world_size);
        let dy = wrap_delta(head_y - self.pos_y[target], self.world_size);
        let forward = dx * self.dir_x[target] + dy * self.dir_y[target];
        let lateral = (dx * -self.dir_y[target] + dy * self.dir_x[target]).abs();

        (BODY_BACK..=BODY_FRONT).contains(&forward) && lateral <= BODY_HALF_WIDTH
    }

    fn is_head_on(
        &self,
        attacker: usize,
        target: usize,
        attacker_head_x: f32,
        attacker_head_y: f32,
    ) -> bool {
        if self.dir_x[attacker] * self.dir_x[target] + self.dir_y[attacker] * self.dir_y[target]
            > HEAD_ON_DOT
        {
            return false;
        }

        let target_head_x = wrap(
            self.pos_x[target] + self.dir_x[target] * HEAD_OFFSET,
            self.world_size,
        );
        let target_head_y = wrap(
            self.pos_y[target] + self.dir_y[target] * HEAD_OFFSET,
            self.world_size,
        );
        let dx = wrap_delta(target_head_x - attacker_head_x, self.world_size);
        let dy = wrap_delta(target_head_y - attacker_head_y, self.world_size);

        dx * dx + dy * dy <= HEAD_ON_RADIUS_SQUARED
    }

    fn update_breeding(&mut self, dt: f32) {
        for index in 0..self.population {
            let current = self.mate_partner[index];
            if current >= 0 && self.is_mate_eligible(index, current as usize).is_some() {
                // Sticky courtship: stay committed to the current partner while it remains
                // eligible, even if a closer candidate appears, so the timer can mature.
                self.mate_timer[index] += dt;
            } else if let Some(partner) = self.find_mate_partner(index) {
                self.mate_partner[index] = partner as i32;
                self.mate_timer[index] = dt;
            } else {
                self.mate_partner[index] = -1;
                self.mate_timer[index] = 0.0;
            }
        }

        self.resolve_breeding();
    }

    fn resolve_breeding(&mut self) {
        self.bred.fill(false);

        for index in 0..self.population {
            let partner = self.mate_partner[index];
            if partner < 0 {
                continue;
            }

            let partner = partner as usize;
            if index >= partner
                || self.bred[index]
                || self.bred[partner]
                || self.mate_partner[partner] != index as i32
                || self.mate_timer[index] < MATE_DURATION_SECONDS
                || self.mate_timer[partner] < MATE_DURATION_SECONDS
            {
                continue;
            }

            self.spawn_child_from_pair(index, partner);
            self.bred[index] = true;
            self.bred[partner] = true;
            self.mate_partner[index] = -1;
            self.mate_partner[partner] = -1;
            self.mate_timer[index] = 0.0;
            self.mate_timer[partner] = 0.0;
        }
    }

    fn find_mate_partner(&self, index: usize) -> Option<usize> {
        if self.age[index] < MIN_MATE_AGE_SECONDS {
            return None;
        }

        let center_cell_x = self.cell_coord(self.pos_x[index]);
        let center_cell_y = self.cell_coord(self.pos_y[index]);
        let mut best_partner = None;
        let mut best_distance_squared = MATE_RADIUS_SQUARED;

        for cell_y in center_cell_y - 1..=center_cell_y + 1 {
            for cell_x in center_cell_x - 1..=center_cell_x + 1 {
                let cell = self.wrapped_cell_index(cell_x, cell_y);
                let mut cursor = self.grid_heads[cell];

                while cursor >= 0 {
                    let other = cursor as usize;
                    cursor = self.grid_next[other];

                    if let Some(distance_squared) = self.is_mate_eligible(index, other) {
                        if distance_squared <= best_distance_squared {
                            best_distance_squared = distance_squared;
                            best_partner = Some(other);
                        }
                    }
                }
            }
        }

        best_partner
    }

    /// Returns the wrapped squared distance to `b` when `a` may currently mate with it:
    /// distinct agents, both old enough, nearly aligned, and within the mate radius.
    fn is_mate_eligible(&self, a: usize, b: usize) -> Option<f32> {
        if b == a || self.age[a] < MIN_MATE_AGE_SECONDS || self.age[b] < MIN_MATE_AGE_SECONDS {
            return None;
        }

        let alignment = self.dir_x[a] * self.dir_x[b] + self.dir_y[a] * self.dir_y[b];
        if alignment < MATE_ALIGNMENT_DOT {
            return None;
        }

        let dx = wrap_delta(self.pos_x[b] - self.pos_x[a], self.world_size);
        let dy = wrap_delta(self.pos_y[b] - self.pos_y[a], self.world_size);
        let distance_squared = dx * dx + dy * dy;
        if distance_squared > MATE_RADIUS_SQUARED {
            return None;
        }

        Some(distance_squared)
    }

    fn spawn_child_from_pair(&mut self, parent_a: usize, parent_b: usize) {
        let generation = self.generation[parent_a]
            .max(self.generation[parent_b])
            .wrapping_add(1);

        for gene_index in 0..GENOME_LEN {
            let source = if self.rng.next_bool() {
                parent_a
            } else {
                parent_b
            };
            self.child_genome[gene_index] = self.genomes[source * GENOME_LEN + gene_index];
        }

        mutate_genome_with_rate(&mut self.child_genome, &mut self.rng, MUTATION_RATE);

        let victim = if self.rng.next_bool() {
            parent_a
        } else {
            parent_b
        };
        let destination = victim * GENOME_LEN;
        self.genomes[destination..destination + GENOME_LEN].copy_from_slice(&self.child_genome);
        self.reset_child_pose(victim, parent_a, parent_b, generation);

        self.births = self.births.wrapping_add(1);
    }

    fn rebuild_grid(&mut self) {
        self.grid_heads.fill(-1);

        for index in 0..self.population {
            let cell = self.cell_index(self.pos_x[index], self.pos_y[index]);
            self.grid_next[index] = self.grid_heads[cell];
            self.grid_heads[cell] = index as i32;
        }
    }

    fn refresh_render_agents(&mut self) {
        let inv_world = 1.0 / self.world_size;
        let inv_speed_range = 1.0 / (MAX_SPEED - MIN_SPEED);

        for index in 0..self.population {
            let base = index * RENDER_STRIDE_FLOATS;
            self.render_agents[base] = self.pos_x[index] * inv_world;
            self.render_agents[base + 1] = self.pos_y[index] * inv_world;
            self.render_agents[base + 2] = self.dir_x[index];
            self.render_agents[base + 3] = self.dir_y[index];
            self.render_agents[base + 4] = self.color_r[index].clamp(0.0, 1.0);
            self.render_agents[base + 5] = self.color_g[index].clamp(0.0, 1.0);
            self.render_agents[base + 6] = self.color_b[index].clamp(0.0, 1.0);
            self.render_agents[base + 7] =
                ((self.speed[index] - MIN_SPEED) * inv_speed_range).clamp(0.0, 1.0);
        }
    }

    fn cell_index(&self, x: f32, y: f32) -> usize {
        self.wrapped_cell_index(self.cell_coord(x), self.cell_coord(y))
    }

    fn cell_coord(&self, value: f32) -> isize {
        (value / GRID_CELL_SIZE).floor() as isize
    }

    fn wrapped_cell_index(&self, cell_x: isize, cell_y: isize) -> usize {
        let cols = self.grid_cols as isize;
        let x = cell_x.rem_euclid(cols) as usize;
        let y = cell_y.rem_euclid(cols) as usize;
        y * self.grid_cols + x
    }
}

#[derive(Clone)]
struct SmallRng {
    state: u64,
}

impl SmallRng {
    fn new(seed: u64) -> SmallRng {
        SmallRng {
            state: seed ^ 0x9e37_79b9_7f4a_7c15,
        }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut value = self.state;
        value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        value ^ (value >> 31)
    }

    fn next_f32(&mut self) -> f32 {
        let bits = (self.next_u64() >> 40) as u32;
        bits as f32 * (1.0 / 16_777_216.0)
    }

    fn next_signed_f32(&mut self) -> f32 {
        self.next_f32() * 2.0 - 1.0
    }

    fn next_bool(&mut self) -> bool {
        self.next_u64() & 1 == 1
    }
}

fn sanitize_world_size(world_size: f32) -> f32 {
    if world_size.is_finite() && world_size >= 128.0 {
        world_size
    } else {
        DEFAULT_WORLD_SIZE
    }
}

fn sanitize_population(population: u32) -> usize {
    if population == 0 {
        DEFAULT_POPULATION
    } else {
        (population as usize).clamp(1, MAX_POPULATION)
    }
}

fn output_to_color(output: f32) -> f32 {
    (output * 0.5 + 0.5).clamp(0.0, 1.0)
}

fn mutate_genome_with_rate(genome: &mut [f32], rng: &mut SmallRng, mutation_rate: f32) {
    for gene in genome {
        if rng.next_f32() < mutation_rate {
            *gene =
                (*gene + rng.next_signed_f32() * MUTATION_MAGNITUDE).clamp(-GENE_LIMIT, GENE_LIMIT);
        }
    }
}

fn wrap(value: f32, world_size: f32) -> f32 {
    value.rem_euclid(world_size)
}

fn wrap_delta(delta: f32, world_size: f32) -> f32 {
    let half = world_size * 0.5;
    if delta > half {
        delta - world_size
    } else if delta < -half {
        delta + world_size
    } else {
        delta
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wraps_positions_and_deltas_across_toroid() {
        assert_eq!(wrap(1030.0, 1024.0), 6.0);
        assert_eq!(wrap(-2.0, 1024.0), 1022.0);
        assert_eq!(wrap_delta(900.0, 1024.0), -124.0);
        assert_eq!(wrap_delta(-900.0, 1024.0), 124.0);
    }

    #[test]
    fn rng_is_deterministic_for_same_seed() {
        let mut a = SmallRng::new(42);
        let mut b = SmallRng::new(42);
        let mut c = SmallRng::new(43);

        for _ in 0..16 {
            assert_eq!(a.next_u64(), b.next_u64());
        }
        assert_ne!(SmallRng::new(42).next_u64(), c.next_u64());
    }

    #[test]
    fn genome_len_matches_network_contract() {
        assert_eq!(INPUT_COUNT, 37);
        assert_eq!(GENOME_LEN, 349);
    }

    #[test]
    fn mutation_clamps_gene_values() {
        let mut rng = SmallRng::new(7);
        let mut genome = vec![3.95; GENOME_LEN];
        mutate_genome_with_rate(&mut genome, &mut rng, 1.0);

        assert!(genome
            .iter()
            .all(|gene| (-GENE_LIMIT..=GENE_LIMIT).contains(gene)));
    }

    #[test]
    fn breeding_replaces_one_parent_with_child() {
        let mut sim = Simulation::new(128.0, 2, 99);
        sim.pos_x[0] = 50.0;
        sim.pos_y[0] = 50.0;
        sim.pos_x[1] = 55.0;
        sim.pos_y[1] = 50.0;
        sim.dir_x[0] = 1.0;
        sim.dir_y[0] = 0.0;
        sim.dir_x[1] = 1.0;
        sim.dir_y[1] = 0.0;
        sim.age[0] = MIN_MATE_AGE_SECONDS + 1.0;
        sim.age[1] = MIN_MATE_AGE_SECONDS + 1.0;
        sim.mate_partner[0] = 1;
        sim.mate_partner[1] = 0;
        sim.mate_timer[0] = MATE_DURATION_SECONDS;
        sim.mate_timer[1] = MATE_DURATION_SECONDS;

        sim.resolve_breeding();

        assert_eq!(sim.population(), 2);
        assert_eq!(sim.births(), 1);
        assert_eq!(sim.generation(), 1);
        assert!(sim.generation.iter().any(|generation| *generation == 1));
    }

    #[test]
    fn side_collision_replaces_dead_agent_with_random_agent() {
        let mut sim = Simulation::new(128.0, 2, 11);
        sim.pos_x[0] = 50.0;
        sim.pos_y[0] = 50.0;
        sim.dir_x[0] = 1.0;
        sim.dir_y[0] = 0.0;
        sim.pos_x[1] = 55.5;
        sim.pos_y[1] = 50.0;
        sim.dir_x[1] = 1.0;
        sim.dir_y[1] = 0.0;
        sim.rebuild_grid();

        sim.resolve_collisions();

        assert_eq!(sim.population(), 2);
        assert_eq!(sim.deaths(), 1);
        assert_eq!(sim.age[0], 0.0);
    }

    #[test]
    fn head_on_collision_is_nonlethal() {
        let mut sim = Simulation::new(128.0, 2, 12);
        sim.pos_x[0] = 50.0;
        sim.pos_y[0] = 50.0;
        sim.dir_x[0] = 1.0;
        sim.dir_y[0] = 0.0;
        sim.pos_x[1] = 61.0;
        sim.pos_y[1] = 50.0;
        sim.dir_x[1] = -1.0;
        sim.dir_y[1] = 0.0;
        sim.rebuild_grid();

        sim.resolve_collisions();

        assert_eq!(sim.deaths(), 0);
        assert_eq!(sim.population(), 2);
    }

    // ----------------------------------------------------------------------------------------
    // Shared helpers
    // ----------------------------------------------------------------------------------------

    fn place(sim: &mut Simulation, index: usize, x: f32, y: f32, dir_x: f32, dir_y: f32) {
        sim.pos_x[index] = x;
        sim.pos_y[index] = y;
        sim.dir_x[index] = dir_x;
        sim.dir_y[index] = dir_y;
    }

    fn setup_ready_pair(sim: &mut Simulation) {
        place(sim, 0, 50.0, 50.0, 1.0, 0.0);
        place(sim, 1, 55.0, 50.0, 1.0, 0.0);
        sim.age[0] = MIN_MATE_AGE_SECONDS + 1.0;
        sim.age[1] = MIN_MATE_AGE_SECONDS + 1.0;
        sim.mate_partner[0] = 1;
        sim.mate_partner[1] = 0;
        sim.mate_timer[0] = MATE_DURATION_SECONDS;
        sim.mate_timer[1] = MATE_DURATION_SECONDS;
    }

    /// Asserts the invariants that must hold after any construction, step, or tick.
    fn assert_core_invariants(sim: &Simulation) {
        let pop = sim.population() as usize;
        assert_eq!(sim.population, pop, "population field diverged from getter");
        assert_eq!(sim.render_agents.len(), pop * RENDER_STRIDE_FLOATS);
        assert_eq!(sim.agent_f32_len(), pop * RENDER_STRIDE_FLOATS);
        assert_eq!(sim.agent_stride_f32(), RENDER_STRIDE_FLOATS);

        for value in &sim.render_agents {
            assert!(value.is_finite(), "render value not finite: {value}");
        }

        for index in 0..pop {
            let base = index * RENDER_STRIDE_FLOATS;
            let x = sim.render_agents[base];
            let y = sim.render_agents[base + 1];
            // Inclusive upper bound tolerates the rare f32 rem_euclid edge that returns world.
            assert!((0.0..=1.0).contains(&x), "x norm out of range: {x}");
            assert!((0.0..=1.0).contains(&y), "y norm out of range: {y}");
            for channel in 4..=6 {
                let c = sim.render_agents[base + channel];
                assert!((0.0..=1.0).contains(&c), "color out of range: {c}");
            }
            let speed_norm = sim.render_agents[base + 7];
            assert!((0.0..=1.0).contains(&speed_norm), "speed_norm out of range: {speed_norm}");

            assert!((0.0..=sim.world_size).contains(&sim.pos_x[index]), "pos_x out of range");
            assert!((0.0..=sim.world_size).contains(&sim.pos_y[index]), "pos_y out of range");
            assert!(
                (MIN_SPEED..=MAX_SPEED).contains(&sim.speed[index]),
                "speed out of range: {}",
                sim.speed[index]
            );
            let dir_len =
                (sim.dir_x[index] * sim.dir_x[index] + sim.dir_y[index] * sim.dir_y[index]).sqrt();
            assert!((dir_len - 1.0).abs() < 1e-3, "direction not unit length: {dir_len}");
        }
    }

    // ----------------------------------------------------------------------------------------
    // A. Sanitization & construction
    // ----------------------------------------------------------------------------------------

    #[test]
    fn sanitize_world_size_rejects_invalid_and_small() {
        assert_eq!(sanitize_world_size(f32::NAN), DEFAULT_WORLD_SIZE);
        assert_eq!(sanitize_world_size(f32::INFINITY), DEFAULT_WORLD_SIZE);
        assert_eq!(sanitize_world_size(f32::NEG_INFINITY), DEFAULT_WORLD_SIZE);
        assert_eq!(sanitize_world_size(-10.0), DEFAULT_WORLD_SIZE);
        assert_eq!(sanitize_world_size(0.0), DEFAULT_WORLD_SIZE);
        assert_eq!(sanitize_world_size(127.9), DEFAULT_WORLD_SIZE);
        assert_eq!(sanitize_world_size(128.0), 128.0);
        assert_eq!(sanitize_world_size(4096.0), 4096.0);
        assert_eq!(sanitize_world_size(10_000.0), 10_000.0);
    }

    #[test]
    fn sanitize_population_clamps_to_bounds() {
        assert_eq!(sanitize_population(0), DEFAULT_POPULATION);
        assert_eq!(sanitize_population(1), 1);
        assert_eq!(sanitize_population(50), 50);
        assert_eq!(sanitize_population(MAX_POPULATION as u32), MAX_POPULATION);
        assert_eq!(sanitize_population(MAX_POPULATION as u32 + 1), MAX_POPULATION);
        assert_eq!(sanitize_population(u32::MAX), MAX_POPULATION);
    }

    #[test]
    fn new_sizes_all_buffers_and_zeroes_counters() {
        let sim = Simulation::new(256.0, 8, 1);
        let pop = 8usize;
        assert_eq!(sim.population(), 8);
        assert_eq!(sim.world_size(), 256.0);
        assert_eq!(sim.pos_x.len(), pop);
        assert_eq!(sim.pos_y.len(), pop);
        assert_eq!(sim.dir_x.len(), pop);
        assert_eq!(sim.dir_y.len(), pop);
        assert_eq!(sim.speed.len(), pop);
        assert_eq!(sim.age.len(), pop);
        assert_eq!(sim.generation.len(), pop);
        assert_eq!(sim.genomes.len(), pop * GENOME_LEN);
        assert_eq!(sim.child_genome.len(), GENOME_LEN);
        assert_eq!(sim.render_agents.len(), pop * RENDER_STRIDE_FLOATS);
        assert_eq!(sim.grid_next.len(), pop);

        let expected_cols = (256.0_f32 / GRID_CELL_SIZE).ceil() as usize;
        assert_eq!(sim.grid_cols, expected_cols);
        assert_eq!(sim.grid_heads.len(), expected_cols * expected_cols);

        assert_eq!(sim.births(), 0);
        assert_eq!(sim.deaths(), 0);
        assert_eq!(sim.sim_steps(), 0);
        assert_eq!(sim.generation(), 0);
        assert_eq!(sim.accumulator, 0.0);
    }

    #[test]
    fn new_sanitizes_arguments() {
        let sim = Simulation::new(10.0, 0, 5);
        assert_eq!(sim.world_size(), DEFAULT_WORLD_SIZE);
        assert_eq!(sim.population(), DEFAULT_POPULATION as u32);
    }

    #[test]
    fn new_clamps_grid_cols_to_at_least_one() {
        let sim = Simulation::new(128.0, 4, 1);
        assert!(sim.grid_cols >= 1);
        assert_eq!(sim.grid_cols, (128.0_f32 / GRID_CELL_SIZE).ceil() as usize);
    }

    #[test]
    fn new_agents_satisfy_initial_invariants() {
        let sim = Simulation::new(512.0, 64, 7);
        assert_core_invariants(&sim);
        for index in 0..(sim.population() as usize) {
            assert_eq!(sim.age[index], 0.0);
            assert_eq!(sim.generation[index], 0);
            assert!((0.25..=1.0).contains(&sim.color_r[index]));
            assert!((0.25..=1.0).contains(&sim.color_g[index]));
            assert!((0.25..=1.0).contains(&sim.color_b[index]));
        }
    }

    // ----------------------------------------------------------------------------------------
    // B. SmallRng
    // ----------------------------------------------------------------------------------------

    #[test]
    fn rng_next_f32_in_unit_interval() {
        let mut rng = SmallRng::new(123);
        for _ in 0..100_000 {
            let v = rng.next_f32();
            assert!((0.0..1.0).contains(&v), "next_f32 out of range: {v}");
        }
    }

    #[test]
    fn rng_next_signed_f32_in_range() {
        let mut rng = SmallRng::new(321);
        for _ in 0..100_000 {
            let v = rng.next_signed_f32();
            assert!((-1.0..1.0).contains(&v), "next_signed_f32 out of range: {v}");
        }
    }

    #[test]
    fn rng_next_bool_produces_both_values() {
        let mut rng = SmallRng::new(9);
        let mut seen_true = false;
        let mut seen_false = false;
        for _ in 0..1000 {
            if rng.next_bool() {
                seen_true = true;
            } else {
                seen_false = true;
            }
        }
        assert!(seen_true && seen_false);
    }

    #[test]
    fn rng_next_f32_mean_is_near_half() {
        let mut rng = SmallRng::new(55);
        let samples = 200_000;
        let mut sum = 0.0f64;
        for _ in 0..samples {
            sum += rng.next_f32() as f64;
        }
        let mean = sum / samples as f64;
        assert!((mean - 0.5).abs() < 0.01, "mean was {mean}");
    }

    // ----------------------------------------------------------------------------------------
    // C. Math / contracts
    // ----------------------------------------------------------------------------------------

    #[test]
    fn wrap_handles_all_regions() {
        assert_eq!(wrap(0.0, 100.0), 0.0);
        assert_eq!(wrap(50.0, 100.0), 50.0);
        assert_eq!(wrap(100.0, 100.0), 0.0);
        assert_eq!(wrap(250.0, 100.0), 50.0);
        assert_eq!(wrap(-30.0, 100.0), 70.0);
        assert_eq!(wrap(-100.0, 100.0), 0.0);
    }

    #[test]
    fn wrap_delta_takes_shortest_path() {
        assert_eq!(wrap_delta(0.0, 100.0), 0.0);
        assert_eq!(wrap_delta(30.0, 100.0), 30.0);
        assert_eq!(wrap_delta(60.0, 100.0), -40.0);
        assert_eq!(wrap_delta(-60.0, 100.0), 40.0);
        assert_eq!(wrap_delta(50.0, 100.0), 50.0);
        assert_eq!(wrap_delta(-50.0, 100.0), -50.0);
    }

    #[test]
    fn output_to_color_maps_tanh_range() {
        assert_eq!(output_to_color(-1.0), 0.0);
        assert_eq!(output_to_color(0.0), 0.5);
        assert_eq!(output_to_color(1.0), 1.0);
        assert_eq!(output_to_color(-5.0), 0.0);
        assert_eq!(output_to_color(5.0), 1.0);
    }

    #[test]
    fn shape_contracts_hold() {
        assert_eq!(RENDER_STRIDE_FLOATS, 8);
        assert_eq!(INPUT_COUNT, RAY_COUNT * 5 + 2);
        assert_eq!(
            GENOME_LEN,
            HIDDEN_COUNT * (INPUT_COUNT + 1) + OUTPUT_COUNT * (HIDDEN_COUNT + 1)
        );
    }

    // ----------------------------------------------------------------------------------------
    // D. Genome & network
    // ----------------------------------------------------------------------------------------

    #[test]
    fn randomize_genome_bounds_and_isolation() {
        let mut sim = Simulation::new(256.0, 4, 3);
        for gene in sim.genomes.iter_mut() {
            *gene = 0.0;
        }
        sim.randomize_genome(2);
        let start = 2 * GENOME_LEN;
        for i in 0..sim.genomes.len() {
            if (start..start + GENOME_LEN).contains(&i) {
                assert!((-0.75..=0.75).contains(&sim.genomes[i]), "gene out of range");
            } else {
                assert_eq!(sim.genomes[i], 0.0, "neighbor slice mutated at {i}");
            }
        }
    }

    #[test]
    fn mutate_rate_zero_leaves_genome_unchanged() {
        let mut rng = SmallRng::new(1);
        let original = vec![0.5; GENOME_LEN];
        let mut genome = original.clone();
        mutate_genome_with_rate(&mut genome, &mut rng, 0.0);
        assert_eq!(genome, original);
    }

    #[test]
    fn mutate_changes_stay_within_magnitude_and_limit() {
        let mut rng = SmallRng::new(2);
        let mut genome = vec![0.0f32; GENOME_LEN];
        mutate_genome_with_rate(&mut genome, &mut rng, 1.0);
        for g in &genome {
            assert!((-GENE_LIMIT..=GENE_LIMIT).contains(g));
            assert!(g.abs() <= MUTATION_MAGNITUDE + 1e-6, "change exceeded magnitude: {g}");
        }
    }

    #[test]
    fn evaluate_network_outputs_in_tanh_range() {
        let sim = Simulation::new(256.0, 4, 8);
        let inputs = [0.5f32; INPUT_COUNT];
        let mut hidden = [0.0; HIDDEN_COUNT];
        let mut outputs = [0.0; OUTPUT_COUNT];
        for index in 0..4 {
            sim.evaluate_network(index, &inputs, &mut hidden, &mut outputs);
            for o in &outputs {
                assert!((-1.0..=1.0).contains(o));
            }
            for h in &hidden {
                assert!((-1.0..=1.0).contains(h));
            }
        }
    }

    #[test]
    fn evaluate_network_bias_only_returns_tanh_bias() {
        let mut sim = Simulation::new(256.0, 1, 1);
        for gene in sim.genomes.iter_mut() {
            *gene = 0.0;
        }
        let hidden_bias = 0.5f32;
        let output_bias = -0.3f32;
        for h in 0..HIDDEN_COUNT {
            let bias_idx = h * (INPUT_COUNT + 1) + INPUT_COUNT;
            sim.genomes[bias_idx] = hidden_bias;
        }
        let output_base = HIDDEN_COUNT * (INPUT_COUNT + 1);
        for o in 0..OUTPUT_COUNT {
            let bias_idx = output_base + o * (HIDDEN_COUNT + 1) + HIDDEN_COUNT;
            sim.genomes[bias_idx] = output_bias;
        }

        let inputs = [0.0f32; INPUT_COUNT];
        let mut hidden = [0.0; HIDDEN_COUNT];
        let mut outputs = [0.0; OUTPUT_COUNT];
        sim.evaluate_network(0, &inputs, &mut hidden, &mut outputs);

        let expected_hidden = hidden_bias.tanh();
        for h in &hidden {
            assert!((h - expected_hidden).abs() < 1e-6);
        }
        let expected_output = output_bias.tanh();
        for o in &outputs {
            assert!((o - expected_output).abs() < 1e-6);
        }
    }

    #[test]
    fn evaluate_network_isolated_per_agent() {
        let mut sim = Simulation::new(256.0, 2, 1);
        for gene in sim.genomes.iter_mut() {
            *gene = 0.0;
        }
        for gene in sim.genomes[GENOME_LEN..2 * GENOME_LEN].iter_mut() {
            *gene = 1.0;
        }
        let inputs = [1.0f32; INPUT_COUNT];
        let mut hidden = [0.0; HIDDEN_COUNT];
        let mut outputs = [0.0; OUTPUT_COUNT];

        sim.evaluate_network(0, &inputs, &mut hidden, &mut outputs);
        for o in &outputs {
            assert_eq!(*o, 0.0, "zero genome must yield tanh(0) = 0");
        }

        sim.evaluate_network(1, &inputs, &mut hidden, &mut outputs);
        assert!(outputs.iter().any(|o| *o != 0.0), "nonzero genome must produce output");
    }

    #[test]
    fn update_decisions_writes_clamped_targets() {
        let mut sim = Simulation::new(256.0, 4, 4);
        sim.rebuild_grid();
        sim.update_decisions();
        for index in 0..4 {
            assert!((0.0..=1.0).contains(&sim.target_r[index]));
            assert!((0.0..=1.0).contains(&sim.target_g[index]));
            assert!((0.0..=1.0).contains(&sim.target_b[index]));
            assert!((-1.0..=1.0).contains(&sim.turn_command[index]));
            assert!((-1.0..=1.0).contains(&sim.accel_command[index]));
        }
    }

    // ----------------------------------------------------------------------------------------
    // E. Vision
    // ----------------------------------------------------------------------------------------

    #[test]
    fn vision_empty_world_sets_only_self_inputs() {
        let mut sim = Simulation::new(256.0, 1, 1);
        place(&mut sim, 0, 100.0, 100.0, 1.0, 0.0);
        sim.speed[0] = MIN_SPEED;
        sim.age[0] = 0.0;
        sim.rebuild_grid();

        let mut inputs = [9.0f32; INPUT_COUNT];
        sim.write_vision_inputs(0, &mut inputs);

        for ray in 0..RAY_COUNT {
            let base = ray * 5;
            for k in 0..5 {
                assert_eq!(inputs[base + k], 0.0, "ray {ray} component {k} should be empty");
            }
        }
        let self_base = RAY_COUNT * 5;
        assert_eq!(inputs[self_base], 0.0);
        assert_eq!(inputs[self_base + 1], 0.0);
    }

    #[test]
    fn vision_self_inputs_normalize_speed_and_age() {
        let mut sim = Simulation::new(256.0, 1, 1);
        place(&mut sim, 0, 100.0, 100.0, 1.0, 0.0);
        sim.rebuild_grid();
        let self_base = RAY_COUNT * 5;
        let mut inputs = [0.0f32; INPUT_COUNT];

        sim.speed[0] = (MIN_SPEED + MAX_SPEED) * 0.5;
        sim.age[0] = AGE_INPUT_CAP_SECONDS * 0.5;
        sim.write_vision_inputs(0, &mut inputs);
        assert!((inputs[self_base] - 0.5).abs() < 1e-5);
        assert!((inputs[self_base + 1] - 0.5).abs() < 1e-5);

        sim.speed[0] = MAX_SPEED;
        sim.age[0] = AGE_INPUT_CAP_SECONDS * 2.0;
        sim.write_vision_inputs(0, &mut inputs);
        assert!((inputs[self_base] - 1.0).abs() < 1e-5);
        assert_eq!(inputs[self_base + 1], 1.0);
    }

    #[test]
    fn vision_detects_agent_directly_ahead() {
        let mut sim = Simulation::new(256.0, 2, 1);
        place(&mut sim, 0, 100.0, 100.0, 1.0, 0.0);
        place(&mut sim, 1, 140.0, 100.0, 1.0, 0.0);
        sim.color_r[1] = 0.2;
        sim.color_g[1] = 0.4;
        sim.color_b[1] = 0.6;
        sim.rebuild_grid();

        let mut inputs = [0.0f32; INPUT_COUNT];
        sim.write_vision_inputs(0, &mut inputs);
        let base = 3 * 5;
        assert_eq!(inputs[base], 1.0);
        let expected_dist = 1.0 - 40.0 / VISION_RANGE;
        assert!((inputs[base + 1] - expected_dist).abs() < 1e-4);
        assert!((inputs[base + 2] - 0.2).abs() < 1e-5);
        assert!((inputs[base + 3] - 0.4).abs() < 1e-5);
        assert!((inputs[base + 4] - 0.6).abs() < 1e-5);
    }

    #[test]
    fn vision_ignores_agent_behind() {
        let mut sim = Simulation::new(256.0, 2, 1);
        place(&mut sim, 0, 100.0, 100.0, 1.0, 0.0);
        place(&mut sim, 1, 60.0, 100.0, 1.0, 0.0);
        sim.rebuild_grid();

        let mut inputs = [0.0f32; INPUT_COUNT];
        sim.write_vision_inputs(0, &mut inputs);
        for ray in 0..RAY_COUNT {
            assert_eq!(inputs[ray * 5], 0.0);
        }
    }

    #[test]
    fn vision_ignores_agent_beyond_range() {
        let mut sim = Simulation::new(512.0, 2, 1);
        place(&mut sim, 0, 100.0, 100.0, 1.0, 0.0);
        place(&mut sim, 1, 100.0 + VISION_RANGE + 20.0, 100.0, 1.0, 0.0);
        sim.rebuild_grid();

        let mut inputs = [0.0f32; INPUT_COUNT];
        sim.write_vision_inputs(0, &mut inputs);
        assert_eq!(inputs[3 * 5], 0.0);
    }

    #[test]
    fn vision_lateral_gate_includes_and_excludes() {
        let mut sim = Simulation::new(256.0, 2, 1);
        place(&mut sim, 0, 100.0, 100.0, 1.0, 0.0);
        place(&mut sim, 1, 140.0, 109.0, 1.0, 0.0);
        sim.rebuild_grid();
        let mut inside = [0.0f32; INPUT_COUNT];
        sim.write_vision_inputs(0, &mut inside);
        assert_eq!(inside[3 * 5], 1.0, "just inside lateral width");

        place(&mut sim, 1, 140.0, 112.0, 1.0, 0.0);
        sim.rebuild_grid();
        let mut outside = [0.0f32; INPUT_COUNT];
        sim.write_vision_inputs(0, &mut outside);
        assert_eq!(outside[3 * 5], 0.0, "just outside center-ray lateral width");
    }

    #[test]
    fn vision_reports_nearest_along_ray() {
        let mut sim = Simulation::new(256.0, 3, 1);
        place(&mut sim, 0, 100.0, 100.0, 1.0, 0.0);
        place(&mut sim, 1, 170.0, 100.0, 1.0, 0.0);
        sim.color_r[1] = 0.9;
        place(&mut sim, 2, 130.0, 100.0, 1.0, 0.0);
        sim.color_r[2] = 0.1;
        sim.rebuild_grid();

        let mut inputs = [0.0f32; INPUT_COUNT];
        sim.write_vision_inputs(0, &mut inputs);
        let base = 3 * 5;
        assert_eq!(inputs[base], 1.0);
        let expected = 1.0 - 30.0 / VISION_RANGE;
        assert!((inputs[base + 1] - expected).abs() < 1e-4);
        assert!((inputs[base + 2] - 0.1).abs() < 1e-5);
    }

    #[test]
    fn vision_sees_across_wrap_boundary() {
        let world = 256.0;
        let mut sim = Simulation::new(world, 2, 1);
        place(&mut sim, 0, world - 10.0, 100.0, 1.0, 0.0);
        place(&mut sim, 1, 30.0, 100.0, 1.0, 0.0);
        sim.rebuild_grid();

        let mut inputs = [0.0f32; INPUT_COUNT];
        sim.write_vision_inputs(0, &mut inputs);
        let base = 3 * 5;
        assert_eq!(inputs[base], 1.0);
        let expected = 1.0 - 40.0 / VISION_RANGE;
        assert!((inputs[base + 1] - expected).abs() < 1e-3);
    }

    #[test]
    fn vision_buckets_into_correct_side_ray() {
        let mut sim = Simulation::new(256.0, 2, 1);
        place(&mut sim, 0, 100.0, 100.0, 1.0, 0.0);
        let d = 40.0f32;
        // Ray 0 direction for a +x facing agent is (RAY_COS[0], RAY_SIN[0]).
        place(&mut sim, 1, 100.0 + d * RAY_COS[0], 100.0 + d * RAY_SIN[0], 1.0, 0.0);
        sim.rebuild_grid();

        let mut inputs = [0.0f32; INPUT_COUNT];
        sim.write_vision_inputs(0, &mut inputs);
        assert_eq!(inputs[0], 1.0, "target should register in ray 0");
        assert_eq!(inputs[3 * 5], 0.0, "center ray should stay empty");
    }

    #[test]
    fn vision_excludes_self() {
        let mut sim = Simulation::new(256.0, 1, 1);
        place(&mut sim, 0, 100.0, 100.0, 1.0, 0.0);
        sim.rebuild_grid();

        let mut inputs = [0.0f32; INPUT_COUNT];
        sim.write_vision_inputs(0, &mut inputs);
        for ray in 0..RAY_COUNT {
            assert_eq!(inputs[ray * 5], 0.0);
        }
    }

    // ----------------------------------------------------------------------------------------
    // F. apply_decisions
    // ----------------------------------------------------------------------------------------

    #[test]
    fn apply_decisions_keeps_direction_unit_length() {
        let mut sim = Simulation::new(256.0, 1, 1);
        place(&mut sim, 0, 100.0, 100.0, 1.0, 0.0);
        sim.turn_command[0] = 1.0;
        sim.accel_command[0] = 0.0;
        sim.apply_decisions(FIXED_STEP_SECONDS);
        let len = (sim.dir_x[0] * sim.dir_x[0] + sim.dir_y[0] * sim.dir_y[0]).sqrt();
        assert!((len - 1.0).abs() < 1e-5);
    }

    #[test]
    fn apply_decisions_turns_by_expected_angle() {
        let mut sim = Simulation::new(256.0, 1, 1);
        place(&mut sim, 0, 100.0, 100.0, 1.0, 0.0);
        sim.turn_command[0] = 1.0;
        sim.accel_command[0] = 0.0;
        sim.target_r[0] = sim.color_r[0];
        sim.apply_decisions(FIXED_STEP_SECONDS);
        let expected = MAX_TURN_RATE * FIXED_STEP_SECONDS;
        assert!((sim.dir_x[0] - expected.cos()).abs() < 1e-5);
        assert!((sim.dir_y[0] - expected.sin()).abs() < 1e-5);
    }

    #[test]
    fn apply_decisions_accelerates_and_clamps_speed() {
        let mut sim = Simulation::new(256.0, 1, 1);
        place(&mut sim, 0, 100.0, 100.0, 1.0, 0.0);
        sim.speed[0] = 40.0;
        sim.turn_command[0] = 0.0;
        sim.accel_command[0] = 1.0;
        sim.apply_decisions(FIXED_STEP_SECONDS);
        let expected = (40.0 + ACCELERATION * FIXED_STEP_SECONDS).min(MAX_SPEED);
        assert!((sim.speed[0] - expected).abs() < 1e-4);

        sim.speed[0] = MAX_SPEED;
        sim.accel_command[0] = 5.0;
        sim.apply_decisions(FIXED_STEP_SECONDS);
        assert_eq!(sim.speed[0], MAX_SPEED);

        sim.speed[0] = MIN_SPEED;
        sim.accel_command[0] = -5.0;
        sim.apply_decisions(FIXED_STEP_SECONDS);
        assert_eq!(sim.speed[0], MIN_SPEED);
    }

    #[test]
    fn apply_decisions_moves_and_wraps_position() {
        let world = 256.0;
        let mut sim = Simulation::new(world, 1, 1);
        place(&mut sim, 0, world - 1.0, 50.0, 1.0, 0.0);
        sim.speed[0] = 60.0;
        sim.turn_command[0] = 0.0;
        sim.accel_command[0] = 0.0;
        let dt = FIXED_STEP_SECONDS;
        sim.apply_decisions(dt);
        let expected = wrap(world - 1.0 + 60.0 * dt, world);
        assert!((sim.pos_x[0] - expected).abs() < 1e-4);
        assert!(sim.pos_x[0] >= 0.0 && sim.pos_x[0] < world);
    }

    #[test]
    fn apply_decisions_blends_color_toward_target() {
        let mut sim = Simulation::new(256.0, 1, 1);
        place(&mut sim, 0, 100.0, 100.0, 1.0, 0.0);
        sim.color_r[0] = 0.0;
        sim.target_r[0] = 1.0;
        sim.turn_command[0] = 0.0;
        sim.accel_command[0] = 0.0;
        let dt = FIXED_STEP_SECONDS;
        sim.apply_decisions(dt);
        let blend = (COLOR_BLEND_RATE * dt).min(1.0);
        assert!((sim.color_r[0] - blend).abs() < 1e-5);
    }

    #[test]
    fn apply_decisions_color_snaps_when_blend_full() {
        let mut sim = Simulation::new(256.0, 1, 1);
        place(&mut sim, 0, 100.0, 100.0, 1.0, 0.0);
        sim.color_r[0] = 0.0;
        sim.target_r[0] = 1.0;
        sim.apply_decisions(1.0);
        assert!((sim.color_r[0] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn apply_decisions_increments_age() {
        let mut sim = Simulation::new(256.0, 1, 1);
        sim.age[0] = 1.0;
        sim.apply_decisions(0.5);
        assert!((sim.age[0] - 1.5).abs() < 1e-6);
    }

    // ----------------------------------------------------------------------------------------
    // G. Grid
    // ----------------------------------------------------------------------------------------

    #[test]
    fn rebuild_grid_places_agents_in_cells() {
        let mut sim = Simulation::new(256.0, 3, 1);
        place(&mut sim, 0, 10.0, 10.0, 1.0, 0.0);
        place(&mut sim, 1, 10.0, 10.0, 1.0, 0.0);
        place(&mut sim, 2, 200.0, 200.0, 1.0, 0.0);
        sim.rebuild_grid();

        let cell_a = sim.cell_index(10.0, 10.0);
        let mut members = Vec::new();
        let mut cursor = sim.grid_heads[cell_a];
        while cursor >= 0 {
            members.push(cursor as usize);
            cursor = sim.grid_next[cursor as usize];
        }
        members.sort();
        assert_eq!(members, vec![0, 1]);

        let cell_b = sim.cell_index(200.0, 200.0);
        assert_eq!(sim.grid_heads[cell_b], 2);
    }

    #[test]
    fn rebuild_grid_clears_stale_entries() {
        let mut sim = Simulation::new(256.0, 1, 1);
        place(&mut sim, 0, 10.0, 10.0, 1.0, 0.0);
        sim.rebuild_grid();
        let old_cell = sim.cell_index(10.0, 10.0);

        place(&mut sim, 0, 200.0, 200.0, 1.0, 0.0);
        sim.rebuild_grid();
        assert_eq!(sim.grid_heads[old_cell], -1);
        let new_cell = sim.cell_index(200.0, 200.0);
        assert_eq!(sim.grid_heads[new_cell], 0);
    }

    #[test]
    fn cell_coord_and_wrapped_index() {
        let sim = Simulation::new(256.0, 1, 1);
        assert_eq!(sim.cell_coord(0.0), 0);
        assert_eq!(sim.cell_coord(31.9), 0);
        assert_eq!(sim.cell_coord(32.0), 1);

        let cols = sim.grid_cols as isize;
        assert_eq!(
            sim.wrapped_cell_index(-1, -1),
            (cols - 1) as usize * sim.grid_cols + (cols - 1) as usize
        );
        assert_eq!(sim.wrapped_cell_index(cols, cols), 0);
    }

    // ----------------------------------------------------------------------------------------
    // H. Collisions
    // ----------------------------------------------------------------------------------------

    #[test]
    fn collision_ignores_lone_agent() {
        let mut sim = Simulation::new(128.0, 1, 1);
        place(&mut sim, 0, 50.0, 50.0, 1.0, 0.0);
        sim.rebuild_grid();
        sim.resolve_collisions();
        assert_eq!(sim.deaths(), 0);
    }

    #[test]
    fn hits_body_side_geometry() {
        let mut sim = Simulation::new(128.0, 1, 1);
        place(&mut sim, 0, 50.0, 50.0, 1.0, 0.0);
        assert!(sim.hits_body_side(50.0 + 1.0, 50.0 + 2.0, 0));
        assert!(!sim.hits_body_side(50.0 + 2.0, 50.0, 0));
        assert!(!sim.hits_body_side(50.0 - 5.5, 50.0, 0));
        assert!(!sim.hits_body_side(50.0, 50.0 + 3.0, 0));
    }

    #[test]
    fn is_head_on_requires_opposing_and_close_heads() {
        let mut sim = Simulation::new(128.0, 2, 1);
        place(&mut sim, 0, 50.0, 50.0, 1.0, 0.0);
        place(&mut sim, 1, 60.0, 50.0, -1.0, 0.0);
        let head0_x = wrap(50.0 + HEAD_OFFSET, sim.world_size);
        let head0_y = 50.0;
        assert!(sim.is_head_on(0, 1, head0_x, head0_y));

        place(&mut sim, 1, 60.0, 70.0, -1.0, 0.0);
        assert!(!sim.is_head_on(0, 1, head0_x, head0_y));

        place(&mut sim, 1, 60.0, 50.0, 1.0, 0.0);
        assert!(!sim.is_head_on(0, 1, head0_x, head0_y));
    }

    #[test]
    fn mutual_side_collision_kills_both() {
        let mut sim = Simulation::new(128.0, 2, 1);
        place(&mut sim, 0, 50.0, 50.0, 1.0, 0.0);
        place(&mut sim, 1, 56.0, 52.0, -1.0, 0.0);
        sim.age[0] = 5.0;
        sim.age[1] = 5.0;
        sim.rebuild_grid();
        sim.resolve_collisions();
        assert_eq!(sim.deaths(), 2);
        assert_eq!(sim.population(), 2);
        assert_eq!(sim.age[0], 0.0);
        assert_eq!(sim.age[1], 0.0);
    }

    #[test]
    fn resolve_collisions_resets_dead_flags_each_call() {
        let mut sim = Simulation::new(128.0, 2, 1);
        place(&mut sim, 0, 50.0, 50.0, 1.0, 0.0);
        place(&mut sim, 1, 55.5, 50.0, 1.0, 0.0);
        sim.rebuild_grid();
        sim.resolve_collisions();
        let deaths_first = sim.deaths();
        assert!(deaths_first >= 1);

        place(&mut sim, 0, 10.0, 10.0, 1.0, 0.0);
        place(&mut sim, 1, 100.0, 100.0, 1.0, 0.0);
        sim.rebuild_grid();
        sim.resolve_collisions();
        assert_eq!(sim.deaths(), deaths_first);
        assert!(sim.dead.iter().all(|d| !*d));
    }

    #[test]
    fn collision_detected_across_wrap_boundary() {
        let world = 128.0;
        let mut sim = Simulation::new(world, 2, 1);
        place(&mut sim, 0, world - 1.0, 50.0, 1.0, 0.0);
        place(&mut sim, 1, 3.5, 50.0, 1.0, 0.0);
        sim.rebuild_grid();
        sim.resolve_collisions();
        assert_eq!(sim.deaths(), 1);
    }

    #[test]
    fn sparse_world_has_no_collisions() {
        let mut sim = Simulation::new(256.0, 2, 1);
        place(&mut sim, 0, 20.0, 20.0, 1.0, 0.0);
        place(&mut sim, 1, 200.0, 200.0, 1.0, 0.0);
        sim.age[0] = 3.0;
        sim.age[1] = 4.0;
        sim.rebuild_grid();
        sim.resolve_collisions();
        assert_eq!(sim.deaths(), 0);
        assert_eq!(sim.age[0], 3.0);
        assert_eq!(sim.age[1], 4.0);
    }

    // ----------------------------------------------------------------------------------------
    // I. Breeding (sticky-partner)
    // ----------------------------------------------------------------------------------------

    #[test]
    fn is_mate_eligible_gates_all_conditions() {
        let mut sim = Simulation::new(128.0, 2, 1);
        place(&mut sim, 0, 50.0, 50.0, 1.0, 0.0);
        place(&mut sim, 1, 55.0, 50.0, 1.0, 0.0);
        sim.age[0] = MIN_MATE_AGE_SECONDS;
        sim.age[1] = MIN_MATE_AGE_SECONDS;
        assert!(sim.is_mate_eligible(0, 1).is_some());
        assert!(sim.is_mate_eligible(0, 0).is_none());

        sim.age[1] = MIN_MATE_AGE_SECONDS - 0.1;
        assert!(sim.is_mate_eligible(0, 1).is_none());
        sim.age[1] = MIN_MATE_AGE_SECONDS;

        place(&mut sim, 1, 55.0, 50.0, 0.0, 1.0);
        assert!(sim.is_mate_eligible(0, 1).is_none());

        place(&mut sim, 1, 50.0 + MATE_RADIUS + 1.0, 50.0, 1.0, 0.0);
        assert!(sim.is_mate_eligible(0, 1).is_none());
    }

    #[test]
    fn find_mate_partner_returns_nearest_eligible() {
        let mut sim = Simulation::new(128.0, 3, 1);
        place(&mut sim, 0, 50.0, 50.0, 1.0, 0.0);
        place(&mut sim, 1, 58.0, 50.0, 1.0, 0.0);
        place(&mut sim, 2, 53.0, 50.0, 1.0, 0.0);
        for i in 0..3 {
            sim.age[i] = MIN_MATE_AGE_SECONDS + 1.0;
        }
        sim.rebuild_grid();
        assert_eq!(sim.find_mate_partner(0), Some(2));
    }

    #[test]
    fn find_mate_partner_none_when_young() {
        let mut sim = Simulation::new(128.0, 2, 1);
        place(&mut sim, 0, 50.0, 50.0, 1.0, 0.0);
        place(&mut sim, 1, 53.0, 50.0, 1.0, 0.0);
        sim.age[0] = MIN_MATE_AGE_SECONDS - 0.5;
        sim.age[1] = MIN_MATE_AGE_SECONDS + 1.0;
        sim.rebuild_grid();
        assert_eq!(sim.find_mate_partner(0), None);
    }

    #[test]
    fn breeding_is_sticky_to_current_partner() {
        let dt = FIXED_STEP_SECONDS;
        let mut sim = Simulation::new(128.0, 3, 1);
        place(&mut sim, 0, 50.0, 50.0, 1.0, 0.0);
        place(&mut sim, 1, 54.0, 50.0, 1.0, 0.0);
        place(&mut sim, 2, 90.0, 90.0, 0.0, 1.0);
        for i in 0..3 {
            sim.age[i] = MIN_MATE_AGE_SECONDS + 1.0;
        }
        sim.rebuild_grid();
        sim.update_breeding(dt);
        assert_eq!(sim.mate_partner[0], 1);
        assert!((sim.mate_timer[0] - dt).abs() < 1e-6);

        // Bring agent 2 closer to 0 than 1 and make it eligible.
        place(&mut sim, 2, 52.0, 50.0, 1.0, 0.0);
        sim.rebuild_grid();
        sim.update_breeding(dt);
        assert_eq!(sim.mate_partner[0], 1, "sticky: must keep current partner");
        assert!((sim.mate_timer[0] - 2.0 * dt).abs() < 1e-6, "timer must keep growing");
    }

    #[test]
    fn breeding_drops_partner_when_ineligible() {
        let dt = FIXED_STEP_SECONDS;
        let mut sim = Simulation::new(128.0, 2, 1);
        place(&mut sim, 0, 50.0, 50.0, 1.0, 0.0);
        place(&mut sim, 1, 54.0, 50.0, 1.0, 0.0);
        sim.age[0] = MIN_MATE_AGE_SECONDS + 1.0;
        sim.age[1] = MIN_MATE_AGE_SECONDS + 1.0;
        sim.rebuild_grid();
        sim.update_breeding(dt);
        assert_eq!(sim.mate_partner[0], 1);

        place(&mut sim, 1, 50.0 + MATE_RADIUS + 5.0, 50.0, 1.0, 0.0);
        sim.rebuild_grid();
        sim.update_breeding(dt);
        assert_eq!(sim.mate_partner[0], -1);
        assert_eq!(sim.mate_timer[0], 0.0);
    }

    #[test]
    fn resolve_breeding_skips_when_not_mutual() {
        let mut sim = Simulation::new(128.0, 2, 1);
        setup_ready_pair(&mut sim);
        sim.mate_partner[1] = -1;
        sim.resolve_breeding();
        assert_eq!(sim.births(), 0);
    }

    #[test]
    fn resolve_breeding_skips_under_duration() {
        let mut sim = Simulation::new(128.0, 2, 1);
        setup_ready_pair(&mut sim);
        sim.mate_timer[1] = MATE_DURATION_SECONDS - 0.01;
        sim.resolve_breeding();
        assert_eq!(sim.births(), 0);
    }

    #[test]
    fn spawn_child_generation_is_max_parents_plus_one() {
        let mut sim = Simulation::new(128.0, 2, 7);
        setup_ready_pair(&mut sim);
        sim.generation[0] = 3;
        sim.generation[1] = 5;
        sim.resolve_breeding();
        assert_eq!(sim.births(), 1);
        assert_eq!(sim.generation(), 6);
        assert!(sim.generation[0] == 6 || sim.generation[1] == 6);
        assert_eq!(sim.population(), 2);
    }

    #[test]
    fn child_genome_within_limits_and_near_identical_parents() {
        let mut sim = Simulation::new(128.0, 2, 7);
        setup_ready_pair(&mut sim);
        for g in sim.genomes.iter_mut() {
            *g = 0.5;
        }
        sim.resolve_breeding();

        let victim = if sim.age[0] == 0.0 { 0 } else { 1 };
        let start = victim * GENOME_LEN;
        for i in 0..GENOME_LEN {
            let g = sim.genomes[start + i];
            assert!((-GENE_LIMIT..=GENE_LIMIT).contains(&g));
            assert!((g - 0.5).abs() <= MUTATION_MAGNITUDE + 1e-6);
        }
    }

    #[test]
    fn reset_child_pose_averages_parents() {
        let mut sim = Simulation::new(256.0, 2, 7);
        place(&mut sim, 0, 100.0, 100.0, 1.0, 0.0);
        place(&mut sim, 1, 104.0, 100.0, 0.6, 0.8);
        sim.speed[0] = 30.0;
        sim.speed[1] = 50.0;
        sim.color_r[0] = 0.2;
        sim.color_r[1] = 0.8;
        let pre_x = sim.pos_x[0];

        sim.reset_child_pose(0, 0, 1, 4);
        assert_eq!(sim.age[0], 0.0);
        assert_eq!(sim.generation[0], 4);
        assert!((sim.pos_x[0] - pre_x).abs() <= 1.5 + 1e-4);

        let dir_len = (sim.dir_x[0] * sim.dir_x[0] + sim.dir_y[0] * sim.dir_y[0]).sqrt();
        assert!((dir_len - 1.0).abs() < 1e-5);

        let expected_speed: f32 = ((30.0 + 50.0) * 0.5_f32).clamp(MIN_SPEED, MAX_SPEED);
        assert!((sim.speed[0] - expected_speed).abs() < 1e-4);
        let expected_r: f32 = ((0.2 + 0.8) * 0.5_f32).clamp(0.0, 1.0);
        assert!((sim.color_r[0] - expected_r).abs() < 1e-5);
        assert_eq!(sim.mate_partner[0], -1);
        assert_eq!(sim.mate_timer[0], 0.0);
    }

    #[test]
    fn reset_child_pose_random_direction_when_parents_cancel() {
        let mut sim = Simulation::new(256.0, 2, 7);
        place(&mut sim, 0, 100.0, 100.0, 1.0, 0.0);
        place(&mut sim, 1, 100.0, 100.0, -1.0, 0.0);
        sim.reset_child_pose(0, 0, 1, 1);
        let dir_len = (sim.dir_x[0] * sim.dir_x[0] + sim.dir_y[0] * sim.dir_y[0]).sqrt();
        assert!((dir_len - 1.0).abs() < 1e-5);
    }

    #[test]
    fn courtship_matures_into_birth_over_steps() {
        let dt = FIXED_STEP_SECONDS;
        let mut sim = Simulation::new(64.0, 2, 1);
        place(&mut sim, 0, 30.0, 30.0, 1.0, 0.0);
        place(&mut sim, 1, 33.0, 30.0, 1.0, 0.0);
        sim.age[0] = MIN_MATE_AGE_SECONDS + 1.0;
        sim.age[1] = MIN_MATE_AGE_SECONDS + 1.0;

        let steps_needed = (MATE_DURATION_SECONDS / dt).ceil() as usize + 2;
        for _ in 0..steps_needed {
            sim.rebuild_grid();
            sim.update_breeding(dt);
        }
        assert_eq!(sim.births(), 1);
        assert_eq!(sim.population(), 2);
        assert!(sim.generation() >= 1);
    }

    // ----------------------------------------------------------------------------------------
    // J. Generation semantics (current max alive)
    // ----------------------------------------------------------------------------------------

    #[test]
    fn generation_reports_current_max_alive() {
        let mut sim = Simulation::new(128.0, 2, 7);
        assert_eq!(sim.generation(), 0);

        setup_ready_pair(&mut sim);
        sim.resolve_breeding();
        assert_eq!(sim.generation(), 1);

        let high = if sim.generation[0] == 1 { 0 } else { 1 };
        sim.reset_random_agent(high);
        assert_eq!(sim.generation(), 0, "generation must reflect current max, not peak");
    }

    // ----------------------------------------------------------------------------------------
    // K. tick / step (wall-clock catch-up)
    // ----------------------------------------------------------------------------------------

    #[test]
    fn tick_ignores_nonpositive_or_nonfinite_dt() {
        let mut sim = Simulation::new(256.0, 4, 1);
        let before = sim.sim_steps();
        sim.tick(0.0);
        sim.tick(-1.0);
        sim.tick(f32::NAN);
        sim.tick(f32::INFINITY);
        assert_eq!(sim.sim_steps(), before);
    }

    #[test]
    fn tick_runs_single_step_at_fixed_dt() {
        let mut sim = Simulation::new(256.0, 4, 1);
        sim.tick(FIXED_STEP_SECONDS);
        assert_eq!(sim.sim_steps(), 1);
    }

    #[test]
    fn tick_catches_up_multiple_steps() {
        let mut sim = Simulation::new(256.0, 4, 1);
        sim.tick(FIXED_STEP_SECONDS * 3.5);
        assert_eq!(sim.sim_steps(), 3);
    }

    #[test]
    fn tick_caps_steps_at_max() {
        let mut sim = Simulation::new(256.0, 4, 1);
        sim.tick(1.0);
        assert_eq!(sim.sim_steps(), MAX_STEPS_PER_TICK);
    }

    #[test]
    fn tick_preserves_remainder_for_wall_clock_rate() {
        let mut sim = Simulation::new(256.0, 4, 1);
        for _ in 0..100 {
            sim.tick(1.0 / 100.0);
        }
        let steps = sim.sim_steps();
        assert!((59..=60).contains(&steps), "expected ~60 steps, got {steps}");
    }

    #[test]
    fn speed_multiplier_runs_multiple_steps() {
        let mut sim = Simulation::new(256.0, 4, 1);
        sim.tick(FIXED_STEP_SECONDS * 5.5);
        assert_eq!(sim.sim_steps(), 5);
    }

    #[test]
    fn advance_steps_zero_is_noop() {
        let mut sim = Simulation::new(256.0, 4, 1);
        let before_steps = sim.sim_steps();
        let before_render = sim.render_agents.clone();

        sim.advance_steps(0);

        assert_eq!(sim.sim_steps(), before_steps);
        assert_eq!(sim.render_agents, before_render);
    }

    #[test]
    fn advance_steps_runs_exact_fixed_steps() {
        let mut sim = Simulation::new(256.0, 4, 1);

        sim.advance_steps(1);
        assert_eq!(sim.sim_steps(), 1);

        sim.advance_steps(5);
        assert_eq!(sim.sim_steps(), 6);
        assert_eq!(sim.population(), 4);
        assert_core_invariants(&sim);
    }

    #[test]
    fn advance_steps_ignores_tick_accumulator() {
        let mut sim = Simulation::new(256.0, 4, 1);
        sim.tick(FIXED_STEP_SECONDS * 0.5);

        sim.advance_steps(3);

        assert_eq!(sim.sim_steps(), 3);
    }

    #[test]
    fn step_advances_counter_and_keeps_population() {
        let mut sim = Simulation::new(256.0, 16, 3);
        sim.step(FIXED_STEP_SECONDS);
        assert_eq!(sim.sim_steps(), 1);
        assert_eq!(sim.population(), 16);
        assert_core_invariants(&sim);
    }

    #[test]
    fn tick_refreshes_render_after_step() {
        let mut sim = Simulation::new(256.0, 4, 1);
        sim.tick(FIXED_STEP_SECONDS);
        for i in 0..4 {
            let base = i * RENDER_STRIDE_FLOATS;
            let expected_x = sim.pos_x[i] / sim.world_size;
            assert!((sim.render_agents[base] - expected_x).abs() < 1e-5);
        }
    }

    #[test]
    fn advance_steps_refreshes_render_after_batch() {
        let mut sim = Simulation::new(256.0, 4, 1);
        sim.advance_steps(3);
        for i in 0..4 {
            let base = i * RENDER_STRIDE_FLOATS;
            let expected_x = sim.pos_x[i] / sim.world_size;
            assert!((sim.render_agents[base] - expected_x).abs() < 1e-5);
        }
    }

    // ----------------------------------------------------------------------------------------
    // L. reset
    // ----------------------------------------------------------------------------------------

    #[test]
    fn reset_clears_counters_and_keeps_population() {
        let mut sim = Simulation::new(256.0, 8, 1);
        for _ in 0..30 {
            sim.tick(FIXED_STEP_SECONDS);
        }
        sim.reset(123);
        assert_eq!(sim.births(), 0);
        assert_eq!(sim.deaths(), 0);
        assert_eq!(sim.sim_steps(), 0);
        assert_eq!(sim.generation(), 0);
        assert_eq!(sim.accumulator, 0.0);
        assert_eq!(sim.population(), 8);
        assert_core_invariants(&sim);
    }

    #[test]
    fn reset_with_same_seed_is_deterministic() {
        let mut a = Simulation::new(256.0, 8, 1);
        let mut b = Simulation::new(256.0, 8, 1);
        a.reset(777);
        b.reset(777);
        assert_eq!(a.pos_x, b.pos_x);
        assert_eq!(a.genomes, b.genomes);
        assert_eq!(a.render_agents, b.render_agents);
    }

    // ----------------------------------------------------------------------------------------
    // M. Determinism, render buffer, invariant fuzz
    // ----------------------------------------------------------------------------------------

    #[test]
    fn construction_is_deterministic() {
        let a = Simulation::new(512.0, 32, 2024);
        let b = Simulation::new(512.0, 32, 2024);
        assert_eq!(a.pos_x, b.pos_x);
        assert_eq!(a.dir_x, b.dir_x);
        assert_eq!(a.genomes, b.genomes);
        assert_eq!(a.render_agents, b.render_agents);
    }

    #[test]
    fn identical_dt_sequence_stays_in_lockstep() {
        let mut a = Simulation::new(512.0, 64, 5);
        let mut b = Simulation::new(512.0, 64, 5);
        for _ in 0..120 {
            a.tick(FIXED_STEP_SECONDS);
            b.tick(FIXED_STEP_SECONDS);
        }
        assert_eq!(a.render_agents, b.render_agents);
        assert_eq!(a.births(), b.births());
        assert_eq!(a.deaths(), b.deaths());
        assert_eq!(a.generation(), b.generation());
    }

    #[test]
    fn refresh_render_agents_layout_and_ranges() {
        let mut sim = Simulation::new(200.0, 1, 1);
        place(&mut sim, 0, 50.0, 150.0, 0.6, 0.8);
        sim.speed[0] = (MIN_SPEED + MAX_SPEED) * 0.5;
        sim.color_r[0] = 1.5;
        sim.color_g[0] = -0.5;
        sim.color_b[0] = 0.3;
        sim.refresh_render_agents();

        assert!((sim.render_agents[0] - 50.0 / 200.0).abs() < 1e-6);
        assert!((sim.render_agents[1] - 150.0 / 200.0).abs() < 1e-6);
        assert!((sim.render_agents[2] - 0.6).abs() < 1e-6);
        assert!((sim.render_agents[3] - 0.8).abs() < 1e-6);
        assert_eq!(sim.render_agents[4], 1.0);
        assert_eq!(sim.render_agents[5], 0.0);
        assert!((sim.render_agents[6] - 0.3).abs() < 1e-6);
        assert!((sim.render_agents[7] - 0.5).abs() < 1e-5);
        assert_eq!(sim.agent_stride_f32(), 8);
        assert_eq!(sim.agent_f32_len(), RENDER_STRIDE_FLOATS);
    }

    #[test]
    fn invariants_hold_under_simulation_fuzz() {
        for seed in [1u32, 7, 99, 2024, 55555] {
            let mut sim = Simulation::new(512.0, 200, seed);
            let mut last_births = 0;
            let mut last_deaths = 0;
            for _ in 0..150 {
                sim.tick(FIXED_STEP_SECONDS);
                assert_core_invariants(&sim);
                assert!(sim.births() >= last_births, "births must be monotonic");
                assert!(sim.deaths() >= last_deaths, "deaths must be monotonic");
                last_births = sim.births();
                last_deaths = sim.deaths();
            }
            assert_eq!(sim.population(), 200, "population must stay constant");
        }
    }
}
