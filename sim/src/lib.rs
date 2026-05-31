use wasm_bindgen::prelude::*;

const DEFAULT_WORLD_SIZE: f32 = 4096.0;
const DEFAULT_POPULATION: usize = 10_000;
const MAX_POPULATION: usize = 100_000;

const FIXED_STEP_SECONDS: f32 = 1.0 / 60.0;
const MAX_STEPS_PER_TICK: u32 = 1;

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
    max_generation: u32,
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
            max_generation: 0,
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

        if steps_this_tick == MAX_STEPS_PER_TICK {
            self.accumulator = 0.0;
        }

        self.refresh_render_agents();
    }

    pub fn reset(&mut self, seed: u32) {
        self.accumulator = 0.0;
        self.steps = 0;
        self.births = 0;
        self.deaths = 0;
        self.max_generation = 0;
        self.rng = SmallRng::new(seed as u64);
        self.randomize_all_agents();
        self.refresh_render_agents();
    }

    pub fn world_size(&self) -> f32 {
        self.world_size
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
        self.max_generation
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
            if let Some(partner) = self.find_mate_partner(index) {
                if self.mate_partner[index] == partner as i32 {
                    self.mate_timer[index] += dt;
                } else {
                    self.mate_partner[index] = partner as i32;
                    self.mate_timer[index] = dt;
                }
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

                    if other == index || self.age[other] < MIN_MATE_AGE_SECONDS {
                        continue;
                    }

                    let alignment = self.dir_x[index] * self.dir_x[other]
                        + self.dir_y[index] * self.dir_y[other];
                    if alignment < MATE_ALIGNMENT_DOT {
                        continue;
                    }

                    let dx = wrap_delta(self.pos_x[other] - self.pos_x[index], self.world_size);
                    let dy = wrap_delta(self.pos_y[other] - self.pos_y[index], self.world_size);
                    let distance_squared = dx * dx + dy * dy;

                    if distance_squared <= best_distance_squared {
                        best_distance_squared = distance_squared;
                        best_partner = Some(other);
                    }
                }
            }
        }

        best_partner
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
        self.max_generation = self.max_generation.max(generation);
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
}
