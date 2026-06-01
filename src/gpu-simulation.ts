import {
  DEFAULT_POPULATION,
  DEFAULT_WORLD_SIZE,
  FIXED_STEP_SECONDS,
  GENOME_LEN,
  GRID_CELL_SIZE,
  CameraState,
  SimulationStats,
  gridColsForWorld,
  sanitizePopulation,
  sanitizeWorldSize,
} from "./simulation-helpers";

const AGENT_STRUCT_BYTES = 96;
const PARAMS_BYTES = 32;
const VIEW_BYTES = 32;
const META_U32_COUNT = 8;
const META_BYTES = META_U32_COUNT * Uint32Array.BYTES_PER_ELEMENT;
const AGENT_WORKGROUP_SIZE = 128;
const GRID_WORKGROUP_SIZE = 256;
const TILE_GRID_WIDTH = 3;
const TILE_COPY_COUNT = TILE_GRID_WIDTH * TILE_GRID_WIDTH;
const ARROW_LENGTH_WORLD_UNITS = 18;
const ARROW_LOCAL_LENGTH = 2.2;
const GLOW_RADIUS_LOCAL = 6.0;

const META_POPULATION = 0;
const META_BIRTHS = 1;
const META_DEATHS = 2;
const META_GENERATION = 3;
const META_SIM_STEPS = 4;
const META_HIGHLIGHT_INDEX = 5;
const META_HIGHLIGHT_AGE_TICKS = 6;
const META_HIGHLIGHT_PACKED = 7;

export type GpuSimulationParams = {
  worldSize?: number;
  population?: number;
  seed?: number;
  format?: GPUTextureFormat;
};

export type DebugAgentState = {
  x: number;
  y: number;
  dirX: number;
  dirY: number;
  r: number;
  g: number;
  b: number;
  speed: number;
  ageTicks: number;
  generation: number;
  matePartnerPlusOne?: number;
  mateTimerTicks?: number;
  rngState?: number;
};

export type DebugCollisionProbe = {
  deathFlags: number[];
  heads: number[];
  next: number[];
};

type ComputePipelines = {
  reset: GPUComputePipeline;
  clearGrid: GPUComputePipeline;
  countGrid: GPUComputePipeline;
  decide: GPUComputePipeline;
  integrate: GPUComputePipeline;
  collide: GPUComputePipeline;
  applyDeaths: GPUComputePipeline;
  updateBreeding: GPUComputePipeline;
  resolveBreeding: GPUComputePipeline;
  finalizeStep: GPUComputePipeline;
  finalizeAgents: GPUComputePipeline;
  finalizeHighlight: GPUComputePipeline;
};

type RenderPipelines = {
  agents: GPURenderPipeline;
  grid: GPURenderPipeline;
  glow: GPURenderPipeline;
};

function roundUpTo(value: number, alignment: number): number {
  return Math.ceil(value / alignment) * alignment;
}

function dispatchCount(itemCount: number, workgroupSize: number): number {
  return Math.max(1, Math.ceil(itemCount / workgroupSize));
}

function writeMixedParams(
  buffer: ArrayBuffer,
  worldSize: number,
  population: number,
  gridCols: number,
  seed: number,
): void {
  const view = new DataView(buffer);
  view.setFloat32(0, worldSize, true);
  view.setFloat32(4, 1 / worldSize, true);
  view.setFloat32(8, FIXED_STEP_SECONDS, true);
  view.setFloat32(12, Math.min(1, 4 * FIXED_STEP_SECONDS), true);
  view.setUint32(16, population, true);
  view.setUint32(20, gridCols, true);
  view.setUint32(24, gridCols * gridCols, true);
  view.setUint32(28, seed >>> 0, true);
}

function createBuffer(
  device: GPUDevice,
  label: string,
  size: number,
  usage: GPUBufferUsageFlags,
): GPUBuffer {
  return device.createBuffer({
    label,
    size: Math.max(4, roundUpTo(size, 4)),
    usage,
  });
}

const SIMULATION_SHADER = `
const GENOME_LEN: u32 = ${GENOME_LEN}u;
const INPUT_COUNT: u32 = 59u;
const HIDDEN_COUNT: u32 = 8u;
const OUTPUT_COUNT: u32 = 5u;
const RAY_COUNT: u32 = 9u;
const RAY_INPUTS: u32 = 6u;
const GRID_CELL_SIZE: f32 = ${GRID_CELL_SIZE.toFixed(1)};
const INV_GRID_CELL_SIZE: f32 = ${(1 / GRID_CELL_SIZE).toFixed(8)};
const MIN_SPEED: f32 = 15.0;
const MAX_SPEED: f32 = 80.0;
const ACCELERATION: f32 = 70.0;
const MAX_TURN_RATE: f32 = 4.0;
const VISION_RANGE: f32 = 96.0;
const RAY_SAMPLE_STEP: f32 = 16.0;
const RAY_SAMPLE_COUNT: u32 = 6u;
const RAY_HALF_WIDTH: f32 = 7.0;
const RAY_FORWARD_SLOP: f32 = 0.055;
const AGENT_LENGTH: f32 = 10.0;
const HEAD_OFFSET: f32 = AGENT_LENGTH * 0.55;
const BODY_BACK: f32 = -HEAD_OFFSET * 0.6;
const BODY_FRONT: f32 = HEAD_OFFSET * 0.2;
const BODY_HALF_WIDTH: f32 = 2.0;
const HEAD_ON_RADIUS_SQUARED: f32 = 20.25;
const HEAD_ON_DOT: f32 = -0.75;
const COLLISION_GRACE_TICKS: u32 = 60u;
const MATE_RADIUS_SQUARED: f32 = 256.0;
const MATE_ALIGNMENT_DOT: f32 = 0.85;
const MIN_MATE_AGE_TICKS: u32 = 120u;
const MATE_DURATION_TICKS: u32 = 48u;
const AGE_INPUT_CAP_TICKS: f32 = 1800.0;
const MUTATION_RATE: f32 = 0.03;
const MUTATION_MAGNITUDE: f32 = 0.25;
const GENE_LIMIT: f32 = 4.0;
const EMPTY_CELL: u32 = 0xffffffffu;
const PI: f32 = 3.141592653589793;
const FRAC_1_PI: f32 = 0.3183098861837907;

const RAY_COS: array<f32, 9> = array<f32, 9>(
  0.0,
  0.5,
  0.76604444,
  0.9396926,
  1.0,
  0.9396926,
  0.76604444,
  0.5,
  0.0,
);
const RAY_SIN: array<f32, 9> = array<f32, 9>(
  -1.0,
  -0.8660254,
  -0.64278764,
  -0.34202015,
  0.0,
  0.34202015,
  0.64278764,
  0.8660254,
  1.0,
);

struct Params {
  world_size: f32,
  inv_world_size: f32,
  fixed_dt: f32,
  color_blend: f32,
  population: u32,
  grid_cols: u32,
  grid_len: u32,
  seed: u32,
};

struct Agent {
  pose: vec4f,
  color_speed: vec4f,
  state: vec4u,
  command: vec4f,
  extra: vec4f,
  rng: vec4u,
};

struct RayHit {
  distance: f32,
  color: vec3f,
  dir: vec2f,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read_write> agents: array<Agent>;
@group(0) @binding(2) var<storage, read_write> genomes: array<f32>;
@group(0) @binding(3) var<storage, read_write> grid_counts: array<atomic<u32>>;
@group(0) @binding(4) var<storage, read_write> grid_offsets: array<u32>;
@group(0) @binding(5) var<storage, read_write> death_flags: array<u32>;
@group(0) @binding(6) var<storage, read_write> sim_meta: array<atomic<u32>>;

fn mix_seed(seed: u32, index: u32) -> u32 {
  var value = seed ^ (index * 0x9e3779b9u) ^ 0x85ebca6bu;
  value = (value ^ (value >> 16u)) * 0x7feb352du;
  value = (value ^ (value >> 15u)) * 0x846ca68bu;
  return value ^ (value >> 16u);
}

fn next_u32(state: ptr<function, u32>) -> u32 {
  *state = *state + 0x9e3779b9u;
  var value = *state;
  value = (value ^ (value >> 16u)) * 0x21f0aaadu;
  value = (value ^ (value >> 15u)) * 0x735a2d97u;
  return value ^ (value >> 15u);
}

fn next_f32(state: ptr<function, u32>) -> f32 {
  return f32(next_u32(state) >> 8u) * (1.0 / 16777216.0);
}

fn next_signed_f32(state: ptr<function, u32>) -> f32 {
  return next_f32(state) * 2.0 - 1.0;
}

fn next_bool(state: ptr<function, u32>) -> bool {
  return (next_u32(state) & 1u) == 1u;
}

fn wrap_near(value: f32) -> f32 {
  if (value >= params.world_size) {
    return value - params.world_size;
  }
  if (value < 0.0) {
    return value + params.world_size;
  }
  return value;
}

fn wrap_delta(delta: f32) -> f32 {
  let half = params.world_size * 0.5;
  if (delta > half) {
    return delta - params.world_size;
  }
  if (delta < -half) {
    return delta + params.world_size;
  }
  return delta;
}

fn wrapped_cell_index(cell_x: i32, cell_y: i32) -> u32 {
  let cols = i32(params.grid_cols);
  let x = ((cell_x % cols) + cols) % cols;
  let y = ((cell_y % cols) + cols) % cols;
  return u32(y) * params.grid_cols + u32(x);
}

fn cell_coord(value: f32) -> i32 {
  return i32(floor(value * INV_GRID_CELL_SIZE));
}

fn cell_index_from_position(position: vec2f) -> u32 {
  return wrapped_cell_index(cell_coord(position.x), cell_coord(position.y));
}

fn fast_tanh(x: f32) -> f32 {
  if (x < -3.0) {
    return -1.0;
  }
  if (x > 3.0) {
    return 1.0;
  }
  let x2 = x * x;
  return clamp(x * (27.0 + x2) / (27.0 + 9.0 * x2), -1.0, 1.0);
}

fn output_to_color(output: f32) -> f32 {
  return clamp(output * 0.5 + 0.5, 0.0, 1.0);
}

fn relative_heading(self_dir: vec2f, other_dir: vec2f) -> f32 {
  let dot_value = dot(self_dir, other_dir);
  let cross_value = self_dir.x * other_dir.y - self_dir.y * other_dir.x;
  return atan2(cross_value, dot_value) * FRAC_1_PI;
}

fn miss_ray() -> RayHit {
  var hit: RayHit;
  hit.distance = VISION_RANGE;
  hit.color = vec3f(0.0);
  hit.dir = vec2f(1.0, 0.0);
  return hit;
}

fn cast_ray(index: u32, origin: vec2f, self_dir: vec2f, ray_dir: vec2f) -> RayHit {
  var hit = miss_ray();
  var previous_cell = EMPTY_CELL;

  for (var sample = 1u; sample <= RAY_SAMPLE_COUNT; sample = sample + 1u) {
    let sample_distance = f32(sample) * RAY_SAMPLE_STEP;
    let sample_position = vec2f(
      wrap_near(origin.x + ray_dir.x * sample_distance),
      wrap_near(origin.y + ray_dir.y * sample_distance),
    );
    let cell = cell_index_from_position(sample_position);

    if (cell == previous_cell) {
      continue;
    }
    previous_cell = cell;

    var cursor = atomicLoad(&grid_counts[cell]);
    while (cursor != EMPTY_CELL) {
      let other = cursor;
      cursor = grid_offsets[other];
      if (other == index || other >= params.population) {
        continue;
      }

      let other_agent = agents[other];
      let delta = vec2f(
        wrap_delta(other_agent.pose.x - origin.x),
        wrap_delta(other_agent.pose.y - origin.y),
      );
      let forward = dot(delta, ray_dir);
      if (forward <= 0.0 || forward > VISION_RANGE || forward >= hit.distance) {
        continue;
      }

      let lateral = abs(delta.x * -ray_dir.y + delta.y * ray_dir.x);
      let width = RAY_HALF_WIDTH + forward * RAY_FORWARD_SLOP;
      if (lateral <= width) {
        hit.distance = forward;
        hit.color = other_agent.color_speed.rgb;
        hit.dir = other_agent.pose.zw;
      }
    }

    if (hit.distance < VISION_RANGE) {
      break;
    }
  }

  return hit;
}

fn randomize_genome(index: u32, state: ptr<function, u32>) {
  let base = index * GENOME_LEN;
  for (var gene = 0u; gene < GENOME_LEN; gene = gene + 1u) {
    genomes[base + gene] = next_signed_f32(state) * 0.75;
  }
}

fn randomize_agent(index: u32, state: ptr<function, u32>, generation: u32) {
  randomize_genome(index, state);

  let angle = next_f32(state) * PI * 2.0;
  let speed = MIN_SPEED + next_f32(state) * (MAX_SPEED - MIN_SPEED);

  var agent: Agent;
  agent.pose = vec4f(
    next_f32(state) * params.world_size,
    next_f32(state) * params.world_size,
    cos(angle),
    sin(angle),
  );
  agent.color_speed = vec4f(
    0.25 + next_f32(state) * 0.75,
    0.25 + next_f32(state) * 0.75,
    0.25 + next_f32(state) * 0.75,
    speed,
  );
  agent.state = vec4u(0u, generation, 0u, 0u);
  agent.command = vec4f(0.0, 0.0, agent.color_speed.r, agent.color_speed.g);
  agent.extra = vec4f(agent.color_speed.b, 0.0, 0.0, 0.0);
  agent.rng = vec4u(*state, 0u, 0u, 0u);
  agents[index] = agent;
}

fn hits_body_side(head: vec2f, target_agent: Agent) -> bool {
  let delta = vec2f(
    wrap_delta(head.x - target_agent.pose.x),
    wrap_delta(head.y - target_agent.pose.y),
  );
  let target_dir = target_agent.pose.zw;
  let forward = dot(delta, target_dir);
  let lateral = abs(delta.x * -target_dir.y + delta.y * target_dir.x);
  return forward >= BODY_BACK && forward <= BODY_FRONT && lateral <= BODY_HALF_WIDTH;
}

fn is_head_on(attacker: Agent, target_agent: Agent, attacker_head: vec2f) -> bool {
  if (dot(attacker.pose.zw, target_agent.pose.zw) > HEAD_ON_DOT) {
    return false;
  }

  let target_head = vec2f(
    wrap_near(target_agent.pose.x + target_agent.pose.z * HEAD_OFFSET),
    wrap_near(target_agent.pose.y + target_agent.pose.w * HEAD_OFFSET),
  );
  let delta = vec2f(
    wrap_delta(target_head.x - attacker_head.x),
    wrap_delta(target_head.y - attacker_head.y),
  );
  return dot(delta, delta) <= HEAD_ON_RADIUS_SQUARED;
}

fn mate_distance_if_eligible(a_index: u32, b_index: u32) -> f32 {
  if (a_index == b_index || b_index >= params.population) {
    return -1.0;
  }

  if (death_flags[a_index] != 0u || death_flags[b_index] != 0u) {
    return -1.0;
  }

  let a = agents[a_index];
  let b = agents[b_index];
  if (a.state.x < MIN_MATE_AGE_TICKS || b.state.x < MIN_MATE_AGE_TICKS) {
    return -1.0;
  }

  if (dot(a.pose.zw, b.pose.zw) < MATE_ALIGNMENT_DOT) {
    return -1.0;
  }

  let delta = vec2f(wrap_delta(b.pose.x - a.pose.x), wrap_delta(b.pose.y - a.pose.y));
  let distance_squared = dot(delta, delta);
  if (distance_squared > MATE_RADIUS_SQUARED) {
    return -1.0;
  }

  return distance_squared;
}

fn find_mate_partner(index: u32) -> u32 {
  if (agents[index].state.x < MIN_MATE_AGE_TICKS) {
    return 0u;
  }

  let center_x = cell_coord(agents[index].pose.x);
  let center_y = cell_coord(agents[index].pose.y);
  var best_partner_plus_one = 0u;
  var best_distance_squared = MATE_RADIUS_SQUARED;

  for (var cell_y = center_y - 1; cell_y <= center_y + 1; cell_y = cell_y + 1) {
    for (var cell_x = center_x - 1; cell_x <= center_x + 1; cell_x = cell_x + 1) {
      let cell = wrapped_cell_index(cell_x, cell_y);
      var cursor = atomicLoad(&grid_counts[cell]);
      while (cursor != EMPTY_CELL) {
        let other = cursor;
        cursor = grid_offsets[other];
        let distance_squared = mate_distance_if_eligible(index, other);
        if (distance_squared >= 0.0 && distance_squared <= best_distance_squared) {
          best_distance_squared = distance_squared;
          best_partner_plus_one = other + 1u;
        }
      }
    }
  }

  return best_partner_plus_one;
}

@compute @workgroup_size(${AGENT_WORKGROUP_SIZE})
fn reset_all(@builtin(global_invocation_id) global_id: vec3u) {
  let index = global_id.x;
  if (index == 0u) {
    atomicStore(&sim_meta[${META_POPULATION}], params.population);
    atomicStore(&sim_meta[${META_BIRTHS}], 0u);
    atomicStore(&sim_meta[${META_DEATHS}], 0u);
    atomicStore(&sim_meta[${META_GENERATION}], 0u);
    atomicStore(&sim_meta[${META_SIM_STEPS}], 0u);
    atomicStore(&sim_meta[${META_HIGHLIGHT_INDEX}], 0u);
    atomicStore(&sim_meta[${META_HIGHLIGHT_AGE_TICKS}], 0u);
  }

  if (index >= params.population) {
    return;
  }

  var state = mix_seed(params.seed, index);
  randomize_agent(index, &state, 0u);
  death_flags[index] = 0u;
}

@compute @workgroup_size(${GRID_WORKGROUP_SIZE})
fn clear_grid(@builtin(global_invocation_id) global_id: vec3u) {
  let index = global_id.x;
  if (index < params.grid_len) {
    atomicStore(&grid_counts[index], EMPTY_CELL);
  }
  if (index < params.population) {
    death_flags[index] = 0u;
    grid_offsets[index] = EMPTY_CELL;
  }
}

@compute @workgroup_size(${AGENT_WORKGROUP_SIZE})
fn count_grid(@builtin(global_invocation_id) global_id: vec3u) {
  let index = global_id.x;
  if (index >= params.population) {
    return;
  }

  let cell = cell_index_from_position(agents[index].pose.xy);
  let previous_head = atomicExchange(&grid_counts[cell], index);
  grid_offsets[index] = previous_head;
}

@compute @workgroup_size(${AGENT_WORKGROUP_SIZE})
fn decide(@builtin(global_invocation_id) global_id: vec3u) {
  let index = global_id.x;
  if (index >= params.population) {
    return;
  }

  let agent = agents[index];
  let self_dir = agent.pose.zw;
  let genome_base = index * GENOME_LEN;
  var hidden: array<f32, 8>;
  for (var h = 0u; h < HIDDEN_COUNT; h = h + 1u) {
    hidden[h] = 0.0;
  }

  for (var ray = 0u; ray < RAY_COUNT; ray = ray + 1u) {
    let ray_dir = vec2f(
      self_dir.x * RAY_COS[ray] - self_dir.y * RAY_SIN[ray],
      self_dir.x * RAY_SIN[ray] + self_dir.y * RAY_COS[ray],
    );
    let hit = cast_ray(index, agent.pose.xy, self_dir, ray_dir);

    if (hit.distance < VISION_RANGE) {
      let base = ray * RAY_INPUTS;
      let distance_input = 1.0 - hit.distance / VISION_RANGE;
      let heading_input = relative_heading(self_dir, hit.dir);
      for (var h = 0u; h < HIDDEN_COUNT; h = h + 1u) {
        let weights = genome_base + h * (INPUT_COUNT + 1u) + base;
        hidden[h] = hidden[h]
          + genomes[weights]
          + distance_input * genomes[weights + 1u]
          + hit.color.r * genomes[weights + 2u]
          + hit.color.g * genomes[weights + 3u]
          + hit.color.b * genomes[weights + 4u]
          + heading_input * genomes[weights + 5u];
      }
    }
  }

  let self_base = RAY_COUNT * RAY_INPUTS;
  let speed_input = clamp((agent.color_speed.a - MIN_SPEED) / (MAX_SPEED - MIN_SPEED), 0.0, 1.0);
  let age_input = clamp(f32(agent.state.x) / AGE_INPUT_CAP_TICKS, 0.0, 1.0);
  for (var h = 0u; h < HIDDEN_COUNT; h = h + 1u) {
    let weights = genome_base + h * (INPUT_COUNT + 1u) + self_base;
    hidden[h] = fast_tanh(hidden[h]
      + speed_input * genomes[weights]
      + age_input * genomes[weights + 1u]
      + agent.color_speed.r * genomes[weights + 2u]
      + agent.color_speed.g * genomes[weights + 3u]
      + agent.color_speed.b * genomes[weights + 4u]
      + genomes[weights + 5u]);
  }

  var outputs: array<f32, 5>;
  var cursor = genome_base + HIDDEN_COUNT * (INPUT_COUNT + 1u);
  for (var output_index = 0u; output_index < OUTPUT_COUNT; output_index = output_index + 1u) {
    var sum = 0.0;
    for (var hidden_index = 0u; hidden_index < HIDDEN_COUNT; hidden_index = hidden_index + 1u) {
      sum = sum + hidden[hidden_index] * genomes[cursor];
      cursor = cursor + 1u;
    }
    sum = sum + genomes[cursor];
    cursor = cursor + 1u;
    outputs[output_index] = fast_tanh(sum);
  }

  agents[index].command = vec4f(
    outputs[0],
    outputs[1],
    output_to_color(outputs[2]),
    output_to_color(outputs[3]),
  );
  agents[index].extra.x = output_to_color(outputs[4]);
}

@compute @workgroup_size(${AGENT_WORKGROUP_SIZE})
fn integrate(@builtin(global_invocation_id) global_id: vec3u) {
  let index = global_id.x;
  if (index >= params.population) {
    return;
  }

  if (death_flags[index] != 0u) {
    agents[index].state.z = 0u;
    agents[index].state.w = 0u;
    return;
  }

  var agent = agents[index];
  let turn = clamp(agent.command.x, -1.0, 1.0) * MAX_TURN_RATE * params.fixed_dt;
  let cos_turn = cos(turn);
  let sin_turn = sin(turn);
  let old_dir = agent.pose.zw;
  var new_dir = vec2f(
    old_dir.x * cos_turn - old_dir.y * sin_turn,
    old_dir.x * sin_turn + old_dir.y * cos_turn,
  );
  new_dir = normalize(new_dir);

  let speed = clamp(
    agent.color_speed.a + clamp(agent.command.y, -1.0, 1.0) * ACCELERATION * params.fixed_dt,
    MIN_SPEED,
    MAX_SPEED,
  );
  let next_position = vec2f(
    wrap_near(agent.pose.x + new_dir.x * speed * params.fixed_dt),
    wrap_near(agent.pose.y + new_dir.y * speed * params.fixed_dt),
  );

  agent.pose = vec4f(next_position, new_dir);
  agent.color_speed = vec4f(
    agent.color_speed.r + (agent.command.z - agent.color_speed.r) * params.color_blend,
    agent.color_speed.g + (agent.command.w - agent.color_speed.g) * params.color_blend,
    agent.color_speed.b + (agent.extra.x - agent.color_speed.b) * params.color_blend,
    speed,
  );
  agent.state.x = agent.state.x + 1u;
  agents[index] = agent;
}

@compute @workgroup_size(${AGENT_WORKGROUP_SIZE})
fn collide(@builtin(global_invocation_id) global_id: vec3u) {
  let index = global_id.x;
  if (index >= params.population) {
    return;
  }

  let attacker = agents[index];
  if (attacker.state.x < COLLISION_GRACE_TICKS) {
    return;
  }

  let head = vec2f(
    wrap_near(attacker.pose.x + attacker.pose.z * HEAD_OFFSET),
    wrap_near(attacker.pose.y + attacker.pose.w * HEAD_OFFSET),
  );
  let center_x = cell_coord(head.x);
  let center_y = cell_coord(head.y);

  for (var cell_y = center_y - 1; cell_y <= center_y + 1; cell_y = cell_y + 1) {
    for (var cell_x = center_x - 1; cell_x <= center_x + 1; cell_x = cell_x + 1) {
      let cell = wrapped_cell_index(cell_x, cell_y);
      var cursor = atomicLoad(&grid_counts[cell]);
      while (cursor != EMPTY_CELL) {
        let other_index = cursor;
        cursor = grid_offsets[other_index];
        if (other_index == index || other_index >= params.population) {
          continue;
        }

        let target_agent = agents[other_index];
        if (target_agent.state.x < COLLISION_GRACE_TICKS) {
          continue;
        }

        if (hits_body_side(head, target_agent) && !is_head_on(attacker, target_agent, head)) {
          death_flags[index] = 1u;
          return;
        }
      }
    }
  }
}

@compute @workgroup_size(${AGENT_WORKGROUP_SIZE})
fn apply_deaths(@builtin(global_invocation_id) global_id: vec3u) {
  let index = global_id.x;
  if (index >= params.population || death_flags[index] == 0u) {
    return;
  }

  var state = agents[index].rng.x + 0xa511e9b3u + atomicLoad(&sim_meta[${META_SIM_STEPS}]);
  randomize_agent(index, &state, 0u);
  _ = atomicAdd(&sim_meta[${META_DEATHS}], 1u);
  death_flags[index] = 0u;
}

@compute @workgroup_size(${AGENT_WORKGROUP_SIZE})
fn update_breeding(@builtin(global_invocation_id) global_id: vec3u) {
  let index = global_id.x;
  if (index >= params.population) {
    return;
  }

  var agent = agents[index];
  let current_partner_plus_one = agent.state.z;
  if (current_partner_plus_one > 0u) {
    let partner = current_partner_plus_one - 1u;
    if (mate_distance_if_eligible(index, partner) >= 0.0) {
      agent.state.w = agent.state.w + 1u;
      agents[index] = agent;
      return;
    }
  }

  let partner_plus_one = find_mate_partner(index);
  agent.state.z = partner_plus_one;
  agent.state.w = select(0u, 1u, partner_plus_one > 0u);
  agents[index] = agent;
}

@compute @workgroup_size(${AGENT_WORKGROUP_SIZE})
fn resolve_breeding(@builtin(global_invocation_id) global_id: vec3u) {
  let index = global_id.x;
  if (index >= params.population) {
    return;
  }

  if (death_flags[index] != 0u) {
    return;
  }

  let parent_a = agents[index];
  let partner_plus_one = parent_a.state.z;
  if (partner_plus_one == 0u) {
    return;
  }

  let partner_index = partner_plus_one - 1u;
  if (index >= partner_index || partner_index >= params.population) {
    return;
  }

  if (death_flags[partner_index] != 0u) {
    return;
  }

  let parent_b = agents[partner_index];
  if (
    parent_b.state.z != index + 1u ||
    parent_a.state.w < MATE_DURATION_TICKS ||
    parent_b.state.w < MATE_DURATION_TICKS
  ) {
    return;
  }

  var state = parent_a.rng.x ^ parent_b.rng.x ^ atomicLoad(&sim_meta[${META_SIM_STEPS}]) ^ 0x68bc21ebu;
  let victim = select(partner_index, index, next_bool(&state));
  let victim_agent = agents[victim];
  let child_generation = max(parent_a.state.y, parent_b.state.y) + 1u;
  let destination = victim * GENOME_LEN;
  let parent_a_genome = index * GENOME_LEN;
  let parent_b_genome = partner_index * GENOME_LEN;

  for (var gene = 0u; gene < GENOME_LEN; gene = gene + 1u) {
    let source_base = select(parent_b_genome, parent_a_genome, next_bool(&state));
    var value = genomes[source_base + gene];
    if (next_f32(&state) < MUTATION_RATE) {
      value = clamp(value + next_signed_f32(&state) * MUTATION_MAGNITUDE, -GENE_LIMIT, GENE_LIMIT);
    }
    genomes[destination + gene] = value;
  }

  var direction = parent_a.pose.zw + parent_b.pose.zw;
  let direction_len = length(direction);
  if (direction_len > 0.0001) {
    direction = direction / direction_len;
  } else {
    let angle = next_f32(&state) * PI * 2.0;
    direction = vec2f(cos(angle), sin(angle));
  }

  var child = victim_agent;
  child.pose = vec4f(
    wrap_near(victim_agent.pose.x + next_signed_f32(&state) * 1.5),
    wrap_near(victim_agent.pose.y + next_signed_f32(&state) * 1.5),
    direction,
  );
  child.color_speed = vec4f(
    clamp((parent_a.color_speed.r + parent_b.color_speed.r) * 0.5, 0.0, 1.0),
    clamp((parent_a.color_speed.g + parent_b.color_speed.g) * 0.5, 0.0, 1.0),
    clamp((parent_a.color_speed.b + parent_b.color_speed.b) * 0.5, 0.0, 1.0),
    clamp((parent_a.color_speed.a + parent_b.color_speed.a) * 0.5, MIN_SPEED, MAX_SPEED),
  );
  child.state = vec4u(0u, child_generation, 0u, 0u);
  child.command = vec4f(0.0, 0.0, child.color_speed.r, child.color_speed.g);
  child.extra = vec4f(child.color_speed.b, 0.0, 0.0, 0.0);
  child.rng = vec4u(state, 0u, 0u, 0u);
  agents[victim] = child;

  agents[index].state.z = 0u;
  agents[index].state.w = 0u;
  agents[partner_index].state.z = 0u;
  agents[partner_index].state.w = 0u;
  _ = atomicAdd(&sim_meta[${META_BIRTHS}], 1u);
}

@compute @workgroup_size(1)
fn finalize_step() {
  _ = atomicAdd(&sim_meta[${META_SIM_STEPS}], 1u);
  atomicStore(&sim_meta[${META_POPULATION}], params.population);
  atomicStore(&sim_meta[${META_GENERATION}], 0u);
  atomicStore(&sim_meta[${META_HIGHLIGHT_PACKED}], 0u);
}

@compute @workgroup_size(${AGENT_WORKGROUP_SIZE})
fn finalize_agents(@builtin(global_invocation_id) global_id: vec3u) {
  let index = global_id.x;
  if (index >= params.population) {
    return;
  }
  let agent = agents[index];
  _ = atomicMax(&sim_meta[${META_GENERATION}], agent.state.y);
  _ = atomicMax(&sim_meta[${META_HIGHLIGHT_PACKED}], agent.state.x * params.population + (params.population - 1u - index));
}

@compute @workgroup_size(1)
fn finalize_highlight() {
  let packed = atomicLoad(&sim_meta[${META_HIGHLIGHT_PACKED}]);
  atomicStore(&sim_meta[${META_HIGHLIGHT_AGE_TICKS}], packed / params.population);
  atomicStore(&sim_meta[${META_HIGHLIGHT_INDEX}], params.population - 1u - (packed % params.population));
}
`;

const AGENT_RENDER_SHADER = `
const TILE_GRID_WIDTH: u32 = ${TILE_GRID_WIDTH}u;
const TILE_COPY_COUNT: u32 = ${TILE_COPY_COUNT}u;
const TILE_CENTER: i32 = ${Math.floor(TILE_GRID_WIDTH / 2)};

struct Agent {
  pose: vec4f,
  color_speed: vec4f,
  state: vec4u,
  command: vec4f,
  extra: vec4f,
  rng: vec4u,
};

struct View {
  aspect: f32,
  arrow_scale: f32,
  camera_x: f32,
  camera_y: f32,
  zoom: f32,
  inv_world_size: f32,
  _pad1: f32,
  _pad2: f32,
};

struct VertexOutput {
  @builtin(position) position: vec4f,
  @location(0) color: vec4f,
};

@group(0) @binding(0) var<storage, read> agents: array<Agent>;
@group(0) @binding(1) var<uniform> view: View;
@group(0) @binding(2) var<storage, read> sim_meta: array<u32>;

fn delta_to_clip(delta: vec2f) -> vec2f {
  let centered = delta * (2.0 * view.zoom);
  if (view.aspect > 1.0) {
    return vec2f(centered.x / view.aspect, -centered.y);
  }

  return vec2f(centered.x, -centered.y * view.aspect);
}

@vertex
fn vertex_main(
  @builtin(vertex_index) vertex_index: u32,
  @builtin(instance_index) instance_index: u32,
) -> VertexOutput {
  let agent_index = instance_index / TILE_COPY_COUNT;
  let tile_index = instance_index % TILE_COPY_COUNT;
  let tile_x = i32(tile_index % TILE_GRID_WIDTH) - TILE_CENTER;
  let tile_y = i32(tile_index / TILE_GRID_WIDTH) - TILE_CENTER;
  let tile_offset = vec2f(f32(tile_x), f32(tile_y));
  let agent = agents[agent_index];

  var local = vec2f(0.0, 0.0);
  if (vertex_index == 0u) {
    local = vec2f(1.35, 0.0);
  } else if (vertex_index == 1u) {
    local = vec2f(-0.85, -0.48);
  } else {
    local = vec2f(-0.85, 0.48);
  }

  let highlight_index = sim_meta[${META_HIGHLIGHT_INDEX}];
  let is_highlight = agent_index == highlight_index;
  let size_scale = select(1.0, 1.35, is_highlight);
  let brightness = select(1.0, 1.7, is_highlight);
  let direction = normalize(agent.pose.zw);
  let side = vec2f(-direction.y, direction.x);
  let normalized_position = agent.pose.xy * view.inv_world_size;
  let agent_delta = normalized_position + tile_offset - vec2f(view.camera_x, view.camera_y);
  let world_delta =
    agent_delta + (direction * local.x + side * local.y) * view.arrow_scale * size_scale;
  let speed_norm = clamp((agent.color_speed.a - 15.0) / 65.0, 0.0, 1.0);
  let speed_glow = 0.65 + speed_norm * 0.35;

  var output: VertexOutput;
  output.position = vec4f(delta_to_clip(world_delta), 0.0, 1.0);
  output.color = vec4f(min(agent.color_speed.rgb * speed_glow * brightness, vec3f(1.0)), 1.0);
  return output;
}

@fragment
fn fragment_main(input: VertexOutput) -> @location(0) vec4f {
  return input.color;
}
`;

const GRID_RENDER_SHADER = `
struct View {
  aspect: f32,
  arrow_scale: f32,
  camera_x: f32,
  camera_y: f32,
  zoom: f32,
  inv_world_size: f32,
  _pad1: f32,
  _pad2: f32,
};

struct VertexOutput {
  @builtin(position) position: vec4f,
  @location(0) clip_position: vec2f,
};

@group(0) @binding(0) var<uniform> view: View;

fn clip_to_world_delta(clip_position: vec2f) -> vec2f {
  let zoom_scale = 2.0 * view.zoom;
  if (view.aspect > 1.0) {
    return vec2f(
      clip_position.x * view.aspect / zoom_scale,
      -clip_position.y / zoom_scale,
    );
  }

  return vec2f(
    clip_position.x / zoom_scale,
    -clip_position.y / (view.aspect * zoom_scale),
  );
}

fn line_alpha(value: f32, spacing: f32, width_pixels: f32) -> f32 {
  let scaled = value / spacing;
  let dist = abs(fract(scaled + 0.5) - 0.5);
  let pixel = max(fwidth(scaled), 0.000001);
  return 1.0 - smoothstep(pixel * width_pixels, pixel * (width_pixels + 1.0), dist);
}

@vertex
fn vertex_main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
  var position = vec2f(-1.0, -1.0);
  if (vertex_index == 1u) {
    position = vec2f(3.0, -1.0);
  } else if (vertex_index == 2u) {
    position = vec2f(-1.0, 3.0);
  }

  var output: VertexOutput;
  output.position = vec4f(position, 0.0, 1.0);
  output.clip_position = position;
  return output;
}

@fragment
fn fragment_main(input: VertexOutput) -> @location(0) vec4f {
  let world = vec2f(view.camera_x, view.camera_y) + clip_to_world_delta(input.clip_position);
  let minor_spacing = 1.0 / 8.0;
  let minor = max(
    line_alpha(world.x, minor_spacing, 0.42),
    line_alpha(world.y, minor_spacing, 0.42),
  );
  let boundary = max(
    line_alpha(world.x, 1.0, 1.15),
    line_alpha(world.y, 1.0, 1.15),
  );
  let color =
    vec3f(0.10, 0.18, 0.20) * minor +
    vec3f(0.58, 0.78, 0.72) * boundary;
  let alpha = min(0.55, minor * 0.18 + boundary * 0.48);

  return vec4f(color, alpha);
}
`;

const GLOW_RENDER_SHADER = `
const TILE_GRID_WIDTH: u32 = ${TILE_GRID_WIDTH}u;
const TILE_CENTER: i32 = ${Math.floor(TILE_GRID_WIDTH / 2)};

struct Agent {
  pose: vec4f,
  color_speed: vec4f,
  state: vec4u,
  command: vec4f,
  extra: vec4f,
  rng: vec4u,
};

struct View {
  aspect: f32,
  arrow_scale: f32,
  camera_x: f32,
  camera_y: f32,
  zoom: f32,
  inv_world_size: f32,
  _pad1: f32,
  _pad2: f32,
};

struct VertexOutput {
  @builtin(position) position: vec4f,
  @location(0) local: vec2f,
};

@group(0) @binding(0) var<storage, read> agents: array<Agent>;
@group(0) @binding(1) var<uniform> view: View;
@group(0) @binding(2) var<storage, read> sim_meta: array<u32>;

fn delta_to_clip(delta: vec2f) -> vec2f {
  let centered = delta * (2.0 * view.zoom);
  if (view.aspect > 1.0) {
    return vec2f(centered.x / view.aspect, -centered.y);
  }

  return vec2f(centered.x, -centered.y * view.aspect);
}

@vertex
fn vertex_main(
  @builtin(vertex_index) vertex_index: u32,
  @builtin(instance_index) instance_index: u32,
) -> VertexOutput {
  let tile_x = i32(instance_index % TILE_GRID_WIDTH) - TILE_CENTER;
  let tile_y = i32(instance_index / TILE_GRID_WIDTH) - TILE_CENTER;
  let tile_offset = vec2f(f32(tile_x), f32(tile_y));
  let agent = agents[sim_meta[${META_HIGHLIGHT_INDEX}]];

  var corners = array<vec2f, 6>(
    vec2f(-1.0, -1.0),
    vec2f(1.0, -1.0),
    vec2f(-1.0, 1.0),
    vec2f(-1.0, 1.0),
    vec2f(1.0, -1.0),
    vec2f(1.0, 1.0),
  );
  let local = corners[vertex_index];
  let glow_radius = view.arrow_scale * ${GLOW_RADIUS_LOCAL.toFixed(1)};
  let normalized_position = agent.pose.xy * view.inv_world_size;
  let agent_delta = normalized_position + tile_offset - vec2f(view.camera_x, view.camera_y);
  let world_delta = agent_delta + local * glow_radius;

  var output: VertexOutput;
  output.position = vec4f(delta_to_clip(world_delta), 0.0, 1.0);
  output.local = local;
  return output;
}

@fragment
fn fragment_main(input: VertexOutput) -> @location(0) vec4f {
  let dist = length(input.local);
  let falloff = pow(clamp(1.0 - dist, 0.0, 1.0), 2.0);
  let color = vec3f(1.0, 0.86, 0.45);
  return vec4f(color * falloff, falloff);
}
`;

export class GpuSimulation {
  readonly worldSize: number;
  readonly population: number;
  readonly gridCols: number;

  private readonly device: GPUDevice;
  private readonly paramsBuffer: GPUBuffer;
  private readonly viewBuffer: GPUBuffer;
  private readonly agentsBuffer: GPUBuffer;
  private readonly genomesBuffer: GPUBuffer;
  private readonly gridCountsBuffer: GPUBuffer;
  private readonly gridOffsetsBuffer: GPUBuffer;
  private readonly deathFlagsBuffer: GPUBuffer;
  private readonly metaBuffer: GPUBuffer;
  private readonly statsReadbackBuffer: GPUBuffer;
  private readonly computePipelines: ComputePipelines;
  private readonly renderPipelines: RenderPipelines | null = null;
  private readonly computeBindGroup: GPUBindGroup;
  private readonly agentRenderBindGroup: GPUBindGroup | null = null;
  private readonly gridRenderBindGroup: GPUBindGroup | null = null;
  private readonly glowRenderBindGroup: GPUBindGroup | null = null;
  private readonly paramsArray = new ArrayBuffer(PARAMS_BYTES);
  private statsPromise: Promise<SimulationStats> | null = null;
  private lastStatsAt = performance.now();
  private lastStatsStepCount = 0;
  private stepsPerSecond = 0;
  private seed: number;

  private constructor(device: GPUDevice, params: Required<Omit<GpuSimulationParams, "format">> & Pick<GpuSimulationParams, "format">) {
    this.device = device;
    this.worldSize = sanitizeWorldSize(params.worldSize);
    this.population = sanitizePopulation(params.population);
    this.gridCols = gridColsForWorld(this.worldSize);
    this.seed = params.seed >>> 0;

    const gridLen = this.gridCols * this.gridCols;
    const agentBytes = this.population * AGENT_STRUCT_BYTES;
    const genomeBytes = this.population * GENOME_LEN * Float32Array.BYTES_PER_ELEMENT;
    const gridBytes = gridLen * Uint32Array.BYTES_PER_ELEMENT;
    const entryBytes = this.population * Uint32Array.BYTES_PER_ELEMENT;

    this.paramsBuffer = createBuffer(device, "simulation-params", PARAMS_BYTES, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
    this.viewBuffer = createBuffer(device, "simulation-view", VIEW_BYTES, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
    this.agentsBuffer = createBuffer(
      device,
      "gpu-agents",
      agentBytes,
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    );
    this.genomesBuffer = createBuffer(
      device,
      "gpu-genomes",
      genomeBytes,
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    );
    this.gridCountsBuffer = createBuffer(
      device,
      "gpu-grid-counts",
      gridBytes,
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    );
    this.gridOffsetsBuffer = createBuffer(
      device,
      "gpu-grid-offsets",
      entryBytes,
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    );
    this.deathFlagsBuffer = createBuffer(
      device,
      "gpu-death-flags",
      entryBytes,
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    );
    this.metaBuffer = createBuffer(device, "gpu-simulation-meta", META_BYTES, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST);
    this.statsReadbackBuffer = createBuffer(device, "gpu-stats-readback", META_BYTES, GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST);

    const simulationModule = device.createShaderModule({
      label: "gpu-simulation-compute-shader",
      code: SIMULATION_SHADER,
    });
    const computeBindGroupLayout = device.createBindGroupLayout({
      label: "gpu-simulation-compute-bind-group-layout",
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
        { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
        { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
        { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
        { binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      ],
    });
    const computePipelineLayout = device.createPipelineLayout({
      label: "gpu-simulation-compute-pipeline-layout",
      bindGroupLayouts: [computeBindGroupLayout],
    });
    this.computePipelines = {
      reset: this.createComputePipeline(simulationModule, "reset_all", computePipelineLayout),
      clearGrid: this.createComputePipeline(simulationModule, "clear_grid", computePipelineLayout),
      countGrid: this.createComputePipeline(simulationModule, "count_grid", computePipelineLayout),
      decide: this.createComputePipeline(simulationModule, "decide", computePipelineLayout),
      integrate: this.createComputePipeline(simulationModule, "integrate", computePipelineLayout),
      collide: this.createComputePipeline(simulationModule, "collide", computePipelineLayout),
      applyDeaths: this.createComputePipeline(simulationModule, "apply_deaths", computePipelineLayout),
      updateBreeding: this.createComputePipeline(simulationModule, "update_breeding", computePipelineLayout),
      resolveBreeding: this.createComputePipeline(simulationModule, "resolve_breeding", computePipelineLayout),
      finalizeStep: this.createComputePipeline(simulationModule, "finalize_step", computePipelineLayout),
      finalizeAgents: this.createComputePipeline(simulationModule, "finalize_agents", computePipelineLayout),
      finalizeHighlight: this.createComputePipeline(simulationModule, "finalize_highlight", computePipelineLayout),
    };

    this.computeBindGroup = device.createBindGroup({
      label: "gpu-simulation-compute-bind-group",
      layout: computeBindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: this.paramsBuffer } },
        { binding: 1, resource: { buffer: this.agentsBuffer } },
        { binding: 2, resource: { buffer: this.genomesBuffer } },
        { binding: 3, resource: { buffer: this.gridCountsBuffer } },
        { binding: 4, resource: { buffer: this.gridOffsetsBuffer } },
        { binding: 5, resource: { buffer: this.deathFlagsBuffer } },
        { binding: 6, resource: { buffer: this.metaBuffer } },
      ],
    });

    if (params.format) {
      this.renderPipelines = this.createRenderPipelines(params.format);
      this.agentRenderBindGroup = device.createBindGroup({
        label: "agent-render-bind-group",
        layout: this.renderPipelines.agents.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: this.agentsBuffer } },
          { binding: 1, resource: { buffer: this.viewBuffer } },
          { binding: 2, resource: { buffer: this.metaBuffer } },
        ],
      });
      this.gridRenderBindGroup = device.createBindGroup({
        label: "grid-render-bind-group",
        layout: this.renderPipelines.grid.getBindGroupLayout(0),
        entries: [{ binding: 0, resource: { buffer: this.viewBuffer } }],
      });
      this.glowRenderBindGroup = device.createBindGroup({
        label: "glow-render-bind-group",
        layout: this.renderPipelines.glow.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: this.agentsBuffer } },
          { binding: 1, resource: { buffer: this.viewBuffer } },
          { binding: 2, resource: { buffer: this.metaBuffer } },
        ],
      });
    }

    this.writeParams(this.seed);
    this.reset(this.seed);
  }

  static async create(device: GPUDevice, params: GpuSimulationParams): Promise<GpuSimulation> {
    return new GpuSimulation(device, {
      worldSize: params.worldSize ?? DEFAULT_WORLD_SIZE,
      population: params.population ?? DEFAULT_POPULATION,
      seed: params.seed ?? 1,
      format: params.format,
    });
  }

  reset(seed: number): void {
    this.seed = seed >>> 0;
    this.writeParams(this.seed);
    this.lastStatsAt = performance.now();
    this.lastStatsStepCount = 0;
    this.stepsPerSecond = 0;

    const encoder = this.device.createCommandEncoder({ label: "gpu-simulation-reset-encoder" });
    let pass = encoder.beginComputePass({ label: "gpu-simulation-reset-pass" });
    pass.setBindGroup(0, this.computeBindGroup);
    pass.setPipeline(this.computePipelines.reset);
    pass.dispatchWorkgroups(dispatchCount(this.population, AGENT_WORKGROUP_SIZE));
    pass.end();
    this.encodeBuildGrid(encoder);
    this.device.queue.submit([encoder.finish()]);
  }

  encodeSteps(encoder: GPUCommandEncoder, stepCount: number): void {
    const steps = Math.max(0, Math.floor(stepCount));
    if (steps === 0) {
      return;
    }

    for (let step = 0; step < steps; step += 1) {
      this.encodeBuildGrid(encoder);

      const pass = encoder.beginComputePass({ label: "gpu-simulation-step-pass" });
      pass.setBindGroup(0, this.computeBindGroup);

      pass.setPipeline(this.computePipelines.decide);
      pass.dispatchWorkgroups(dispatchCount(this.population, AGENT_WORKGROUP_SIZE));

      pass.setPipeline(this.computePipelines.integrate);
      pass.dispatchWorkgroups(dispatchCount(this.population, AGENT_WORKGROUP_SIZE));
      pass.end();

      this.encodeBuildGrid(encoder);

      const interactions = encoder.beginComputePass({ label: "gpu-simulation-interaction-pass" });
      interactions.setBindGroup(0, this.computeBindGroup);

      interactions.setPipeline(this.computePipelines.collide);
      interactions.dispatchWorkgroups(dispatchCount(this.population, AGENT_WORKGROUP_SIZE));

      interactions.setPipeline(this.computePipelines.updateBreeding);
      interactions.dispatchWorkgroups(dispatchCount(this.population, AGENT_WORKGROUP_SIZE));

      interactions.setPipeline(this.computePipelines.resolveBreeding);
      interactions.dispatchWorkgroups(dispatchCount(this.population, AGENT_WORKGROUP_SIZE));

      interactions.setPipeline(this.computePipelines.applyDeaths);
      interactions.dispatchWorkgroups(dispatchCount(this.population, AGENT_WORKGROUP_SIZE));

      interactions.setPipeline(this.computePipelines.finalizeStep);
      interactions.dispatchWorkgroups(1);
      interactions.setPipeline(this.computePipelines.finalizeAgents);
      interactions.dispatchWorkgroups(dispatchCount(this.population, AGENT_WORKGROUP_SIZE));
      interactions.setPipeline(this.computePipelines.finalizeHighlight);
      interactions.dispatchWorkgroups(1);
      interactions.end();
    }
  }

  encodeRender(pass: GPURenderPassEncoder, camera: CameraState, aspect: number): void {
    if (!this.renderPipelines || !this.gridRenderBindGroup || !this.glowRenderBindGroup || !this.agentRenderBindGroup) {
      throw new Error("GpuSimulation was created without render pipelines.");
    }

    this.writeView(camera, aspect);

    pass.setPipeline(this.renderPipelines.grid);
    pass.setBindGroup(0, this.gridRenderBindGroup);
    pass.draw(3);

    pass.setPipeline(this.renderPipelines.glow);
    pass.setBindGroup(0, this.glowRenderBindGroup);
    pass.draw(6, TILE_COPY_COUNT);

    pass.setPipeline(this.renderPipelines.agents);
    pass.setBindGroup(0, this.agentRenderBindGroup);
    pass.draw(3, this.population * TILE_COPY_COUNT);
  }

  async readStatsAsync(): Promise<SimulationStats> {
    if (this.statsPromise) {
      return this.statsPromise;
    }

    const encoder = this.device.createCommandEncoder({ label: "gpu-stats-readback-encoder" });
    encoder.copyBufferToBuffer(this.metaBuffer, 0, this.statsReadbackBuffer, 0, META_BYTES);
    this.device.queue.submit([encoder.finish()]);

    this.statsPromise = this.statsReadbackBuffer
      .mapAsync(GPUMapMode.READ)
      .then(() => {
        const values = new Uint32Array(this.statsReadbackBuffer.getMappedRange().slice(0));
        this.statsReadbackBuffer.unmap();

        const now = performance.now();
        const simSteps = values[META_SIM_STEPS] ?? 0;
        const elapsed = Math.max(1, now - this.lastStatsAt);
        this.stepsPerSecond = Math.round(((simSteps - this.lastStatsStepCount) * 1000) / elapsed);
        this.lastStatsAt = now;
        this.lastStatsStepCount = simSteps;

        return {
          population: values[META_POPULATION] ?? this.population,
          births: values[META_BIRTHS] ?? 0,
          deaths: values[META_DEATHS] ?? 0,
          generation: values[META_GENERATION] ?? 0,
          simSteps,
          stepsPerSecond: this.stepsPerSecond,
        };
      })
      .finally(() => {
        this.statsPromise = null;
      });

    return this.statsPromise;
  }

  debugOverwriteAgents(agents: DebugAgentState[]): void {
    if (agents.length > this.population) {
      throw new Error(`Cannot write ${agents.length} debug agents into population ${this.population}.`);
    }

    const buffer = new ArrayBuffer(agents.length * AGENT_STRUCT_BYTES);
    const view = new DataView(buffer);

    agents.forEach((agent, index) => {
      const base = index * AGENT_STRUCT_BYTES;
      const floats = [
        agent.x,
        agent.y,
        agent.dirX,
        agent.dirY,
        agent.r,
        agent.g,
        agent.b,
        agent.speed,
        0,
        0,
        agent.r,
        agent.g,
        agent.b,
        0,
        0,
        0,
      ];
      const uints = [
        Math.max(0, Math.floor(agent.ageTicks)) >>> 0,
        Math.max(0, Math.floor(agent.generation)) >>> 0,
        Math.max(0, Math.floor(agent.matePartnerPlusOne ?? 0)) >>> 0,
        Math.max(0, Math.floor(agent.mateTimerTicks ?? 0)) >>> 0,
        (agent.rngState ?? (0x9e3779b9 ^ index)) >>> 0,
        0,
        0,
        0,
      ];

      for (let i = 0; i < 8; i += 1) {
        view.setFloat32(base + i * 4, floats[i], true);
      }
      for (let i = 0; i < 4; i += 1) {
        view.setUint32(base + 32 + i * 4, uints[i], true);
      }
      for (let i = 8; i < 16; i += 1) {
        view.setFloat32(base + 16 + i * 4, floats[i], true);
      }
      for (let i = 0; i < 4; i += 1) {
        view.setUint32(base + 80 + i * 4, uints[i + 4], true);
      }
    });

    this.device.queue.writeBuffer(this.agentsBuffer, 0, new Uint8Array(buffer));
  }

  debugZeroGenomes(): void {
    const zeroes = new Uint8Array(this.population * GENOME_LEN * Float32Array.BYTES_PER_ELEMENT);
    this.device.queue.writeBuffer(this.genomesBuffer, 0, zeroes);
  }

  async debugFlushWritesAsync(): Promise<void> {
    await this.device.queue.onSubmittedWorkDone();
  }

  async debugReadAgentsAsync(count = this.population): Promise<DebugAgentState[]> {
    const readCount = Math.min(this.population, Math.max(0, Math.floor(count)));
    const byteSize = readCount * AGENT_STRUCT_BYTES;
    const readback = createBuffer(
      this.device,
      "debug-agent-readback",
      byteSize,
      GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    );
    const encoder = this.device.createCommandEncoder({ label: "debug-agent-readback-encoder" });
    encoder.copyBufferToBuffer(this.agentsBuffer, 0, readback, 0, byteSize);
    this.device.queue.submit([encoder.finish()]);
    await readback.mapAsync(GPUMapMode.READ);

    const bytes = readback.getMappedRange().slice(0);
    const view = new DataView(bytes);
    const agents: DebugAgentState[] = [];

    for (let index = 0; index < readCount; index += 1) {
      const base = index * AGENT_STRUCT_BYTES;
      agents.push({
        x: view.getFloat32(base, true),
        y: view.getFloat32(base + 4, true),
        dirX: view.getFloat32(base + 8, true),
        dirY: view.getFloat32(base + 12, true),
        r: view.getFloat32(base + 16, true),
        g: view.getFloat32(base + 20, true),
        b: view.getFloat32(base + 24, true),
        speed: view.getFloat32(base + 28, true),
        ageTicks: view.getUint32(base + 32, true),
        generation: view.getUint32(base + 36, true),
        matePartnerPlusOne: view.getUint32(base + 40, true),
        mateTimerTicks: view.getUint32(base + 44, true),
        rngState: view.getUint32(base + 80, true),
      });
    }

    readback.unmap();
    readback.destroy();
    return agents;
  }

  async debugProbeCollisionsAsync(): Promise<DebugCollisionProbe> {
    const gridLen = this.gridCols * this.gridCols;
    const headsBytes = gridLen * Uint32Array.BYTES_PER_ELEMENT;
    const nextBytes = this.population * Uint32Array.BYTES_PER_ELEMENT;
    const flagsBytes = this.population * Uint32Array.BYTES_PER_ELEMENT;
    const headsOffset = 0;
    const nextOffset = roundUpTo(headsOffset + headsBytes, 4);
    const flagsOffset = roundUpTo(nextOffset + nextBytes, 4);
    const totalBytes = roundUpTo(flagsOffset + flagsBytes, 4);
    const readback = createBuffer(
      this.device,
      "debug-collision-probe-readback",
      totalBytes,
      GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    );

    const encoder = this.device.createCommandEncoder({ label: "debug-collision-probe-encoder" });
    this.encodeBuildGrid(encoder);
    this.encodeDispatch(
      encoder,
      this.computePipelines.collide,
      dispatchCount(this.population, AGENT_WORKGROUP_SIZE),
      "debug-collision-probe-pass",
    );
    encoder.copyBufferToBuffer(this.gridCountsBuffer, 0, readback, headsOffset, headsBytes);
    encoder.copyBufferToBuffer(this.gridOffsetsBuffer, 0, readback, nextOffset, nextBytes);
    encoder.copyBufferToBuffer(this.deathFlagsBuffer, 0, readback, flagsOffset, flagsBytes);
    this.device.queue.submit([encoder.finish()]);
    await readback.mapAsync(GPUMapMode.READ);

    const bytes = readback.getMappedRange().slice(0);
    const heads = [...new Uint32Array(bytes.slice(headsOffset, headsOffset + headsBytes))];
    const next = [...new Uint32Array(bytes.slice(nextOffset, nextOffset + nextBytes))];
    const deathFlags = [...new Uint32Array(bytes.slice(flagsOffset, flagsOffset + flagsBytes))];

    readback.unmap();
    readback.destroy();
    return { deathFlags, heads, next };
  }

  destroy(): void {
    this.paramsBuffer.destroy();
    this.viewBuffer.destroy();
    this.agentsBuffer.destroy();
    this.genomesBuffer.destroy();
    this.gridCountsBuffer.destroy();
    this.gridOffsetsBuffer.destroy();
    this.deathFlagsBuffer.destroy();
    this.metaBuffer.destroy();
    this.statsReadbackBuffer.destroy();
  }

  private createComputePipeline(
    module: GPUShaderModule,
    entryPoint: string,
    layout: GPUPipelineLayout,
  ): GPUComputePipeline {
    return this.device.createComputePipeline({
      label: `gpu-simulation-${entryPoint}`,
      layout,
      compute: { module, entryPoint },
    });
  }

  private createRenderPipelines(format: GPUTextureFormat): RenderPipelines {
    const agentModule = this.device.createShaderModule({
      label: "gpu-agent-render-shader",
      code: AGENT_RENDER_SHADER,
    });
    const gridModule = this.device.createShaderModule({
      label: "gpu-grid-render-shader",
      code: GRID_RENDER_SHADER,
    });
    const glowModule = this.device.createShaderModule({
      label: "gpu-glow-render-shader",
      code: GLOW_RENDER_SHADER,
    });

    const agents = this.device.createRenderPipeline({
      label: "gpu-agent-render-pipeline",
      layout: "auto",
      vertex: { module: agentModule, entryPoint: "vertex_main" },
      fragment: {
        module: agentModule,
        entryPoint: "fragment_main",
        targets: [
          {
            format,
            blend: {
              color: { srcFactor: "src-alpha", dstFactor: "one-minus-src-alpha", operation: "add" },
              alpha: { srcFactor: "one", dstFactor: "one-minus-src-alpha", operation: "add" },
            },
          },
        ],
      },
      primitive: { topology: "triangle-list" },
    });

    const grid = this.device.createRenderPipeline({
      label: "gpu-grid-render-pipeline",
      layout: "auto",
      vertex: { module: gridModule, entryPoint: "vertex_main" },
      fragment: {
        module: gridModule,
        entryPoint: "fragment_main",
        targets: [
          {
            format,
            blend: {
              color: { srcFactor: "src-alpha", dstFactor: "one-minus-src-alpha", operation: "add" },
              alpha: { srcFactor: "one", dstFactor: "one-minus-src-alpha", operation: "add" },
            },
          },
        ],
      },
      primitive: { topology: "triangle-list" },
    });

    const glow = this.device.createRenderPipeline({
      label: "gpu-glow-render-pipeline",
      layout: "auto",
      vertex: { module: glowModule, entryPoint: "vertex_main" },
      fragment: {
        module: glowModule,
        entryPoint: "fragment_main",
        targets: [
          {
            format,
            blend: {
              color: { srcFactor: "one", dstFactor: "one", operation: "add" },
              alpha: { srcFactor: "one", dstFactor: "one", operation: "add" },
            },
          },
        ],
      },
      primitive: { topology: "triangle-list" },
    });

    return { agents, grid, glow };
  }

  private encodeDispatch(
    encoder: GPUCommandEncoder,
    pipeline: GPUComputePipeline,
    workgroupCount: number,
    label: string,
  ): void {
    const pass = encoder.beginComputePass({ label });
    pass.setBindGroup(0, this.computeBindGroup);
    pass.setPipeline(pipeline);
    pass.dispatchWorkgroups(workgroupCount);
    pass.end();
  }

  private encodeBuildGrid(encoder: GPUCommandEncoder): void {
    const pass = encoder.beginComputePass({ label: "gpu-simulation-build-grid-pass" });
    pass.setBindGroup(0, this.computeBindGroup);

    pass.setPipeline(this.computePipelines.clearGrid);
    pass.dispatchWorkgroups(dispatchCount(Math.max(this.gridCols * this.gridCols, this.population), GRID_WORKGROUP_SIZE));

    pass.setPipeline(this.computePipelines.countGrid);
    pass.dispatchWorkgroups(dispatchCount(this.population, AGENT_WORKGROUP_SIZE));
    pass.end();
  }

  private writeParams(seed: number): void {
    writeMixedParams(this.paramsArray, this.worldSize, this.population, this.gridCols, seed);
    this.device.queue.writeBuffer(this.paramsBuffer, 0, this.paramsArray);
  }

  private writeView(camera: CameraState, aspect: number): void {
    const arrowScale = ARROW_LENGTH_WORLD_UNITS / ARROW_LOCAL_LENGTH / this.worldSize;
    const values = new Float32Array([
      aspect,
      arrowScale,
      camera.centerX,
      camera.centerY,
      camera.zoom,
      1 / this.worldSize,
      0,
      0,
    ]);
    this.device.queue.writeBuffer(this.viewBuffer, 0, values);
  }
}
