// Torus agent simulation.
//
// Agents live on the unit square wrapped at the edges (a torus). State is held
// in a fixed-capacity slot array; every step rebuilds a spatial grid via a
// counting sort that compacts the live agents into a dense, cell-sorted array
// for cheap N-closest queries. A k-NN gather pass writes each agent's sensory
// vector (the future brain input); for now movement is a random walk with a
// rainbow hue drift, and population is sine-waved around half capacity by
// GPU-side births and deaths. No CPU<->GPU readback occurs: the live count
// drives an indirect draw and demographics are computed on the GPU.

import {
  GRID_DIM,
  HEADING_JITTER,
  HUE_DRIFT,
  MAX_AGENTS,
  N_NEIGHBORS,
  POPULATION_AMPLITUDE,
  POPULATION_MID,
  POPULATION_OMEGA,
  STEP_DT,
  WORLD_SIZE,
} from "./config";

const WG = 64;
const SCAN_WG = 256;
const NUM_CELLS = GRID_DIM * GRID_DIM;
const SCAN_CHUNK = NUM_CELLS / SCAN_WG;
// Per-agent sensory stride: N neighbours x 7 fields + 4 own-state fields.
const SENSORY_STRIDE = N_NEIGHBORS * 7 + 4;

// Agent struct stride in bytes (must match the WGSL `Agent` layout below).
const AGENT_F32 = 10; // pos.xy, dir, vel, hue, sat, val, alive, id, pad
const AGENT_BYTES = AGENT_F32 * 4;

const BASE_VEL_MIN = 0.001;
const BASE_VEL_MAX = 0.004;

if (NUM_CELLS % SCAN_WG !== 0) {
  throw new Error(`GRID_DIM^2 (${NUM_CELLS}) must be a multiple of ${SCAN_WG} for the prefix scan`);
}

const SHADER = /* wgsl */ `
const N: u32 = ${N_NEIGHBORS}u;
const STRIDE: u32 = ${SENSORY_STRIDE}u;
const NUM_CELLS: u32 = ${NUM_CELLS}u;
const SCAN_CHUNK: u32 = ${SCAN_CHUNK}u;
const TWO_PI: f32 = 6.28318530717958647;

struct Params {
  dt: f32,
  headingJitter: f32,
  hueDrift: f32,
  worldSize: f32,
  populationMid: f32,
  populationAmplitude: f32,
  populationOmega: f32,
  baseVelMin: f32,
  baseVelMax: f32,
  maxAgents: u32,
  gridDim: u32,
  numCells: u32,
  nNeighbors: u32,
  pad: u32,
}

struct Agent {
  pos: vec2f,
  dir: f32,
  vel: f32,
  hue: f32,
  sat: f32,
  val: f32,
  alive: u32,
  id: u32,
}

struct Dense {
  pos: vec2f,
  slot: u32,
  pad: u32,
}

struct Meta {
  step: u32,
  spawnCount: u32,
  killFraction: f32,
  pad0: u32,
  denseCount: atomic<u32>,
  freeCount: atomic<u32>,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read_write> agents: array<Agent>;
@group(0) @binding(2) var<storage, read_write> globals: Meta;
@group(0) @binding(3) var<storage, read_write> cellCount: array<atomic<u32>>;
@group(0) @binding(4) var<storage, read_write> cellStart: array<u32>;
@group(0) @binding(5) var<storage, read_write> dense: array<Dense>;
@group(0) @binding(6) var<storage, read_write> freeList: array<u32>;
@group(0) @binding(7) var<storage, read_write> sensory: array<f32>;
@group(0) @binding(8) var<storage, read_write> indirect: array<u32>;

fn pcg(v: u32) -> u32 {
  var s = v * 747796405u + 2891336453u;
  s = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u;
  return (s >> 22u) ^ s;
}

fn hash2(a: u32, b: u32) -> u32 {
  return pcg(a + pcg(b));
}

fn randf(seed: u32) -> f32 {
  return f32(pcg(seed)) * (1.0 / 4294967296.0);
}

fn cellOf(p: vec2f) -> u32 {
  let dim = f32(params.gridDim);
  var cx = i32(floor(p.x / params.worldSize * dim));
  var cy = i32(floor(p.y / params.worldSize * dim));
  cx = clamp(cx, 0, i32(params.gridDim) - 1);
  cy = clamp(cy, 0, i32(params.gridDim) - 1);
  return u32(cy) * params.gridDim + u32(cx);
}

fn tDelta(a: f32, b: f32) -> f32 {
  let s = params.worldSize;
  var d = b - a;
  d = d - s * round(d / s);
  return d;
}

fn wrapAngle(a: f32) -> f32 {
  return a - TWO_PI * round(a / TWO_PI);
}

@compute @workgroup_size(1)
fn bump() {
  globals.step = globals.step + 1u;
}

@compute @workgroup_size(${WG})
fn clearCells(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i < params.numCells) {
    atomicStore(&cellCount[i], 0u);
  }
  if (i == 0u) {
    atomicStore(&globals.freeCount, 0u);
  }
}

@compute @workgroup_size(${WG})
fn count(@builtin(global_invocation_id) gid: vec3u) {
  let slot = gid.x;
  if (slot >= params.maxAgents) {
    return;
  }
  let a = agents[slot];
  if (a.alive == 1u) {
    atomicAdd(&cellCount[cellOf(a.pos)], 1u);
  } else {
    let f = atomicAdd(&globals.freeCount, 1u);
    freeList[f] = slot;
  }
}

var<workgroup> blockSum: array<u32, ${SCAN_WG}>;
var<workgroup> blockOff: array<u32, ${SCAN_WG}>;

@compute @workgroup_size(${SCAN_WG})
fn scan(@builtin(local_invocation_index) lid: u32) {
  let t = lid;
  var localv: array<u32, ${SCAN_CHUNK}>;
  var running = 0u;
  for (var k = 0u; k < SCAN_CHUNK; k = k + 1u) {
    let c = t * SCAN_CHUNK + k;
    let v = atomicLoad(&cellCount[c]);
    localv[k] = running;
    running = running + v;
  }
  blockSum[t] = running;
  workgroupBarrier();

  if (t == 0u) {
    var acc = 0u;
    for (var i = 0u; i < ${SCAN_WG}u; i = i + 1u) {
      blockOff[i] = acc;
      acc = acc + blockSum[i];
    }
    cellStart[NUM_CELLS] = acc;
    atomicStore(&globals.denseCount, acc);
  }
  workgroupBarrier();

  let base = blockOff[t];
  for (var k = 0u; k < SCAN_CHUNK; k = k + 1u) {
    let c = t * SCAN_CHUNK + k;
    let val = base + localv[k];
    cellStart[c] = val;
    atomicStore(&cellCount[c], val); // reuse cellCount as the scatter cursor
  }
}

@compute @workgroup_size(${WG})
fn scatter(@builtin(global_invocation_id) gid: vec3u) {
  let slot = gid.x;
  if (slot >= params.maxAgents) {
    return;
  }
  let a = agents[slot];
  if (a.alive != 1u) {
    return;
  }
  let d = atomicAdd(&cellCount[cellOf(a.pos)], 1u);
  dense[d].pos = a.pos;
  dense[d].slot = slot;
}

@compute @workgroup_size(1)
fn demographics() {
  let step = globals.step;
  let live = atomicLoad(&globals.denseCount);
  let freeC = atomicLoad(&globals.freeCount);
  let targetPop = params.populationMid
    + params.populationAmplitude * sin(f32(step) * params.populationOmega);

  var spawn = 0u;
  var killF = 0.0;
  if (f32(live) > targetPop) {
    killF = (f32(live) - targetPop) / max(f32(live), 1.0);
  } else {
    let want = u32(round(targetPop - f32(live)));
    spawn = min(want, freeC);
  }
  globals.spawnCount = spawn;
  globals.killFraction = killF;

  indirect[0] = 3u; // vertices per agent triangle
  indirect[1] = live; // instance count
  indirect[2] = 0u;
  indirect[3] = 0u;
}

@compute @workgroup_size(${WG})
fn knn(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= atomicLoad(&globals.denseCount)) {
    return;
  }

  let selfPos = dense[i].pos;
  let selfSlot = dense[i].slot;
  let dim = i32(params.gridDim);
  let homeCx = clamp(i32(floor(selfPos.x / params.worldSize * f32(dim))), 0, dim - 1);
  let homeCy = clamp(i32(floor(selfPos.y / params.worldSize * f32(dim))), 0, dim - 1);
  let cellW = params.worldSize / f32(dim);
  let maxR = (dim - 1) / 2;

  var nbrD2: array<f32, N>;
  var nbrSlot: array<u32, N>;
  for (var k = 0u; k < N; k = k + 1u) {
    nbrD2[k] = 1e30;
    nbrSlot[k] = 0xffffffffu;
  }
  var found = 0u;

  for (var r = 0; r <= maxR; r = r + 1) {
    for (var dy = -r; dy <= r; dy = dy + 1) {
      for (var dx = -r; dx <= r; dx = dx + 1) {
        if (max(abs(dx), abs(dy)) != r) {
          continue; // only the border cells of this ring
        }
        let cx = u32(((homeCx + dx) % dim + dim) % dim);
        let cy = u32(((homeCy + dy) % dim + dim) % dim);
        let cell = cy * params.gridDim + cx;
        let start = cellStart[cell];
        let endi = cellStart[cell + 1u];
        for (var j = start; j < endi; j = j + 1u) {
          if (j == i) {
            continue;
          }
          let nPos = dense[j].pos;
          let ddx = tDelta(selfPos.x, nPos.x);
          let ddy = tDelta(selfPos.y, nPos.y);
          let d2 = ddx * ddx + ddy * ddy;
          if (found < N) {
            nbrD2[found] = d2;
            nbrSlot[found] = dense[j].slot;
            found = found + 1u;
          } else {
            var worstK = 0u;
            var worst = nbrD2[0];
            for (var k = 1u; k < N; k = k + 1u) {
              if (nbrD2[k] > worst) {
                worst = nbrD2[k];
                worstK = k;
              }
            }
            if (d2 < worst) {
              nbrD2[worstK] = d2;
              nbrSlot[worstK] = dense[j].slot;
            }
          }
        }
      }
    }

    if (found >= params.nNeighbors) {
      var worst = 0.0;
      for (var k = 0u; k < N; k = k + 1u) {
        worst = max(worst, nbrD2[k]);
      }
      let bound = f32(r) * cellW;
      if (worst <= bound * bound) {
        break;
      }
    }
  }

  let selfAgent = agents[selfSlot];
  let baseIdx = selfSlot * STRIDE;
  sensory[baseIdx + 0u] = selfAgent.vel;
  sensory[baseIdx + 1u] = selfAgent.hue;
  sensory[baseIdx + 2u] = selfAgent.sat;
  sensory[baseIdx + 3u] = selfAgent.val;
  for (var k = 0u; k < N; k = k + 1u) {
    let off = baseIdx + 4u + k * 7u;
    if (nbrSlot[k] != 0xffffffffu) {
      let na = agents[nbrSlot[k]];
      let ddx = tDelta(selfPos.x, na.pos.x);
      let ddy = tDelta(selfPos.y, na.pos.y);
      sensory[off + 0u] = wrapAngle(atan2(ddy, ddx) - selfAgent.dir);
      sensory[off + 1u] = sqrt(nbrD2[k]);
      sensory[off + 2u] = wrapAngle(na.dir - selfAgent.dir);
      sensory[off + 3u] = na.vel;
      sensory[off + 4u] = na.hue;
      sensory[off + 5u] = na.sat;
      sensory[off + 6u] = na.val;
    } else {
      for (var m = 0u; m < 7u; m = m + 1u) {
        sensory[off + m] = 0.0;
      }
    }
  }
}

@compute @workgroup_size(${WG})
fn integrate(@builtin(global_invocation_id) gid: vec3u) {
  let slot = gid.x;
  if (slot >= params.maxAgents) {
    return;
  }
  var a = agents[slot];
  if (a.alive != 1u) {
    return;
  }
  let h = hash2(a.id, globals.step);
  a.dir = a.dir + (randf(h) * 2.0 - 1.0) * params.headingJitter;
  var p = a.pos + vec2f(cos(a.dir), sin(a.dir)) * a.vel * params.dt;
  p = p - floor(p / params.worldSize) * params.worldSize;
  a.pos = p;
  a.hue = fract(a.hue + params.hueDrift);
  agents[slot] = a;
}

@compute @workgroup_size(${WG})
fn death(@builtin(global_invocation_id) gid: vec3u) {
  let slot = gid.x;
  if (slot >= params.maxAgents) {
    return;
  }
  let kf = globals.killFraction;
  if (kf <= 0.0) {
    return;
  }
  let a = agents[slot];
  if (a.alive != 1u) {
    return;
  }
  let h = hash2(a.id ^ 0x5bd1e995u, globals.step);
  if (randf(h) < kf) {
    agents[slot].alive = 0u;
  }
}

@compute @workgroup_size(${WG})
fn birth(@builtin(global_invocation_id) gid: vec3u) {
  let t = gid.x;
  if (t >= globals.spawnCount) {
    return;
  }
  let slot = freeList[t];
  let step = globals.step;
  let h = hash2(slot ^ 0x1000193u, step);
  let velRange = params.baseVelMax - params.baseVelMin;

  var a: Agent;
  a.pos = vec2f(randf(h) * params.worldSize, randf(h ^ 0xa5a5a5a5u) * params.worldSize);
  a.dir = randf(h ^ 0x12345u) * TWO_PI;
  a.vel = params.baseVelMin + randf(h ^ 0x999u) * velRange;
  a.hue = randf(h ^ 0x777u);
  a.sat = 0.85;
  a.val = 1.0;
  a.alive = 1u;
  a.id = pcg((slot * 2654435761u) ^ step);
  agents[slot] = a;
}
`;

export class Simulation {
  readonly maxAgents: number;

  private readonly device: GPUDevice;
  private readonly bindGroup: GPUBindGroup;
  private readonly pipelines: Record<string, GPUComputePipeline>;
  private readonly agents: GPUBuffer;
  private readonly dense: GPUBuffer;
  private readonly indirect: GPUBuffer;

  private readonly cellWorkgroups: number;
  private readonly agentWorkgroups: number;

  constructor(device: GPUDevice, maxAgents: number) {
    this.device = device;
    this.maxAgents = maxAgents;
    this.cellWorkgroups = Math.ceil(NUM_CELLS / WG);
    this.agentWorkgroups = Math.ceil(maxAgents / WG);

    const storage = GPUBufferUsage.STORAGE;
    this.agents = device.createBuffer({
      size: maxAgents * AGENT_BYTES,
      usage: storage,
      mappedAtCreation: true,
      label: "agents",
    });
    writeInitialAgents(this.agents.getMappedRange(), maxAgents);
    this.agents.unmap();

    const meta = device.createBuffer({ size: 24, usage: storage, label: "meta" });
    const cellCount = device.createBuffer({ size: NUM_CELLS * 4, usage: storage, label: "cell-count" });
    const cellStart = device.createBuffer({ size: (NUM_CELLS + 1) * 4, usage: storage, label: "cell-start" });
    this.dense = device.createBuffer({ size: maxAgents * 16, usage: storage, label: "dense" });
    const freeList = device.createBuffer({ size: maxAgents * 4, usage: storage, label: "free-list" });
    const sensory = device.createBuffer({
      size: maxAgents * SENSORY_STRIDE * 4,
      usage: storage,
      label: "sensory",
    });
    this.indirect = device.createBuffer({
      size: 16,
      usage: GPUBufferUsage.INDIRECT | GPUBufferUsage.STORAGE,
      mappedAtCreation: true,
      label: "draw-indirect",
    });
    new Uint32Array(this.indirect.getMappedRange()).set([3, 0, 0, 0]);
    this.indirect.unmap();

    const paramsBuffer = device.createBuffer({
      size: 64,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      label: "sim-params",
    });
    device.queue.writeBuffer(paramsBuffer, 0, buildParams());

    const layout = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
        ...[1, 2, 3, 4, 5, 6, 7, 8].map((binding) => ({
          binding,
          visibility: GPUShaderStage.COMPUTE,
          buffer: { type: "storage" as const },
        })),
      ],
    });

    this.bindGroup = device.createBindGroup({
      layout,
      entries: [
        { binding: 0, resource: { buffer: paramsBuffer } },
        { binding: 1, resource: { buffer: this.agents } },
        { binding: 2, resource: { buffer: meta } },
        { binding: 3, resource: { buffer: cellCount } },
        { binding: 4, resource: { buffer: cellStart } },
        { binding: 5, resource: { buffer: this.dense } },
        { binding: 6, resource: { buffer: freeList } },
        { binding: 7, resource: { buffer: sensory } },
        { binding: 8, resource: { buffer: this.indirect } },
      ],
    });

    const module = device.createShaderModule({ code: SHADER });
    const pipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [layout] });
    const make = (entryPoint: string): GPUComputePipeline =>
      device.createComputePipeline({ layout: pipelineLayout, compute: { module, entryPoint } });
    this.pipelines = {
      bump: make("bump"),
      clearCells: make("clearCells"),
      count: make("count"),
      scan: make("scan"),
      scatter: make("scatter"),
      demographics: make("demographics"),
      knn: make("knn"),
      integrate: make("integrate"),
      death: make("death"),
      birth: make("birth"),
    };
  }

  /** Latest live-agent state, addressed by dense index -> slot. */
  get agentsBuffer(): GPUBuffer {
    return this.agents;
  }
  get denseBuffer(): GPUBuffer {
    return this.dense;
  }
  /** Draw-indirect args (vertexCount=3, instanceCount=liveCount). */
  get indirectBuffer(): GPUBuffer {
    return this.indirect;
  }

  /** Encode `steps` dependent simulation steps into a single compute pass. */
  encode(encoder: GPUCommandEncoder, steps: number): void {
    if (steps <= 0) {
      return;
    }
    const pass = encoder.beginComputePass({ label: "simulation" });
    pass.setBindGroup(0, this.bindGroup);
    for (let i = 0; i < steps; i += 1) {
      this.dispatch(pass, "bump", 1);
      this.dispatch(pass, "clearCells", this.cellWorkgroups);
      this.dispatch(pass, "count", this.agentWorkgroups);
      this.dispatch(pass, "scan", 1);
      this.dispatch(pass, "scatter", this.agentWorkgroups);
      this.dispatch(pass, "demographics", 1);
      this.dispatch(pass, "knn", this.agentWorkgroups);
      this.dispatch(pass, "integrate", this.agentWorkgroups);
      this.dispatch(pass, "death", this.agentWorkgroups);
      this.dispatch(pass, "birth", this.agentWorkgroups);
    }
    pass.end();
  }

  private dispatch(pass: GPUComputePassEncoder, name: string, workgroups: number): void {
    pass.setPipeline(this.pipelines[name]);
    pass.dispatchWorkgroups(workgroups);
  }
}

function buildParams(): ArrayBuffer {
  const buf = new ArrayBuffer(64);
  const f = new Float32Array(buf);
  const u = new Uint32Array(buf);
  f[0] = STEP_DT;
  f[1] = HEADING_JITTER;
  f[2] = HUE_DRIFT;
  f[3] = WORLD_SIZE;
  f[4] = POPULATION_MID;
  f[5] = POPULATION_AMPLITUDE;
  f[6] = POPULATION_OMEGA;
  f[7] = BASE_VEL_MIN;
  f[8] = BASE_VEL_MAX;
  u[9] = MAX_AGENTS;
  u[10] = GRID_DIM;
  u[11] = NUM_CELLS;
  u[12] = N_NEIGHBORS;
  u[13] = 0;
  return buf;
}

// Seed roughly half the slots with live agents so population starts near the
// sine midpoint; the rest begin dead and are filled by births over time.
function writeInitialAgents(range: ArrayBuffer, count: number): void {
  const f = new Float32Array(range);
  const u = new Uint32Array(range);
  let seed = 0x9e3779b9;
  const rng = (): number => {
    seed = (Math.imul(seed ^ (seed >>> 15), 2246822519) + 1) >>> 0;
    return seed / 0xffffffff;
  };
  const initialAlive = Math.round(POPULATION_MID);
  for (let i = 0; i < count; i += 1) {
    const b = i * AGENT_F32;
    const alive = i < initialAlive ? 1 : 0;
    f[b + 0] = rng() * WORLD_SIZE;
    f[b + 1] = rng() * WORLD_SIZE;
    f[b + 2] = rng() * Math.PI * 2;
    f[b + 3] = BASE_VEL_MIN + rng() * (BASE_VEL_MAX - BASE_VEL_MIN);
    f[b + 4] = rng();
    f[b + 5] = 0.85;
    f[b + 6] = 1.0;
    u[b + 7] = alive;
    u[b + 8] = (Math.imul(i + 1, 2654435761) >>> 0) || 1;
    u[b + 9] = 0;
  }
}
