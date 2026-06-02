// Torus agent simulation.
//
// Agents live on the unit square wrapped at the edges (a torus). State is held
// in fixed-capacity slot arrays. Each step builds a cell-sorted neighbor index,
// runs neural movement, resolves collision deaths/head-on breeding, and writes
// children through fixed GPU buffers. There is no CPU<->GPU simulation readback.

import {
  AGENT_HIT_RADIUS,
  AGENT_MAX_SPEED,
  AGENT_MAX_TURN,
  AGENT_MIN_SPEED,
  BRAIN_WEIGHT_COUNT,
  CONTACT_DOT,
  GRID_DIM,
  HEAD_ON_DOT,
  HUE_MUTATION_SCALE,
  INITIAL_AGENTS,
  MAX_AGENTS,
  MUTATION_RATE,
  MUTATION_SCALE,
  MUTATION_WEIGHT_LIMIT,
  NEURAL_HIDDEN,
  NEURAL_INPUTS,
  NEURAL_NEIGHBORS,
  SENSOR_RADIUS,
  SPEED_MUTATION_SCALE,
  STEP_DT,
  WORLD_SIZE,
} from "./config";
import {
  AGENT_BYTES,
  AGENT_F32,
  AGENT_STRUCT_WGSL,
  BIRTH_EVENT_BYTES,
  BIRTH_EVENT_STRUCT_WGSL,
  DENSE_BYTES,
  DENSE_STRUCT_WGSL,
  DRAW_INDIRECT_BYTES,
  PLANNED_BYTES,
  PLANNED_STRUCT_WGSL,
  SIM_PARAMS_BYTES,
} from "./layout";

const WG = 64;
const SCAN_WG = 256;
const NUM_CELLS = GRID_DIM * GRID_DIM;
const SCAN_CHUNK = NUM_CELLS / SCAN_WG;
const BRAIN_BYTES = BRAIN_WEIGHT_COUNT * 4;

const PIPELINE_NAMES = [
  "clearStep",
  "clearIndex",
  "count",
  "scan",
  "scatter",
  "planMove",
  "chooseContacts",
  "resolveMates",
  "commitAgents",
  "spawnChildren",
  "writeIndirect",
] as const;
type PipelineName = (typeof PIPELINE_NAMES)[number];

if (NUM_CELLS % SCAN_WG !== 0) {
  throw new Error(`GRID_DIM^2 (${NUM_CELLS}) must be a multiple of ${SCAN_WG} for the prefix scan`);
}

if (AGENT_HIT_RADIUS * 2 + AGENT_MAX_SPEED * 2 >= WORLD_SIZE / GRID_DIM) {
  throw new Error("collision broadphase assumes hit diameter plus relative step motion fits within one cell");
}

const SHADER = /* wgsl */ `
const NUM_CELLS: u32 = ${NUM_CELLS}u;
const SCAN_CHUNK: u32 = ${SCAN_CHUNK}u;
const TWO_PI: f32 = 6.28318530717958647;
const NO_TARGET: u32 = 0xffffffffu;

struct Params {
  dt: f32,
  worldSize: f32,
  hitRadius: f32,
  collisionDistanceSq: f32,
  contactDot: f32,
  headOnDot: f32,
  maxTurn: f32,
  minSpeed: f32,
  maxSpeed: f32,
  mutationRate: f32,
  mutationScale: f32,
  mutationWeightLimit: f32,
  speedMutationScale: f32,
  hueMutationScale: f32,
  sensorRadius: f32,
  pad0: f32,
  maxAgents: u32,
  gridDim: u32,
  numCells: u32,
  neuralNeighbors: u32,
  brainWeightCount: u32,
  pad1: u32,
  pad2: u32,
  pad3: u32,
}

${AGENT_STRUCT_WGSL}
${DENSE_STRUCT_WGSL}
${PLANNED_STRUCT_WGSL}
${BIRTH_EVENT_STRUCT_WGSL}

struct Meta {
  step: u32,
  denseCount: atomic<u32>,
  freeCount: atomic<u32>,
  birthCount: atomic<u32>,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read_write> agents: array<Agent>;
@group(0) @binding(2) var<storage, read_write> globals: Meta;
@group(0) @binding(3) var<storage, read_write> cellCount: array<atomic<u32>>;
@group(0) @binding(4) var<storage, read_write> cellStart: array<u32>;
@group(0) @binding(5) var<storage, read_write> dense: array<Dense>;
@group(0) @binding(6) var<storage, read_write> freeList: array<u32>;
@group(0) @binding(7) var<storage, read_write> indirect: array<u32>;
@group(0) @binding(8) var<storage, read_write> brains: array<f32>;
@group(0) @binding(9) var<storage, read_write> planned: array<Planned>;
@group(0) @binding(10) var<storage, read_write> killMarks: array<atomic<u32>>;
@group(0) @binding(11) var<storage, read_write> mateTargets: array<u32>;
@group(0) @binding(12) var<storage, read_write> birthEvents: array<BirthEvent>;

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

fn wrapPos(p: vec2f) -> vec2f {
  return p - floor(p / params.worldSize) * params.worldSize;
}

fn torusDelta(a: vec2f, b: vec2f) -> vec2f {
  var d = b - a;
  d = d - params.worldSize * round(d / params.worldSize);
  return d;
}

fn cellCoord(p: vec2f) -> vec2u {
  let dim = f32(params.gridDim);
  var cx = i32(floor(p.x / params.worldSize * dim));
  var cy = i32(floor(p.y / params.worldSize * dim));
  cx = clamp(cx, 0, i32(params.gridDim) - 1);
  cy = clamp(cy, 0, i32(params.gridDim) - 1);
  return vec2u(u32(cx), u32(cy));
}

fn cellOf(p: vec2f) -> u32 {
  let c = cellCoord(p);
  return c.y * params.gridDim + c.x;
}

fn wrapCell(c: i32) -> u32 {
  let dim = i32(params.gridDim);
  var v = c;
  if (v < 0) {
    v = v + dim;
  }
  if (v >= dim) {
    v = v - dim;
  }
  return u32(v);
}

fn wrappedCellOf(cx: i32, cy: i32) -> u32 {
  return wrapCell(cy) * params.gridDim + wrapCell(cx);
}

fn speedNorm(vel: f32) -> f32 {
  return clamp((vel - params.minSpeed) / max(params.maxSpeed - params.minSpeed, 0.000001) * 2.0 - 1.0, -1.0, 1.0);
}

fn squash(x: f32) -> f32 {
  return x / (1.0 + abs(x));
}

fn brainAt(slot: u32, weight: u32) -> f32 {
  return brains[slot * params.brainWeightCount + weight];
}

fn classifyImpact(a: Planned, b: Planned) -> vec4f {
  let d = torusDelta(a.pos, b.pos);
  let distSq = dot(d, d);
  if (distSq > params.collisionDistanceSq || distSq <= 0.000000000001) {
    return vec4f(0.0, 0.0, 0.0, distSq);
  }

  let n = d * inverseSqrt(distSq);
  let af = vec2f(cos(a.dir), sin(a.dir));
  let bf = vec2f(cos(b.dir), sin(b.dir));
  let aImpact = dot(af, n);
  let bImpact = dot(bf, -n);
  return vec4f(aImpact, bImpact, 1.0, distSq);
}

@compute @workgroup_size(${WG})
fn clearStep(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i < params.numCells) {
    atomicStore(&cellCount[i], 0u);
  }
  if (i < params.maxAgents) {
    atomicStore(&killMarks[i], 0u);
    mateTargets[i] = NO_TARGET;
  }
  if (i == 0u) {
    globals.step = globals.step + 1u;
    atomicStore(&globals.freeCount, 0u);
    atomicStore(&globals.birthCount, 0u);
  }
}

@compute @workgroup_size(${WG})
fn clearIndex(@builtin(global_invocation_id) gid: vec3u) {
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

@compute @workgroup_size(${WG})
fn planMove(@builtin(global_invocation_id) gid: vec3u) {
  let slot = gid.x;
  if (slot >= params.maxAgents) {
    return;
  }
  let a = agents[slot];
  if (a.alive != 1u) {
    return;
  }

  var bestDist: array<f32, ${NEURAL_NEIGHBORS}>;
  var bestSlot: array<u32, ${NEURAL_NEIGHBORS}>;
  for (var i = 0u; i < ${NEURAL_NEIGHBORS}u; i = i + 1u) {
    bestDist[i] = 1e20;
    bestSlot[i] = NO_TARGET;
  }

  let cc = cellCoord(a.pos);
  for (var oy = -1; oy <= 1; oy = oy + 1) {
    for (var ox = -1; ox <= 1; ox = ox + 1) {
      let c = wrappedCellOf(i32(cc.x) + ox, i32(cc.y) + oy);
      let start = cellStart[c];
      let end = cellStart[c + 1u];
      for (var di = start; di < end; di = di + 1u) {
        let otherSlot = dense[di].slot;
        if (otherSlot == slot) {
          continue;
        }
        let other = agents[otherSlot];
        if (other.alive != 1u) {
          continue;
        }
        let delta = torusDelta(a.pos, dense[di].pos);
        let distSq = dot(delta, delta);
        var insertAt = ${NEURAL_NEIGHBORS}u;
        for (var k = 0u; k < ${NEURAL_NEIGHBORS}u; k = k + 1u) {
          if (distSq < bestDist[k]) {
            insertAt = k;
            break;
          }
        }
        if (insertAt < ${NEURAL_NEIGHBORS}u) {
          var k = ${NEURAL_NEIGHBORS - 1}u;
          loop {
            if (k <= insertAt) {
              break;
            }
            bestDist[k] = bestDist[k - 1u];
            bestSlot[k] = bestSlot[k - 1u];
            k = k - 1u;
          }
          bestDist[insertAt] = distSq;
          bestSlot[insertAt] = otherSlot;
        }
      }
    }
  }

  var inputs: array<f32, ${NEURAL_INPUTS}>;
  for (var i = 0u; i < ${NEURAL_INPUTS}u; i = i + 1u) {
    inputs[i] = 0.0;
  }
  inputs[0] = 1.0;
  inputs[1] = speedNorm(a.vel);
  inputs[2] = randf(hash2(a.id ^ 0x51ed270bu, globals.step)) * 2.0 - 1.0;

  let af = vec2f(cos(a.dir), sin(a.dir));
  let ar = vec2f(-af.y, af.x);
  for (var n = 0u; n < ${NEURAL_NEIGHBORS}u; n = n + 1u) {
    let otherSlot = bestSlot[n];
    if (otherSlot == NO_TARGET) {
      continue;
    }
    let other = agents[otherSlot];
    let delta = torusDelta(a.pos, other.pos);
    let distSq = max(dot(delta, delta), 0.000000000001);
    let dist = sqrt(distSq);
    let dirToOther = delta / dist;
    let otherForward = vec2f(cos(other.dir), sin(other.dir));
    let base = 3u + n * 6u;
    inputs[base] = 1.0;
    inputs[base + 1u] = clamp(1.0 - dist / params.sensorRadius, 0.0, 1.0);
    inputs[base + 2u] = dot(af, dirToOther);
    inputs[base + 3u] = dot(ar, dirToOther);
    inputs[base + 4u] = dot(af, otherForward);
    inputs[base + 5u] = speedNorm(other.vel);
  }

  var hidden: array<f32, ${NEURAL_HIDDEN}>;
  for (var h = 0u; h < ${NEURAL_HIDDEN}u; h = h + 1u) {
    var sum = 0.0;
    for (var i = 0u; i < ${NEURAL_INPUTS}u; i = i + 1u) {
      sum = sum + inputs[i] * brainAt(slot, h * ${NEURAL_INPUTS}u + i);
    }
    hidden[h] = squash(sum);
  }

  let outBase = ${NEURAL_INPUTS * NEURAL_HIDDEN}u;
  var turnRaw = 0.0;
  var speedRaw = 0.0;
  for (var h = 0u; h < ${NEURAL_HIDDEN}u; h = h + 1u) {
    turnRaw = turnRaw + hidden[h] * brainAt(slot, outBase + h);
    speedRaw = speedRaw + hidden[h] * brainAt(slot, outBase + ${NEURAL_HIDDEN}u + h);
  }

  var p: Planned;
  p.dir = a.dir + squash(turnRaw) * params.maxTurn;
  let speed01 = squash(speedRaw) * 0.5 + 0.5;
  p.vel = params.minSpeed + speed01 * (params.maxSpeed - params.minSpeed);
  p.pos = wrapPos(a.pos + vec2f(cos(p.dir), sin(p.dir)) * p.vel * params.dt);
  planned[slot] = p;
}

@compute @workgroup_size(${WG})
fn chooseContacts(@builtin(global_invocation_id) gid: vec3u) {
  let slot = gid.x;
  if (slot >= params.maxAgents) {
    return;
  }
  let a = agents[slot];
  if (a.alive != 1u) {
    return;
  }

  let ap = planned[slot];
  let cc = cellCoord(a.pos);
  var mate = NO_TARGET;
  var mateDist = 1e20;
  var killSelf = 0u;
  var killDist = 1e20;

  for (var oy = -1; oy <= 1; oy = oy + 1) {
    for (var ox = -1; ox <= 1; ox = ox + 1) {
      let c = wrappedCellOf(i32(cc.x) + ox, i32(cc.y) + oy);
      let start = cellStart[c];
      let end = cellStart[c + 1u];
      for (var di = start; di < end; di = di + 1u) {
        let otherSlot = dense[di].slot;
        if (otherSlot == slot) {
          continue;
        }
        let other = agents[otherSlot];
        if (other.alive != 1u) {
          continue;
        }
        let impact = classifyImpact(ap, planned[otherSlot]);
        if (impact.z == 0.0) {
          continue;
        }
        let aImpact = impact.x;
        let bImpact = impact.y;
        let distSq = impact.w;
        if (aImpact >= params.headOnDot && bImpact >= params.headOnDot) {
          if (distSq < mateDist) {
            mateDist = distSq;
            mate = otherSlot;
          }
        } else if (aImpact >= bImpact && aImpact >= params.contactDot && distSq < killDist) {
          killDist = distSq;
          killSelf = 1u;
        }
      }
    }
  }

  if (killSelf == 1u) {
    atomicStore(&killMarks[slot], 1u);
  } else {
    mateTargets[slot] = mate;
  }
}

@compute @workgroup_size(${WG})
fn resolveMates(@builtin(global_invocation_id) gid: vec3u) {
  let slot = gid.x;
  if (slot >= params.maxAgents) {
    return;
  }
  let mate = mateTargets[slot];
  if (mate == NO_TARGET || slot >= mate) {
    return;
  }
  if (agents[slot].alive != 1u || agents[mate].alive != 1u) {
    return;
  }
  if (atomicLoad(&killMarks[slot]) != 0u || atomicLoad(&killMarks[mate]) != 0u) {
    return;
  }
  if (mateTargets[mate] != slot) {
    return;
  }

  let e = atomicAdd(&globals.birthCount, 1u);
  if (e < params.maxAgents) {
    birthEvents[e].parentA = slot;
    birthEvents[e].parentB = mate;
  }
}

@compute @workgroup_size(${WG})
fn commitAgents(@builtin(global_invocation_id) gid: vec3u) {
  let slot = gid.x;
  if (slot >= params.maxAgents) {
    return;
  }
  var a = agents[slot];
  if (a.alive != 1u) {
    return;
  }
  if (atomicLoad(&killMarks[slot]) != 0u) {
    a.alive = 0u;
    agents[slot] = a;
    return;
  }

  let p = planned[slot];
  a.pos = p.pos;
  a.dir = p.dir;
  a.vel = p.vel;
  agents[slot] = a;
}

@compute @workgroup_size(${WG})
fn spawnChildren(@builtin(global_invocation_id) gid: vec3u) {
  let t = gid.x;
  let birthCount = min(atomicLoad(&globals.birthCount), params.maxAgents);
  if (t >= birthCount) {
    return;
  }

  let event = birthEvents[t];
  let freeCount = atomicLoad(&globals.freeCount);
  let seed = hash2(t ^ 0x9e3779b9u, globals.step);
  var childSlot: u32;
  if (t < freeCount) {
    childSlot = freeList[t];
  } else if (randf(seed) < 0.5) {
    childSlot = event.parentA;
  } else {
    childSlot = event.parentB;
  }

  let a = agents[event.parentA];
  let b = agents[event.parentB];
  if (a.alive != 1u || b.alive != 1u) {
    return;
  }

  let delta = torusDelta(a.pos, b.pos);
  let posSeed = hash2(seed, childSlot ^ 0x85ebca6bu);
  let headingFromA = randf(posSeed) < 0.5;
  let speedMutation = (randf(posSeed ^ 0x632be5abu) * 2.0 - 1.0) * params.speedMutationScale;
  let hueMutation = (randf(posSeed ^ 0x85157af5u) * 2.0 - 1.0) * params.hueMutationScale;

  var child: Agent;
  child.pos = wrapPos(a.pos + delta * 0.5);
  child.dir = select(b.dir, a.dir, headingFromA);
  child.vel = clamp((a.vel + b.vel) * 0.5 + speedMutation, params.minSpeed, params.maxSpeed);
  child.hue = fract((a.hue + b.hue) * 0.5 + hueMutation);
  child.sat = 0.85;
  child.val = 1.0;
  child.alive = 1u;
  child.id = pcg((childSlot * 2654435761u) ^ globals.step ^ t ^ seed);

  let childBase = childSlot * params.brainWeightCount;
  let aBase = event.parentA * params.brainWeightCount;
  let bBase = event.parentB * params.brainWeightCount;
  for (var i = 0u; i < ${BRAIN_WEIGHT_COUNT}u; i = i + 1u) {
    let wSeed = hash2(child.id ^ i, globals.step ^ t);
    let inherited = select(brains[bBase + i], brains[aBase + i], randf(wSeed) < 0.5);
    var w = inherited;
    if (randf(wSeed ^ 0x27d4eb2du) < params.mutationRate) {
      w = w + (randf(wSeed ^ 0x165667b1u) * 2.0 - 1.0) * params.mutationScale;
    }
    brains[childBase + i] = clamp(w, -params.mutationWeightLimit, params.mutationWeightLimit);
  }

  agents[childSlot] = child;
}

@compute @workgroup_size(1)
fn writeIndirect() {
  let live = atomicLoad(&globals.denseCount);
  indirect[0] = 3u; // vertices per agent triangle
  indirect[1] = live; // instance count
  indirect[2] = 0u;
  indirect[3] = 0u;
}
`;

export class Simulation {
  private readonly device: GPUDevice;
  private readonly bindGroup: GPUBindGroup;
  private readonly pipelines: Record<PipelineName, GPUComputePipeline>;
  private readonly agents: GPUBuffer;
  private readonly dense: GPUBuffer;
  private readonly indirect: GPUBuffer;

  private readonly clearWorkgroups: number;
  private readonly agentWorkgroups: number;

  constructor(device: GPUDevice) {
    this.device = device;
    this.clearWorkgroups = Math.ceil(Math.max(NUM_CELLS, MAX_AGENTS) / WG);
    this.agentWorkgroups = Math.ceil(MAX_AGENTS / WG);

    const storage = GPUBufferUsage.STORAGE;
    this.agents = device.createBuffer({
      size: MAX_AGENTS * AGENT_BYTES,
      usage: storage,
      mappedAtCreation: true,
      label: "agents",
    });
    writeInitialAgents(this.agents.getMappedRange(), MAX_AGENTS);
    this.agents.unmap();

    const brains = device.createBuffer({
      size: MAX_AGENTS * BRAIN_BYTES,
      usage: storage,
      mappedAtCreation: true,
      label: "brains",
    });
    writeInitialBrains(brains.getMappedRange(), MAX_AGENTS);
    brains.unmap();

    // Zero-initialized: step and counters start at 0.
    const meta = device.createBuffer({ size: 16, usage: storage, label: "meta" });
    const cellCount = device.createBuffer({ size: NUM_CELLS * 4, usage: storage, label: "cell-count" });
    const cellStart = device.createBuffer({ size: (NUM_CELLS + 1) * 4, usage: storage, label: "cell-start" });
    this.dense = device.createBuffer({ size: MAX_AGENTS * DENSE_BYTES, usage: storage, label: "dense" });
    const freeList = device.createBuffer({ size: MAX_AGENTS * 4, usage: storage, label: "free-list" });
    const planned = device.createBuffer({ size: MAX_AGENTS * PLANNED_BYTES, usage: storage, label: "planned" });
    const killMarks = device.createBuffer({ size: MAX_AGENTS * 4, usage: storage, label: "kill-marks" });
    const mateTargets = device.createBuffer({ size: MAX_AGENTS * 4, usage: storage, label: "mate-targets" });
    const birthEvents = device.createBuffer({
      size: MAX_AGENTS * BIRTH_EVENT_BYTES,
      usage: storage,
      label: "birth-events",
    });
    this.indirect = device.createBuffer({
      size: DRAW_INDIRECT_BYTES,
      usage: GPUBufferUsage.INDIRECT | GPUBufferUsage.STORAGE,
      mappedAtCreation: true,
      label: "draw-indirect",
    });
    new Uint32Array(this.indirect.getMappedRange()).set([3, 0, 0, 0]);
    this.indirect.unmap();

    const paramsBuffer = device.createBuffer({
      size: SIM_PARAMS_BYTES,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      label: "sim-params",
    });
    device.queue.writeBuffer(paramsBuffer, 0, buildSimulationParams());

    const layout = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
        ...Array.from({ length: 12 }, (_, i) => i + 1).map((binding) => ({
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
        { binding: 7, resource: { buffer: this.indirect } },
        { binding: 8, resource: { buffer: brains } },
        { binding: 9, resource: { buffer: planned } },
        { binding: 10, resource: { buffer: killMarks } },
        { binding: 11, resource: { buffer: mateTargets } },
        { binding: 12, resource: { buffer: birthEvents } },
      ],
    });

    const module = device.createShaderModule({ code: SHADER });
    const pipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [layout] });
    const make = (entryPoint: PipelineName): GPUComputePipeline =>
      device.createComputePipeline({ layout: pipelineLayout, compute: { module, entryPoint } });
    this.pipelines = Object.fromEntries(
      PIPELINE_NAMES.map((entryPoint) => [entryPoint, make(entryPoint)]),
    ) as Record<PipelineName, GPUComputePipeline>;
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
  encode(
    encoder: GPUCommandEncoder,
    steps: number,
    timestampWrites?: GPUComputePassTimestampWrites,
  ): void {
    if (steps <= 0) {
      return;
    }
    const pass = encoder.beginComputePass({ label: "simulation", timestampWrites });
    pass.setBindGroup(0, this.bindGroup);
    // WebGPU orders dispatches in one compute pass; each pass sees prior storage writes.
    for (let i = 0; i < steps; i += 1) {
      this.dispatch(pass, "clearStep", this.clearWorkgroups);
      this.dispatch(pass, "count", this.agentWorkgroups);
      this.dispatch(pass, "scan", 1);
      this.dispatch(pass, "scatter", this.agentWorkgroups);
      this.dispatch(pass, "planMove", this.agentWorkgroups);
      this.dispatch(pass, "chooseContacts", this.agentWorkgroups);
      this.dispatch(pass, "resolveMates", this.agentWorkgroups);
      this.dispatch(pass, "commitAgents", this.agentWorkgroups);
      this.dispatch(pass, "spawnChildren", this.agentWorkgroups);
    }
    // Rebuild the live index after movement, deaths, and births so render sees final state.
    this.dispatch(pass, "clearIndex", this.clearWorkgroups);
    this.dispatch(pass, "count", this.agentWorkgroups);
    this.dispatch(pass, "scan", 1);
    this.dispatch(pass, "scatter", this.agentWorkgroups);
    this.dispatch(pass, "writeIndirect", 1);
    pass.end();
  }

  private dispatch(pass: GPUComputePassEncoder, name: PipelineName, workgroups: number): void {
    pass.setPipeline(this.pipelines[name]);
    pass.dispatchWorkgroups(workgroups);
  }
}

export function buildSimulationParams(): ArrayBuffer {
  const buf = new ArrayBuffer(SIM_PARAMS_BYTES);
  const f = new Float32Array(buf);
  const u = new Uint32Array(buf);
  f[0] = STEP_DT;
  f[1] = WORLD_SIZE;
  f[2] = AGENT_HIT_RADIUS;
  f[3] = (AGENT_HIT_RADIUS * 2) ** 2;
  f[4] = CONTACT_DOT;
  f[5] = HEAD_ON_DOT;
  f[6] = AGENT_MAX_TURN;
  f[7] = AGENT_MIN_SPEED;
  f[8] = AGENT_MAX_SPEED;
  f[9] = MUTATION_RATE;
  f[10] = MUTATION_SCALE;
  f[11] = MUTATION_WEIGHT_LIMIT;
  f[12] = SPEED_MUTATION_SCALE;
  f[13] = HUE_MUTATION_SCALE;
  f[14] = SENSOR_RADIUS;
  f[15] = 0;
  u[16] = MAX_AGENTS;
  u[17] = GRID_DIM;
  u[18] = NUM_CELLS;
  u[19] = NEURAL_NEIGHBORS;
  u[20] = BRAIN_WEIGHT_COUNT;
  u[21] = 0;
  u[22] = 0;
  u[23] = 0;
  return buf;
}

// Seed the initial population with random agents; the rest begin dead and can be
// filled only by collision breeding.
function writeInitialAgents(range: ArrayBuffer, count: number): void {
  const f = new Float32Array(range);
  const u = new Uint32Array(range);
  let seed = 0x9e3779b9;
  const rng = (): number => {
    seed = (Math.imul(seed ^ (seed >>> 15), 2246822519) + 1) >>> 0;
    return seed / 0xffffffff;
  };
  const initialAlive = Math.round(INITIAL_AGENTS);
  for (let i = 0; i < count; i += 1) {
    const b = i * AGENT_F32;
    const alive = i < initialAlive ? 1 : 0;
    f[b + 0] = rng() * WORLD_SIZE;
    f[b + 1] = rng() * WORLD_SIZE;
    f[b + 2] = rng() * Math.PI * 2;
    f[b + 3] = AGENT_MIN_SPEED + rng() * (AGENT_MAX_SPEED - AGENT_MIN_SPEED);
    f[b + 4] = rng();
    f[b + 5] = 0.85;
    f[b + 6] = 1.0;
    u[b + 7] = alive;
    u[b + 8] = (Math.imul(i + 1, 2654435761) >>> 0) || 1;
    u[b + 9] = 0;
  }
}

function writeInitialBrains(range: ArrayBuffer, count: number): void {
  const f = new Float32Array(range);
  let seed = 0x6a09e667;
  const rng = (): number => {
    seed = (Math.imul(seed ^ (seed >>> 16), 2246822507) + 0x9e3779b9) >>> 0;
    return seed / 0xffffffff;
  };
  for (let slot = 0; slot < count; slot += 1) {
    const base = slot * BRAIN_WEIGHT_COUNT;
    for (let i = 0; i < BRAIN_WEIGHT_COUNT; i += 1) {
      f[base + i] = (rng() * 2 - 1) * 0.5;
    }
  }
}
