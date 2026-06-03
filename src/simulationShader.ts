import {
  BRAIN_WEIGHT_COUNT,
  GRID_DIM,
  NEURAL_HIDDEN,
  NEURAL_INPUTS,
  NEURAL_NEIGHBORS,
  WORLD_SIZE,
} from "./config";
import {
  AGENT_STRUCT_WGSL,
  BIRTH_EVENT_STRUCT_WGSL,
  DENSE_STRUCT_WGSL,
  PLANNED_STRUCT_WGSL,
} from "./layout";
import {
  NUM_CELLS,
  SCAN_CHUNK,
  SCAN_WORKGROUP_SIZE,
  SENSOR_CELL_RADIUS,
  WORKGROUP_SIZE,
} from "./simulationPolicy";

export const PIPELINE_NAMES = [
  "clearStep",
  "count",
  "scan",
  "scatter",
  "planMove",
  "chooseContacts",
  "commitContacts",
  "spawnChildren",
  "spawnImmigrants",
] as const;

export type PipelineName = (typeof PIPELINE_NAMES)[number];

export const SHADER = /* wgsl */ `
const NUM_CELLS: u32 = ${NUM_CELLS}u;
const SCAN_CHUNK: u32 = ${SCAN_CHUNK}u;
const SENSOR_CELL_RADIUS: i32 = ${SENSOR_CELL_RADIUS};
const GRID_DIM: u32 = ${GRID_DIM}u;
const WORLD_SIZE: f32 = ${WORLD_SIZE};
const INV_WORLD_SIZE: f32 = ${1 / WORLD_SIZE};
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
  maxAgents: u32,
  gridDim: u32,
  numCells: u32,
  populationFloor: u32,
  pad0: u32, // pads the uniform struct to a 16-byte multiple
}

${AGENT_STRUCT_WGSL}
${DENSE_STRUCT_WGSL}
${PLANNED_STRUCT_WGSL}
${BIRTH_EVENT_STRUCT_WGSL}

struct Meta {
  step: u32,
  freeCount: atomic<u32>,
  birthCount: atomic<u32>,
  childCount: atomic<u32>,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read_write> agents: array<Agent>;
@group(0) @binding(2) var<storage, read_write> globals: Meta;
@group(0) @binding(3) var<storage, read_write> cellCount: array<atomic<u32>>;
@group(0) @binding(4) var<storage, read_write> cellStart: array<u32>;
@group(0) @binding(5) var<storage, read_write> dense: array<Dense>;
@group(0) @binding(6) var<storage, read_write> freeList: array<u32>;
@group(0) @binding(7) var<storage, read_write> brains: array<f32>;
@group(0) @binding(8) var<storage, read_write> planned: array<Planned>;
@group(0) @binding(9) var<storage, read_write> killMarks: array<u32>;
@group(0) @binding(10) var<storage, read_write> mateTargets: array<u32>;
@group(0) @binding(11) var<storage, read_write> birthEvents: array<BirthEvent>;

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

fn unitFromBits(bits: u32) -> f32 {
  return f32(bits) * (1.0 / 4294967296.0);
}

fn unit16(bits: u32) -> f32 {
  return f32(bits & 0xffffu) * (1.0 / 65536.0);
}

fn wrapPos(p: vec2f) -> vec2f {
  return p - floor(p * INV_WORLD_SIZE) * WORLD_SIZE;
}

fn torusDelta(a: vec2f, b: vec2f) -> vec2f {
  var d = b - a;
  d = d - WORLD_SIZE * round(d * INV_WORLD_SIZE);
  return d;
}

fn cellCoord(p: vec2f) -> vec2u {
  let dim = f32(GRID_DIM);
  var cx = i32(floor(p.x * INV_WORLD_SIZE * dim));
  var cy = i32(floor(p.y * INV_WORLD_SIZE * dim));
  cx = clamp(cx, 0, i32(GRID_DIM) - 1);
  cy = clamp(cy, 0, i32(GRID_DIM) - 1);
  return vec2u(u32(cx), u32(cy));
}

fn cellOf(p: vec2f) -> u32 {
  let c = cellCoord(p);
  return c.y * GRID_DIM + c.x;
}

fn wrapCell(c: i32) -> u32 {
  let dim = i32(GRID_DIM);
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
  return wrapCell(cy) * GRID_DIM + wrapCell(cx);
}

fn speedNorm(vel: f32) -> f32 {
  return clamp((vel - params.minSpeed) / max(params.maxSpeed - params.minSpeed, 0.000001) * 2.0 - 1.0, -1.0, 1.0);
}

fn writeRandomBrain(slot: u32, seed: u32) {
  let base = slot * ${BRAIN_WEIGHT_COUNT}u;
  var rng = seed ^ (slot * 0x9e3779b9u);
  for (var i = 0u; i < ${BRAIN_WEIGHT_COUNT}u; i = i + 1u) {
    rng = pcg(rng + i);
    brains[base + i] = (unitFromBits(rng) * 2.0 - 1.0) * 0.5;
  }
}

fn spawnRandomAgent(slot: u32, seed: u32) -> Agent {
  var rng = hash2(seed, slot ^ 0x85ebca6bu);
  var a: Agent;
  rng = pcg(rng);
  a.pos.x = unitFromBits(rng) * WORLD_SIZE;
  rng = pcg(rng ^ 0x27d4eb2du);
  a.pos.y = unitFromBits(rng) * WORLD_SIZE;
  rng = pcg(rng ^ 0x632be5abu);
  a.dir = unitFromBits(rng) * TWO_PI;
  rng = pcg(rng ^ 0x165667b1u);
  a.vel = params.minSpeed + unitFromBits(rng) * (params.maxSpeed - params.minSpeed);
  rng = pcg(rng ^ 0x85157af5u ^ globals.step);
  a.hue = unitFromBits(rng);
  a.sat = 0.85;
  a.val = 1.0;
  a.alive = 1u;
  a.id = pcg(slot ^ globals.step ^ seed);
  return a;
}

fn squash(x: f32) -> f32 {
  return x / (1.0 + abs(x));
}

fn brainAt(slot: u32, weight: u32) -> f32 {
  return brains[slot * ${BRAIN_WEIGHT_COUNT}u + weight];
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

@compute @workgroup_size(${WORKGROUP_SIZE})
fn clearStep(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i < params.numCells) {
    atomicStore(&cellCount[i], 0u);
  }
  if (i < params.maxAgents) {
    killMarks[i] = 0u;
    mateTargets[i] = NO_TARGET;
  }
  if (i == 0u) {
    globals.step = globals.step + 1u;
    atomicStore(&globals.freeCount, 0u);
    atomicStore(&globals.birthCount, 0u);
    atomicStore(&globals.childCount, 0u);
  }
}

@compute @workgroup_size(${WORKGROUP_SIZE})
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

var<workgroup> blockSum: array<u32, ${SCAN_WORKGROUP_SIZE}>;
var<workgroup> blockOff: array<u32, ${SCAN_WORKGROUP_SIZE}>;

@compute @workgroup_size(${SCAN_WORKGROUP_SIZE})
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
    for (var i = 0u; i < ${SCAN_WORKGROUP_SIZE}u; i = i + 1u) {
      blockOff[i] = acc;
      acc = acc + blockSum[i];
    }
    cellStart[NUM_CELLS] = acc;
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

@compute @workgroup_size(${WORKGROUP_SIZE})
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

@compute @workgroup_size(${WORKGROUP_SIZE})
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

  let sensorRadiusSq = params.sensorRadius * params.sensorRadius;
  let cc = cellCoord(a.pos);
  for (var oy = -SENSOR_CELL_RADIUS; oy <= SENSOR_CELL_RADIUS; oy = oy + 1) {
    for (var ox = -SENSOR_CELL_RADIUS; ox <= SENSOR_CELL_RADIUS; ox = ox + 1) {
      let c = wrappedCellOf(i32(cc.x) + ox, i32(cc.y) + oy);
      let start = cellStart[c];
      let end = cellStart[c + 1u];
      for (var di = start; di < end; di = di + 1u) {
        let otherSlot = dense[di].slot;
        if (otherSlot == slot) {
          continue;
        }
        let delta = torusDelta(a.pos, dense[di].pos);
        let distSq = dot(delta, delta);
        if (distSq > sensorRadiusSq) {
          continue;
        }
        if (distSq < bestDist[3]) {
          if (distSq < bestDist[1]) {
            if (distSq < bestDist[0]) {
              bestDist[3] = bestDist[2];
              bestSlot[3] = bestSlot[2];
              bestDist[2] = bestDist[1];
              bestSlot[2] = bestSlot[1];
              bestDist[1] = bestDist[0];
              bestSlot[1] = bestSlot[0];
              bestDist[0] = distSq;
              bestSlot[0] = otherSlot;
            } else {
              bestDist[3] = bestDist[2];
              bestSlot[3] = bestSlot[2];
              bestDist[2] = bestDist[1];
              bestSlot[2] = bestSlot[1];
              bestDist[1] = distSq;
              bestSlot[1] = otherSlot;
            }
          } else if (distSq < bestDist[2]) {
            bestDist[3] = bestDist[2];
            bestSlot[3] = bestSlot[2];
            bestDist[2] = distSq;
            bestSlot[2] = otherSlot;
          } else {
            bestDist[3] = distSq;
            bestSlot[3] = otherSlot;
          }
        }
      }
    }
  }

  let selfSpeed = speedNorm(a.vel);
  let selfNoise = randf(hash2(a.id ^ 0x51ed270bu, globals.step)) * 2.0 - 1.0;
  let af = vec2f(cos(a.dir), sin(a.dir));
  let ar = vec2f(-af.y, af.x);
  var hidden: array<f32, ${NEURAL_HIDDEN}>;
  for (var h = 0u; h < ${NEURAL_HIDDEN}u; h = h + 1u) {
    let weightBase = h * ${NEURAL_INPUTS}u;
    hidden[h] =
      brainAt(slot, weightBase) +
      selfSpeed * brainAt(slot, weightBase + 1u) +
      selfNoise * brainAt(slot, weightBase + 2u);
  }

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
    let f1 = clamp(1.0 - dist / params.sensorRadius, 0.0, 1.0);
    let f2 = dot(af, dirToOther);
    let f3 = dot(ar, dirToOther);
    let f4 = dot(af, otherForward);
    let f5 = speedNorm(other.vel);
    for (var h = 0u; h < ${NEURAL_HIDDEN}u; h = h + 1u) {
      let wb = h * ${NEURAL_INPUTS}u + base;
      hidden[h] = hidden[h] +
        brainAt(slot, wb) +
        f1 * brainAt(slot, wb + 1u) +
        f2 * brainAt(slot, wb + 2u) +
        f3 * brainAt(slot, wb + 3u) +
        f4 * brainAt(slot, wb + 4u) +
        f5 * brainAt(slot, wb + 5u);
    }
  }

  for (var h = 0u; h < ${NEURAL_HIDDEN}u; h = h + 1u) {
    hidden[h] = squash(hidden[h]);
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

@compute @workgroup_size(${WORKGROUP_SIZE})
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
        } else if (aImpact >= params.contactDot && distSq < killDist) {
          // The deeper impact is the aggressor and dies; ties break by slot so
          // exactly one of a symmetric pair is removed.
          if (aImpact > bImpact || (aImpact == bImpact && slot < otherSlot)) {
            killDist = distSq;
            killSelf = 1u;
          }
        }
      }
    }
  }

  if (killSelf == 1u) {
    killMarks[slot] = 1u;
  } else {
    mateTargets[slot] = mate;
  }
}

@compute @workgroup_size(${WORKGROUP_SIZE})
fn commitContacts(@builtin(global_invocation_id) gid: vec3u) {
  let slot = gid.x;
  if (slot >= params.maxAgents) {
    return;
  }

  var a = agents[slot];
  if (a.alive != 1u) {
    return;
  }

  let mate = mateTargets[slot];
  if (mate != NO_TARGET) {
    if (slot < mate) {
      let slotAliveForBirth = killMarks[slot] == 0u;
      let mateAliveForBirth = killMarks[mate] == 0u;
      if (slotAliveForBirth && mateAliveForBirth && mateTargets[mate] == slot) {
        let e = atomicAdd(&globals.birthCount, 1u);
        if (e < params.maxAgents) {
          birthEvents[e].parentA = slot;
          birthEvents[e].parentB = mate;
        }
      }
    }
  }

  if (killMarks[slot] != 0u) {
    a.alive = 0u;
    agents[slot] = a;
    let f = atomicAdd(&globals.freeCount, 1u);
    freeList[f] = slot;
    return;
  }

  let p = planned[slot];
  a.pos = p.pos;
  a.dir = p.dir;
  a.vel = p.vel;
  agents[slot] = a;
}

@compute @workgroup_size(${WORKGROUP_SIZE})
fn spawnChildren(@builtin(global_invocation_id) gid: vec3u) {
  let t = gid.x;
  let birthCount = min(atomicLoad(&globals.birthCount), params.maxAgents);
  if (t >= birthCount) {
    return;
  }

  let event = birthEvents[t];
  let a = agents[event.parentA];
  let b = agents[event.parentB];
  if (a.alive != 1u || b.alive != 1u) {
    return;
  }

  let seed = hash2(t ^ 0x9e3779b9u, globals.step);
  let childOrdinal = atomicAdd(&globals.childCount, 1u);
  let freeCount = atomicLoad(&globals.freeCount);
  var childSlot: u32;
  if (childOrdinal < freeCount) {
    childSlot = freeList[childOrdinal];
  } else if (randf(seed) < 0.5) {
    childSlot = event.parentA;
  } else {
    childSlot = event.parentB;
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

  let childBase = childSlot * ${BRAIN_WEIGHT_COUNT}u;
  let aBase = event.parentA * ${BRAIN_WEIGHT_COUNT}u;
  let bBase = event.parentB * ${BRAIN_WEIGHT_COUNT}u;
  var rng = child.id ^ globals.step ^ (t * 0x9e3779b9u);
  for (var i = 0u; i < ${BRAIN_WEIGHT_COUNT}u; i = i + 1u) {
    rng = pcg(rng + i);
    let inherited = select(brains[bBase + i], brains[aBase + i], (rng & 1u) == 0u);
    var w = inherited;
    if (unit16(rng >> 1u) < params.mutationRate) {
      rng = pcg(rng ^ 0x165667b1u);
      w = w + (unitFromBits(rng) * 2.0 - 1.0) * params.mutationScale;
    }
    brains[childBase + i] = clamp(w, -params.mutationWeightLimit, params.mutationWeightLimit);
  }

  agents[childSlot] = child;
}

@compute @workgroup_size(${WORKGROUP_SIZE})
fn spawnImmigrants(@builtin(global_invocation_id) gid: vec3u) {
  let t = gid.x;
  let freeCount = atomicLoad(&globals.freeCount);
  let consumedByChildren = min(atomicLoad(&globals.childCount), freeCount);
  let remainingFree = freeCount - consumedByChildren;
  let live = params.maxAgents - remainingFree;
  if (live >= params.populationFloor) {
    return;
  }
  let needed = min(params.populationFloor - live, remainingFree);
  if (t >= needed) {
    return;
  }

  let slot = freeList[consumedByChildren + t];
  let seed = hash2(slot ^ globals.step, 0xdeadbeefu ^ t);
  agents[slot] = spawnRandomAgent(slot, seed);
  writeRandomBrain(slot, seed);
}
`;
