import {
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
  POPULATION_FLOOR,
  MUTATION_RATE,
  MUTATION_SCALE,
  MUTATION_WEIGHT_LIMIT,
  SENSOR_RADIUS,
  SPEED_MUTATION_SCALE,
  STEP_DT,
  WORLD_SIZE,
  AGENT_HIT_RADIUS,
} from "./config";
import { AGENT_F32, SIM_PARAMS_BYTES } from "./layout";
import { NUM_CELLS } from "./simulationPolicy";

export const BRAIN_BYTES = BRAIN_WEIGHT_COUNT * 4;

export const SIM_PARAM_F32 = {
  dt: 0,
  worldSize: 1,
  hitRadius: 2,
  collisionDistanceSq: 3,
  contactDot: 4,
  headOnDot: 5,
  maxTurn: 6,
  minSpeed: 7,
  maxSpeed: 8,
  mutationRate: 9,
  mutationScale: 10,
  mutationWeightLimit: 11,
  speedMutationScale: 12,
  hueMutationScale: 13,
  sensorRadius: 14,
} as const;

export const SIM_PARAM_U32 = {
  maxAgents: 15,
  gridDim: 16,
  numCells: 17,
  populationFloor: 18,
} as const;

export function initialAliveCount(): number {
  return Math.round(INITIAL_AGENTS);
}

export function buildSimulationParams(): ArrayBuffer {
  const buf = new ArrayBuffer(SIM_PARAMS_BYTES);
  const f = new Float32Array(buf);
  const u = new Uint32Array(buf);
  f[SIM_PARAM_F32.dt] = STEP_DT;
  f[SIM_PARAM_F32.worldSize] = WORLD_SIZE;
  f[SIM_PARAM_F32.hitRadius] = AGENT_HIT_RADIUS;
  f[SIM_PARAM_F32.collisionDistanceSq] = (AGENT_HIT_RADIUS * 2) ** 2;
  f[SIM_PARAM_F32.contactDot] = CONTACT_DOT;
  f[SIM_PARAM_F32.headOnDot] = HEAD_ON_DOT;
  f[SIM_PARAM_F32.maxTurn] = AGENT_MAX_TURN;
  f[SIM_PARAM_F32.minSpeed] = AGENT_MIN_SPEED;
  f[SIM_PARAM_F32.maxSpeed] = AGENT_MAX_SPEED;
  f[SIM_PARAM_F32.mutationRate] = MUTATION_RATE;
  f[SIM_PARAM_F32.mutationScale] = MUTATION_SCALE;
  f[SIM_PARAM_F32.mutationWeightLimit] = MUTATION_WEIGHT_LIMIT;
  f[SIM_PARAM_F32.speedMutationScale] = SPEED_MUTATION_SCALE;
  f[SIM_PARAM_F32.hueMutationScale] = HUE_MUTATION_SCALE;
  f[SIM_PARAM_F32.sensorRadius] = SENSOR_RADIUS;
  u[SIM_PARAM_U32.maxAgents] = MAX_AGENTS;
  u[SIM_PARAM_U32.gridDim] = GRID_DIM;
  u[SIM_PARAM_U32.numCells] = NUM_CELLS;
  u[SIM_PARAM_U32.populationFloor] = POPULATION_FLOOR;
  return buf;
}

// Seed the initial population with random agents; the rest begin dead and are
// filled by collision breeding or random immigrants when below POPULATION_FLOOR.
export function writeInitialAgents(range: ArrayBuffer, count: number = MAX_AGENTS): void {
  const f = new Float32Array(range);
  const u = new Uint32Array(range);
  let seed = 0x9e3779b9;
  const rng = (): number => {
    seed = (Math.imul(seed ^ (seed >>> 15), 2246822519) + 1) >>> 0;
    return seed / 0xffffffff;
  };
  const initialAlive = initialAliveCount();
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

export function writeInitialBrains(range: ArrayBuffer, count: number = MAX_AGENTS): void {
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

export function writeInitialDense(
  denseRange: ArrayBuffer,
  agentsRange: ArrayBuffer,
  count: number = MAX_AGENTS,
): void {
  const denseF = new Float32Array(denseRange);
  const denseU = new Uint32Array(denseRange);
  const agentsF = new Float32Array(agentsRange);
  const initialAlive = Math.min(initialAliveCount(), count);
  for (let slot = 0; slot < initialAlive; slot += 1) {
    const agentBase = slot * AGENT_F32;
    const denseBase = slot * 4;
    denseF[denseBase + 0] = agentsF[agentBase + 0];
    denseF[denseBase + 1] = agentsF[agentBase + 1];
    denseU[denseBase + 2] = slot;
    denseU[denseBase + 3] = 0;
  }
}
