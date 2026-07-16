import {
  AGENT_MAX_SPEED,
  AGENT_MAX_TURN,
  AGENT_MIN_SPEED,
  BRAIN_WEIGHT_COUNT,
  NEURAL_HIDDEN,
  NEURAL_INPUTS,
  NEURAL_NEIGHBORS,
  SENSOR_RADIUS,
  STEP_DT,
  WORLD_SIZE,
} from "./config";
import type { NeuralTrace, ProbeAgent, ProbeScenario, ProbeTrace } from "./interpretabilityTypes";
import { classifyCollision } from "./collision";
import { toroidalDelta, toroidalDistanceSq } from "./spatial";

const TWO_PI = Math.PI * 2;

export function evaluateNeuralStep(
  focal: ProbeAgent,
  neighbors: readonly ProbeAgent[],
  step: number,
): NeuralTrace {
  assertGenomeSize(focal.genome);
  const nearest = nearestNeighbors(focal, neighbors);
  const inputs = buildInputs(focal, nearest, step);
  const hidden: number[] = [];
  for (let h = 0; h < NEURAL_HIDDEN; h += 1) {
    let value = 0;
    const weightBase = h * NEURAL_INPUTS;
    for (let i = 0; i < NEURAL_INPUTS; i += 1) {
      value += inputs[i] * focal.genome[weightBase + i];
    }
    hidden.push(squash(value));
  }

  const outBase = NEURAL_INPUTS * NEURAL_HIDDEN;
  let turnRaw = 0;
  let speedRaw = 0;
  for (let h = 0; h < NEURAL_HIDDEN; h += 1) {
    turnRaw += hidden[h] * focal.genome[outBase + h];
    speedRaw += hidden[h] * focal.genome[outBase + NEURAL_HIDDEN + h];
  }
  const nextDir = wrapAngle(focal.dir + squash(turnRaw) * AGENT_MAX_TURN);
  const speed01 = squash(speedRaw) * 0.5 + 0.5;
  const nextVel = AGENT_MIN_SPEED + speed01 * (AGENT_MAX_SPEED - AGENT_MIN_SPEED);
  return { inputs, hidden, turnRaw, speedRaw, nextDir, nextVel };
}

export function runProbeScenario(scenario: ProbeScenario): ProbeTrace {
  let focal = { ...scenario.focal, genome: scenario.focal.genome.slice() };
  const neighbors = scenario.neighbors.map((neighbor) => ({ ...neighbor, genome: neighbor.genome.slice() }));
  const positions = [{ x: focal.x, y: focal.y }];
  const neural: NeuralTrace[] = [];
  const collisionKinds: string[] = [];
  for (let step = 0; step < scenario.steps; step += 1) {
    const trace = evaluateNeuralStep(focal, neighbors, step);
    neural.push(trace);
    focal = {
      ...focal,
      dir: trace.nextDir,
      vel: trace.nextVel,
      x: wrapPosition(focal.x + Math.cos(trace.nextDir) * trace.nextVel * STEP_DT),
      y: wrapPosition(focal.y + Math.sin(trace.nextDir) * trace.nextVel * STEP_DT),
    };
    positions.push({ x: focal.x, y: focal.y });
    const firstCollision = neighbors
      .map((neighbor) => classifyCollision(focal.x, focal.y, focal.dir, neighbor.x, neighbor.y, neighbor.dir).kind)
      .find((kind) => kind !== "none") ?? "none";
    collisionKinds.push(firstCollision);
  }
  return { scenarioName: scenario.name, positions, neural, collisionKinds };
}

export function standardProbeScenarios(focal: ProbeAgent): ProbeScenario[] {
  const center = { ...focal, x: WORLD_SIZE / 2, y: WORLD_SIZE / 2, dir: 0, vel: AGENT_MIN_SPEED };
  const neighbor = (x: number, y: number, dir: number): ProbeAgent => ({
    id: 900_000 + Math.floor(x * 1000) + Math.floor(y * 100),
    x,
    y,
    dir,
    vel: AGENT_MIN_SPEED,
    genome: focal.genome,
  });
  return [
    { name: "Alone", focal: center, neighbors: [], steps: 80 },
    { name: "Neighbor ahead", focal: center, neighbors: [neighbor(center.x + SENSOR_RADIUS * 0.45, center.y, Math.PI)], steps: 80 },
    { name: "Neighbor behind", focal: center, neighbors: [neighbor(center.x - SENSOR_RADIUS * 0.45, center.y, 0)], steps: 80 },
    { name: "Head-on", focal: center, neighbors: [neighbor(center.x + SENSOR_RADIUS * 0.25, center.y, Math.PI)], steps: 80 },
    {
      name: "Crowd",
      focal: center,
      neighbors: [
        neighbor(center.x + SENSOR_RADIUS * 0.35, center.y, Math.PI),
        neighbor(center.x, center.y + SENSOR_RADIUS * 0.35, -Math.PI / 2),
        neighbor(center.x - SENSOR_RADIUS * 0.35, center.y, 0),
        neighbor(center.x, center.y - SENSOR_RADIUS * 0.35, Math.PI / 2),
      ],
      steps: 80,
    },
  ];
}

function buildInputs(focal: ProbeAgent, nearest: readonly ProbeAgent[], step: number): number[] {
  const inputs = new Array(NEURAL_INPUTS).fill(0);
  inputs[0] = 1;
  inputs[1] = speedNorm(focal.vel);
  inputs[2] = selfNoise(focal.id, step);
  const afx = Math.cos(focal.dir);
  const afy = Math.sin(focal.dir);
  const arx = -afy;
  const ary = afx;
  for (let n = 0; n < Math.min(NEURAL_NEIGHBORS, nearest.length); n += 1) {
    const other = nearest[n];
    const dx = toroidalDelta(focal.x, other.x, WORLD_SIZE);
    const dy = toroidalDelta(focal.y, other.y, WORLD_SIZE);
    const dist = Math.max(Math.hypot(dx, dy), 1e-9);
    const dirX = dx / dist;
    const dirY = dy / dist;
    const otherForwardX = Math.cos(other.dir);
    const otherForwardY = Math.sin(other.dir);
    const base = 3 + n * 6;
    inputs[base + 0] = 1;
    inputs[base + 1] = clamp(1 - dist / SENSOR_RADIUS, 0, 1);
    inputs[base + 2] = afx * dirX + afy * dirY;
    inputs[base + 3] = arx * dirX + ary * dirY;
    inputs[base + 4] = afx * otherForwardX + afy * otherForwardY;
    inputs[base + 5] = speedNorm(other.vel);
  }
  return inputs;
}

function nearestNeighbors(focal: ProbeAgent, neighbors: readonly ProbeAgent[]): ProbeAgent[] {
  return neighbors
    .filter((neighbor) => toroidalDistanceSq(focal.x, focal.y, neighbor.x, neighbor.y, WORLD_SIZE) <= SENSOR_RADIUS * SENSOR_RADIUS)
    .sort((a, b) =>
      toroidalDistanceSq(focal.x, focal.y, a.x, a.y, WORLD_SIZE) -
      toroidalDistanceSq(focal.x, focal.y, b.x, b.y, WORLD_SIZE)
    )
    .slice(0, NEURAL_NEIGHBORS);
}

function speedNorm(vel: number): number {
  return clamp(((vel - AGENT_MIN_SPEED) / Math.max(AGENT_MAX_SPEED - AGENT_MIN_SPEED, 0.000001)) * 2 - 1, -1, 1);
}

function selfNoise(id: number, step: number): number {
  return randf(hash2((id ^ 0x51ed270b) >>> 0, step >>> 0)) * 2 - 1;
}

function pcg(v: number): number {
  let s = (Math.imul(v >>> 0, 747796405) + 2891336453) >>> 0;
  s = (Math.imul(((s >>> ((s >>> 28) + 4)) ^ s) >>> 0, 277803737)) >>> 0;
  return ((s >>> 22) ^ s) >>> 0;
}

function hash2(a: number, b: number): number {
  return pcg((a + pcg(b)) >>> 0);
}

function randf(seed: number): number {
  return pcg(seed) / 4294967296;
}

function squash(x: number): number {
  return x / (1 + Math.abs(x));
}

function wrapPosition(x: number): number {
  return x - Math.floor(x / WORLD_SIZE) * WORLD_SIZE;
}

function wrapAngle(angle: number): number {
  return angle - Math.floor(angle / TWO_PI) * TWO_PI;
}

function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value));
}

function assertGenomeSize(genome: Float32Array): void {
  if (genome.length !== BRAIN_WEIGHT_COUNT) {
    throw new Error(`probe genome has ${genome.length} weights, expected ${BRAIN_WEIGHT_COUNT}`);
  }
}
