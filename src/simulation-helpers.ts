export const DEFAULT_WORLD_SIZE = 8192;
export const DEFAULT_POPULATION = 10_000;
export const MAX_POPULATION = 100_000;
export const INITIAL_SEED = 1;

export const FIXED_STEP_SECONDS = 1 / 60;
export const TARGET_MAX_STEPS_PER_FRAME = 8;
export const MAX_MODE_STEPS_PER_FRAME = 8;

export const RAY_COUNT = 9;
export const RAY_INPUTS = 6;
export const SELF_INPUTS = 5;
export const INPUT_COUNT = RAY_COUNT * RAY_INPUTS + SELF_INPUTS;
export const HIDDEN_COUNT = 8;
export const OUTPUT_COUNT = 5;
export const GENOME_LEN = HIDDEN_COUNT * (INPUT_COUNT + 1) + OUTPUT_COUNT * (HIDDEN_COUNT + 1);

export const MIN_SPEED = 15;
export const MAX_SPEED = 80;
export const GRID_CELL_SIZE = 64;

export type SimRate = number | "max";

export type SimulationStats = {
  population: number;
  births: number;
  deaths: number;
  generation: number;
  simSteps: number;
  stepsPerSecond: number;
};

export type CameraState = {
  centerX: number;
  centerY: number;
  zoom: number;
};

export function sanitizeWorldSize(worldSize: number): number {
  return Number.isFinite(worldSize) && worldSize >= 128 ? worldSize : DEFAULT_WORLD_SIZE;
}

export function sanitizePopulation(population: number): number {
  if (!Number.isFinite(population) || population <= 0) {
    return DEFAULT_POPULATION;
  }

  return Math.min(MAX_POPULATION, Math.max(1, Math.floor(population)));
}

export function gridColsForWorld(worldSize: number): number {
  return Math.max(1, Math.ceil(sanitizeWorldSize(worldSize) / GRID_CELL_SIZE));
}

export function wrapUnit(value: number): number {
  return value - Math.floor(value);
}

export function wrapNear(value: number, worldSize: number): number {
  if (value >= worldSize) {
    return value - worldSize;
  }

  if (value < 0) {
    return value + worldSize;
  }

  return value;
}

export function wrapDelta(delta: number, worldSize: number): number {
  const half = worldSize * 0.5;

  if (delta > half) {
    return delta - worldSize;
  }

  if (delta < -half) {
    return delta + worldSize;
  }

  return delta;
}

export function outputToColor(output: number): number {
  return Math.min(1, Math.max(0, output * 0.5 + 0.5));
}

export function fastTanh(value: number): number {
  if (value < -3) {
    return -1;
  }

  if (value > 3) {
    return 1;
  }

  const squared = value * value;
  return Math.min(1, Math.max(-1, (value * (27 + squared)) / (27 + 9 * squared)));
}

export function normalizeSimRate(value: SimRate): SimRate {
  if (value === "max") {
    return value;
  }

  return Number.isFinite(value) && value > 0 ? value : 1;
}

export function stepsDueForFrame(
  elapsedSeconds: number,
  simRate: SimRate,
  stepRemainderSeconds: number,
  maxModeStepsPerFrame = MAX_MODE_STEPS_PER_FRAME,
): { steps: number; remainder: number } {
  if (simRate === "max") {
    return { steps: Math.max(1, Math.floor(maxModeStepsPerFrame)), remainder: 0 };
  }

  const boundedElapsed = Math.min(Math.max(0, elapsedSeconds), 0.25);
  let remainder = stepRemainderSeconds + boundedElapsed * simRate;
  const due = Math.floor(remainder / FIXED_STEP_SECONDS);
  const steps = Math.min(due, TARGET_MAX_STEPS_PER_FRAME);
  remainder -= steps * FIXED_STEP_SECONDS;

  if (steps === TARGET_MAX_STEPS_PER_FRAME) {
    remainder = Math.min(remainder, FIXED_STEP_SECONDS);
  }

  return { steps, remainder };
}
