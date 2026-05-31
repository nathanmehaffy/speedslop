import initWasm, { Simulation, type InitOutput } from "../sim/pkg/speedslop.js";
import type {
  MainToWorkerMessage,
  SimRate,
  SimulationStats,
  WorkerToMainMessage,
} from "./simulation-messages";

const REAL_TIME_MAX_DELTA_SECONDS = 0.25;
const TARGET_MAX_STEPS_PER_CHUNK = 24;
const MAX_MODE_BATCH_STEPS = 8;
// Snapshots are published on this fixed wall-clock heartbeat so motion is evenly paced at every
// sim rate. In Max mode the worker steps the simulation continuously between heartbeats; in target
// modes it sleeps until the next one.
const SNAPSHOT_INTERVAL_MS = 1000 / 120;
const STATS_INTERVAL_MS = 500;
const PAUSED_LOOP_DELAY_MS = 50;

type WorkerScope = {
  addEventListener(
    type: "message",
    listener: (event: MessageEvent<MainToWorkerMessage>) => void,
  ): void;
  postMessage(message: WorkerToMainMessage, transfer?: Transferable[]): void;
  setTimeout(handler: () => void, timeout?: number): number;
};

const workerScope = self as unknown as WorkerScope;

let wasm: InitOutput | null = null;
let simulation: Simulation | null = null;
let fixedStepSeconds = 1 / 60;
let paused = false;
let simRate: SimRate = 1;
let epoch = 0;
let availableBuffers: ArrayBuffer[] = [];
let snapshotDirty = false;
let nextSnapshotAt = performance.now();
let previousTickAt = performance.now();
let stepRemainderSeconds = 0;
let lastStatsAt = performance.now();
let lastStatsStepCount = 0;
let stepsPerSecond = 0;
let loopTimer: number | null = null;
let initializing = false;

function postMessage(message: WorkerToMainMessage, transfer: Transferable[] = []): void {
  workerScope.postMessage(message, transfer);
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

function postError(error: unknown): void {
  postMessage({ type: "error", message: errorMessage(error) });
}

function normalizeSimRate(value: SimRate): SimRate {
  if (value === "max") {
    return value;
  }

  return Number.isFinite(value) && value > 0 ? value : 1;
}

function resetStepClock(now = performance.now()): void {
  previousTickAt = now;
  stepRemainderSeconds = 0;
}

function readStats(): SimulationStats {
  if (!simulation) {
    return {
      population: 0,
      births: 0,
      deaths: 0,
      generation: 0,
      simSteps: 0,
      stepsPerSecond,
    };
  }

  return {
    population: simulation.population(),
    births: simulation.births(),
    deaths: simulation.deaths(),
    generation: simulation.generation(),
    simSteps: simulation.sim_steps(),
    stepsPerSecond,
  };
}

function stepDelta(current: number, previous: number): number {
  return current >= previous ? current - previous : current + 0xffffffff - previous + 1;
}

function maybePostStats(now: number, force = false): void {
  if (!simulation || (!force && now - lastStatsAt < STATS_INTERVAL_MS)) {
    return;
  }

  const currentSteps = simulation.sim_steps();
  const elapsedMs = Math.max(1, now - lastStatsAt);
  stepsPerSecond = Math.round((stepDelta(currentSteps, lastStatsStepCount) * 1000) / elapsedMs);
  lastStatsAt = now;
  lastStatsStepCount = currentSteps;

  postMessage({ type: "stats", epoch, stats: readStats() });
}

function advanceSimulationSteps(stepCount: number): void {
  if (!simulation) {
    return;
  }

  const steps = Math.max(0, Math.floor(stepCount));
  if (steps === 0) {
    return;
  }

  simulation.advance_steps(steps);
  snapshotDirty = true;
}

function publishSnapshot(): void {
  if (!simulation || !wasm || !snapshotDirty || availableBuffers.length === 0) {
    return;
  }

  const buffer = availableBuffers.pop();
  if (!buffer) {
    return;
  }

  simulation.refresh_render_agents();

  const agentF32Len = simulation.agent_f32_len();
  const expectedBytes = agentF32Len * Float32Array.BYTES_PER_ELEMENT;
  if (buffer.byteLength !== expectedBytes) {
    postError(`Snapshot buffer has ${buffer.byteLength} bytes; expected ${expectedBytes}.`);
    return;
  }

  const source = new Float32Array(wasm.memory.buffer, simulation.agent_ptr(), agentF32Len);
  new Float32Array(buffer).set(source);

  snapshotDirty = false;
  postMessage(
    {
      type: "snapshot",
      epoch,
      buffer,
      stats: readStats(),
      highlightIndex: simulation.oldest_agent_index(),
    },
    [buffer],
  );
}

function resetSnapshotClock(now = performance.now()): void {
  nextSnapshotAt = now + SNAPSHOT_INTERVAL_MS;
}

function runTargetRate(now: number): void {
  const elapsedSeconds = Math.min(
    Math.max(0, (now - previousTickAt) / 1000),
    REAL_TIME_MAX_DELTA_SECONDS,
  );
  previousTickAt = now;

  const rate = typeof simRate === "number" ? simRate : 1;
  stepRemainderSeconds += elapsedSeconds * rate;

  const stepsDue = Math.floor(stepRemainderSeconds / fixedStepSeconds);
  const stepsToRun = Math.min(stepsDue, TARGET_MAX_STEPS_PER_CHUNK);
  if (stepsToRun > 0) {
    advanceSimulationSteps(stepsToRun);
    stepRemainderSeconds -= stepsToRun * fixedStepSeconds;
  }
}

function runMaxRate(): void {
  stepRemainderSeconds = 0;

  // Step continuously until the next snapshot heartbeat, so each published frame advances a
  // similar amount of wall-clock time and the cadence stays even.
  do {
    advanceSimulationSteps(MAX_MODE_BATCH_STEPS);
  } while (performance.now() < nextSnapshotAt);
}

function scheduleLoop(delayMs: number): void {
  if (loopTimer !== null) {
    return;
  }

  loopTimer = workerScope.setTimeout(runLoop, delayMs);
}

function snapshotIsDue(now: number): boolean {
  return now >= nextSnapshotAt;
}

function advanceSnapshotClock(now: number): void {
  // Skip over any missed heartbeats (e.g. after a stall) so we resume on the grid instead of
  // emitting a catch-up burst.
  const missed = Math.floor((now - nextSnapshotAt) / SNAPSHOT_INTERVAL_MS) + 1;
  nextSnapshotAt += missed * SNAPSHOT_INTERVAL_MS;
}

function nextLoopDelayMs(now: number): number {
  if (paused) {
    return PAUSED_LOOP_DELAY_MS;
  }

  if (simRate === "max") {
    return 0;
  }

  return Math.max(0, Math.min(SNAPSHOT_INTERVAL_MS, nextSnapshotAt - now));
}

function runLoop(): void {
  loopTimer = null;

  try {
    if (!simulation) {
      return;
    }

    let now = performance.now();
    if (paused) {
      resetStepClock(now);
      resetSnapshotClock(now);
      maybePostStats(now);
      scheduleLoop(nextLoopDelayMs(now));
      return;
    }

    if (simRate === "max") {
      runMaxRate();
    } else {
      runTargetRate(now);
    }

    now = performance.now();
    if (snapshotIsDue(now)) {
      publishSnapshot();
      advanceSnapshotClock(now);
    }

    maybePostStats(now);
    scheduleLoop(nextLoopDelayMs(now));
  } catch (error) {
    postError(error);
  }
}

async function initialize(message: Extract<MainToWorkerMessage, { type: "init" }>): Promise<void> {
  if (initializing) {
    return;
  }

  initializing = true;
  epoch = message.epoch;
  paused = message.paused;
  simRate = normalizeSimRate(message.simRate);

  wasm = await initWasm();
  simulation = new Simulation(message.worldSize, message.population, message.seed);
  fixedStepSeconds = simulation.fixed_step_seconds();
  resetStepClock();

  lastStatsAt = performance.now();
  lastStatsStepCount = simulation.sim_steps();
  stepsPerSecond = 0;
  snapshotDirty = true;
  nextSnapshotAt = performance.now();

  postMessage({
    type: "ready",
    epoch,
    agentF32Len: simulation.agent_f32_len(),
    agentStrideF32: simulation.agent_stride_f32(),
    fixedStepSeconds,
    stats: readStats(),
  });
  scheduleLoop(0);
}

function resetSimulation(message: Extract<MainToWorkerMessage, { type: "reset" }>): void {
  epoch = message.epoch;

  if (!simulation) {
    return;
  }

  simulation.reset(message.seed);
  resetStepClock();

  lastStatsAt = performance.now();
  lastStatsStepCount = simulation.sim_steps();
  stepsPerSecond = 0;
  snapshotDirty = true;

  const now = performance.now();
  resetSnapshotClock(now);
  maybePostStats(now, true);
  publishSnapshot();
}

function returnSnapshotBuffer(buffer: ArrayBuffer): void {
  if (buffer.byteLength > 0) {
    availableBuffers.push(buffer);
  }
}

async function handleMessage(message: MainToWorkerMessage): Promise<void> {
  switch (message.type) {
    case "init":
      await initialize(message);
      break;
    case "setPaused":
      paused = message.paused;
      resetStepClock();
      scheduleLoop(0);
      break;
    case "setSimRate":
      simRate = normalizeSimRate(message.simRate);
      resetStepClock();
      scheduleLoop(0);
      break;
    case "reset":
      resetSimulation(message);
      scheduleLoop(0);
      break;
    case "returnSnapshotBuffer":
      returnSnapshotBuffer(message.buffer);
      scheduleLoop(0);
      break;
  }
}

workerScope.addEventListener("message", (event) => {
  void handleMessage(event.data).catch(postError);
});
