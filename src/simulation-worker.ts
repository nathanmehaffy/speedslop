import initWasm, { Simulation, type InitOutput } from "../sim/pkg/speedslop.js";
import type {
  MainToWorkerMessage,
  SimRate,
  SimulationStats,
  WorkerToMainMessage,
} from "./simulation-messages";

const REAL_TIME_MAX_DELTA_SECONDS = 0.25;
const TARGET_MAX_STEPS_PER_CHUNK = 24;
const MAX_MODE_CHUNK_MS = 8;
const MAX_MODE_BATCH_STEPS = 8;
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
let lastSnapshotAt = Number.NEGATIVE_INFINITY;
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

function maybePublishSnapshot(now: number, force = false): void {
  if (!simulation || !wasm || availableBuffers.length === 0) {
    return;
  }

  if (!snapshotDirty && !force) {
    return;
  }

  if (!force && now - lastSnapshotAt < SNAPSHOT_INTERVAL_MS) {
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
  lastSnapshotAt = now;

  postMessage({ type: "snapshot", epoch, buffer, stats: readStats() }, [buffer]);
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

function runMaxRate(now: number): void {
  previousTickAt = now;
  stepRemainderSeconds = 0;

  const deadline = now + MAX_MODE_CHUNK_MS;
  do {
    advanceSimulationSteps(MAX_MODE_BATCH_STEPS);
  } while (performance.now() < deadline);
}

function scheduleLoop(delayMs: number): void {
  if (loopTimer !== null) {
    return;
  }

  loopTimer = workerScope.setTimeout(runLoop, delayMs);
}

function runLoop(): void {
  loopTimer = null;

  try {
    if (!simulation) {
      return;
    }

    const now = performance.now();
    if (!paused) {
      if (simRate === "max") {
        runMaxRate(now);
      } else {
        runTargetRate(now);
      }
    } else {
      resetStepClock(now);
    }

    maybePublishSnapshot(performance.now());
    maybePostStats(performance.now());
    scheduleLoop(paused ? PAUSED_LOOP_DELAY_MS : 0);
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
  lastSnapshotAt = Number.NEGATIVE_INFINITY;

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
  lastSnapshotAt = Number.NEGATIVE_INFINITY;

  const now = performance.now();
  maybePostStats(now, true);
  maybePublishSnapshot(now, true);
}

function returnSnapshotBuffer(buffer: ArrayBuffer): void {
  if (buffer.byteLength > 0) {
    availableBuffers.push(buffer);
  }

  maybePublishSnapshot(performance.now());
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
