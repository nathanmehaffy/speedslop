import { GpuSimulation } from "./gpu-simulation";
import { DEFAULT_POPULATION, DEFAULT_WORLD_SIZE, INITIAL_SEED } from "./simulation-helpers";

const RESULT_MARKER = "SPEEDSLOP_SIM_BENCH_RESULT";
const ERROR_MARKER = "SPEEDSLOP_SIM_BENCH_ERROR";
const DEFAULT_STEPS = 1200;
const DEFAULT_WARMUP_STEPS = 120;
const DEFAULT_BATCH_SIZE = 20;

type BenchConfig = {
  population: number;
  worldSize: number;
  seed: number;
  steps: number;
  warmup: number;
  batch: number;
};

type AdapterDescription = {
  vendor?: string;
  architecture?: string;
  device?: string;
  description?: string;
};

function positiveIntegerParam(params: URLSearchParams, name: string, fallback: number): number {
  const raw = params.get(name);
  if (raw === null) {
    return fallback;
  }

  const value = Number(raw);
  if (!Number.isFinite(value) || value <= 0) {
    throw new Error(`Expected ${name} to be a positive number.`);
  }

  return Math.floor(value);
}

function nonNegativeIntegerParam(params: URLSearchParams, name: string, fallback: number): number {
  const raw = params.get(name);
  if (raw === null) {
    return fallback;
  }

  const value = Number(raw);
  if (!Number.isFinite(value) || value < 0) {
    throw new Error(`Expected ${name} to be zero or a positive number.`);
  }

  return Math.floor(value);
}

function parseConfig(): BenchConfig {
  const params = new URLSearchParams(window.location.search);

  return {
    population: positiveIntegerParam(params, "population", DEFAULT_POPULATION),
    worldSize: positiveIntegerParam(params, "world-size", DEFAULT_WORLD_SIZE),
    seed: nonNegativeIntegerParam(params, "seed", INITIAL_SEED),
    steps: positiveIntegerParam(params, "steps", DEFAULT_STEPS),
    warmup: nonNegativeIntegerParam(params, "warmup", DEFAULT_WARMUP_STEPS),
    batch: positiveIntegerParam(params, "batch", DEFAULT_BATCH_SIZE),
  };
}

function describeAdapter(adapter: GPUAdapter): AdapterDescription {
  const info = adapter.info as GPUAdapterInfo & AdapterDescription;
  const description: AdapterDescription = {};

  for (const key of ["vendor", "architecture", "device", "description"] as const) {
    const value = info[key];
    if (typeof value === "string" && value.length > 0) {
      description[key] = value;
    }
  }

  return description;
}

async function runSteps(device: GPUDevice, simulation: GpuSimulation, stepCount: number, batchSize: number): Promise<void> {
  let remaining = stepCount;

  while (remaining > 0) {
    const batchSteps = Math.min(batchSize, remaining);
    const encoder = device.createCommandEncoder({ label: "simulation-benchmark-steps" });
    simulation.encodeSteps(encoder, batchSteps);
    device.queue.submit([encoder.finish()]);
    await device.queue.onSubmittedWorkDone();
    remaining -= batchSteps;
  }
}

async function runBenchmark(): Promise<void> {
  if (!navigator.gpu) {
    throw new Error("WebGPU is not available in this browser.");
  }

  const config = parseConfig();
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });

  if (!adapter) {
    throw new Error("No compatible WebGPU adapter was found.");
  }

  const device = await adapter.requestDevice();
  const simulation = await GpuSimulation.create(device, {
    worldSize: config.worldSize,
    population: config.population,
    seed: config.seed,
  });

  try {
    await device.queue.onSubmittedWorkDone();
    await runSteps(device, simulation, config.warmup, config.batch);

    const startedAt = performance.now();
    await runSteps(device, simulation, config.steps, config.batch);
    const elapsedMs = performance.now() - startedAt;
    const stepsPerSecond = Math.round((config.steps * 1000) / Math.max(elapsedMs, 0.001));
    const agentUpdatesPerSecond = Math.round((config.steps * config.population * 1000) / Math.max(elapsedMs, 0.001));

    console.log(
      `${RESULT_MARKER} ${JSON.stringify({
        adapter: describeAdapter(adapter),
        population: config.population,
        worldSize: config.worldSize,
        seed: config.seed,
        warmupSteps: config.warmup,
        measuredSteps: config.steps,
        batchSize: config.batch,
        elapsedMs,
        stepsPerSecond,
        agentUpdatesPerSecond,
      })}`,
    );
  } finally {
    simulation.destroy();
    device.destroy();
  }
}

runBenchmark().catch((error: unknown) => {
  const message = error instanceof Error ? error.message : String(error);
  console.error(`${ERROR_MARKER} ${JSON.stringify({ message })}`);
});
