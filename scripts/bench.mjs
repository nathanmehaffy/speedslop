import { readFileSync } from "node:fs";
import { performance } from "node:perf_hooks";
import initWasm, { Simulation } from "../sim/pkg/speedslop.js";

const DEFAULT_WORLD_SIZE = 4096;
const DEFAULT_POPULATIONS = [1_000, 5_000, 10_000];
const DEFAULT_WARMUP_STEPS = 80;
const DEFAULT_TARGET_MS = 1_500;
const DEFAULT_BATCH_STEPS = 8;

function readNumberEnv(name, fallback) {
  const raw = process.env[name];
  if (!raw) {
    return fallback;
  }

  const value = Number(raw);
  return Number.isFinite(value) && value > 0 ? value : fallback;
}

function readPopulationList() {
  const raw = process.env.POPS;
  if (!raw) {
    return DEFAULT_POPULATIONS;
  }

  const populations = raw
    .split(",")
    .map((value) => Number(value.trim()))
    .filter((value) => Number.isInteger(value) && value > 0);

  return populations.length > 0 ? populations : DEFAULT_POPULATIONS;
}

function formatNumber(value) {
  return Math.round(value).toLocaleString("en-US");
}

function measureCore(population, warmupSteps, targetMs, batchSteps) {
  const simulation = new Simulation(DEFAULT_WORLD_SIZE, population, 1);
  simulation.advance_steps(warmupSteps);

  let steps = 0;
  const startedAt = performance.now();
  while (performance.now() - startedAt < targetMs) {
    simulation.advance_steps(batchSteps);
    steps += batchSteps;
  }
  const elapsedMs = performance.now() - startedAt;
  const stepsPerSecond = (steps * 1000) / elapsedMs;
  const deaths = simulation.deaths();
  const births = simulation.births();

  simulation.free();

  return {
    population,
    steps,
    elapsedMs,
    stepsPerSecond,
    deaths,
    births,
  };
}

function measureSnapshotCopy(wasm, population) {
  const simulation = new Simulation(DEFAULT_WORLD_SIZE, population, 1);
  simulation.refresh_render_agents();

  const agentF32Len = simulation.agent_f32_len();
  const bytes = agentF32Len * Float32Array.BYTES_PER_ELEMENT;
  const source = new Float32Array(wasm.memory.buffer, simulation.agent_ptr(), agentF32Len);
  const buffers = [new ArrayBuffer(bytes), new ArrayBuffer(bytes), new ArrayBuffer(bytes)];
  const repetitions = Math.max(100, Math.floor(1_000_000_000 / bytes));

  const startedAt = performance.now();
  for (let i = 0; i < repetitions; i += 1) {
    new Float32Array(buffers[i % buffers.length]).set(source);
  }
  const elapsedMs = performance.now() - startedAt;

  simulation.free();

  return {
    bytes,
    repetitions,
    copiesPerSecond: (repetitions * 1000) / elapsedMs,
    gbPerSecond: (bytes * repetitions) / elapsedMs / 1_000_000,
  };
}

const wasmBytes = readFileSync(new URL("../sim/pkg/speedslop_bg.wasm", import.meta.url));
const wasm = await initWasm({ module_or_path: wasmBytes });
const populations = readPopulationList();
const warmupSteps = readNumberEnv("WARMUP_STEPS", DEFAULT_WARMUP_STEPS);
const targetMs = readNumberEnv("TARGET_MS", DEFAULT_TARGET_MS);
const batchSteps = readNumberEnv("BATCH_STEPS", DEFAULT_BATCH_STEPS);

console.log(
  `Core benchmark: world=${DEFAULT_WORLD_SIZE}, warmup=${warmupSteps}, target_ms=${targetMs}, batch=${batchSteps}`,
);

for (const population of populations) {
  const result = measureCore(population, warmupSteps, targetMs, batchSteps);
  console.log(
    [
      `${formatNumber(result.population)} agents`,
      `${formatNumber(result.stepsPerSecond)} steps/s`,
      `${formatNumber(result.steps)} measured steps`,
      `${result.elapsedMs.toFixed(1)} ms`,
      `${formatNumber(result.deaths)} deaths`,
      `${formatNumber(result.births)} births`,
    ].join(" | "),
  );
}

console.log("");
console.log("Snapshot copy benchmark:");

for (const population of populations) {
  const result = measureSnapshotCopy(wasm, population);
  console.log(
    [
      `${formatNumber(population)} agents`,
      `${formatNumber(result.bytes)} bytes`,
      `${formatNumber(result.copiesPerSecond)} copies/s`,
      `${result.gbPerSecond.toFixed(2)} GB/s`,
    ].join(" | "),
  );
}
