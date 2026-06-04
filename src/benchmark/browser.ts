import { Camera, type Viewport } from "../camera";
import {
  GRID_DIM,
  MAX_AGENTS,
  MAX_DEVICE_PIXEL_RATIO,
  WORLD_SIZE,
} from "../config";
import { initGpu, installGpuErrorHandlers, resizeCanvas } from "../gpu";
import { Renderer } from "../renderer";
import { Simulation } from "../simulation";
import { PIPELINE_NAMES } from "../simulationShader";

const TIMESTAMP_COUNT = 2;
const TIMESTAMP_BYTES = TIMESTAMP_COUNT * 8;

export interface BenchmarkOptions {
  samples?: number;
  warmup?: number;
  width?: number;
  height?: number;
  stepBatches?: number[];
  profileStages?: boolean;
}

export interface BenchmarkStats {
  min: number;
  median: number;
  mean: number;
  p95: number;
  max: number;
}

export interface BenchmarkCaseResult {
  name: string;
  workload: Record<string, number | string>;
  samples: number;
  warmup: number;
  gpuMs: BenchmarkStats;
  cpuEncodeMs: BenchmarkStats;
  stepsPerSecond?: number;
}

export interface BenchmarkReport {
  version: 1;
  createdAt: string;
  environment: {
    userAgent: string;
    devicePixelRatio: number;
    canvasWidth: number;
    canvasHeight: number;
    maxAgents: number;
    gridDim: number;
    worldSize: number;
  };
  options: Required<BenchmarkOptions>;
  cases: BenchmarkCaseResult[];
}

declare global {
  interface Window {
    speedSlopRunBenchmark: (options?: BenchmarkOptions) => Promise<BenchmarkReport>;
  }
}

interface TimingSample {
  gpuMs: number;
  cpuEncodeMs: number;
}

const DEFAULT_OPTIONS: Required<BenchmarkOptions> = {
  samples: 25,
  warmup: 5,
  width: 1280,
  height: 720,
  stepBatches: [1, 4, 16, 64],
  profileStages: false,
};

class TimestampTimer {
  private readonly querySet: GPUQuerySet;
  private readonly resolveBuffer: GPUBuffer;
  private readonly readbackBuffer: GPUBuffer;

  constructor(private readonly device: GPUDevice) {
    this.querySet = device.createQuerySet({ type: "timestamp", count: TIMESTAMP_COUNT });
    this.resolveBuffer = device.createBuffer({
      size: TIMESTAMP_BYTES,
      usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC,
      label: "benchmark-timestamp-resolve",
    });
    this.readbackBuffer = device.createBuffer({
      size: TIMESTAMP_BYTES,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
      label: "benchmark-timestamp-readback",
    });
  }

  async measure(encode: (encoder: GPUCommandEncoder, querySet: GPUQuerySet) => void): Promise<TimingSample> {
    const encoder = this.device.createCommandEncoder({ label: "benchmark-encoder" });
    const cpuStart = performance.now();
    encode(encoder, this.querySet);
    encoder.resolveQuerySet(this.querySet, 0, TIMESTAMP_COUNT, this.resolveBuffer, 0);
    encoder.copyBufferToBuffer(this.resolveBuffer, 0, this.readbackBuffer, 0, TIMESTAMP_BYTES);
    const cpuEncodeMs = performance.now() - cpuStart;

    this.device.queue.submit([encoder.finish()]);
    await this.readbackBuffer.mapAsync(GPUMapMode.READ);
    const stamps = new BigUint64Array(this.readbackBuffer.getMappedRange());
    const deltaNs = stamps[1] - stamps[0];
    this.readbackBuffer.unmap();
    if (deltaNs <= 0n) {
      throw new Error("GPU timestamp query returned a non-positive duration");
    }
    return {
      gpuMs: Number(deltaNs) / 1e6,
      cpuEncodeMs,
    };
  }
}

export async function runBenchmarks(rawOptions: BenchmarkOptions = {}): Promise<BenchmarkReport> {
  const options = normalizeOptions(rawOptions);
  const canvas = benchmarkCanvas(options);
  const { device, context, format } = await initGpu(canvas);
  const removeGpuErrorHandlers = installGpuErrorHandlers(device, (error) => {
    throw error;
  });

  try {
    const timer = new TimestampTimer(device);
    const viewport = resizeBenchmarkCanvas(canvas, options);
    const cases: BenchmarkCaseResult[] = [];

    for (const steps of options.stepBatches) {
      cases.push(await runSimOnlyCase(device, timer, options, steps));
    }

    cases.push(await runRenderOnlyCase(device, context, format, timer, options, viewport));

    for (const steps of options.stepBatches) {
      cases.push(await runSimRenderCase(device, context, format, timer, options, viewport, steps));
    }

    if (options.profileStages) {
      for (const stage of PIPELINE_NAMES) {
        cases.push(await runStageProfileCase(device, timer, options, stage));
      }
    }

    return {
      version: 1,
      createdAt: new Date().toISOString(),
      environment: {
        userAgent: navigator.userAgent,
        devicePixelRatio: window.devicePixelRatio,
        canvasWidth: canvas.width,
        canvasHeight: canvas.height,
        maxAgents: MAX_AGENTS,
        gridDim: GRID_DIM,
        worldSize: WORLD_SIZE,
      },
      options,
      cases,
    };
  } finally {
    removeGpuErrorHandlers();
  }
}

async function runStageProfileCase(
  device: GPUDevice,
  timer: TimestampTimer,
  options: Required<BenchmarkOptions>,
  stage: (typeof PIPELINE_NAMES)[number],
): Promise<BenchmarkCaseResult> {
  const simulation = new Simulation(device);
  const samples = await collectSamples(options, (querySet) => (encoder) => {
    simulation.encodeProfiledStep(encoder, stage, {
      querySet,
      beginningOfPassWriteIndex: 0,
      endOfPassWriteIndex: 1,
    });
  }, timer);
  return summarizeCase("sim-stage", { stage }, options, samples);
}

async function runSimOnlyCase(
  device: GPUDevice,
  timer: TimestampTimer,
  options: Required<BenchmarkOptions>,
  steps: number,
): Promise<BenchmarkCaseResult> {
  const simulation = new Simulation(device);
  const samples = await collectSamples(options, (querySet) => (encoder) => {
    simulation.encode(encoder, steps, {
      querySet,
      beginningOfPassWriteIndex: 0,
      endOfPassWriteIndex: 1,
    });
  }, timer);
  return summarizeCase("sim-only", { steps }, options, samples, steps);
}

async function runRenderOnlyCase(
  device: GPUDevice,
  context: GPUCanvasContext,
  format: GPUTextureFormat,
  timer: TimestampTimer,
  options: Required<BenchmarkOptions>,
  viewport: Viewport,
): Promise<BenchmarkCaseResult> {
  const simulation = new Simulation(device);
  const renderer = fixedRenderer(device, format, simulation.agentsBuffer, viewport);
  const samples = await collectSamples(options, (querySet) => (encoder) => {
    renderer.snapshotAgents(encoder);
    const view = context.getCurrentTexture().createView();
    renderer.encode(encoder, view, {
      querySet,
      beginningOfPassWriteIndex: 0,
      endOfPassWriteIndex: 1,
    });
  }, timer);
  return summarizeCase("render-only", { viewport: `${viewport.width}x${viewport.height}` }, options, samples);
}

async function runSimRenderCase(
  device: GPUDevice,
  context: GPUCanvasContext,
  format: GPUTextureFormat,
  timer: TimestampTimer,
  options: Required<BenchmarkOptions>,
  viewport: Viewport,
  steps: number,
): Promise<BenchmarkCaseResult> {
  const simulation = new Simulation(device);
  const renderer = fixedRenderer(device, format, simulation.agentsBuffer, viewport);
  const samples = await collectSamples(options, (querySet) => (encoder) => {
    renderer.snapshotAgents(encoder);
    simulation.encode(encoder, steps, {
      querySet,
      beginningOfPassWriteIndex: 0,
    });
    const view = context.getCurrentTexture().createView();
    renderer.encode(encoder, view, {
      querySet,
      endOfPassWriteIndex: 1,
    });
  }, timer);
  return summarizeCase("sim-plus-render", { steps, viewport: `${viewport.width}x${viewport.height}` }, options, samples, steps);
}

async function collectSamples(
  options: Required<BenchmarkOptions>,
  encodeForQuerySet: (querySet: GPUQuerySet) => (encoder: GPUCommandEncoder) => void,
  timer: TimestampTimer,
): Promise<TimingSample[]> {
  const samples: TimingSample[] = [];
  const totalRuns = options.warmup + options.samples;
  for (let i = 0; i < totalRuns; i += 1) {
    const sample = await timer.measure((encoder, querySet) => {
      encodeForQuerySet(querySet)(encoder);
    });
    if (i >= options.warmup) {
      samples.push(sample);
    }
  }
  return samples;
}

function fixedRenderer(
  device: GPUDevice,
  format: GPUTextureFormat,
  agentsBuffer: GPUBuffer,
  viewport: Viewport,
): Renderer {
  const renderer = new Renderer(device, format, agentsBuffer);
  const camera = new Camera();
  camera.fitWorld(viewport);
  renderer.update(camera.center, camera.zoom, viewport.width, viewport.height, camera.visibleTileOffsets(viewport));
  return renderer;
}

function summarizeCase(
  name: string,
  workload: Record<string, number | string>,
  options: Required<BenchmarkOptions>,
  samples: TimingSample[],
  steps?: number,
): BenchmarkCaseResult {
  const gpuMs = stats(samples.map((sample) => sample.gpuMs));
  const result: BenchmarkCaseResult = {
    name,
    workload,
    samples: options.samples,
    warmup: options.warmup,
    gpuMs,
    cpuEncodeMs: stats(samples.map((sample) => sample.cpuEncodeMs)),
  };
  if (steps !== undefined) {
    result.stepsPerSecond = (steps * 1000) / gpuMs.median;
  }
  return result;
}

function stats(values: number[]): BenchmarkStats {
  if (values.length === 0) {
    throw new Error("Cannot summarize an empty benchmark sample set");
  }
  const sorted = [...values].sort((a, b) => a - b);
  const sum = sorted.reduce((total, value) => total + value, 0);
  return {
    min: sorted[0],
    median: percentile(sorted, 0.5),
    mean: sum / sorted.length,
    p95: percentile(sorted, 0.95),
    max: sorted[sorted.length - 1],
  };
}

function percentile(sortedValues: readonly number[], percentileValue: number): number {
  const index = Math.min(sortedValues.length - 1, Math.ceil(sortedValues.length * percentileValue) - 1);
  return sortedValues[index];
}

function normalizeOptions(rawOptions: BenchmarkOptions): Required<BenchmarkOptions> {
  const options = { ...DEFAULT_OPTIONS, ...rawOptions };
  return {
    samples: positiveInteger(options.samples, "samples"),
    warmup: nonNegativeInteger(options.warmup, "warmup"),
    width: positiveInteger(options.width, "width"),
    height: positiveInteger(options.height, "height"),
    stepBatches: options.stepBatches.map((steps) => positiveInteger(steps, "stepBatches")),
    profileStages: Boolean(options.profileStages),
  };
}

function positiveInteger(value: number, name: string): number {
  if (!Number.isInteger(value) || value <= 0) {
    throw new Error(`${name} must be a positive integer`);
  }
  return value;
}

function nonNegativeInteger(value: number, name: string): number {
  if (!Number.isInteger(value) || value < 0) {
    throw new Error(`${name} must be a non-negative integer`);
  }
  return value;
}

function benchmarkCanvas(options: Required<BenchmarkOptions>): HTMLCanvasElement {
  const canvas = document.querySelector<HTMLCanvasElement>("#benchmark-canvas");
  if (!canvas) {
    throw new Error("Benchmark canvas #benchmark-canvas not found");
  }
  canvas.style.width = `${options.width}px`;
  canvas.style.height = `${options.height}px`;
  return canvas;
}

function resizeBenchmarkCanvas(canvas: HTMLCanvasElement, options: Required<BenchmarkOptions>): Viewport {
  resizeCanvas(canvas, MAX_DEVICE_PIXEL_RATIO);
  return { width: options.width, height: options.height };
}

window.speedSlopRunBenchmark = runBenchmarks;
