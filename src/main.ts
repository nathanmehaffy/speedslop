import "./style.css";
import { GpuSimulation } from "./gpu-simulation";
import { runGpuSelfCheck } from "./gpu-self-check";
import {
  CameraState,
  DEFAULT_POPULATION,
  DEFAULT_WORLD_SIZE,
  FIXED_STEP_SECONDS,
  INITIAL_SEED,
  MAX_MODE_STEPS_PER_FRAME,
  SimRate,
  SimulationStats,
  normalizeSimRate,
  stepsDueForFrame,
  wrapUnit,
} from "./simulation-helpers";

const MIN_ZOOM = 1;
const MAX_ZOOM = 64;
const WHEEL_ZOOM_SENSITIVITY = 0.0015;
const STATS_INTERVAL_MS = 500;
const DISPLAY_MODE_MAX_EXTRA_STEPS_PER_FRAME = MAX_MODE_STEPS_PER_FRAME - 1;
const DISPLAY_MODE_P_GAIN = 0.08;
const DISPLAY_MODE_EXTRA_STEP_DELTA_LIMIT = 0.35;
const DISPLAY_MODE_FPS_HEADROOM = 0.985;
const DISPLAY_MODE_FPS_EMA_ALPHA = 0.22;
const DISPLAY_MODE_ERROR_DEADBAND_FPS = 1.0;

type WebGpuState = {
  adapter: GPUAdapter;
  adapterLabel: string;
  device: GPUDevice;
  context: GPUCanvasContext;
  format: GPUTextureFormat;
  simulation: GpuSimulation;
};

type FpsState = {
  frames: number;
  last: number;
};

type PointerPoint = {
  clientX: number;
  clientY: number;
};

type Vec2 = {
  x: number;
  y: number;
};

type PointerGesture = {
  centerX: number;
  centerY: number;
  distance: number;
};

function queryRequired<T extends Element>(selector: string): T {
  const element = document.querySelector<T>(selector);

  if (!element) {
    throw new Error(`Required element is missing: ${selector}`);
  }

  return element;
}

const canvas = queryRequired<HTMLCanvasElement>("#viewport");
const statusEl = queryRequired<HTMLSpanElement>("#status");
const fpsEl = queryRequired<HTMLSpanElement>("#fps");
const stepsPerSecondEl = queryRequired<HTMLSpanElement>("#steps-per-second");
const populationEl = queryRequired<HTMLSpanElement>("#population");
const birthsEl = queryRequired<HTMLSpanElement>("#births");
const deathsEl = queryRequired<HTMLSpanElement>("#deaths");
const generationEl = queryRequired<HTMLSpanElement>("#generation");
const pauseButton = queryRequired<HTMLButtonElement>("#pause");
const resetButton = queryRequired<HTMLButtonElement>("#reset");
const resetViewButton = queryRequired<HTMLButtonElement>("#reset-view");
const seedInput = queryRequired<HTMLInputElement>("#seed");
const speedSelect = queryRequired<HTMLSelectElement>("#speed");
const renderFpsSelect = queryRequired<HTMLSelectElement>("#render-fps");
const searchParams = new URLSearchParams(window.location.search);
const benchmarkMode = searchParams.has("bench");
const selfCheckMode = searchParams.has("selfcheck");

function setStatus(message: string): void {
  statusEl.textContent = message;
}

function resizeCanvasToDisplaySize(): boolean {
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const width = Math.max(1, Math.floor(canvas.clientWidth * dpr));
  const height = Math.max(1, Math.floor(canvas.clientHeight * dpr));

  if (canvas.width !== width || canvas.height !== height) {
    canvas.width = width;
    canvas.height = height;
    return true;
  }

  return false;
}

function canvasAspect(): number {
  return canvas.height > 0 ? canvas.width / canvas.height : 1;
}

function createDefaultCamera(): CameraState {
  return { centerX: 0.5, centerY: 0.5, zoom: MIN_ZOOM };
}

function resetCamera(camera: CameraState): void {
  camera.centerX = 0.5;
  camera.centerY = 0.5;
  camera.zoom = MIN_ZOOM;
}

function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value));
}

function clientOffsetToWorldDelta(camera: CameraState, clientX: number, clientY: number): Vec2 {
  const rect = canvas.getBoundingClientRect();

  if (rect.width <= 0 || rect.height <= 0) {
    return { x: 0, y: 0 };
  }

  const clipX = ((clientX - rect.left) / rect.width) * 2 - 1;
  const clipY = 1 - ((clientY - rect.top) / rect.height) * 2;
  const zoomScale = 2 * camera.zoom;
  const aspect = canvasAspect();

  if (aspect > 1) {
    return {
      x: (clipX * aspect) / zoomScale,
      y: -clipY / zoomScale,
    };
  }

  return {
    x: clipX / zoomScale,
    y: -clipY / (aspect * zoomScale),
  };
}

function clientToWorld(camera: CameraState, clientX: number, clientY: number): Vec2 {
  const offset = clientOffsetToWorldDelta(camera, clientX, clientY);

  return {
    x: wrapUnit(camera.centerX + offset.x),
    y: wrapUnit(camera.centerY + offset.y),
  };
}

function anchorCameraAtClientPoint(
  camera: CameraState,
  beforeClientX: number,
  beforeClientY: number,
  afterClientX: number,
  afterClientY: number,
  nextZoom: number,
): void {
  const anchor = clientToWorld(camera, beforeClientX, beforeClientY);
  camera.zoom = clamp(nextZoom, MIN_ZOOM, MAX_ZOOM);

  const offset = clientOffsetToWorldDelta(camera, afterClientX, afterClientY);
  camera.centerX = wrapUnit(anchor.x - offset.x);
  camera.centerY = wrapUnit(anchor.y - offset.y);
}

function readPointerGesture(pointers: Map<number, PointerPoint>): PointerGesture | null {
  const points = [...pointers.values()].slice(0, 2);

  if (points.length === 0) {
    return null;
  }

  if (points.length === 1) {
    return {
      centerX: points[0].clientX,
      centerY: points[0].clientY,
      distance: 0,
    };
  }

  const [a, b] = points;

  return {
    centerX: (a.clientX + b.clientX) * 0.5,
    centerY: (a.clientY + b.clientY) * 0.5,
    distance: Math.hypot(a.clientX - b.clientX, a.clientY - b.clientY),
  };
}

async function createWebGpuState(seed: number): Promise<WebGpuState> {
  if (!navigator.gpu) {
    throw new Error("WebGPU is not available in this browser.");
  }

  const adapter = await navigator.gpu.requestAdapter({
    powerPreference: "high-performance",
  });

  if (!adapter) {
    throw new Error("No compatible WebGPU adapter was found.");
  }

  const adapterLabel = describeAdapter(adapter);
  const device = await adapter.requestDevice();
  device.addEventListener("uncapturederror", (event) => {
    const message = event.error.message || "Uncaptured WebGPU error";
    setStatus(message);
    console.error(event.error);
  });

  const context = canvas.getContext("webgpu");

  if (!context) {
    throw new Error("Unable to create a WebGPU canvas context.");
  }

  const format = navigator.gpu.getPreferredCanvasFormat();
  context.configure({
    device,
    format,
    alphaMode: "opaque",
  });

  const simulation = await GpuSimulation.create(device, {
    worldSize: DEFAULT_WORLD_SIZE,
    population: DEFAULT_POPULATION,
    seed,
    format,
  });

  return { adapter, adapterLabel, device, context, format, simulation };
}

function describeAdapter(adapter: GPUAdapter): string {
  const info = adapter.info as GPUAdapterInfo & {
    architecture?: string;
    description?: string;
    device?: string;
    powerPreference?: string;
    type?: string;
    vendor?: string;
  };
  const details = [
    info.description,
    info.vendor,
    info.architecture,
    info.device,
    info.type,
    info.powerPreference,
  ].filter((value): value is string => Boolean(value && value.trim()));

  return details.length > 0 ? details.join(" / ") : "WebGPU";
}

function setUpCameraControls(camera: CameraState): void {
  const pointers = new Map<number, PointerPoint>();

  const applyGesture = (before: PointerGesture | null, after: PointerGesture | null): void => {
    if (!before || !after) {
      return;
    }

    let nextZoom = camera.zoom;

    if (before.distance > 0 && after.distance > 0) {
      nextZoom = camera.zoom * (after.distance / before.distance);
    }

    anchorCameraAtClientPoint(
      camera,
      before.centerX,
      before.centerY,
      after.centerX,
      after.centerY,
      nextZoom,
    );
  };

  canvas.addEventListener("pointerdown", (event) => {
    event.preventDefault();
    canvas.setPointerCapture(event.pointerId);
    pointers.set(event.pointerId, {
      clientX: event.clientX,
      clientY: event.clientY,
    });
    canvas.classList.add("is-panning");
  });

  canvas.addEventListener("pointermove", (event) => {
    if (!pointers.has(event.pointerId)) {
      return;
    }

    event.preventDefault();
    const before = readPointerGesture(pointers);
    pointers.set(event.pointerId, {
      clientX: event.clientX,
      clientY: event.clientY,
    });
    const after = readPointerGesture(pointers);
    applyGesture(before, after);
  });

  const forgetPointer = (event: PointerEvent): void => {
    pointers.delete(event.pointerId);

    if (canvas.hasPointerCapture(event.pointerId)) {
      canvas.releasePointerCapture(event.pointerId);
    }

    if (pointers.size === 0) {
      canvas.classList.remove("is-panning");
    }
  };

  canvas.addEventListener("pointerup", forgetPointer);
  canvas.addEventListener("pointercancel", forgetPointer);
  canvas.addEventListener("lostpointercapture", forgetPointer);

  canvas.addEventListener(
    "wheel",
    (event) => {
      event.preventDefault();

      const delta =
        event.deltaY *
        (event.deltaMode === WheelEvent.DOM_DELTA_LINE
          ? 16
          : event.deltaMode === WheelEvent.DOM_DELTA_PAGE
            ? Math.max(1, canvas.clientHeight)
            : 1);
      const nextZoom = camera.zoom * Math.exp(-delta * WHEEL_ZOOM_SENSITIVITY);

      anchorCameraAtClientPoint(
        camera,
        event.clientX,
        event.clientY,
        event.clientX,
        event.clientY,
        nextZoom,
      );
    },
    { passive: false },
  );

  resetViewButton.addEventListener("click", () => {
    resetCamera(camera);
  });
}

function submitSimulationSteps(gpu: WebGpuState, stepCount: number): void {
  if (stepCount <= 0) {
    return;
  }

  const encoder = gpu.device.createCommandEncoder({
    label: "gpu-simulation-step-encoder",
  });
  gpu.simulation.encodeSteps(encoder, stepCount);
  gpu.device.queue.submit([encoder.finish()]);
}

function submitRenderFrame(gpu: WebGpuState, camera: CameraState): void {
  const encoder = gpu.device.createCommandEncoder({
    label: "gpu-simulation-render-encoder",
  });
  const pass = encoder.beginRenderPass({
    label: "gpu-simulation-render-pass",
    colorAttachments: [
      {
        view: gpu.context.getCurrentTexture().createView(),
        clearValue: { r: 0.012, g: 0.014, b: 0.018, a: 1 },
        loadOp: "clear",
        storeOp: "store",
      },
    ],
  });

  gpu.simulation.encodeRender(pass, camera, canvasAspect());
  pass.end();
  gpu.device.queue.submit([encoder.finish()]);
}

function updateFps(now: number, state: FpsState): number | null {
  state.frames += 1;

  if (now - state.last >= 500) {
    const fps = Math.round((state.frames * 1000) / (now - state.last));
    fpsEl.textContent = `${fps} FPS`;
    state.frames = 0;
    state.last = now;
    return fps;
  }

  return null;
}

function updateSimulationHud(stats: SimulationStats): void {
  stepsPerSecondEl.textContent = `${stats.stepsPerSecond.toLocaleString()} steps/s`;
  populationEl.textContent = `${stats.population.toLocaleString()} agents`;
  birthsEl.textContent = `${stats.births.toLocaleString()} births`;
  deathsEl.textContent = `${stats.deaths.toLocaleString()} deaths`;
  generationEl.textContent = `gen ${stats.generation.toLocaleString()}`;

  if (benchmarkMode) {
    setStatus(`${stats.stepsPerSecond.toLocaleString()} GPU steps/s, no per-agent readback`);
  }
}

function parseSimRate(): SimRate {
  if (speedSelect.value === "max") {
    return "max";
  }

  return normalizeSimRate(Number(speedSelect.value));
}

function renderFrameIntervalMs(): number {
  if (renderFpsSelect.value === "display") {
    return 0;
  }

  const fps = Number(renderFpsSelect.value);
  return Number.isFinite(fps) && fps > 0 ? 1000 / fps : 0;
}

function maxModeStepsPerFrame(): number {
  if (benchmarkMode || renderFpsSelect.value !== "display") {
    return MAX_MODE_STEPS_PER_FRAME;
  }

  return 1;
}

function parseSeed(): number {
  const value = Number(seedInput.value);
  if (!Number.isFinite(value)) {
    return INITIAL_SEED;
  }

  return Math.max(0, Math.floor(value)) >>> 0;
}

function initialStats(): SimulationStats {
  return {
    population: DEFAULT_POPULATION,
    births: 0,
    deaths: 0,
    generation: 0,
    simSteps: 0,
    stepsPerSecond: 0,
  };
}

async function run(): Promise<void> {
  resizeCanvasToDisplaySize();
  window.addEventListener("resize", resizeCanvasToDisplaySize);

  seedInput.value = String(INITIAL_SEED);
  if (benchmarkMode) {
    speedSelect.value = "max";
    renderFpsSelect.value = "15";
  }

  const camera = createDefaultCamera();
  const fpsState = { frames: 0, last: performance.now() };
  let gpu: WebGpuState | null = null;
  let paused = false;
  let previousTickAt = performance.now();
  let stepRemainderSeconds = 0;
  let lastRenderAt = Number.NEGATIVE_INFINITY;
  let lastStatsReadAt = Number.NEGATIVE_INFINITY;
  let bestDisplayFps = 0;
  let displayModeSmoothedFps = 0;
  let displayModeExtraStepBudget = 0;
  let displayModeExtraStepAccumulator = 0;

  const setSimulationControlsEnabled = (enabled: boolean): void => {
    pauseButton.disabled = !enabled;
    resetButton.disabled = !enabled;
    resetViewButton.disabled = !enabled;
    speedSelect.disabled = !enabled;
    renderFpsSelect.disabled = !enabled;
  };

  const updatePauseButton = (): void => {
    pauseButton.textContent = paused ? "Resume" : "Pause";
    pauseButton.setAttribute("aria-pressed", String(paused));
  };

  pauseButton.addEventListener("click", () => {
    paused = !paused;
    previousTickAt = performance.now();
    stepRemainderSeconds = 0;
    updatePauseButton();
  });

  speedSelect.addEventListener("change", () => {
    previousTickAt = performance.now();
    stepRemainderSeconds = 0;
    displayModeExtraStepBudget = 0;
    displayModeExtraStepAccumulator = 0;
    displayModeSmoothedFps = 0;
  });

  renderFpsSelect.addEventListener("change", () => {
    lastRenderAt = Number.NEGATIVE_INFINITY;
    displayModeExtraStepBudget = 0;
    displayModeExtraStepAccumulator = 0;
    displayModeSmoothedFps = 0;
  });

  resetButton.addEventListener("click", () => {
    if (!gpu) {
      return;
    }

    gpu.simulation.reset(parseSeed());
    previousTickAt = performance.now();
    stepRemainderSeconds = 0;
    lastStatsReadAt = Number.NEGATIVE_INFINITY;
    displayModeExtraStepBudget = 0;
    displayModeExtraStepAccumulator = 0;
    displayModeSmoothedFps = 0;
    updateSimulationHud(initialStats());
  });

  const readStats = (now: number): void => {
    if (!gpu || now - lastStatsReadAt < STATS_INTERVAL_MS) {
      return;
    }

    lastStatsReadAt = now;
    void gpu.simulation
      .readStatsAsync()
      .then(updateSimulationHud)
      .catch((error: unknown) => {
        const message = error instanceof Error ? error.message : String(error);
        setStatus(message);
        console.error(error);
      });
  };

  const loop = (now: number): void => {
    if (!gpu) {
      requestAnimationFrame(loop);
      return;
    }

    resizeCanvasToDisplaySize();

    const renderIntervalMs = renderFrameIntervalMs();
    const shouldRender =
      renderIntervalMs === 0 ||
      lastRenderAt === Number.NEGATIVE_INFINITY ||
      now - lastRenderAt >= renderIntervalMs - 0.5;
    let steps = 0;

    if (paused) {
      previousTickAt = now;
      stepRemainderSeconds = 0;
    } else {
      const elapsedSeconds = Math.max(0, (now - previousTickAt) / 1000);
      previousTickAt = now;
      const simRate = parseSimRate();

      if (renderIntervalMs === 0 && !benchmarkMode) {
        if (simRate === "max") {
          displayModeExtraStepAccumulator += displayModeExtraStepBudget;
          const extraSteps = Math.min(
            DISPLAY_MODE_MAX_EXTRA_STEPS_PER_FRAME,
            Math.floor(displayModeExtraStepAccumulator),
          );
          displayModeExtraStepAccumulator -= extraSteps;
          steps = 1 + extraSteps;
        } else {
          stepRemainderSeconds += elapsedSeconds * simRate;
          stepRemainderSeconds = Math.min(stepRemainderSeconds, FIXED_STEP_SECONDS);

          if (stepRemainderSeconds >= FIXED_STEP_SECONDS) {
            steps = 1;
            stepRemainderSeconds -= FIXED_STEP_SECONDS;
          }
        }
      } else {
        const due = stepsDueForFrame(
          elapsedSeconds,
          simRate,
          stepRemainderSeconds,
          maxModeStepsPerFrame(),
        );
        steps = due.steps;
        stepRemainderSeconds = due.remainder;
      }
    }

    if (shouldRender) {
      submitRenderFrame(gpu, camera);
      const fps = updateFps(now, fpsState);
      if (fps !== null && renderIntervalMs === 0 && !benchmarkMode) {
        displayModeSmoothedFps =
          displayModeSmoothedFps === 0
            ? fps
            : displayModeSmoothedFps +
              (fps - displayModeSmoothedFps) * DISPLAY_MODE_FPS_EMA_ALPHA;
        bestDisplayFps = Math.max(bestDisplayFps, displayModeSmoothedFps);

        if (bestDisplayFps > 0) {
          const targetFps = bestDisplayFps * DISPLAY_MODE_FPS_HEADROOM;
          const errorFps = displayModeSmoothedFps - targetFps;
          const controlledError =
            Math.abs(errorFps) <= DISPLAY_MODE_ERROR_DEADBAND_FPS ? 0 : errorFps;
          const delta = clamp(
            controlledError * DISPLAY_MODE_P_GAIN,
            -DISPLAY_MODE_EXTRA_STEP_DELTA_LIMIT,
            DISPLAY_MODE_EXTRA_STEP_DELTA_LIMIT,
          );

          displayModeExtraStepBudget = clamp(
            displayModeExtraStepBudget + delta,
            0,
            DISPLAY_MODE_MAX_EXTRA_STEPS_PER_FRAME,
          );
          displayModeExtraStepAccumulator = Math.min(
            displayModeExtraStepAccumulator,
            displayModeExtraStepBudget,
          );
        }
      }
      lastRenderAt = now;
    }

    submitSimulationSteps(gpu, steps);
    readStats(now);
    requestAnimationFrame(loop);
  };

  setSimulationControlsEnabled(false);
  setStatus("Starting GPU simulation");
  updatePauseButton();
  updateSimulationHud(initialStats());

  gpu = await createWebGpuState(INITIAL_SEED);
  let selfCheckSummary = "";
  if (selfCheckMode) {
    setStatus("Running GPU self-check");
    const results = await runGpuSelfCheck(gpu.device, gpu.format);
    console.table(results);
    selfCheckSummary = `${results.length} GPU self-checks passed; `;
  }
  setUpCameraControls(camera);
  setStatus(
    benchmarkMode
      ? "Benchmark mode: GPU simulation active"
      : `${selfCheckSummary}${gpu.adapterLabel} ready`,
  );
  setSimulationControlsEnabled(true);
  requestAnimationFrame(loop);
}

run().catch((error: unknown) => {
  const message = error instanceof Error ? error.message : String(error);
  setStatus(message);
  fpsEl.textContent = "WebGPU error";
  console.error(error);
});
