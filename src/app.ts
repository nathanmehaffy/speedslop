import { Camera, type Viewport } from "./camera";
import { MAX_DEVICE_PIXEL_RATIO, TELEMETRY_SAMPLE_MS } from "./config";
import { ThroughputController } from "./controller";
import { initGpu, installGpuErrorHandlers, resizeCanvas } from "./gpu";
import { InterpretabilityPanel } from "./interpretabilityPanel";
import { InterpretabilityRecorder } from "./interpretabilityRecorder";
import {
  parseAgents,
  parseGenomeSamples,
  parseLifeRecords,
  parseMeta,
  readSimulationSnapshot,
} from "./interpretabilitySnapshot";
import { LiveGraphs } from "./liveGraphs";
import { GpuProfiler } from "./profiler";
import { Renderer } from "./renderer";
import { Simulation } from "./simulation";
import { renderStepTelemetry, renderTelemetry } from "./telemetry";

type RunMode = "pause" | "slow" | "fast" | "turbo" | "max";
type ViewMode = "simulation" | "analysis";

const SLOW_STEPS_PER_SECOND = 50;
const FAST_STEPS_PER_SECOND = 500;
const MAX_TARGET_BATCH_MS = 96;
const MAX_IN_FLIGHT_BATCHES = 8;
const MAX_QUEUED_MS = 480;
const MAX_INITIAL_STEPS = 256;
const MAX_MIN_STEPS = 1;
const MAX_STEPS_PER_BATCH = 16_384;
const MAX_SMOOTHING = 0.35;
const MAX_STEP_FACTOR = 4;

interface MaxBatch {
  steps: number;
  estimatedMs: number;
  queueDepth: number;
  submittedAt: number;
  done: Promise<void>;
}

export interface AppElements {
  canvas: HTMLCanvasElement;
  monitor: HTMLElement | null;
  controls: HTMLElement | null;
  liveGraphs: HTMLElement | null;
  analysisRoot: HTMLElement | null;
}

export interface AppOptions {
  onFatalError: (error: unknown) => void;
}

export interface RunningApp {
  stop: () => void;
}

export async function startApp(elements: AppElements, options: AppOptions): Promise<RunningApp> {
  const { canvas, monitor, controls, liveGraphs: liveGraphsRoot, analysisRoot } = elements;
  const { device, context, format } = await initGpu(canvas);
  const simulation = new Simulation(device);
  const renderer = new Renderer(
    device,
    format,
    simulation.agentsBuffer,
  );
  const displayController = new ThroughputController();
  const profiler = new GpuProfiler(device);
  const recorder = new InterpretabilityRecorder(device, simulation);
  const liveGraphs = new LiveGraphs(liveGraphsRoot);
  const camera = new Camera();
  const panelRoot = analysisRoot ?? createAnalysisRoot();
  const panel = new InterpretabilityPanel(panelRoot, () => {
    leaveAnalysisMode();
  });
  let cameraFitted = false;

  let animationFrame: number | null = null;
  let maxLoopActive = false;
  let maxGeneration = 0;
  let stopped = false;
  let started = false;
  let lastTimestamp = 0;
  let mode: RunMode = "turbo";
  let viewMode: ViewMode = "simulation";
  let modeBeforeAnalysis: RunMode = mode;
  let fixedStepAccumulator = 0;
  let maxStepsPerBatch = MAX_INITIAL_STEPS;
  let maxEstimatedBatchMs = MAX_TARGET_BATCH_MS;

  let windowStart = performance.now();
  let windowFrames = 0;
  let windowSteps = 0;
  let removeGpuErrorHandlers = (): void => {};

  const detachInput = attachCameraControls(canvas, camera);
  const detachControls = attachRunModeControls(controls, mode, (nextMode) => {
    if (viewMode === "analysis") {
      return;
    }
    if (nextMode === mode) {
      return;
    }
    mode = nextMode;
    maxGeneration += 1;
    started = false;
    fixedStepAccumulator = 0;
    maxStepsPerBatch = MAX_INITIAL_STEPS;
    maxEstimatedBatchMs = MAX_TARGET_BATCH_MS;
    resetTelemetryWindow(performance.now());
    if (monitor && mode === "max") {
      monitor.textContent = "-- sim steps/s";
    }
    document.body.dataset.simMode = mode;
    reschedule();
  });
  const detachInterpretabilityControls = attachInterpretabilityControls(
    controls,
    (enabled) => {
      recorder.deepRecording = enabled;
    },
    () => {
      void enterAnalysisMode();
    },
  );
  document.body.dataset.simMode = mode;
  document.body.dataset.viewMode = viewMode;

  const stop = (): void => {
    if (stopped) {
      return;
    }
    stopped = true;
    cancelScheduledFrame();
    removeGpuErrorHandlers();
    detachInput();
    detachControls();
    detachInterpretabilityControls();
    recorder.destroy();
  };

  const fail = (error: unknown): void => {
    stop();
    options.onFatalError(error);
  };

  removeGpuErrorHandlers = installGpuErrorHandlers(device, fail);

  const frame = (timestamp: number): void => {
    animationFrame = null;
    if (stopped) {
      return;
    }

    try {
      const previousTimestamp = lastTimestamp;
      if (viewMode === "analysis") {
        return;
      }
      const sample = profiler.takeSample();
      if (sample && mode === "turbo") {
        displayController.recordFrameCost(sample.gpuMs, sample.cpuMs, sample.steps);
      }
      if (started) {
        const deltaMs = timestamp - previousTimestamp;
        if (mode !== "max") {
          displayController.recordFrameDelta(deltaMs);
        }
      }
      const deltaMs = started ? Math.max(0, timestamp - previousTimestamp) : 0;
      started = true;
      lastTimestamp = timestamp;

      const steps = stepsForMode(mode, deltaMs);
      const shouldRender = mode !== "max";

      if (shouldRender) {
        resizeCanvas(canvas, MAX_DEVICE_PIXEL_RATIO);
        const viewport: Viewport = {
          width: canvas.clientWidth || 1,
          height: canvas.clientHeight || 1,
        };
        if (!cameraFitted && viewport.width > 1 && viewport.height > 1) {
          camera.fitWorld(viewport);
          cameraFitted = true;
        }
        renderer.update(camera.center, camera.zoom, viewport.width, viewport.height, camera.visibleTileOffsets(viewport));
      }

      const workStart = performance.now();
      const encoder = device.createCommandEncoder();
      if (steps > 0) {
        renderer.snapshotAgents(encoder);
      }
      simulation.encode(
        encoder,
        steps,
        steps > 0
          ? profiler.computePassWrites()
          : undefined,
      );
      if (shouldRender) {
        const view = context.getCurrentTexture().createView();
        renderer.encode(encoder, view, profiler.renderPassWrites(steps > 0));
      }
      recorder.encodeSamples(encoder, performance.now(), mode);
      if (shouldRender || steps > 0) {
        profiler.resolve(encoder, steps, performance.now() - workStart);
      }
      device.queue.submit([encoder.finish()]);
      profiler.poll();
      recorder.poll();

      windowFrames += 1;
      windowSteps += steps;
      const elapsed = timestamp - windowStart;
      if (elapsed >= TELEMETRY_SAMPLE_MS) {
        if (monitor) {
          monitor.textContent = renderTelemetry({
            elapsedMs: elapsed,
            frames: windowFrames,
            steps: windowSteps,
          });
        }
        liveGraphs.render(recorder.metaSamples);
        windowStart = timestamp;
        windowFrames = 0;
        windowSteps = 0;
      }

      scheduleNextFrame();
    } catch (error) {
      fail(error);
    }
  };

  function stepsForMode(currentMode: RunMode, deltaMs: number): number {
    if (currentMode === "pause") {
      return 0;
    }
    if (currentMode === "slow") {
      return drawFixedRateSteps(deltaMs, SLOW_STEPS_PER_SECOND);
    }
    if (currentMode === "fast") {
      return drawFixedRateSteps(deltaMs, FAST_STEPS_PER_SECOND);
    }
    return displayController.nextSteps();
  }

  function drawFixedRateSteps(deltaMs: number, stepsPerSecond: number): number {
    fixedStepAccumulator += (Math.min(deltaMs, 250) * stepsPerSecond) / 1000;
    const steps = Math.floor(fixedStepAccumulator);
    fixedStepAccumulator -= steps;
    return steps;
  }

  function scheduleNextFrame(): void {
    if (stopped || viewMode === "analysis" || animationFrame !== null || maxLoopActive) {
      return;
    }
    if (mode === "max") {
      maxLoopActive = true;
      void runMaxLoop(maxGeneration);
      return;
    }
    animationFrame = requestAnimationFrame(frame);
  }

  function cancelScheduledFrame(): void {
    if (animationFrame !== null) {
      cancelAnimationFrame(animationFrame);
      animationFrame = null;
    }
  }

  function reschedule(): void {
    cancelScheduledFrame();
    scheduleNextFrame();
  }

  async function runMaxLoop(generation: number): Promise<void> {
    const pending: MaxBatch[] = [];
    let lastCompletionTime: number | null = null;
    try {
      while (!stopped && mode === "max" && generation === maxGeneration) {
        while (
          !stopped &&
          viewMode === "simulation" &&
          mode === "max" &&
          generation === maxGeneration &&
          canSubmitMaxBatch(pending)
        ) {
          pending.push(submitMaxBatch(pending));
        }

        const batch = pending.shift();
        if (!batch) {
          continue;
        }
        await batch.done;

        const completedAt = performance.now();
        windowFrames += 1;
        windowSteps += batch.steps;
        const batchMs = lastCompletionTime === null
          ? (completedAt - batch.submittedAt) / batch.queueDepth
          : completedAt - lastCompletionTime;
        lastCompletionTime = completedAt;
        recordMaxBatchCost(batchMs, batch.steps);
        const elapsed = performance.now() - windowStart;
        if (elapsed >= TELEMETRY_SAMPLE_MS) {
          if (monitor) {
            monitor.textContent = renderStepTelemetry({
              elapsedMs: elapsed,
              steps: windowSteps,
            });
          }
          liveGraphs.render(recorder.metaSamples);
          resetTelemetryWindow(performance.now());
        }
      }
    } catch (error) {
      fail(error);
    } finally {
      maxLoopActive = false;
      if (!stopped && viewMode === "simulation" && mode !== "max") {
        scheduleNextFrame();
      }
    }
  }

  function submitMaxBatch(pending: readonly MaxBatch[]): MaxBatch {
    const steps = Math.round(maxStepsPerBatch);
    const submittedAt = performance.now();
    const queueDepth = pending.length + 1;
    const encoder = device.createCommandEncoder();
    renderer.snapshotAgents(encoder);
    simulation.encode(encoder, steps);
    recorder.encodeSamples(encoder, submittedAt, mode);
    device.queue.submit([encoder.finish()]);
    recorder.poll();
    return {
      steps,
      estimatedMs: maxEstimatedBatchMs,
      queueDepth,
      submittedAt,
      done: device.queue.onSubmittedWorkDone(),
    };
  }

  function canSubmitMaxBatch(pending: readonly MaxBatch[]): boolean {
    if (pending.length >= MAX_IN_FLIGHT_BATCHES) {
      return false;
    }
    return pending.length === 0 || maxQueuedMs(pending) + maxEstimatedBatchMs <= MAX_QUEUED_MS;
  }

  function maxQueuedMs(pending: readonly MaxBatch[]): number {
    return pending.reduce((total, batch) => total + batch.estimatedMs, 0);
  }

  function resetTelemetryWindow(timestamp: number): void {
    windowStart = timestamp;
    windowFrames = 0;
    windowSteps = 0;
  }

  function recordMaxBatchCost(batchMs: number, steps: number): void {
    if (!(batchMs > 0) || !isFinite(batchMs)) {
      return;
    }
    const factor = clamp(MAX_TARGET_BATCH_MS / batchMs, 1 / MAX_STEP_FACTOR, MAX_STEP_FACTOR);
    const target = clamp(steps * factor, MAX_MIN_STEPS, MAX_STEPS_PER_BATCH);
    const logCurrent = Math.log(clamp(maxStepsPerBatch, MAX_MIN_STEPS, MAX_STEPS_PER_BATCH));
    const logTarget = Math.log(target);
    maxStepsPerBatch = Math.exp(logCurrent + MAX_SMOOTHING * (logTarget - logCurrent));
    maxEstimatedBatchMs += MAX_SMOOTHING * (batchMs - maxEstimatedBatchMs);
  }

  async function enterAnalysisMode(): Promise<void> {
    if (stopped || viewMode === "analysis") {
      return;
    }
    modeBeforeAnalysis = mode;
    viewMode = "analysis";
    mode = "pause";
    maxGeneration += 1;
    document.body.dataset.viewMode = viewMode;
    document.body.dataset.simMode = mode;
    cancelScheduledFrame();
    await device.queue.onSubmittedWorkDone();
    if (stopped) {
      return;
    }
    const capturedAtMs = performance.now();
    const raw = await readSimulationSnapshot(device, simulation);
    const agents = parseAgents(raw.agentsBuffer);
    const lifeRecords = parseLifeRecords(raw.lifeRecordsBuffer);
    const meta = parseMeta(raw.metaBuffer, capturedAtMs);
    const genomes = parseGenomeSamples(raw.brainsBuffer, agents);
    panel.show({
      capturedAtMs,
      step: meta.step,
      agents,
      lifeRecords,
      genomes,
      metaSamples: [...recorder.metaSamples, meta],
      agentHistory: recorder.agentSamples.map((sample) => sample.agents),
      camera: {
        center: { x: camera.center.x, y: camera.center.y },
        zoom: camera.zoom,
      },
    });
  }

  function leaveAnalysisMode(): void {
    if (stopped || viewMode !== "analysis") {
      return;
    }
    panel.hide();
    viewMode = "simulation";
    mode = modeBeforeAnalysis;
    started = false;
    fixedStepAccumulator = 0;
    document.body.dataset.viewMode = viewMode;
    document.body.dataset.simMode = mode;
    resetTelemetryWindow(performance.now());
    reschedule();
  }

  scheduleNextFrame();
  return { stop };
}

function attachRunModeControls(
  controls: HTMLElement | null,
  initialMode: RunMode,
  onChange: (mode: RunMode) => void,
): () => void {
  if (!controls) {
    return () => {};
  }

  const buttons = Array.from(controls.querySelectorAll<HTMLButtonElement>("button[data-mode]"));
  const setPressed = (mode: RunMode): void => {
    for (const button of buttons) {
      button.setAttribute("aria-pressed", String(button.dataset.mode === mode));
    }
  };
  const onClick = (event: MouseEvent): void => {
    const button = event.currentTarget as HTMLButtonElement;
    const nextMode = button.dataset.mode as RunMode | undefined;
    if (!isRunMode(nextMode)) {
      return;
    }
    setPressed(nextMode);
    onChange(nextMode);
  };

  for (const button of buttons) {
    button.addEventListener("click", onClick);
  }
  setPressed(initialMode);

  return () => {
    for (const button of buttons) {
      button.removeEventListener("click", onClick);
    }
  };
}

function attachInterpretabilityControls(
  controls: HTMLElement | null,
  onDeepRecordingChange: (enabled: boolean) => void,
  onAnalyze: () => void,
): () => void {
  if (!controls) {
    return () => {};
  }
  const deep = controls.querySelector<HTMLInputElement>("#deep-recording");
  const analyze = controls.querySelector<HTMLButtonElement>("#analyze-mode");
  const onDeepChange = (): void => {
    onDeepRecordingChange(Boolean(deep?.checked));
  };
  const onAnalyzeClick = (): void => {
    onAnalyze();
  };
  deep?.addEventListener("change", onDeepChange);
  analyze?.addEventListener("click", onAnalyzeClick);
  return () => {
    deep?.removeEventListener("change", onDeepChange);
    analyze?.removeEventListener("click", onAnalyzeClick);
  };
}

function createAnalysisRoot(): HTMLElement {
  const root = document.createElement("div");
  root.id = "analysis-panel";
  root.hidden = true;
  document.body.appendChild(root);
  return root;
}

function isRunMode(value: string | undefined): value is RunMode {
  return value === "pause" || value === "slow" || value === "fast" || value === "turbo" || value === "max";
}

function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value));
}

function attachCameraControls(canvas: HTMLCanvasElement, camera: Camera): () => void {
  let dragging = false;
  let lastX = 0;
  let lastY = 0;

  const viewport = (): Viewport => ({
    width: canvas.clientWidth || 1,
    height: canvas.clientHeight || 1,
  });

  const onPointerDown = (e: PointerEvent): void => {
    dragging = true;
    lastX = e.clientX;
    lastY = e.clientY;
    canvas.setPointerCapture(e.pointerId);
  };
  const onPointerMove = (e: PointerEvent): void => {
    if (!dragging) {
      return;
    }
    camera.pan(e.clientX - lastX, e.clientY - lastY);
    lastX = e.clientX;
    lastY = e.clientY;
  };
  const onPointerUp = (e: PointerEvent): void => {
    dragging = false;
    if (canvas.hasPointerCapture(e.pointerId)) {
      canvas.releasePointerCapture(e.pointerId);
    }
  };
  const onWheel = (e: WheelEvent): void => {
    e.preventDefault();
    const rect = canvas.getBoundingClientRect();
    camera.zoomBy(e.deltaY, { x: e.clientX - rect.left, y: e.clientY - rect.top }, viewport());
  };

  canvas.addEventListener("pointerdown", onPointerDown);
  canvas.addEventListener("pointermove", onPointerMove);
  canvas.addEventListener("pointerup", onPointerUp);
  canvas.addEventListener("pointerleave", onPointerUp);
  canvas.addEventListener("wheel", onWheel, { passive: false });

  return () => {
    canvas.removeEventListener("pointerdown", onPointerDown);
    canvas.removeEventListener("pointermove", onPointerMove);
    canvas.removeEventListener("pointerup", onPointerUp);
    canvas.removeEventListener("pointerleave", onPointerUp);
    canvas.removeEventListener("wheel", onWheel);
  };
}
