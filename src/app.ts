import { Camera, type Viewport } from "./camera";
import { MAX_DEVICE_PIXEL_RATIO, TELEMETRY_SAMPLE_MS } from "./config";
import { ThroughputController } from "./controller";
import { initGpu, installGpuErrorHandlers, resizeCanvas } from "./gpu";
import { GpuProfiler } from "./profiler";
import { Renderer } from "./renderer";
import { Simulation } from "./simulation";
import { renderTelemetry } from "./telemetry";

export interface AppElements {
  canvas: HTMLCanvasElement;
  monitor: HTMLElement | null;
}

export interface AppOptions {
  onFatalError: (error: unknown) => void;
}

export interface RunningApp {
  stop: () => void;
}

export async function startApp(elements: AppElements, options: AppOptions): Promise<RunningApp> {
  const { canvas, monitor } = elements;
  const { device, context, format } = await initGpu(canvas);
  const simulation = new Simulation(device);
  const renderer = new Renderer(
    device,
    format,
    simulation.agentsBuffer,
    simulation.denseBuffer,
    simulation.indirectBuffer,
  );
  const controller = new ThroughputController();
  const profiler = new GpuProfiler(device);
  const camera = new Camera();
  let cameraFitted = false;

  let animationFrame = 0;
  let stopped = false;
  let started = false;
  let lastTimestamp = 0;

  let windowStart = performance.now();
  let windowFrames = 0;
  let windowSteps = 0;
  let removeGpuErrorHandlers = (): void => {};

  const detachInput = attachCameraControls(canvas, camera);

  const stop = (): void => {
    if (stopped) {
      return;
    }
    stopped = true;
    cancelAnimationFrame(animationFrame);
    removeGpuErrorHandlers();
    detachInput();
  };

  const fail = (error: unknown): void => {
    stop();
    options.onFatalError(error);
  };

  removeGpuErrorHandlers = installGpuErrorHandlers(device, fail);

  const frame = (timestamp: number): void => {
    if (stopped) {
      return;
    }

    try {
      const sample = profiler.takeSample();
      if (sample) {
        controller.recordFrameCost(sample.gpuMs, sample.cpuMs, sample.steps);
      }
      if (started) {
        controller.recordFrameDelta(timestamp - lastTimestamp);
      }
      started = true;
      lastTimestamp = timestamp;

      const steps = controller.nextSteps();

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

      const workStart = performance.now();
      const encoder = device.createCommandEncoder();
      simulation.encode(encoder, steps, steps > 0 ? profiler.computePassWrites() : undefined);
      const view = context.getCurrentTexture().createView();
      renderer.encode(encoder, view, profiler.renderPassWrites(steps > 0));
      profiler.resolve(encoder, steps, performance.now() - workStart);
      device.queue.submit([encoder.finish()]);
      profiler.poll();

      windowFrames += 1;
      windowSteps += steps;
      const elapsed = timestamp - windowStart;
      if (elapsed >= TELEMETRY_SAMPLE_MS && monitor) {
        monitor.textContent = renderTelemetry({
          elapsedMs: elapsed,
          frames: windowFrames,
          steps: windowSteps,
        });
        windowStart = timestamp;
        windowFrames = 0;
        windowSteps = 0;
      }

      animationFrame = requestAnimationFrame(frame);
    } catch (error) {
      fail(error);
    }
  };

  animationFrame = requestAnimationFrame(frame);
  return { stop };
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
