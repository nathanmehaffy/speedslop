import { CONTROLLER_WARMUP_FRAMES, DEMO_AGENT_COUNT, MAX_DEVICE_PIXEL_RATIO, TELEMETRY_SAMPLE_MS } from "./config";
import { ThroughputController } from "./controller";
import { initGpu, installGpuErrorHandlers, resizeCanvas } from "./gpu";
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
  const simulation = new Simulation(device, DEMO_AGENT_COUNT);
  const renderer = new Renderer(device, format, DEMO_AGENT_COUNT);
  const controller = new ThroughputController({ warmupFrames: CONTROLLER_WARMUP_FRAMES });

  let animationFrame = 0;
  let stopped = false;
  let started = false;
  let lastTimestamp = 0;

  let windowStart = performance.now();
  let windowFrames = 0;
  let windowSteps = 0;
  let removeGpuErrorHandlers = (): void => {};

  const stop = (): void => {
    if (stopped) {
      return;
    }
    stopped = true;
    cancelAnimationFrame(animationFrame);
    removeGpuErrorHandlers();
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
      const steps = decideSteps(controller, timestamp, lastTimestamp, started);
      started = true;
      lastTimestamp = timestamp;

      const size = resizeCanvas(canvas, MAX_DEVICE_PIXEL_RATIO);
      const aspect = size.width / size.height;

      renderer.updateParams(aspect);

      const encoder = device.createCommandEncoder();
      simulation.encode(encoder, steps);
      const view = context.getCurrentTexture().createView();
      renderer.encode(encoder, view, simulation.currentPositionBuffer);
      device.queue.submit([encoder.finish()]);

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

function decideSteps(
  controller: ThroughputController,
  timestamp: number,
  lastTimestamp: number,
  started: boolean,
): number {
  if (!started) {
    return controller.bootstrap();
  }

  const decision = controller.recordFrame(timestamp - lastTimestamp);
  return decision.steps;
}
