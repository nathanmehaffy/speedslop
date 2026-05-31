import "./style.css";
import type {
  MainToWorkerMessage,
  SimRate,
  SimulationStats,
  WorkerSnapshotMessage,
  WorkerToMainMessage,
} from "./simulation-messages";

const WORLD_SIZE = 4096;
const POPULATION = 10_000;
const INITIAL_SEED = 1;
const SNAPSHOT_BUFFER_COUNT = 3;
const ARROW_LENGTH_WORLD_UNITS = 18;
const ARROW_LOCAL_LENGTH = 2.2;
const MIN_ZOOM = 1;
const MAX_ZOOM = 64;
const WHEEL_ZOOM_SENSITIVITY = 0.0015;
const GRID_CELLS_PER_WORLD = 8;
const TILE_GRID_WIDTH = 5;
const TILE_COPY_COUNT = TILE_GRID_WIDTH * TILE_GRID_WIDTH;

type WebGpuState = {
  adapter: GPUAdapter;
  device: GPUDevice;
  context: GPUCanvasContext;
  agentBuffer: GPUBuffer;
  viewBuffer: GPUBuffer;
  bindGroup: GPUBindGroup;
  gridBindGroup: GPUBindGroup;
  pipeline: GPURenderPipeline;
  gridPipeline: GPURenderPipeline;
  format: GPUTextureFormat;
};

type FpsState = {
  frames: number;
  last: number;
};

type CameraState = {
  centerX: number;
  centerY: number;
  zoom: number;
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

function createDefaultCamera(): CameraState {
  return { centerX: 0.5, centerY: 0.5, zoom: MIN_ZOOM };
}

function resetCamera(camera: CameraState): void {
  camera.centerX = 0.5;
  camera.centerY = 0.5;
  camera.zoom = MIN_ZOOM;
}

function wrapUnit(value: number): number {
  return value - Math.floor(value);
}

function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value));
}

function canvasAspect(): number {
  return canvas.height > 0 ? canvas.width / canvas.height : 1;
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

async function createWebGpuState(agentBufferByteSize: number): Promise<WebGpuState> {
  if (!navigator.gpu) {
    throw new Error("WebGPU is not available in this browser.");
  }

  const adapter = await navigator.gpu.requestAdapter({
    powerPreference: "high-performance",
  });

  if (!adapter) {
    throw new Error("No compatible WebGPU adapter was found.");
  }

  const device = await adapter.requestDevice();
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

  const agentBuffer = device.createBuffer({
    label: "agent-render-buffer",
    size: Math.max(4, agentBufferByteSize),
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });

  const viewBuffer = device.createBuffer({
    label: "view-uniform-buffer",
    size: 32,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });

  const shader = device.createShaderModule({
    label: "agent-instancing-shader",
    code: `
      struct View {
        aspect: f32,
        arrow_scale: f32,
        camera_x: f32,
        camera_y: f32,
        zoom: f32,
        _pad0: f32,
        _pad1: f32,
        _pad2: f32,
      };

      struct VertexOutput {
        @builtin(position) position: vec4f,
        @location(0) color: vec4f,
      };

      @group(0) @binding(0) var<storage, read> agents: array<vec4f>;
      @group(0) @binding(1) var<uniform> view: View;

      fn delta_to_clip(delta: vec2f) -> vec2f {
        let centered = delta * (2.0 * view.zoom);
        if (view.aspect > 1.0) {
          return vec2f(centered.x / view.aspect, -centered.y);
        }

        return vec2f(centered.x, -centered.y * view.aspect);
      }

      @vertex
      fn vertex_main(
        @builtin(vertex_index) vertex_index: u32,
        @builtin(instance_index) instance_index: u32,
      ) -> VertexOutput {
        let agent_index = instance_index / ${TILE_COPY_COUNT}u;
        let tile_index = instance_index % ${TILE_COPY_COUNT}u;
        let tile_x = i32(tile_index % ${TILE_GRID_WIDTH}u) - ${Math.floor(TILE_GRID_WIDTH / 2)};
        let tile_y = i32(tile_index / ${TILE_GRID_WIDTH}u) - ${Math.floor(TILE_GRID_WIDTH / 2)};
        let tile_offset = vec2f(f32(tile_x), f32(tile_y));
        let base = agent_index * 2u;
        let pose = agents[base];
        let appearance = agents[base + 1u];

        var local = vec2f(0.0, 0.0);
        if (vertex_index == 0u) {
          local = vec2f(1.35, 0.0);
        } else if (vertex_index == 1u) {
          local = vec2f(-0.85, -0.48);
        } else {
          local = vec2f(-0.85, 0.48);
        }

        let direction = normalize(pose.zw);
        let side = vec2f(-direction.y, direction.x);
        let agent_delta = pose.xy + tile_offset - vec2f(view.camera_x, view.camera_y);
        let world_delta =
          agent_delta + (direction * local.x + side * local.y) * view.arrow_scale;
        let speed_glow = 0.65 + appearance.a * 0.35;

        var output: VertexOutput;
        output.position = vec4f(delta_to_clip(world_delta), 0.0, 1.0);
        output.color = vec4f(appearance.rgb * speed_glow, 1.0);
        return output;
      }

      @fragment
      fn fragment_main(input: VertexOutput) -> @location(0) vec4f {
        return input.color;
      }
    `,
  });

  const gridShader = device.createShaderModule({
    label: "world-grid-shader",
    code: `
      struct View {
        aspect: f32,
        arrow_scale: f32,
        camera_x: f32,
        camera_y: f32,
        zoom: f32,
        _pad0: f32,
        _pad1: f32,
        _pad2: f32,
      };

      struct VertexOutput {
        @builtin(position) position: vec4f,
        @location(0) clip_position: vec2f,
      };

      @group(0) @binding(0) var<uniform> view: View;

      fn clip_to_world_delta(clip_position: vec2f) -> vec2f {
        let zoom_scale = 2.0 * view.zoom;
        if (view.aspect > 1.0) {
          return vec2f(
            clip_position.x * view.aspect / zoom_scale,
            -clip_position.y / zoom_scale,
          );
        }

        return vec2f(
          clip_position.x / zoom_scale,
          -clip_position.y / (view.aspect * zoom_scale),
        );
      }

      fn line_alpha(value: f32, spacing: f32, width_pixels: f32) -> f32 {
        let scaled = value / spacing;
        let dist = abs(fract(scaled + 0.5) - 0.5);
        let pixel = max(fwidth(scaled), 0.000001);
        return 1.0 - smoothstep(pixel * width_pixels, pixel * (width_pixels + 1.0), dist);
      }

      @vertex
      fn vertex_main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
        var position = vec2f(-1.0, -1.0);
        if (vertex_index == 1u) {
          position = vec2f(3.0, -1.0);
        } else if (vertex_index == 2u) {
          position = vec2f(-1.0, 3.0);
        }

        var output: VertexOutput;
        output.position = vec4f(position, 0.0, 1.0);
        output.clip_position = position;
        return output;
      }

      @fragment
      fn fragment_main(input: VertexOutput) -> @location(0) vec4f {
        let world =
          vec2f(view.camera_x, view.camera_y) + clip_to_world_delta(input.clip_position);
        let minor_spacing = 1.0 / ${GRID_CELLS_PER_WORLD.toFixed(1)};
        let minor = max(
          line_alpha(world.x, minor_spacing, 0.42),
          line_alpha(world.y, minor_spacing, 0.42),
        );
        let boundary = max(
          line_alpha(world.x, 1.0, 1.15),
          line_alpha(world.y, 1.0, 1.15),
        );
        let color =
          vec3f(0.10, 0.18, 0.20) * minor +
          vec3f(0.58, 0.78, 0.72) * boundary;
        let alpha = min(0.55, minor * 0.18 + boundary * 0.48);

        return vec4f(color, alpha);
      }
    `,
  });

  const pipeline = device.createRenderPipeline({
    label: "agent-instancing-pipeline",
    layout: "auto",
    vertex: {
      module: shader,
      entryPoint: "vertex_main",
    },
    fragment: {
      module: shader,
      entryPoint: "fragment_main",
      targets: [
        {
          format,
          blend: {
            color: {
              srcFactor: "src-alpha",
              dstFactor: "one-minus-src-alpha",
              operation: "add",
            },
            alpha: {
              srcFactor: "one",
              dstFactor: "one-minus-src-alpha",
              operation: "add",
            },
          },
        },
      ],
    },
    primitive: {
      topology: "triangle-list",
    },
  });

  const gridPipeline = device.createRenderPipeline({
    label: "world-grid-pipeline",
    layout: "auto",
    vertex: {
      module: gridShader,
      entryPoint: "vertex_main",
    },
    fragment: {
      module: gridShader,
      entryPoint: "fragment_main",
      targets: [
        {
          format,
          blend: {
            color: {
              srcFactor: "src-alpha",
              dstFactor: "one-minus-src-alpha",
              operation: "add",
            },
            alpha: {
              srcFactor: "one",
              dstFactor: "one-minus-src-alpha",
              operation: "add",
            },
          },
        },
      ],
    },
    primitive: {
      topology: "triangle-list",
    },
  });

  const bindGroup = device.createBindGroup({
    label: "agent-instancing-bind-group",
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: agentBuffer } },
      { binding: 1, resource: { buffer: viewBuffer } },
    ],
  });

  const gridBindGroup = device.createBindGroup({
    label: "world-grid-bind-group",
    layout: gridPipeline.getBindGroupLayout(0),
    entries: [{ binding: 0, resource: { buffer: viewBuffer } }],
  });

  return {
    adapter,
    device,
    context,
    agentBuffer,
    viewBuffer,
    bindGroup,
    gridBindGroup,
    pipeline,
    gridPipeline,
    format,
  };
}

function uploadSnapshot(gpu: WebGpuState, snapshot: WorkerSnapshotMessage): void {
  gpu.device.queue.writeBuffer(gpu.agentBuffer, 0, new Float32Array(snapshot.buffer));
}

function updateViewUniform(gpu: WebGpuState, camera: CameraState): void {
  const aspect = canvasAspect();
  const arrowScale = ARROW_LENGTH_WORLD_UNITS / ARROW_LOCAL_LENGTH / WORLD_SIZE;
  const uniform = new Float32Array([
    aspect,
    arrowScale,
    camera.centerX,
    camera.centerY,
    camera.zoom,
    0,
    0,
    0,
  ]);
  gpu.device.queue.writeBuffer(gpu.viewBuffer, 0, uniform);
}

function setUpCameraControls(gpu: WebGpuState, camera: CameraState): void {
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
    updateViewUniform(gpu, camera);
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
      updateViewUniform(gpu, camera);
    },
    { passive: false },
  );

  resetViewButton.addEventListener("click", () => {
    resetCamera(camera);
    updateViewUniform(gpu, camera);
  });
}

function renderFrame(gpu: WebGpuState, population: number): void {
  const encoder = gpu.device.createCommandEncoder({
    label: "agent-render-encoder",
  });
  const view = gpu.context.getCurrentTexture().createView();

  const pass = encoder.beginRenderPass({
    label: "agent-render-pass",
    colorAttachments: [
      {
        view,
        clearValue: { r: 0.012, g: 0.014, b: 0.018, a: 1 },
        loadOp: "clear",
        storeOp: "store",
      },
    ],
  });

  pass.setPipeline(gpu.gridPipeline);
  pass.setBindGroup(0, gpu.gridBindGroup);
  pass.draw(3);

  pass.setPipeline(gpu.pipeline);
  pass.setBindGroup(0, gpu.bindGroup);
  pass.draw(3, population * TILE_COPY_COUNT);
  pass.end();

  gpu.device.queue.submit([encoder.finish()]);
}

function updateFps(now: number, state: FpsState): void {
  state.frames += 1;

  if (now - state.last >= 500) {
    const fps = Math.round((state.frames * 1000) / (now - state.last));
    fpsEl.textContent = `${fps} FPS`;
    state.frames = 0;
    state.last = now;
  }
}

function updateSimulationHud(stats: SimulationStats): void {
  stepsPerSecondEl.textContent = `${stats.stepsPerSecond.toLocaleString()} steps/s`;
  populationEl.textContent = `${stats.population.toLocaleString()} agents`;
  birthsEl.textContent = `${stats.births.toLocaleString()} births`;
  deathsEl.textContent = `${stats.deaths.toLocaleString()} deaths`;
  generationEl.textContent = `gen ${stats.generation.toLocaleString()}`;
}

function parseSimRate(): SimRate {
  if (speedSelect.value === "max") {
    return "max";
  }

  const value = Number(speedSelect.value);
  return Number.isFinite(value) && value > 0 ? value : 1;
}

function renderFrameIntervalMs(): number {
  if (renderFpsSelect.value === "display") {
    return 0;
  }

  const fps = Number(renderFpsSelect.value);
  return Number.isFinite(fps) && fps > 0 ? 1000 / fps : 0;
}

function parseSeed(): number {
  const value = Number(seedInput.value);
  if (!Number.isFinite(value)) {
    return INITIAL_SEED;
  }

  return Math.max(0, Math.floor(value)) >>> 0;
}

async function run(): Promise<void> {
  resizeCanvasToDisplaySize();
  window.addEventListener("resize", resizeCanvasToDisplaySize);

  seedInput.value = String(INITIAL_SEED);

  const worker = new Worker(new URL("./simulation-worker.ts", import.meta.url), {
    type: "module",
  });
  const camera = createDefaultCamera();
  const fpsState = { frames: 0, last: performance.now() };
  let activeEpoch = 0;
  let gpu: WebGpuState | null = null;
  let latestSnapshot: WorkerSnapshotMessage | null = null;
  let lastRenderAt = Number.NEGATIVE_INFINITY;
  let population = 0;
  let renderLoopStarted = false;
  let paused = false;

  const setSimulationControlsEnabled = (enabled: boolean): void => {
    pauseButton.disabled = !enabled;
    resetButton.disabled = !enabled;
    resetViewButton.disabled = !enabled;
    speedSelect.disabled = !enabled;
    renderFpsSelect.disabled = !enabled;
  };

  const sendToWorker = (message: MainToWorkerMessage, transfer: Transferable[] = []): void => {
    worker.postMessage(message, transfer);
  };

  const returnSnapshotBuffer = (buffer: ArrayBuffer): void => {
    sendToWorker({ type: "returnSnapshotBuffer", buffer }, [buffer]);
  };

  const clearLatestSnapshot = (): void => {
    if (!latestSnapshot) {
      return;
    }

    returnSnapshotBuffer(latestSnapshot.buffer);
    latestSnapshot = null;
  };

  const updatePauseButton = (): void => {
    pauseButton.textContent = paused ? "Resume" : "Pause";
    pauseButton.setAttribute("aria-pressed", String(paused));
  };

  pauseButton.addEventListener("click", () => {
    paused = !paused;
    updatePauseButton();
    sendToWorker({ type: "setPaused", paused });
  });

  speedSelect.addEventListener("change", () => {
    sendToWorker({ type: "setSimRate", simRate: parseSimRate() });
  });

  renderFpsSelect.addEventListener("change", () => {
    lastRenderAt = Number.NEGATIVE_INFINITY;
  });

  resetButton.addEventListener("click", () => {
    activeEpoch += 1;
    population = 0;
    clearLatestSnapshot();
    sendToWorker({ type: "reset", seed: parseSeed(), epoch: activeEpoch });
  });

  const loop = (now: number): void => {
    if (!gpu) {
      requestAnimationFrame(loop);
      return;
    }

    if (resizeCanvasToDisplaySize()) {
      updateViewUniform(gpu, camera);
    }

    const renderIntervalMs = renderFrameIntervalMs();
    const shouldRender =
      renderIntervalMs === 0 ||
      lastRenderAt === Number.NEGATIVE_INFINITY ||
      now - lastRenderAt >= renderIntervalMs - 0.5;

    if (shouldRender) {
      if (latestSnapshot) {
        const snapshot = latestSnapshot;
        latestSnapshot = null;
        uploadSnapshot(gpu, snapshot);
        population = snapshot.stats.population;
        updateSimulationHud(snapshot.stats);
        returnSnapshotBuffer(snapshot.buffer);
      }

      renderFrame(gpu, population);
      updateFps(now, fpsState);
      lastRenderAt = now;
    }

    requestAnimationFrame(loop);
  };

  const startRenderLoop = (): void => {
    if (renderLoopStarted) {
      return;
    }

    renderLoopStarted = true;
    requestAnimationFrame(loop);
  };

  const handleReady = async (
    message: Extract<WorkerToMainMessage, { type: "ready" }>,
  ): Promise<void> => {
    if (!gpu) {
      const agentByteSize = message.agentF32Len * Float32Array.BYTES_PER_ELEMENT;
      const nextGpu = await createWebGpuState(agentByteSize);
      gpu = nextGpu;
      setUpCameraControls(nextGpu, camera);
      updateViewUniform(nextGpu, camera);

      for (let i = 0; i < SNAPSHOT_BUFFER_COUNT; i += 1) {
        returnSnapshotBuffer(new ArrayBuffer(agentByteSize));
      }

      setStatus(`${nextGpu.adapter.info?.description || "WebGPU"} ready`);
      setSimulationControlsEnabled(true);
      startRenderLoop();
    }

    if (message.epoch === activeEpoch) {
      updateSimulationHud(message.stats);
    }
  };

  worker.addEventListener("message", (event: MessageEvent<WorkerToMainMessage>) => {
    const message = event.data;

    switch (message.type) {
      case "ready":
        void handleReady(message).catch((error: unknown) => {
          const text = error instanceof Error ? error.message : String(error);
          setStatus(text);
          fpsEl.textContent = "WebGPU unavailable";
          console.error(error);
        });
        break;
      case "snapshot":
        if (message.epoch !== activeEpoch) {
          returnSnapshotBuffer(message.buffer);
          break;
        }

        clearLatestSnapshot();
        latestSnapshot = message;
        updateSimulationHud(message.stats);
        break;
      case "stats":
        if (message.epoch === activeEpoch) {
          updateSimulationHud(message.stats);
        }
        break;
      case "error":
        setStatus(message.message);
        console.error(message.message);
        break;
    }
  });

  setSimulationControlsEnabled(false);
  setStatus("Starting simulation worker");
  updatePauseButton();
  updateSimulationHud({
    population: 0,
    births: 0,
    deaths: 0,
    generation: 0,
    simSteps: 0,
    stepsPerSecond: 0,
  });

  sendToWorker({
    type: "init",
    worldSize: WORLD_SIZE,
    population: POPULATION,
    seed: INITIAL_SEED,
    epoch: activeEpoch,
    paused,
    simRate: parseSimRate(),
  });
}

run().catch((error: unknown) => {
  const message = error instanceof Error ? error.message : String(error);
  setStatus(message);
  fpsEl.textContent = "WebGPU unavailable";
  console.error(error);
});
