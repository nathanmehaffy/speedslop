import initWasm, { Simulation, type InitOutput } from "../sim/pkg/speedslop.js";
import "./style.css";

const WORLD_SIZE = 4096;
const POPULATION = 10_000;
const INITIAL_SEED = 1;
const MAX_DELTA_SECONDS = 1 / 10;
const ARROW_LENGTH_WORLD_UNITS = 18;
const ARROW_LOCAL_LENGTH = 2.2;

type WebGpuState = {
  adapter: GPUAdapter;
  device: GPUDevice;
  context: GPUCanvasContext;
  agentBuffer: GPUBuffer;
  viewBuffer: GPUBuffer;
  bindGroup: GPUBindGroup;
  pipeline: GPURenderPipeline;
  format: GPUTextureFormat;
};

type FpsState = {
  frames: number;
  last: number;
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
const populationEl = queryRequired<HTMLSpanElement>("#population");
const birthsEl = queryRequired<HTMLSpanElement>("#births");
const deathsEl = queryRequired<HTMLSpanElement>("#deaths");
const generationEl = queryRequired<HTMLSpanElement>("#generation");
const pauseButton = queryRequired<HTMLButtonElement>("#pause");
const resetButton = queryRequired<HTMLButtonElement>("#reset");
const seedInput = queryRequired<HTMLInputElement>("#seed");
const speedSelect = queryRequired<HTMLSelectElement>("#speed");

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
    size: 16,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });

  const shader = device.createShaderModule({
    label: "agent-instancing-shader",
    code: `
      struct View {
        aspect: f32,
        arrow_scale: f32,
        _pad0: f32,
        _pad1: f32,
      };

      struct VertexOutput {
        @builtin(position) position: vec4f,
        @location(0) color: vec4f,
      };

      @group(0) @binding(0) var<storage, read> agents: array<vec4f>;
      @group(0) @binding(1) var<uniform> view: View;

      fn world_to_clip(position: vec2f) -> vec2f {
        let centered = position * 2.0 - vec2f(1.0, 1.0);
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
        let base = instance_index * 2u;
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
        let world_position =
          pose.xy + (direction * local.x + side * local.y) * view.arrow_scale;
        let speed_glow = 0.65 + appearance.a * 0.35;

        var output: VertexOutput;
        output.position = vec4f(world_to_clip(world_position), 0.0, 1.0);
        output.color = vec4f(appearance.rgb * speed_glow, 1.0);
        return output;
      }

      @fragment
      fn fragment_main(input: VertexOutput) -> @location(0) vec4f {
        return input.color;
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

  const bindGroup = device.createBindGroup({
    label: "agent-instancing-bind-group",
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: agentBuffer } },
      { binding: 1, resource: { buffer: viewBuffer } },
    ],
  });

  return { adapter, device, context, agentBuffer, viewBuffer, bindGroup, pipeline, format };
}

function uploadAgents(gpu: WebGpuState, wasm: InitOutput, simulation: Simulation): void {
  const ptr = simulation.agent_ptr();
  const len = simulation.agent_f32_len();
  const agents = new Float32Array(wasm.memory.buffer, ptr, len);

  gpu.device.queue.writeBuffer(gpu.agentBuffer, 0, agents);
}

function updateViewUniform(gpu: WebGpuState): void {
  const aspect = canvas.height > 0 ? canvas.width / canvas.height : 1;
  const arrowScale = ARROW_LENGTH_WORLD_UNITS / ARROW_LOCAL_LENGTH / WORLD_SIZE;
  const uniform = new Float32Array([aspect, arrowScale, 0, 0]);
  gpu.device.queue.writeBuffer(gpu.viewBuffer, 0, uniform);
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

  pass.setPipeline(gpu.pipeline);
  pass.setBindGroup(0, gpu.bindGroup);
  pass.draw(3, population);
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

function updateSimulationHud(simulation: Simulation): void {
  populationEl.textContent = `${simulation.population().toLocaleString()} agents`;
  birthsEl.textContent = `${simulation.births().toLocaleString()} births`;
  deathsEl.textContent = `${simulation.deaths().toLocaleString()} deaths`;
  generationEl.textContent = `gen ${simulation.generation().toLocaleString()}`;
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

  const wasm = await initWasm();
  const simulation = new Simulation(WORLD_SIZE, POPULATION, INITIAL_SEED);
  const agentByteSize = simulation.agent_f32_len() * Float32Array.BYTES_PER_ELEMENT;
  const gpu = await createWebGpuState(agentByteSize);
  const fpsState = { frames: 0, last: performance.now() };
  let previous = performance.now();
  let paused = false;

  const updatePauseButton = (): void => {
    pauseButton.textContent = paused ? "Resume" : "Pause";
    pauseButton.setAttribute("aria-pressed", String(paused));
  };

  pauseButton.addEventListener("click", () => {
    paused = !paused;
    updatePauseButton();
  });

  resetButton.addEventListener("click", () => {
    simulation.reset(parseSeed());
    uploadAgents(gpu, wasm, simulation);
    updateSimulationHud(simulation);
  });

  setStatus(`${gpu.adapter.info?.description || "WebGPU"} ready`);
  updatePauseButton();
  uploadAgents(gpu, wasm, simulation);
  updateViewUniform(gpu);
  updateSimulationHud(simulation);

  const loop = (now: number): void => {
    if (resizeCanvasToDisplaySize()) {
      updateViewUniform(gpu);
    }

    const dt = Math.min((now - previous) / 1000, MAX_DELTA_SECONDS);
    previous = now;

    if (!paused) {
      const speedScale = Number(speedSelect.value);
      simulation.tick(dt * (Number.isFinite(speedScale) ? speedScale : 1));
    }

    uploadAgents(gpu, wasm, simulation);
    renderFrame(gpu, simulation.population());
    updateFps(now, fpsState);
    updateSimulationHud(simulation);

    requestAnimationFrame(loop);
  };

  requestAnimationFrame(loop);
}

run().catch((error: unknown) => {
  const message = error instanceof Error ? error.message : String(error);
  setStatus(message);
  fpsEl.textContent = "WebGPU unavailable";
  console.error(error);
});
