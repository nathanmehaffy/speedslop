import initWasm, { Simulation, type InitOutput } from "../sim/pkg/speedslop.js";
import "./style.css";

const SIM_WIDTH = 512;
const SIM_HEIGHT = 512;
const MAX_DELTA_SECONDS = 1 / 20;

type WebGpuState = {
  adapter: GPUAdapter;
  device: GPUDevice;
  context: GPUCanvasContext;
  texture: GPUTexture;
  bindGroup: GPUBindGroup;
  pipeline: GPURenderPipeline;
  format: GPUTextureFormat;
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

function setStatus(message: string): void {
  statusEl.textContent = message;
}

function resizeCanvasToDisplaySize(): void {
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const width = Math.max(1, Math.floor(canvas.clientWidth * dpr));
  const height = Math.max(1, Math.floor(canvas.clientHeight * dpr));

  if (canvas.width !== width || canvas.height !== height) {
    canvas.width = width;
    canvas.height = height;
  }
}

async function createWebGpuState(): Promise<WebGpuState> {
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

  const texture = device.createTexture({
    label: "simulation-frame",
    size: [SIM_WIDTH, SIM_HEIGHT],
    format: "rgba8unorm",
    usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
  });

  const sampler = device.createSampler({
    magFilter: "linear",
    minFilter: "linear",
  });

  const shader = device.createShaderModule({
    label: "simulation-present-shader",
    code: `
      struct VertexOutput {
        @builtin(position) position: vec4f,
        @location(0) uv: vec2f,
      };

      @vertex
      fn vertex_main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
        var positions = array<vec2f, 3>(
          vec2f(-1.0, -3.0),
          vec2f(-1.0,  1.0),
          vec2f( 3.0,  1.0)
        );

        let position = positions[vertex_index];

        var output: VertexOutput;
        output.position = vec4f(position, 0.0, 1.0);
        output.uv = position * vec2f(0.5, -0.5) + vec2f(0.5, 0.5);
        return output;
      }

      @group(0) @binding(0) var simulation_sampler: sampler;
      @group(0) @binding(1) var simulation_texture: texture_2d<f32>;

      @fragment
      fn fragment_main(input: VertexOutput) -> @location(0) vec4f {
        return textureSample(simulation_texture, simulation_sampler, input.uv);
      }
    `,
  });

  const pipeline = device.createRenderPipeline({
    label: "simulation-present-pipeline",
    layout: "auto",
    vertex: {
      module: shader,
      entryPoint: "vertex_main",
    },
    fragment: {
      module: shader,
      entryPoint: "fragment_main",
      targets: [{ format }],
    },
    primitive: {
      topology: "triangle-list",
    },
  });

  const bindGroup = device.createBindGroup({
    label: "simulation-present-bind-group",
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: sampler },
      { binding: 1, resource: texture.createView() },
    ],
  });

  return { adapter, device, context, texture, bindGroup, pipeline, format };
}

function uploadSimulationFrame(
  device: GPUDevice,
  texture: GPUTexture,
  wasm: InitOutput,
  simulation: Simulation,
): void {
  const ptr = simulation.frame_ptr();
  const len = simulation.frame_len();
  const frame = new Uint8Array(wasm.memory.buffer, ptr, len);

  device.queue.writeTexture(
    { texture },
    frame,
    { bytesPerRow: SIM_WIDTH * 4, rowsPerImage: SIM_HEIGHT },
    { width: SIM_WIDTH, height: SIM_HEIGHT },
  );
}

function renderFrame(gpu: WebGpuState): void {
  const encoder = gpu.device.createCommandEncoder({
    label: "simulation-present-encoder",
  });
  const view = gpu.context.getCurrentTexture().createView();

  const pass = encoder.beginRenderPass({
    label: "simulation-present-pass",
    colorAttachments: [
      {
        view,
        clearValue: { r: 0.02, g: 0.03, b: 0.04, a: 1 },
        loadOp: "clear",
        storeOp: "store",
      },
    ],
  });

  pass.setPipeline(gpu.pipeline);
  pass.setBindGroup(0, gpu.bindGroup);
  pass.draw(3);
  pass.end();

  gpu.device.queue.submit([encoder.finish()]);
}

function updateFps(now: number, state: { frames: number; last: number }): void {
  state.frames += 1;

  if (now - state.last >= 500) {
    const fps = Math.round((state.frames * 1000) / (now - state.last));
    fpsEl.textContent = `${fps} FPS`;
    state.frames = 0;
    state.last = now;
  }
}

async function run(): Promise<void> {
  resizeCanvasToDisplaySize();
  window.addEventListener("resize", resizeCanvasToDisplaySize);

  const wasm = await initWasm();
  const simulation = new Simulation(SIM_WIDTH, SIM_HEIGHT);
  const gpu = await createWebGpuState();
  const fpsState = { frames: 0, last: performance.now() };
  let previous = performance.now();

  setStatus(`${gpu.adapter.info?.description || "WebGPU"} ready`);

  const loop = (now: number): void => {
    resizeCanvasToDisplaySize();

    const dt = Math.min((now - previous) / 1000, MAX_DELTA_SECONDS);
    previous = now;

    simulation.tick(dt);
    uploadSimulationFrame(gpu.device, gpu.texture, wasm, simulation);
    renderFrame(gpu);
    updateFps(now, fpsState);

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
