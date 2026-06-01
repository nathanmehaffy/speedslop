import "./style.css";

const shaderSource = /* wgsl */ `
struct Uniforms {
  time: f32,
  aspect: f32,
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;

struct VertexOutput {
  @builtin(position) position: vec4f,
  @location(0) uv: vec2f,
}

@vertex
fn vs_main(@builtin(vertex_index) vertexIndex: u32) -> VertexOutput {
  var positions = array(
    vec2f(-1.0, -1.0),
    vec2f( 3.0, -1.0),
    vec2f(-1.0,  3.0),
  );
  var output: VertexOutput;
  output.position = vec4f(positions[vertexIndex], 0.0, 1.0);
  output.uv = positions[vertexIndex] * vec2f(0.5, -0.5) + vec2f(0.5, 0.5);
  return output;
}

fn spiral(uv: vec2f, time: f32) -> f32 {
  let centered = uv - vec2f(0.5, 0.5);
  let aspect = vec2f(uniforms.aspect, 1.0);
  let p = centered * aspect;
  let radius = length(p);
  let angle = atan2(p.y, p.x) + time;
  let arms = 5.0;
  let tightness = 18.0;
  return sin(angle * arms - radius * tightness);
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4f {
  let wave = spiral(input.uv, uniforms.time);
  let glow = smoothstep(-0.15, 0.85, wave);
  let hue = fract(uniforms.time * 0.05 + input.uv.x * 0.2 + wave * 0.08);
  let color = vec3f(
    0.5 + 0.5 * cos(6.28318 * (hue + 0.0)),
    0.5 + 0.5 * cos(6.28318 * (hue + 0.33)),
    0.5 + 0.5 * cos(6.28318 * (hue + 0.67)),
  );
  return vec4f(color * glow, 1.0);
}
`;

function resizeCanvas(canvas: HTMLCanvasElement): { width: number; height: number } {
  const dpr = Math.min(window.devicePixelRatio, 2);
  const width = Math.max(1, Math.floor(canvas.clientWidth * dpr));
  const height = Math.max(1, Math.floor(canvas.clientHeight * dpr));
  if (canvas.width !== width || canvas.height !== height) {
    canvas.width = width;
    canvas.height = height;
  }
  return { width, height };
}

async function main(): Promise<void> {
  const canvasElement = document.querySelector<HTMLCanvasElement>("#canvas");
  if (!canvasElement) {
    throw new Error("Canvas element #canvas not found");
  }
  const canvas: HTMLCanvasElement = canvasElement;

  if (!navigator.gpu) {
    throw new Error("WebGPU is not available in this browser");
  }

  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) {
    throw new Error("No WebGPU adapter found");
  }

  const device = await adapter.requestDevice();
  const webgpuContext = canvas.getContext("webgpu");
  if (!webgpuContext) {
    throw new Error("Failed to get WebGPU canvas context");
  }
  const context: GPUCanvasContext = webgpuContext;

  const format = navigator.gpu.getPreferredCanvasFormat();
  let size = resizeCanvas(canvas);

  context.configure({
    device,
    format,
    alphaMode: "opaque",
  });

  const uniformBuffer = device.createBuffer({
    size: 8,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });

  const bindGroupLayout = device.createBindGroupLayout({
    entries: [
      {
        binding: 0,
        visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
        buffer: { type: "uniform" },
      },
    ],
  });

  const pipelineLayout = device.createPipelineLayout({
    bindGroupLayouts: [bindGroupLayout],
  });

  const shaderModule = device.createShaderModule({ code: shaderSource });

  const pipeline = device.createRenderPipeline({
    layout: pipelineLayout,
    vertex: {
      module: shaderModule,
      entryPoint: "vs_main",
    },
    fragment: {
      module: shaderModule,
      entryPoint: "fs_main",
      targets: [{ format }],
    },
    primitive: {
      topology: "triangle-list",
    },
  });

  const bindGroup = device.createBindGroup({
    layout: bindGroupLayout,
    entries: [{ binding: 0, resource: { buffer: uniformBuffer } }],
  });

  const resizeObserver = new ResizeObserver(() => {
    size = resizeCanvas(canvas);
  });
  resizeObserver.observe(canvas);

  const start = performance.now();
  const fpsElement = document.querySelector<HTMLElement>("#fps-monitor");
  let fpsFrameCount = 0;
  let fpsLastSample = performance.now();

  function frame(timestamp: number): void {
    fpsFrameCount += 1;
    const fpsElapsed = timestamp - fpsLastSample;
    if (fpsElapsed >= 500 && fpsElement) {
      const fps = Math.round((fpsFrameCount * 1000) / fpsElapsed);
      fpsElement.textContent = `${fps} FPS`;
      fpsFrameCount = 0;
      fpsLastSample = timestamp;
    }

    size = resizeCanvas(canvas);
    const aspect = size.width / size.height;
    const time = (performance.now() - start) * 0.001;

    device.queue.writeBuffer(
      uniformBuffer,
      0,
      new Float32Array([time, aspect]),
    );

    const commandEncoder = device.createCommandEncoder();
    const textureView = context.getCurrentTexture().createView();
    const pass = commandEncoder.beginRenderPass({
      colorAttachments: [
        {
          view: textureView,
          clearValue: { r: 0.02, g: 0.03, b: 0.06, a: 1 },
          loadOp: "clear",
          storeOp: "store",
        },
      ],
    });

    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.draw(3);
    pass.end();

    device.queue.submit([commandEncoder.finish()]);
    requestAnimationFrame(frame);
  }

  requestAnimationFrame(frame);
}

main().catch((error: unknown) => {
  console.error(error);
  const message = error instanceof Error ? error.message : String(error);
  document.body.textContent = `WebGPU error: ${message}`;
});
