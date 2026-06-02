// Placeholder renderer.
//
// Draws each agent as a small instanced quad. Positions are supplied as an
// instanced vertex buffer (the simulation's current position buffer), so no
// storage-buffer reads happen in the vertex stage. The buffer is bound per
// frame to whichever ping-pong buffer holds the latest state.

import { CLEAR_COLOR, POINT_SIZE } from "./config";

const SHADER = /* wgsl */ `
struct RenderParams {
  aspect: f32,
  pointSize: f32,
}

@group(0) @binding(0) var<uniform> rp: RenderParams;

struct VSOut {
  @builtin(position) pos: vec4f,
  @location(0) color: vec3f,
}

fn hue(h: f32) -> vec3f {
  let k = fract(h + vec3f(0.0, 2.0 / 3.0, 1.0 / 3.0)) * 6.0 - 3.0;
  return clamp(abs(k) - 1.0, vec3f(0.0), vec3f(1.0));
}

@vertex
fn vs(
  @builtin(vertex_index) vi: u32,
  @builtin(instance_index) ii: u32,
  @location(0) agent: vec2f,
) -> VSOut {
  var corners = array<vec2f, 6>(
    vec2f(-1.0, -1.0), vec2f(1.0, -1.0), vec2f(-1.0, 1.0),
    vec2f(-1.0, 1.0), vec2f(1.0, -1.0), vec2f(1.0, 1.0),
  );

  var ndc = agent * 2.0 - vec2f(1.0, 1.0);
  ndc.y = -ndc.y;
  var offset = corners[vi] * rp.pointSize;

  // Keep the world square regardless of canvas aspect ratio.
  if (rp.aspect >= 1.0) {
    ndc.x = ndc.x / rp.aspect;
    offset.x = offset.x / rp.aspect;
  } else {
    ndc.y = ndc.y * rp.aspect;
    offset.y = offset.y * rp.aspect;
  }

  var out: VSOut;
  out.pos = vec4f(ndc + offset, 0.0, 1.0);
  out.color = hue(f32(ii) * 0.618034);
  return out;
}

@fragment
fn fs(in: VSOut) -> @location(0) vec4f {
  return vec4f(in.color, 1.0);
}
`;

export class Renderer {
  private readonly device: GPUDevice;
  private readonly agentCount: number;
  private readonly pipeline: GPURenderPipeline;
  private readonly paramsBuffer: GPUBuffer;
  private readonly bindGroup: GPUBindGroup;

  constructor(device: GPUDevice, format: GPUTextureFormat, agentCount: number) {
    this.device = device;
    this.agentCount = agentCount;

    this.paramsBuffer = device.createBuffer({
      size: 8,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      label: "render-params",
    });

    const layout = device.createBindGroupLayout({
      entries: [{ binding: 0, visibility: GPUShaderStage.VERTEX, buffer: { type: "uniform" } }],
    });

    const module = device.createShaderModule({ code: SHADER });
    this.pipeline = device.createRenderPipeline({
      layout: device.createPipelineLayout({ bindGroupLayouts: [layout] }),
      vertex: {
        module,
        entryPoint: "vs",
        buffers: [
          {
            arrayStride: 8,
            stepMode: "instance",
            attributes: [{ shaderLocation: 0, offset: 0, format: "float32x2" }],
          },
        ],
      },
      fragment: { module, entryPoint: "fs", targets: [{ format }] },
      primitive: { topology: "triangle-list" },
    });

    this.bindGroup = device.createBindGroup({
      layout,
      entries: [{ binding: 0, resource: { buffer: this.paramsBuffer } }],
    });
  }

  updateParams(aspect: number): void {
    this.device.queue.writeBuffer(this.paramsBuffer, 0, new Float32Array([aspect, POINT_SIZE]));
  }

  encode(
    encoder: GPUCommandEncoder,
    view: GPUTextureView,
    positions: GPUBuffer,
  ): void {
    const pass = encoder.beginRenderPass({
      colorAttachments: [
        {
          view,
          clearValue: CLEAR_COLOR,
          loadOp: "clear",
          storeOp: "store",
        },
      ],
    });
    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, this.bindGroup);
    pass.setVertexBuffer(0, positions);
    pass.draw(6, this.agentCount);
    pass.end();
  }
}
