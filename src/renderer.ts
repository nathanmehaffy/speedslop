// Renderer for the torus agent world.
//
// Agents are drawn as direction-facing, HSV-coloured triangles via an indirect
// draw whose instance count is the GPU-resident live count (no readback). The
// world is the unit square; to visualise the torus wrapping it is tiled across
// the visible viewport, and each tile gets a thin grey border. Pan/zoom come
// from the pure `Camera`; the renderer only consumes the resulting transform.

import { AGENT_TRIANGLE_SIZE, BORDER_COLOR, CLEAR_COLOR, WORLD_SIZE } from "./config";
import { MAX_VISIBLE_TILES, type TileOffset } from "./camera";
import { AGENT_STRUCT_WGSL, DENSE_STRUCT_WGSL } from "./layout";

const TILE_STRIDE = 256; // min uniform buffer dynamic-offset alignment

const SHADER = /* wgsl */ `
struct Camera {
  center: vec2f,
  scale: vec2f,
  triSize: f32,
}

struct Tile {
  offset: vec2f,
  pad: vec2f,
}

${AGENT_STRUCT_WGSL}
${DENSE_STRUCT_WGSL}

@group(0) @binding(0) var<uniform> camera: Camera;
@group(0) @binding(1) var<uniform> tile: Tile;
@group(0) @binding(2) var<storage, read> agents: array<Agent>;
@group(0) @binding(3) var<storage, read> dense: array<Dense>;

struct VSOut {
  @builtin(position) pos: vec4f,
  @location(0) color: vec3f,
}

fn hsv2rgb(hsv: vec3f) -> vec3f {
  let h = fract(hsv.x) * 6.0;
  let c = hsv.z * hsv.y;
  let x = c * (1.0 - abs(fract(h * 0.5) * 2.0 - 1.0));
  var rgb = vec3f(0.0);
  if (h < 1.0) { rgb = vec3f(c, x, 0.0); }
  else if (h < 2.0) { rgb = vec3f(x, c, 0.0); }
  else if (h < 3.0) { rgb = vec3f(0.0, c, x); }
  else if (h < 4.0) { rgb = vec3f(0.0, x, c); }
  else if (h < 5.0) { rgb = vec3f(x, 0.0, c); }
  else { rgb = vec3f(c, 0.0, x); }
  return rgb + (hsv.z - c);
}

fn toClip(world: vec2f) -> vec4f {
  let ndc = (world - camera.center) * camera.scale;
  return vec4f(ndc, 0.0, 1.0);
}

@vertex
fn vsAgent(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> VSOut {
  let a = agents[dense[ii].slot];
  let s = camera.triSize;
  var local = array<vec2f, 3>(
    vec2f(s, 0.0),
    vec2f(-0.6 * s, 0.5 * s),
    vec2f(-0.6 * s, -0.5 * s),
  );
  let lv = local[vi];
  let c = cos(a.dir);
  let sn = sin(a.dir);
  let rotated = vec2f(lv.x * c - lv.y * sn, lv.x * sn + lv.y * c);
  let world = a.pos + tile.offset + rotated;

  var out: VSOut;
  out.pos = toClip(world);
  out.color = hsv2rgb(vec3f(a.hue, a.sat, a.val));
  return out;
}

@vertex
fn vsBorder(@builtin(vertex_index) vi: u32) -> VSOut {
  let w = f32(${WORLD_SIZE});
  var corners = array<vec2f, 8>(
    vec2f(0.0, 0.0), vec2f(w, 0.0),
    vec2f(w, 0.0), vec2f(w, w),
    vec2f(w, w), vec2f(0.0, w),
    vec2f(0.0, w), vec2f(0.0, 0.0),
  );
  var out: VSOut;
  out.pos = toClip(corners[vi] + tile.offset);
  out.color = vec3f(${BORDER_COLOR[0]}, ${BORDER_COLOR[1]}, ${BORDER_COLOR[2]});
  return out;
}

@fragment
fn fs(in: VSOut) -> @location(0) vec4f {
  return vec4f(in.color, 1.0);
}
`;

export class Renderer {
  private readonly device: GPUDevice;
  private readonly agentPipeline: GPURenderPipeline;
  private readonly borderPipeline: GPURenderPipeline;
  private readonly cameraBuffer: GPUBuffer;
  private readonly tileBuffer: GPUBuffer;
  private readonly bindGroup: GPUBindGroup;
  private readonly indirect: GPUBuffer;
  private readonly tileScratch: Uint8Array<ArrayBuffer>;

  private tileCount = 0;

  constructor(
    device: GPUDevice,
    format: GPUTextureFormat,
    agents: GPUBuffer,
    dense: GPUBuffer,
    indirect: GPUBuffer,
  ) {
    this.device = device;
    this.indirect = indirect;

    this.cameraBuffer = device.createBuffer({
      size: 32,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      label: "camera",
    });
    this.tileBuffer = device.createBuffer({
      size: MAX_VISIBLE_TILES * TILE_STRIDE,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      label: "tiles",
    });

    const layout = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.VERTEX, buffer: { type: "uniform" } },
        {
          binding: 1,
          visibility: GPUShaderStage.VERTEX,
          buffer: { type: "uniform", hasDynamicOffset: true, minBindingSize: 16 },
        },
        { binding: 2, visibility: GPUShaderStage.VERTEX, buffer: { type: "read-only-storage" } },
        { binding: 3, visibility: GPUShaderStage.VERTEX, buffer: { type: "read-only-storage" } },
      ],
    });
    this.bindGroup = device.createBindGroup({
      layout,
      entries: [
        { binding: 0, resource: { buffer: this.cameraBuffer } },
        { binding: 1, resource: { buffer: this.tileBuffer, size: 16 } },
        { binding: 2, resource: { buffer: agents } },
        { binding: 3, resource: { buffer: dense } },
      ],
    });

    const module = device.createShaderModule({ code: SHADER });
    const pipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [layout] });
    this.agentPipeline = device.createRenderPipeline({
      layout: pipelineLayout,
      vertex: { module, entryPoint: "vsAgent" },
      fragment: { module, entryPoint: "fs", targets: [{ format }] },
      primitive: { topology: "triangle-list" },
    });
    this.borderPipeline = device.createRenderPipeline({
      layout: pipelineLayout,
      vertex: { module, entryPoint: "vsBorder" },
      fragment: { module, entryPoint: "fs", targets: [{ format }] },
      primitive: { topology: "line-list" },
    });
    this.tileScratch = new Uint8Array(new ArrayBuffer(MAX_VISIBLE_TILES * TILE_STRIDE));
  }

  /** Update the camera transform and the per-tile offsets for this frame. */
  update(
    center: { x: number; y: number },
    zoom: number,
    canvasWidth: number,
    canvasHeight: number,
    offsets: readonly TileOffset[],
  ): void {
    const cam = new Float32Array(8);
    cam[0] = center.x;
    cam[1] = center.y;
    cam[2] = (2 * zoom) / canvasWidth;
    cam[3] = (2 * zoom) / canvasHeight;
    cam[4] = AGENT_TRIANGLE_SIZE;
    this.device.queue.writeBuffer(this.cameraBuffer, 0, cam);

    if (offsets.length > MAX_VISIBLE_TILES) {
      throw new Error(`renderer received ${offsets.length} tile offsets, but the budget is ${MAX_VISIBLE_TILES}`);
    }
    this.tileCount = offsets.length;
    const view = new Float32Array(this.tileScratch.buffer);
    for (let i = 0; i < this.tileCount; i += 1) {
      view[(i * TILE_STRIDE) / 4 + 0] = offsets[i][0];
      view[(i * TILE_STRIDE) / 4 + 1] = offsets[i][1];
    }
    if (this.tileCount > 0) {
      this.device.queue.writeBuffer(
        this.tileBuffer,
        0,
        this.tileScratch,
        0,
        this.tileCount * TILE_STRIDE,
      );
    }
  }

  encode(
    encoder: GPUCommandEncoder,
    view: GPUTextureView,
    timestampWrites?: GPURenderPassTimestampWrites,
  ): void {
    const pass = encoder.beginRenderPass({
      colorAttachments: [
        { view, clearValue: CLEAR_COLOR, loadOp: "clear", storeOp: "store" },
      ],
      timestampWrites,
    });

    pass.setPipeline(this.borderPipeline);
    for (let i = 0; i < this.tileCount; i += 1) {
      pass.setBindGroup(0, this.bindGroup, [i * TILE_STRIDE]);
      pass.draw(8, 1);
    }

    pass.setPipeline(this.agentPipeline);
    for (let i = 0; i < this.tileCount; i += 1) {
      pass.setBindGroup(0, this.bindGroup, [i * TILE_STRIDE]);
      pass.drawIndirect(this.indirect, 0);
    }

    pass.end();
  }
}
