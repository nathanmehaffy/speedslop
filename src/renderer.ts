// Renderer for the torus agent world.
//
// Agents are drawn directly from fixed simulation slots as direction-facing,
// HSV-coloured triangles. Dead slots emit degenerate offscreen triangles, which
// is cheaper than rebuilding a live-only render index after every sim batch.
// Torus tile copies are rendered with instancing so zoomed-out views do not
// multiply draw calls.

import { AGENT_TRIANGLE_SIZE, BORDER_COLOR, CLEAR_COLOR, MAX_AGENTS, WORLD_SIZE } from "./config";
import { MAX_VISIBLE_TILES, type TileOffset } from "./camera";
import { AGENT_BYTES, AGENT_STRUCT_WGSL } from "./layout";

const TILE_BYTES = 16;
const TILE_F32 = TILE_BYTES / 4;
const TRAIL_HISTORY = 4;
const TRAIL_SUBDIVISIONS = 2;
const TAIL_ALPHA = 0.65;
const TAIL_MIN_ALPHA = 0.08;
const TAIL_WIDTH_FACTOR = 0.55;

const SHADER = /* wgsl */ `
const MAX_AGENTS: u32 = ${MAX_AGENTS}u;

struct Camera {
  center: vec2f,
  scale: vec2f,
  triSize: f32,
}

struct Trail {
  head: u32,
  readyCount: u32,
  pad: vec2u,
}

struct Tile {
  offset: vec2f,
  pad: vec2f,
}

${AGENT_STRUCT_WGSL}

@group(0) @binding(0) var<uniform> camera: Camera;
@group(0) @binding(1) var<storage, read> tiles: array<Tile>;
@group(0) @binding(2) var<storage, read> agents: array<Agent>;
@group(0) @binding(3) var<storage, read> historyAgents: array<Agent>;
@group(0) @binding(4) var<uniform> trail: Trail;

struct VSOut {
  @builtin(position) pos: vec4f,
  @location(0) color: vec4f,
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

fn deadVertex() -> VSOut {
  var dead: VSOut;
  dead.pos = vec4f(2.0, 2.0, 0.0, 1.0);
  dead.color = vec4f(0.0);
  return dead;
}

fn wrapDelta(a: vec2f, b: vec2f) -> vec2f {
  let world = f32(${WORLD_SIZE});
  return b - a - world * round((b - a) / world);
}

fn historyAgent(slot: u32, age: u32) -> Agent {
  let historySlot = (trail.head + ${TRAIL_HISTORY}u - age) % ${TRAIL_HISTORY}u;
  return historyAgents[historySlot * MAX_AGENTS + slot];
}

fn sampleAgent(slot: u32, sampleIndex: u32) -> Agent {
  if (sampleIndex == 0u) {
    return agents[slot];
  }
  return historyAgent(slot, sampleIndex - 1u);
}

fn compatibleSample(slot: u32, sampleIndex: u32, id: u32) -> bool {
  if (sampleIndex > trail.readyCount) {
    return false;
  }
  let a = sampleAgent(slot, sampleIndex);
  return a.alive == 1u && a.id == id;
}

fn samplePos(slot: u32, sampleIndex: u32, fallbackIndex: u32, id: u32) -> vec2f {
  if (compatibleSample(slot, sampleIndex, id)) {
    return sampleAgent(slot, sampleIndex).pos;
  }
  return sampleAgent(slot, fallbackIndex).pos;
}

fn catmullRom(p0: vec2f, p1: vec2f, p2: vec2f, p3: vec2f, t: f32) -> vec2f {
  let t2 = t * t;
  let t3 = t2 * t;
  return 0.5 * (
    (2.0 * p1) +
    (-p0 + p2) * t +
    (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t2 +
    (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t3
  );
}

@vertex
fn vsAgent(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> VSOut {
  let tileIndex = ii / MAX_AGENTS;
  let slot = ii - tileIndex * MAX_AGENTS;
  let a = agents[slot];
  if (a.alive != 1u) {
    return deadVertex();
  }

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
  let world = a.pos + tiles[tileIndex].offset + rotated;

  var out: VSOut;
  out.pos = toClip(world);
  out.color = vec4f(hsv2rgb(vec3f(a.hue, a.sat, a.val)), 1.0);
  return out;
}

@vertex
fn vsTail(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> VSOut {
  let trailInstance = ii % (${TRAIL_HISTORY}u * ${TRAIL_SUBDIVISIONS}u);
  let agentInstance = ii / (${TRAIL_HISTORY}u * ${TRAIL_SUBDIVISIONS}u);
  let tileIndex = agentInstance / MAX_AGENTS;
  let slot = agentInstance - tileIndex * MAX_AGENTS;
  let segmentIndex = trailInstance / ${TRAIL_SUBDIVISIONS}u;
  let subdivisionIndex = trailInstance - segmentIndex * ${TRAIL_SUBDIVISIONS}u;

  let a = agents[slot];
  if (a.alive != 1u || segmentIndex >= trail.readyCount) {
    return deadVertex();
  }

  let s1 = segmentIndex;
  let s2 = segmentIndex + 1u;
  if (!compatibleSample(slot, s1, a.id) || !compatibleSample(slot, s2, a.id)) {
    return deadVertex();
  }

  let s0 = select(0u, segmentIndex - 1u, segmentIndex > 0u);
  let s3 = min(segmentIndex + 2u, trail.readyCount);

  let p1 = sampleAgent(slot, s1).pos;
  let p2 = p1 + wrapDelta(p1, sampleAgent(slot, s2).pos);
  let p0Raw = samplePos(slot, s0, s1, a.id);
  let p3Raw = samplePos(slot, s3, s2, a.id);
  let p0 = p1 + wrapDelta(p1, p0Raw);
  let p3 = p2 + wrapDelta(sampleAgent(slot, s2).pos, p3Raw);

  var endpoint = array<f32, 6>(0.0, 0.0, 1.0, 0.0, 1.0, 1.0);
  var side = array<f32, 6>(-1.0, 1.0, -1.0, 1.0, 1.0, -1.0);
  let t = (f32(subdivisionIndex) + endpoint[vi]) / f32(${TRAIL_SUBDIVISIONS});
  let center = catmullRom(p0, p1, p2, p3, t);
  let tangent = catmullRom(p0, p1, p2, p3, min(t + 0.05, 1.0)) - catmullRom(p0, p1, p2, p3, max(t - 0.05, 0.0));
  let lenSq = dot(tangent, tangent);
  if (lenSq < 0.000000000001) {
    return deadVertex();
  }

  let age = f32(segmentIndex) + t;
  let fade = max(0.0, 1.0 - age / max(f32(trail.readyCount), 1.0));
  let normal = normalize(vec2f(-tangent.y, tangent.x));
  let width = camera.triSize * f32(${TAIL_WIDTH_FACTOR}) * (0.45 + 0.55 * fade);
  let world = center + normal * side[vi] * width + tiles[tileIndex].offset;
  let alpha = f32(${TAIL_MIN_ALPHA}) + (f32(${TAIL_ALPHA}) - f32(${TAIL_MIN_ALPHA})) * fade;

  var out: VSOut;
  out.pos = toClip(world);
  out.color = vec4f(hsv2rgb(vec3f(a.hue, a.sat, a.val)), alpha);
  return out;
}

@vertex
fn vsBorder(@builtin(vertex_index) vi: u32, @builtin(instance_index) tileIndex: u32) -> VSOut {
  let w = f32(${WORLD_SIZE});
  var corners = array<vec2f, 8>(
    vec2f(0.0, 0.0), vec2f(w, 0.0),
    vec2f(w, 0.0), vec2f(w, w),
    vec2f(w, w), vec2f(0.0, w),
    vec2f(0.0, w), vec2f(0.0, 0.0),
  );
  var out: VSOut;
  out.pos = toClip(corners[vi] + tiles[tileIndex].offset);
  out.color = vec4f(${BORDER_COLOR[0]}, ${BORDER_COLOR[1]}, ${BORDER_COLOR[2]}, ${BORDER_COLOR[3]});
  return out;
}

@fragment
fn fs(in: VSOut) -> @location(0) vec4f {
  return in.color;
}
`;

export class Renderer {
  private readonly device: GPUDevice;
  private readonly agentPipeline: GPURenderPipeline;
  private readonly tailPipeline: GPURenderPipeline;
  private readonly borderPipeline: GPURenderPipeline;
  private readonly agents: GPUBuffer;
  private readonly historyAgents: GPUBuffer;
  private readonly trailBuffer: GPUBuffer;
  private readonly cameraBuffer: GPUBuffer;
  private readonly tileBuffer: GPUBuffer;
  private readonly bindGroup: GPUBindGroup;
  private readonly tileScratch: Uint8Array<ArrayBuffer>;
  private readonly cameraScratch = new Float32Array(8);
  private readonly trailScratch = new Uint32Array(4);

  private tileCount = 0;
  private historyHead = 0;
  private readyCount = 0;

  constructor(
    device: GPUDevice,
    format: GPUTextureFormat,
    agents: GPUBuffer,
  ) {
    this.device = device;
    this.agents = agents;
    this.historyAgents = device.createBuffer({
      size: TRAIL_HISTORY * MAX_AGENTS * AGENT_BYTES,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      label: "agent-trail-history",
    });
    this.trailBuffer = device.createBuffer({
      size: 16,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      label: "agent-trail-meta",
    });

    this.cameraBuffer = device.createBuffer({
      size: 32,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      label: "camera",
    });
    this.tileBuffer = device.createBuffer({
      size: MAX_VISIBLE_TILES * TILE_BYTES,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      label: "tiles",
    });

    const layout = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.VERTEX, buffer: { type: "uniform" } },
        { binding: 1, visibility: GPUShaderStage.VERTEX, buffer: { type: "read-only-storage" } },
        { binding: 2, visibility: GPUShaderStage.VERTEX, buffer: { type: "read-only-storage" } },
        { binding: 3, visibility: GPUShaderStage.VERTEX, buffer: { type: "read-only-storage" } },
        { binding: 4, visibility: GPUShaderStage.VERTEX, buffer: { type: "uniform" } },
      ],
    });
    this.bindGroup = device.createBindGroup({
      layout,
      entries: [
        { binding: 0, resource: { buffer: this.cameraBuffer } },
        { binding: 1, resource: { buffer: this.tileBuffer } },
        { binding: 2, resource: { buffer: agents } },
        { binding: 3, resource: { buffer: this.historyAgents } },
        { binding: 4, resource: { buffer: this.trailBuffer } },
      ],
    });

    const module = device.createShaderModule({ code: SHADER });
    const pipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [layout] });
    const alphaBlend: GPUBlendState = {
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
    };
    this.tailPipeline = device.createRenderPipeline({
      layout: pipelineLayout,
      vertex: { module, entryPoint: "vsTail" },
      fragment: { module, entryPoint: "fs", targets: [{ format, blend: alphaBlend }] },
      primitive: { topology: "triangle-list" },
    });
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
    this.tileScratch = new Uint8Array(new ArrayBuffer(MAX_VISIBLE_TILES * TILE_BYTES));
  }

  snapshotAgents(encoder: GPUCommandEncoder): void {
    const offset = this.historyHead * MAX_AGENTS * AGENT_BYTES;
    encoder.copyBufferToBuffer(this.agents, 0, this.historyAgents, offset, MAX_AGENTS * AGENT_BYTES);

    this.readyCount = Math.min(this.readyCount + 1, TRAIL_HISTORY);
    const meta = this.trailScratch;
    meta[0] = this.historyHead;
    meta[1] = this.readyCount;
    this.device.queue.writeBuffer(this.trailBuffer, 0, meta);
    this.historyHead = (this.historyHead + 1) % TRAIL_HISTORY;
  }

  /** Update the camera transform and the per-tile offsets for this frame. */
  update(
    center: { x: number; y: number },
    zoom: number,
    canvasWidth: number,
    canvasHeight: number,
    offsets: readonly TileOffset[],
  ): void {
    const cam = this.cameraScratch;
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
      view[i * TILE_F32 + 0] = offsets[i][0];
      view[i * TILE_F32 + 1] = offsets[i][1];
    }
    if (this.tileCount > 0) {
      this.device.queue.writeBuffer(
        this.tileBuffer,
        0,
        this.tileScratch,
        0,
        this.tileCount * TILE_BYTES,
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
    pass.setBindGroup(0, this.bindGroup);
    pass.draw(8, this.tileCount);

    pass.setPipeline(this.tailPipeline);
    pass.draw(6, MAX_AGENTS * this.tileCount * TRAIL_HISTORY * TRAIL_SUBDIVISIONS);

    pass.setPipeline(this.agentPipeline);
    pass.draw(3, MAX_AGENTS * this.tileCount);

    pass.end();
  }
}
