// Placeholder GPU simulation.
//
// This is NOT the real evolutionary simulation; it is the smallest thing that
// exercises the architecture: a structure-of-arrays position buffer with
// ping-pong state, a GPU-resident step counter bumped on the GPU each step, and
// a batch of dependent compute dispatches encoded with no CPU work in between.
// Each agent starts from a deterministic seed position and drifts in a fixed
// (id-hashed) direction that rotates slowly with the step counter; the motion is
// a per-agent translation, so agent density stays uniform and the cloud never
// collapses.

const WORKGROUP_SIZE = 64;

const SHADER = /* wgsl */ `
struct Params {
  dt: f32,
  agentCount: u32,
}

struct Counter { value: u32 }

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read>       posIn:  array<vec2f>;
@group(0) @binding(2) var<storage, read_write> posOut: array<vec2f>;
@group(1) @binding(0) var<storage, read_write> counter: Counter;

fn pcg(seed: u32) -> u32 {
  var n = seed * 747796405u + 2891336453u;
  n = ((n >> ((n >> 28u) + 4u)) ^ n) * 277803737u;
  return (n >> 22u) ^ n;
}

fn randf(seed: u32) -> f32 {
  return f32(pcg(seed)) * (1.0 / 4294967295.0);
}

@compute @workgroup_size(1)
fn bump() {
  counter.value = counter.value + 1u;
}

@compute @workgroup_size(${WORKGROUP_SIZE})
fn update(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= params.agentCount) {
    return;
  }

  let step = counter.value;
  let heading = randf(i) * 6.2831853 + f32(step) * 0.00001;
  let velocity = vec2f(cos(heading), sin(heading)) * 0.0010;

  let p = fract(posIn[i] + velocity * params.dt + vec2f(1.0, 1.0));
  posOut[i] = p;
}
`;

export class Simulation {
  readonly agentCount: number;
  readonly positionBuffers: [GPUBuffer, GPUBuffer];

  private readonly device: GPUDevice;
  private readonly bumpPipeline: GPUComputePipeline;
  private readonly updatePipeline: GPUComputePipeline;
  private readonly counterBindGroup: GPUBindGroup;
  private readonly simBindGroups: [GPUBindGroup, GPUBindGroup];
  private readonly workgroups: number;

  private current = 0;

  constructor(device: GPUDevice, agentCount: number) {
    this.device = device;
    this.agentCount = agentCount;
    this.workgroups = Math.ceil(agentCount / WORKGROUP_SIZE);

    const sizeBytes = agentCount * 2 * 4;
    this.positionBuffers = [
      createSeeded(device, randomPositions(agentCount), "positions-0"),
      device.createBuffer({ size: sizeBytes, usage: STATE_USAGE, label: "positions-1" }),
    ];

    const counterBuffer = device.createBuffer({
      size: 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      label: "step-counter",
    });
    device.queue.writeBuffer(counterBuffer, 0, new Uint32Array([0]));

    const paramsBuffer = device.createBuffer({
      size: 8,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      label: "sim-params",
    });
    const params = new ArrayBuffer(8);
    new Float32Array(params, 0, 1).set([1.0]); // dt
    new Uint32Array(params, 4, 1).set([agentCount]);
    device.queue.writeBuffer(paramsBuffer, 0, params);

    const simLayout = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      ],
    });
    const counterLayout = device.createBindGroupLayout({
      entries: [{ binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } }],
    });
    const emptyLayout = device.createBindGroupLayout({ entries: [] });

    const module = device.createShaderModule({ code: SHADER });
    this.bumpPipeline = device.createComputePipeline({
      layout: device.createPipelineLayout({ bindGroupLayouts: [emptyLayout, counterLayout] }),
      compute: { module, entryPoint: "bump" },
    });
    this.updatePipeline = device.createComputePipeline({
      layout: device.createPipelineLayout({ bindGroupLayouts: [simLayout, counterLayout] }),
      compute: { module, entryPoint: "update" },
    });

    this.counterBindGroup = device.createBindGroup({
      layout: counterLayout,
      entries: [{ binding: 0, resource: { buffer: counterBuffer } }],
    });

    this.simBindGroups = [
      this.makeSimBindGroup(simLayout, paramsBuffer, 0),
      this.makeSimBindGroup(simLayout, paramsBuffer, 1),
    ];
  }

  /** The position buffer holding the latest state, for the renderer to draw. */
  get currentPositionBuffer(): GPUBuffer {
    return this.positionBuffers[this.current];
  }

  /** Encode `steps` dependent simulation steps into a single compute pass. */
  encode(encoder: GPUCommandEncoder, steps: number): void {
    if (steps <= 0) {
      return;
    }
    let current = this.current;
    const pass = encoder.beginComputePass({ label: "simulation" });
    for (let i = 0; i < steps; i += 1) {
      pass.setPipeline(this.bumpPipeline);
      pass.setBindGroup(1, this.counterBindGroup);
      pass.dispatchWorkgroups(1);

      pass.setPipeline(this.updatePipeline);
      pass.setBindGroup(0, this.simBindGroups[current]);
      pass.setBindGroup(1, this.counterBindGroup);
      pass.dispatchWorkgroups(this.workgroups);

      current ^= 1;
    }
    pass.end();
    this.current = current;
  }

  // read = `from`, write = `1 - from`.
  private makeSimBindGroup(
    layout: GPUBindGroupLayout,
    paramsBuffer: GPUBuffer,
    from: number,
  ): GPUBindGroup {
    const to = from ^ 1;
    return this.device.createBindGroup({
      layout,
      entries: [
        { binding: 0, resource: { buffer: paramsBuffer } },
        { binding: 1, resource: { buffer: this.positionBuffers[from] } },
        { binding: 2, resource: { buffer: this.positionBuffers[to] } },
      ],
    });
  }
}

const STATE_USAGE = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.VERTEX;

function randomPositions(count: number): Float32Array {
  const data = new Float32Array(count * 2);
  for (let i = 0; i < count; i += 1) {
    data[i * 2] = randf(i * 2);
    data[i * 2 + 1] = randf(i * 2 + 1);
  }
  return data;
}

function randf(seed: number): number {
  return pcg(seed) / 0xffffffff;
}

function pcg(seed: number): number {
  let n = Math.imul(seed >>> 0, 747_796_405) + 2_891_336_453;
  n >>>= 0;
  const shift = (n >>> 28) + 4;
  n = Math.imul(((n >>> shift) ^ n) >>> 0, 277_803_737);
  n >>>= 0;
  return ((n >>> 22) ^ n) >>> 0;
}

function createSeeded(device: GPUDevice, data: Float32Array, label: string): GPUBuffer {
  const buffer = device.createBuffer({
    size: data.byteLength,
    usage: STATE_USAGE,
    mappedAtCreation: true,
    label,
  });
  new Float32Array(buffer.getMappedRange()).set(data);
  buffer.unmap();
  return buffer;
}
