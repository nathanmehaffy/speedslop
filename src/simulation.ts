// Torus agent simulation.
//
// Agents live on a wrapped square world. `Simulation` owns the GPU resources and
// pass ordering; shader source and CPU-side packing live in focused modules so
// layout and policy invariants can be tested without constructing WebGPU state.

import { MAX_AGENTS } from "./config";
import {
  AGENT_BYTES,
  BIRTH_EVENT_BYTES,
  DENSE_BYTES,
  DRAW_INDIRECT_BYTES,
  PLANNED_BYTES,
  SIM_PARAMS_BYTES,
} from "./layout";
import {
  BRAIN_BYTES,
  buildSimulationParams,
  writeInitialAgents,
  writeInitialBrains,
  writeInitialDense,
  writeInitialIndirect,
} from "./simulationPacking";
import { assertSimulationConfig, NUM_CELLS, WORKGROUP_SIZE } from "./simulationPolicy";
import { PIPELINE_NAMES, SHADER, type PipelineName } from "./simulationShader";

assertSimulationConfig();

export { buildSimulationParams } from "./simulationPacking";

export class Simulation {
  private readonly bindGroup: GPUBindGroup;
  private readonly pipelines: Record<PipelineName, GPUComputePipeline>;
  private readonly agents: GPUBuffer;
  private readonly dense: GPUBuffer;
  private readonly indirect: GPUBuffer;

  private readonly clearWorkgroups: number;
  private readonly agentWorkgroups: number;

  constructor(device: GPUDevice) {
    this.clearWorkgroups = Math.ceil(Math.max(NUM_CELLS, MAX_AGENTS) / WORKGROUP_SIZE);
    this.agentWorkgroups = Math.ceil(MAX_AGENTS / WORKGROUP_SIZE);

    const storage = GPUBufferUsage.STORAGE;
    this.agents = device.createBuffer({
      size: MAX_AGENTS * AGENT_BYTES,
      usage: storage,
      mappedAtCreation: true,
      label: "agents",
    });
    const agentsRange = this.agents.getMappedRange();
    writeInitialAgents(agentsRange, MAX_AGENTS);

    const brains = device.createBuffer({
      size: MAX_AGENTS * BRAIN_BYTES,
      usage: storage,
      mappedAtCreation: true,
      label: "brains",
    });
    writeInitialBrains(brains.getMappedRange(), MAX_AGENTS);
    brains.unmap();

    this.dense = device.createBuffer({
      size: MAX_AGENTS * DENSE_BYTES,
      usage: storage,
      mappedAtCreation: true,
      label: "dense",
    });
    writeInitialDense(this.dense.getMappedRange(), agentsRange, MAX_AGENTS);
    this.dense.unmap();
    this.agents.unmap();

    // Zero-initialized: step and counters start at 0.
    const meta = device.createBuffer({
      size: 32,
      usage: storage,
      label: "meta",
    });
    const cellCount = device.createBuffer({ size: NUM_CELLS * 4, usage: storage, label: "cell-count" });
    const cellStart = device.createBuffer({ size: (NUM_CELLS + 1) * 4, usage: storage, label: "cell-start" });
    const freeList = device.createBuffer({ size: MAX_AGENTS * 4, usage: storage, label: "free-list" });
    const planned = device.createBuffer({ size: MAX_AGENTS * PLANNED_BYTES, usage: storage, label: "planned" });
    const killMarks = device.createBuffer({ size: MAX_AGENTS * 4, usage: storage, label: "kill-marks" });
    const mateTargets = device.createBuffer({ size: MAX_AGENTS * 4, usage: storage, label: "mate-targets" });
    const birthEvents = device.createBuffer({
      size: MAX_AGENTS * BIRTH_EVENT_BYTES,
      usage: storage,
      label: "birth-events",
    });
    this.indirect = device.createBuffer({
      size: DRAW_INDIRECT_BYTES,
      usage: GPUBufferUsage.INDIRECT | GPUBufferUsage.STORAGE,
      mappedAtCreation: true,
      label: "draw-indirect",
    });
    writeInitialIndirect(this.indirect.getMappedRange());
    this.indirect.unmap();

    const paramsBuffer = device.createBuffer({
      size: SIM_PARAMS_BYTES,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      label: "sim-params",
    });
    device.queue.writeBuffer(paramsBuffer, 0, buildSimulationParams());

    const layout = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
        ...Array.from({ length: 12 }, (_, i) => i + 1).map((binding) => ({
          binding,
          visibility: GPUShaderStage.COMPUTE,
          buffer: { type: "storage" as const },
        })),
      ],
    });

    this.bindGroup = device.createBindGroup({
      layout,
      entries: [
        { binding: 0, resource: { buffer: paramsBuffer } },
        { binding: 1, resource: { buffer: this.agents } },
        { binding: 2, resource: { buffer: meta } },
        { binding: 3, resource: { buffer: cellCount } },
        { binding: 4, resource: { buffer: cellStart } },
        { binding: 5, resource: { buffer: this.dense } },
        { binding: 6, resource: { buffer: freeList } },
        { binding: 7, resource: { buffer: this.indirect } },
        { binding: 8, resource: { buffer: brains } },
        { binding: 9, resource: { buffer: planned } },
        { binding: 10, resource: { buffer: killMarks } },
        { binding: 11, resource: { buffer: mateTargets } },
        { binding: 12, resource: { buffer: birthEvents } },
      ],
    });

    const module = device.createShaderModule({ code: SHADER });
    const pipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [layout] });
    const make = (entryPoint: PipelineName): GPUComputePipeline =>
      device.createComputePipeline({ layout: pipelineLayout, compute: { module, entryPoint } });
    this.pipelines = Object.fromEntries(
      PIPELINE_NAMES.map((entryPoint) => [entryPoint, make(entryPoint)]),
    ) as Record<PipelineName, GPUComputePipeline>;
  }

  /** Latest live-agent state, addressed by dense index -> slot. */
  get agentsBuffer(): GPUBuffer {
    return this.agents;
  }

  get denseBuffer(): GPUBuffer {
    return this.dense;
  }

  /** Draw-indirect args (vertexCount=3, instanceCount=liveCount). */
  get indirectBuffer(): GPUBuffer {
    return this.indirect;
  }

  /** Encode `steps` dependent simulation steps into a single compute pass. */
  encode(
    encoder: GPUCommandEncoder,
    steps: number,
    timestampWrites?: GPUComputePassTimestampWrites,
  ): void {
    if (steps <= 0) {
      return;
    }

    const pass = encoder.beginComputePass({ label: "simulation", timestampWrites });
    pass.setBindGroup(0, this.bindGroup);
    // WebGPU orders dispatches in one compute pass; each pass sees prior storage writes.
    for (let i = 0; i < steps; i += 1) {
      this.dispatch(pass, "clearStep", this.clearWorkgroups);
      this.rebuildIndex(pass, true);
      this.dispatch(pass, "planMove", this.agentWorkgroups);
      this.dispatch(pass, "chooseContacts", this.agentWorkgroups);
      this.dispatch(pass, "resolveMates", this.agentWorkgroups);
      this.dispatch(pass, "commitAgents", this.agentWorkgroups);

      // Childbirth and immigration only need a fresh free list plus the live
      // count (maxAgents - freeCount); they never read the cell index, so a
      // lightweight free-list refresh replaces a full counting-sort rebuild.
      this.refreshFreeList(pass);
      this.dispatch(pass, "spawnChildren", this.agentWorkgroups);

      this.refreshFreeList(pass);
      this.dispatch(pass, "spawnImmigrants", this.agentWorkgroups);
    }

    // Rebuild the live index once after the batch so render sees final state and
    // the indirect draw count is current.
    this.rebuildIndex(pass, false);
    this.dispatch(pass, "writeIndirect", 1);
    pass.end();
  }

  private rebuildIndex(pass: GPUComputePassEncoder, afterClearStep: boolean): void {
    if (!afterClearStep) {
      this.dispatch(pass, "clearIndex", this.clearWorkgroups);
    }
    this.dispatch(pass, "count", this.agentWorkgroups);
    this.dispatch(pass, "scan", 1);
    this.dispatch(pass, "scatter", this.agentWorkgroups);
  }

  private refreshFreeList(pass: GPUComputePassEncoder): void {
    this.dispatch(pass, "clearFreeList", 1);
    this.dispatch(pass, "gatherDead", this.agentWorkgroups);
  }

  private dispatch(pass: GPUComputePassEncoder, name: PipelineName, workgroups: number): void {
    pass.setPipeline(this.pipelines[name]);
    pass.dispatchWorkgroups(workgroups);
  }
}
