// Torus agent simulation.
//
// Agents live on a wrapped square world. `Simulation` owns the GPU resources and
// pass ordering; shader source and CPU-side packing live in focused modules so
// layout and policy invariants can be tested without constructing WebGPU state.

import { MAX_AGENTS, POPULATION_FLOOR } from "./config";
import {
  AGENT_BYTES,
  BIRTH_EVENT_BYTES,
  DENSE_BYTES,
  PLANNED_BYTES,
  SIM_PARAMS_BYTES,
} from "./layout";
import {
  BRAIN_BYTES,
  buildSimulationParams,
  writeInitialAgents,
  writeInitialBrains,
  writeInitialDense,
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

  private readonly clearWorkgroups: number;
  private readonly agentWorkgroups: number;
  private readonly birthWorkgroups: number;
  private readonly immigrantWorkgroups: number;

  constructor(device: GPUDevice) {
    this.clearWorkgroups = Math.ceil(Math.max(NUM_CELLS, MAX_AGENTS) / WORKGROUP_SIZE);
    this.agentWorkgroups = Math.ceil(MAX_AGENTS / WORKGROUP_SIZE);
    this.birthWorkgroups = Math.ceil((MAX_AGENTS / 2) / WORKGROUP_SIZE);
    this.immigrantWorkgroups = Math.ceil(POPULATION_FLOOR / WORKGROUP_SIZE);

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

    const paramsBuffer = device.createBuffer({
      size: SIM_PARAMS_BYTES,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      label: "sim-params",
    });
    device.queue.writeBuffer(paramsBuffer, 0, buildSimulationParams());

    const layout = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
        ...Array.from({ length: 11 }, (_, i) => i + 1).map((binding) => ({
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
        { binding: 7, resource: { buffer: brains } },
        { binding: 8, resource: { buffer: planned } },
        { binding: 9, resource: { buffer: killMarks } },
        { binding: 10, resource: { buffer: mateTargets } },
        { binding: 11, resource: { buffer: birthEvents } },
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

  /** Latest fixed-slot agent state. */
  get agentsBuffer(): GPUBuffer {
    return this.agents;
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
      this.rebuildIndex(pass);
      this.dispatch(pass, "planMove", this.agentWorkgroups);
      this.dispatch(pass, "chooseContacts", this.agentWorkgroups);
      this.dispatch(pass, "commitContacts", this.agentWorkgroups);
      this.dispatch(pass, "spawnChildren", this.birthWorkgroups);
      this.dispatch(pass, "spawnImmigrants", this.immigrantWorkgroups);
    }
    pass.end();
  }

  encodeProfiledStep(
    encoder: GPUCommandEncoder,
    measuredStage: PipelineName,
    timestampWrites: GPUComputePassTimestampWrites,
  ): void {
    this.dispatchProfiled(encoder, "clearStep", this.clearWorkgroups, measuredStage, timestampWrites);
    this.dispatchProfiled(encoder, "count", this.agentWorkgroups, measuredStage, timestampWrites);
    this.dispatchProfiled(encoder, "scan", 1, measuredStage, timestampWrites);
    this.dispatchProfiled(encoder, "scatter", this.agentWorkgroups, measuredStage, timestampWrites);
    this.dispatchProfiled(encoder, "planMove", this.agentWorkgroups, measuredStage, timestampWrites);
    this.dispatchProfiled(encoder, "chooseContacts", this.agentWorkgroups, measuredStage, timestampWrites);
    this.dispatchProfiled(encoder, "commitContacts", this.agentWorkgroups, measuredStage, timestampWrites);
    this.dispatchProfiled(encoder, "spawnChildren", this.birthWorkgroups, measuredStage, timestampWrites);
    this.dispatchProfiled(encoder, "spawnImmigrants", this.immigrantWorkgroups, measuredStage, timestampWrites);
  }

  private rebuildIndex(pass: GPUComputePassEncoder): void {
    this.dispatch(pass, "count", this.agentWorkgroups);
    this.dispatch(pass, "scan", 1);
    this.dispatch(pass, "scatter", this.agentWorkgroups);
  }

  private dispatch(pass: GPUComputePassEncoder, name: PipelineName, workgroups: number): void {
    pass.setPipeline(this.pipelines[name]);
    pass.dispatchWorkgroups(workgroups);
  }

  private dispatchProfiled(
    encoder: GPUCommandEncoder,
    name: PipelineName,
    workgroups: number,
    measuredStage: PipelineName,
    timestampWrites: GPUComputePassTimestampWrites,
  ): void {
    const pass = encoder.beginComputePass({
      label: name,
      timestampWrites: name === measuredStage ? timestampWrites : undefined,
    });
    pass.setBindGroup(0, this.bindGroup);
    this.dispatch(pass, name, workgroups);
    pass.end();
  }
}
