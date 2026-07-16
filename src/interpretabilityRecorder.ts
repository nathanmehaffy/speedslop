import { BRAIN_WEIGHT_COUNT, MAX_AGENTS } from "./config";
import { AGENT_BYTES, LIFE_RECORD_BYTES, META_BYTES } from "./layout";
import type { AgentSample, GenomeSample, LifeRecord, MetaSample } from "./interpretabilityTypes";
import { parseAgents, parseGenomeSamples, parseLifeRecords, parseMeta } from "./interpretabilitySnapshot";
import type { Simulation } from "./simulation";

const META_SAMPLE_MS = 500;
const AGENT_SAMPLE_MS = 2_000;
const GENOME_SAMPLE_MS = 30_000;
const MAX_META_SAMPLES = 2_400;
const MAX_DEEP_SAMPLES = 300;

interface PendingMeta {
  buffer: GPUBuffer;
  recordedAtMs: number;
  mapping: boolean;
}

interface PendingDeep {
  agentsBuffer: GPUBuffer;
  lifeBuffer: GPUBuffer;
  brainsBuffer: GPUBuffer | null;
  includesGenome: boolean;
  mapping: boolean;
}

export interface AgentHistorySample {
  recordedAtMs: number;
  agents: AgentSample[];
  lifeRecords: LifeRecord[];
}

export class InterpretabilityRecorder {
  readonly metaSamples: MetaSample[] = [];
  readonly agentSamples: AgentHistorySample[] = [];
  readonly genomeSamples: GenomeSample[][] = [];

  deepRecording = false;

  private readonly metaReadback: GPUBuffer;
  private pendingMeta: PendingMeta | null = null;
  private pendingDeep: PendingDeep | null = null;
  private lastMetaAt = -Infinity;
  private lastAgentAt = -Infinity;
  private lastGenomeAt = -Infinity;

  constructor(
    private readonly device: GPUDevice,
    private readonly simulation: Simulation,
  ) {
    this.metaReadback = device.createBuffer({
      size: META_BYTES,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
      label: "interpretability-meta-readback",
    });
  }

  encodeSamples(encoder: GPUCommandEncoder, nowMs: number, simMode: string): void {
    if (nowMs - this.lastMetaAt >= META_SAMPLE_MS && !this.pendingMeta) {
      encoder.copyBufferToBuffer(this.simulation.metaBuffer, 0, this.metaReadback, 0, META_BYTES);
      this.pendingMeta = { buffer: this.metaReadback, recordedAtMs: nowMs, mapping: false };
      this.lastMetaAt = nowMs;
    }

    if (!this.deepRecording || simMode === "max" || this.pendingDeep) {
      return;
    }

    const shouldSampleAgents = nowMs - this.lastAgentAt >= AGENT_SAMPLE_MS;
    const shouldSampleGenome = nowMs - this.lastGenomeAt >= GENOME_SAMPLE_MS;
    if (!shouldSampleAgents && !shouldSampleGenome) {
      return;
    }

    const agentsSize = MAX_AGENTS * AGENT_BYTES;
    const lifeSize = MAX_AGENTS * LIFE_RECORD_BYTES;
    const brainsSize = MAX_AGENTS * BRAIN_WEIGHT_COUNT * 4;
    const agentsBuffer = this.createDeepReadback(agentsSize, "interpretability-agents-readback");
    const lifeBuffer = this.createDeepReadback(lifeSize, "interpretability-life-readback");
    const brainsBuffer = shouldSampleGenome
      ? this.createDeepReadback(brainsSize, "interpretability-brains-readback")
      : null;
    encoder.copyBufferToBuffer(this.simulation.agentsBuffer, 0, agentsBuffer, 0, agentsSize);
    encoder.copyBufferToBuffer(this.simulation.lifeRecordsBuffer, 0, lifeBuffer, 0, lifeSize);
    if (brainsBuffer) {
      encoder.copyBufferToBuffer(this.simulation.brainsBuffer, 0, brainsBuffer, 0, brainsSize);
      this.lastGenomeAt = nowMs;
    }
    this.pendingDeep = { agentsBuffer, lifeBuffer, brainsBuffer, includesGenome: Boolean(brainsBuffer), mapping: false };
    this.lastAgentAt = nowMs;
  }

  poll(): void {
    this.pollMeta();
    this.pollDeep();
  }

  destroy(): void {
    this.metaReadback.destroy();
  }

  private pollMeta(): void {
    const pending = this.pendingMeta;
    if (!pending || pending.mapping) {
      return;
    }
    pending.mapping = true;
    void pending.buffer.mapAsync(GPUMapMode.READ).then(() => {
      const sample = parseMeta(copyMappedRange(pending.buffer), pending.recordedAtMs);
      pushCapped(this.metaSamples, sample, MAX_META_SAMPLES);
      this.pendingMeta = null;
    }).catch(() => {
      this.pendingMeta = null;
    });
  }

  private pollDeep(): void {
    const pending = this.pendingDeep;
    if (!pending || pending.mapping) {
      return;
    }
    pending.mapping = true;
    const maps = [
      pending.agentsBuffer.mapAsync(GPUMapMode.READ),
      pending.lifeBuffer.mapAsync(GPUMapMode.READ),
    ];
    if (pending.brainsBuffer) {
      maps.push(pending.brainsBuffer.mapAsync(GPUMapMode.READ));
    }
    void Promise.all(maps).then(() => {
      const agentsBuffer = copyMappedRange(pending.agentsBuffer);
      const lifeBuffer = copyMappedRange(pending.lifeBuffer);
      const agents = parseAgents(agentsBuffer);
      const lifeRecords = parseLifeRecords(lifeBuffer);
      pushCapped(this.agentSamples, { recordedAtMs: performance.now(), agents, lifeRecords }, MAX_DEEP_SAMPLES);
      if (pending.brainsBuffer) {
        const brainsBuffer = copyMappedRange(pending.brainsBuffer);
        pushCapped(this.genomeSamples, parseGenomeSamples(brainsBuffer, agents), MAX_DEEP_SAMPLES);
      }
      pending.agentsBuffer.destroy();
      pending.lifeBuffer.destroy();
      pending.brainsBuffer?.destroy();
      this.pendingDeep = null;
    }).catch(() => {
      pending.agentsBuffer.destroy();
      pending.lifeBuffer.destroy();
      pending.brainsBuffer?.destroy();
      this.pendingDeep = null;
    });
  }

  private createDeepReadback(size: number, label: string): GPUBuffer {
    return this.device.createBuffer({
      size,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
      label,
    });
  }
}

function copyMappedRange(buffer: GPUBuffer): ArrayBuffer {
  const copy = buffer.getMappedRange().slice(0);
  buffer.unmap();
  return copy;
}

function pushCapped<T>(items: T[], item: T, cap: number): void {
  items.push(item);
  if (items.length > cap) {
    items.splice(0, items.length - cap);
  }
}
