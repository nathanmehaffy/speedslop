import { MAX_AGENTS, BRAIN_WEIGHT_COUNT } from "./config";
import { AGENT_BYTES, AGENT_F32, LIFE_RECORD_BYTES, LIFE_RECORD_U32, META_BYTES } from "./layout";
import { META_U32_OFFSET } from "./simulationPacking";
import type { AgentSample, GenomeSample, LifeRecord, MetaSample } from "./interpretabilityTypes";
import type { Simulation } from "./simulation";

export const GENOME_SAMPLE_CAP = 512;

export interface ParsedSnapshotBuffers {
  agents: AgentSample[];
  lifeRecords: LifeRecord[];
  genomes: GenomeSample[];
  meta: MetaSample;
}

export function parseAgents(buffer: ArrayBuffer, count: number = MAX_AGENTS): AgentSample[] {
  const f = new Float32Array(buffer);
  const u = new Uint32Array(buffer);
  const agents: AgentSample[] = [];
  for (let slot = 0; slot < count; slot += 1) {
    const b = slot * AGENT_F32;
    agents.push({
      slot,
      x: f[b + 0],
      y: f[b + 1],
      dir: f[b + 2],
      vel: f[b + 3],
      hue: f[b + 4],
      sat: f[b + 5],
      val: f[b + 6],
      alive: u[b + 7] === 1,
      id: u[b + 8],
    });
  }
  return agents;
}

export function parseLifeRecords(buffer: ArrayBuffer, count: number = MAX_AGENTS): LifeRecord[] {
  const u = new Uint32Array(buffer);
  const records: LifeRecord[] = [];
  for (let slot = 0; slot < count; slot += 1) {
    const b = slot * LIFE_RECORD_U32;
    records.push({
      slot,
      lineageId: u[b + 0],
      parentAId: u[b + 1],
      parentBId: u[b + 2],
      birthStep: u[b + 3],
      childCount: u[b + 4],
      originKind: u[b + 5],
    });
  }
  return records;
}

export function parseMeta(buffer: ArrayBuffer, recordedAtMs: number): MetaSample {
  const u = new Uint32Array(buffer);
  const deathTotal = u[META_U32_OFFSET.deathTotal];
  const deathAgeTotal = u[META_U32_OFFSET.deathAgeTotal];
  return {
    recordedAtMs,
    step: u[META_U32_OFFSET.step],
    liveCount: u[META_U32_OFFSET.liveCount],
    birthTotal: u[META_U32_OFFSET.birthTotal],
    deathTotal,
    immigrantTotal: u[META_U32_OFFSET.immigrantTotal],
    overwriteBirthTotal: u[META_U32_OFFSET.overwriteBirthTotal],
    deathAgeTotal,
    meanDeathAge: deathTotal > 0 ? deathAgeTotal / deathTotal : 0,
  };
}

export function parseGenomeSamples(
  buffer: ArrayBuffer,
  agents: readonly AgentSample[],
  cap: number = GENOME_SAMPLE_CAP,
): GenomeSample[] {
  const weights = new Float32Array(buffer);
  const liveAgents = agents.filter((agent) => agent.alive);
  const slots = deterministicLiveSlots(liveAgents, cap);
  return slots.map((agent) => {
    const start = agent.slot * BRAIN_WEIGHT_COUNT;
    const end = start + BRAIN_WEIGHT_COUNT;
    return {
      slot: agent.slot,
      id: agent.id,
      weights: weights.slice(start, end),
    };
  });
}

export function deterministicLiveSlots(liveAgents: readonly AgentSample[], cap: number): AgentSample[] {
  if (liveAgents.length <= cap) {
    return [...liveAgents];
  }
  const sampled: AgentSample[] = [];
  for (let i = 0; i < cap; i += 1) {
    const index = Math.floor((i * liveAgents.length) / cap);
    sampled.push(liveAgents[index]);
  }
  return sampled;
}

export async function readSimulationSnapshot(
  device: GPUDevice,
  simulation: Simulation,
): Promise<{
  agentsBuffer: ArrayBuffer;
  brainsBuffer: ArrayBuffer;
  lifeRecordsBuffer: ArrayBuffer;
  metaBuffer: ArrayBuffer;
}> {
  const agentsSize = MAX_AGENTS * AGENT_BYTES;
  const brainsSize = MAX_AGENTS * BRAIN_WEIGHT_COUNT * 4;
  const lifeSize = MAX_AGENTS * LIFE_RECORD_BYTES;
  const metaSize = META_BYTES;

  const agentsReadback = createReadbackBuffer(device, agentsSize, "snapshot-agents-readback");
  const brainsReadback = createReadbackBuffer(device, brainsSize, "snapshot-brains-readback");
  const lifeReadback = createReadbackBuffer(device, lifeSize, "snapshot-life-readback");
  const metaReadback = createReadbackBuffer(device, metaSize, "snapshot-meta-readback");

  const encoder = device.createCommandEncoder();
  encoder.copyBufferToBuffer(simulation.agentsBuffer, 0, agentsReadback, 0, agentsSize);
  encoder.copyBufferToBuffer(simulation.brainsBuffer, 0, brainsReadback, 0, brainsSize);
  encoder.copyBufferToBuffer(simulation.lifeRecordsBuffer, 0, lifeReadback, 0, lifeSize);
  encoder.copyBufferToBuffer(simulation.metaBuffer, 0, metaReadback, 0, metaSize);
  device.queue.submit([encoder.finish()]);

  await Promise.all([
    agentsReadback.mapAsync(GPUMapMode.READ),
    brainsReadback.mapAsync(GPUMapMode.READ),
    lifeReadback.mapAsync(GPUMapMode.READ),
    metaReadback.mapAsync(GPUMapMode.READ),
  ]);

  const agentsBuffer = copyMappedRange(agentsReadback);
  const brainsBuffer = copyMappedRange(brainsReadback);
  const lifeRecordsBuffer = copyMappedRange(lifeReadback);
  const metaBuffer = copyMappedRange(metaReadback);
  agentsReadback.destroy();
  brainsReadback.destroy();
  lifeReadback.destroy();
  metaReadback.destroy();
  return { agentsBuffer, brainsBuffer, lifeRecordsBuffer, metaBuffer };
}

function createReadbackBuffer(device: GPUDevice, size: number, label: string): GPUBuffer {
  return device.createBuffer({
    size,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    label,
  });
}

function copyMappedRange(buffer: GPUBuffer): ArrayBuffer {
  const mapped = buffer.getMappedRange();
  const copy = mapped.slice(0);
  buffer.unmap();
  return copy;
}
