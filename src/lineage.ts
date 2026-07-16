import type { AgentSample, LifeRecord } from "./interpretabilityTypes";

export interface LineageSummary {
  lineageId: number;
  living: number;
  births: number;
  meanAge: number;
  childCount: number;
  founderSlots: number[];
}

export function summarizeLineages(
  agents: readonly AgentSample[],
  lifeRecords: readonly LifeRecord[],
  currentStep: number,
): LineageSummary[] {
  const summaries = new Map<number, LineageSummary>();
  for (const agent of agents) {
    const record = lifeRecords[agent.slot];
    if (!record || record.lineageId === 0) {
      continue;
    }
    const summary = summaries.get(record.lineageId) ?? {
      lineageId: record.lineageId,
      living: 0,
      births: 0,
      meanAge: 0,
      childCount: 0,
      founderSlots: [],
    };
    summary.births += record.originKind === 1 ? 1 : 0;
    summary.childCount += record.childCount;
    if (record.parentAId === 0 && record.parentBId === 0) {
      summary.founderSlots.push(record.slot);
    }
    if (agent.alive) {
      summary.living += 1;
      summary.meanAge += Math.max(0, currentStep - record.birthStep);
    }
    summaries.set(record.lineageId, summary);
  }
  for (const summary of summaries.values()) {
    if (summary.living > 0) {
      summary.meanAge /= summary.living;
    }
  }
  return [...summaries.values()].sort((a, b) => b.living - a.living || b.childCount - a.childCount);
}

export function ancestryForAgent(
  agent: AgentSample,
  agents: readonly AgentSample[],
  lifeRecords: readonly LifeRecord[],
): LifeRecord[] {
  const byId = new Map<number, LifeRecord>();
  for (const current of agents) {
    if (current.alive) {
      byId.set(current.id, lifeRecords[current.slot]);
    }
  }
  const ancestry: LifeRecord[] = [];
  let current: LifeRecord | undefined = lifeRecords[agent.slot];
  const seen = new Set<string>();
  while (current && current.lineageId !== 0 && !seen.has(recordKey(current))) {
    ancestry.push(current);
    seen.add(recordKey(current));
    current = byId.get(current.parentAId) ?? byId.get(current.parentBId);
  }
  return ancestry;
}

function recordKey(record: LifeRecord): string {
  return `${record.slot}:${record.birthStep}:${record.lineageId}`;
}
