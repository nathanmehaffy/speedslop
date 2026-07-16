import { describe, expect, it } from "vitest";

import type { AgentSample, LifeRecord } from "./interpretabilityTypes";
import { ancestryForAgent, summarizeLineages } from "./lineage";

describe("lineage summaries", () => {
  it("aggregates living descendants and reproductive success by lineage", () => {
    const agents: AgentSample[] = [
      agent(0, 100, true),
      agent(1, 101, true),
      agent(2, 200, true),
      agent(3, 201, false),
    ];
    const records: LifeRecord[] = [
      life(0, 10, 0, 0, 0, 1),
      life(1, 10, 100, 200, 5, 2),
      life(2, 20, 0, 0, 3, 0),
      life(3, 20, 200, 101, 8, 0),
    ];

    const summaries = summarizeLineages(agents, records, 15);

    expect(summaries[0].lineageId).toBe(10);
    expect(summaries[0].living).toBe(2);
    expect(summaries[0].childCount).toBe(3);
    expect(summaries[0].meanAge).toBeCloseTo(12.5);
  });

  it("walks ancestry through currently living parents", () => {
    const agents: AgentSample[] = [
      agent(0, 100, true),
      agent(1, 101, true),
    ];
    const records: LifeRecord[] = [
      life(0, 10, 0, 0, 0, 1),
      life(1, 10, 100, 0, 4, 0),
    ];

    const ancestry = ancestryForAgent(agents[1], agents, records);

    expect(ancestry.map((record) => record.slot)).toEqual([1, 0]);
  });
});

function agent(slot: number, id: number, alive: boolean): AgentSample {
  return { slot, id, alive, x: 0, y: 0, dir: 0, vel: 0, hue: 0, sat: 1, val: 1 };
}

function life(
  slot: number,
  lineageId: number,
  parentAId: number,
  parentBId: number,
  birthStep: number,
  childCount: number,
): LifeRecord {
  return { slot, lineageId, parentAId, parentBId, birthStep, childCount, originKind: parentAId === 0 ? 0 : 1 };
}
