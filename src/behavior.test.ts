import { describe, expect, it } from "vitest";

import { describeBehaviors } from "./behavior";
import type { AgentSample, LifeRecord } from "./interpretabilityTypes";

describe("behavior descriptors", () => {
  it("measures crowding, approach bias, age, children, and persistence", () => {
    const agents: AgentSample[] = [
      agent(0, 1, 1, 0, 1),
      agent(1, 2, 1, Math.PI, 2),
      agent(2, 3, 3, 0, 3),
    ];
    const previous: AgentSample[] = [
      agent(0, 0.99, 1, 0, 1),
    ];
    const lifeRecords: LifeRecord[] = [
      life(0, 10, 5, 2),
      life(1, 20, 7, 0),
      life(2, 30, 9, 0),
    ];

    const descriptors = describeBehaviors(agents, lifeRecords, 15, previous);

    expect(descriptors[0].age).toBe(10);
    expect(descriptors[0].childCount).toBe(2);
    expect(descriptors[0].localCrowding).toBe(0);
    expect(descriptors[0].approachBias).toBe(0);
    expect(descriptors[0].persistence).toBeCloseTo(1);
  });
});

function agent(slot: number, x: number, y: number, dir: number, id: number): AgentSample {
  return { slot, x, y, dir, vel: 0.001, hue: 0, sat: 1, val: 1, alive: true, id };
}

function life(slot: number, lineageId: number, birthStep: number, childCount: number): LifeRecord {
  return { slot, lineageId, parentAId: 0, parentBId: 0, birthStep, childCount, originKind: 0 };
}
