import { describe, expect, it } from "vitest";

import { AGENT_HIT_RADIUS, AGENT_MAX_SPEED, POPULATION_FLOOR, SENSOR_RADIUS } from "./config";
import {
  CELL_WIDTH,
  SENSOR_CELL_RADIUS,
  chooseBirthSlot,
  collisionBroadphaseReach,
  isWithinSensorRadius,
  replenishmentCount,
  sensorCellRadius,
} from "./simulationPolicy";

describe("sensor policy", () => {
  it("scans enough neighboring cells for the configured sensor radius", () => {
    expect(SENSOR_CELL_RADIUS).toBe(sensorCellRadius(SENSOR_RADIUS, CELL_WIDTH));
    expect(SENSOR_CELL_RADIUS).toBeGreaterThanOrEqual(2);
  });

  it("filters candidates by actual sensor distance", () => {
    expect(isWithinSensorRadius((SENSOR_RADIUS * 0.99) ** 2)).toBe(true);
    expect(isWithinSensorRadius((SENSOR_RADIUS * 1.01) ** 2)).toBe(false);
  });
});

describe("collision broadphase policy", () => {
  it("keeps one-cell collision scans valid for the configured motion bounds", () => {
    expect(collisionBroadphaseReach(AGENT_HIT_RADIUS, AGENT_MAX_SPEED)).toBeLessThan(CELL_WIDTH);
  });
});

describe("demographic policy", () => {
  it("replenishes back to the population floor when free slots exist", () => {
    expect(replenishmentCount(POPULATION_FLOOR - 12, 20)).toBe(12);
  });

  it("never asks for more immigrants than available free slots", () => {
    expect(replenishmentCount(POPULATION_FLOOR - 12, 5)).toBe(5);
  });

  it("does not replenish when already at the population floor", () => {
    expect(replenishmentCount(POPULATION_FLOOR, 100)).toBe(0);
  });

  it("allocates same-step free slots before parent overwrite", () => {
    expect(chooseBirthSlot(0, [42], 1, 2, 0.9)).toEqual({ kind: "free", slot: 42 });
  });

  it("overwrites a parent only when no free slot remains", () => {
    expect(chooseBirthSlot(1, [42], 1, 2, 0.9)).toEqual({ kind: "parent", slot: 2 });
  });
});
