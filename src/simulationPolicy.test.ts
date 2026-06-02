import { describe, expect, it } from "vitest";

import { AGENT_HIT_RADIUS, AGENT_MAX_SPEED, SENSOR_RADIUS } from "./config";
import {
  CELL_WIDTH,
  SENSOR_CELL_RADIUS,
  collisionBroadphaseReach,
  isWithinSensorRadius,
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
