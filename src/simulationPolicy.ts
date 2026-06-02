import {
  AGENT_HIT_RADIUS,
  AGENT_MAX_SPEED,
  GRID_DIM,
  POPULATION_FLOOR,
  SENSOR_RADIUS,
  WORLD_SIZE,
} from "./config";

export const WORKGROUP_SIZE = 64;
export const SCAN_WORKGROUP_SIZE = 256;
export const NUM_CELLS = GRID_DIM * GRID_DIM;
export const SCAN_CHUNK = NUM_CELLS / SCAN_WORKGROUP_SIZE;
export const CELL_WIDTH = WORLD_SIZE / GRID_DIM;
export const SENSOR_CELL_RADIUS = sensorCellRadius();

export interface BirthSlotChoice {
  kind: "free" | "parent";
  slot: number;
}

export function sensorCellRadius(sensorRadius = SENSOR_RADIUS, cellWidth = CELL_WIDTH): number {
  return Math.ceil(sensorRadius / cellWidth);
}

export function isWithinSensorRadius(distanceSq: number, sensorRadius = SENSOR_RADIUS): boolean {
  return distanceSq <= sensorRadius * sensorRadius;
}

export function collisionBroadphaseReach(
  hitRadius = AGENT_HIT_RADIUS,
  maxSpeed = AGENT_MAX_SPEED,
): number {
  return hitRadius * 2 + maxSpeed * 2;
}

export function replenishmentCount(
  liveCount: number,
  freeCount: number,
  populationFloor: number = POPULATION_FLOOR,
): number {
  return Math.min(Math.max(0, populationFloor - liveCount), freeCount);
}

export function chooseBirthSlot(
  birthOrdinal: number,
  freeSlots: readonly number[],
  parentA: number,
  parentB: number,
  parentRoll: number,
): BirthSlotChoice {
  if (birthOrdinal < freeSlots.length) {
    return { kind: "free", slot: freeSlots[birthOrdinal] };
  }
  return { kind: "parent", slot: parentRoll < 0.5 ? parentA : parentB };
}

export function assertSimulationConfig(): void {
  if (NUM_CELLS % SCAN_WORKGROUP_SIZE !== 0) {
    throw new Error(
      `GRID_DIM^2 (${NUM_CELLS}) must be a multiple of ${SCAN_WORKGROUP_SIZE} for the prefix scan`,
    );
  }
  if (collisionBroadphaseReach() >= CELL_WIDTH) {
    throw new Error("collision broadphase assumes hit diameter plus relative step motion fits within one cell");
  }
}
