import {
  AGENT_HIT_RADIUS,
  AGENT_MAX_SPEED,
  GRID_DIM,
  SENSOR_RADIUS,
  STEP_DT,
  WORLD_SIZE,
} from "./config";

export const WORKGROUP_SIZE = 8;
export const SCAN_WORKGROUP_SIZE = 96;
export const NUM_CELLS = GRID_DIM * GRID_DIM;
export const SCAN_CHUNK = NUM_CELLS / SCAN_WORKGROUP_SIZE;
export const CELL_WIDTH = WORLD_SIZE / GRID_DIM;
export const SENSOR_CELL_RADIUS = sensorCellRadius();

export function sensorCellRadius(sensorRadius = SENSOR_RADIUS, cellWidth = CELL_WIDTH): number {
  // floor(..) + 1 stays safe even when the radius is an exact multiple of the
  // cell width, where ceil(..) would under-scan a far-edge neighbor by one cell.
  return Math.floor(sensorRadius / cellWidth) + 1;
}

export function isWithinSensorRadius(distanceSq: number, sensorRadius = SENSOR_RADIUS): boolean {
  return distanceSq <= sensorRadius * sensorRadius;
}

// Largest gap between two agents' current cells that can still produce a
// collision: hit diameter plus the per-step displacement of both agents
// (speed scaled by the step dt). The one-cell broadphase is valid only while
// this stays below one cell width.
export function collisionBroadphaseReach(
  hitRadius = AGENT_HIT_RADIUS,
  maxSpeed = AGENT_MAX_SPEED,
  stepDt = STEP_DT,
): number {
  return hitRadius * 2 + maxSpeed * 2 * stepDt;
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
