import { describe, expect, it } from "vitest";

import {
  AGENT_HIT_RADIUS,
  AGENT_MAX_SPEED,
  AGENT_MAX_TURN,
  AGENT_MIN_SPEED,
  CONTACT_DOT,
  GRID_DIM,
  HEAD_ON_DOT,
  HUE_MUTATION_SCALE,
  MAX_AGENTS,
  POPULATION_FLOOR,
  MUTATION_RATE,
  MUTATION_SCALE,
  MUTATION_WEIGHT_LIMIT,
  SENSOR_RADIUS,
  SPEED_MUTATION_SCALE,
  STEP_DT,
  WORLD_SIZE,
} from "./config";
import {
  AGENT_BYTES,
  AGENT_F32,
  BIRTH_EVENT_BYTES,
  DENSE_BYTES,
  DRAW_INDIRECT_BYTES,
  PLANNED_BYTES,
  PLANNED_F32,
  SIM_PARAMS_BYTES,
} from "./layout";
import { buildSimulationParams } from "./simulationPacking";

const NUM_CELLS = GRID_DIM * GRID_DIM;

describe("GPU layout contracts", () => {
  it("keeps buffer strides aligned with the WGSL structs", () => {
    expect(AGENT_F32).toBe(10);
    expect(AGENT_BYTES).toBe(40);
    expect(PLANNED_F32).toBe(4);
    expect(PLANNED_BYTES).toBe(16);
    expect(BIRTH_EVENT_BYTES).toBe(8);
    expect(DENSE_BYTES).toBe(16);
    expect(DRAW_INDIRECT_BYTES).toBe(16);
  });

  it("packs simulation params at the offsets expected by WGSL", () => {
    const params = buildSimulationParams();
    expect(params.byteLength).toBe(SIM_PARAMS_BYTES);

    const f = new Float32Array(params);
    const u = new Uint32Array(params);
    expect(f[0]).toBeCloseTo(STEP_DT);
    expect(f[1]).toBeCloseTo(WORLD_SIZE);
    expect(f[2]).toBeCloseTo(AGENT_HIT_RADIUS);
    expect(f[3]).toBeCloseTo((AGENT_HIT_RADIUS * 2) ** 2);
    expect(f[4]).toBeCloseTo(CONTACT_DOT);
    expect(f[5]).toBeCloseTo(HEAD_ON_DOT);
    expect(f[6]).toBeCloseTo(AGENT_MAX_TURN);
    expect(f[7]).toBeCloseTo(AGENT_MIN_SPEED);
    expect(f[8]).toBeCloseTo(AGENT_MAX_SPEED);
    expect(f[9]).toBeCloseTo(MUTATION_RATE);
    expect(f[10]).toBeCloseTo(MUTATION_SCALE);
    expect(f[11]).toBeCloseTo(MUTATION_WEIGHT_LIMIT);
    expect(f[12]).toBeCloseTo(SPEED_MUTATION_SCALE);
    expect(f[13]).toBeCloseTo(HUE_MUTATION_SCALE);
    expect(f[14]).toBeCloseTo(SENSOR_RADIUS);
    expect(u[15]).toBe(MAX_AGENTS);
    expect(u[16]).toBe(GRID_DIM);
    expect(u[17]).toBe(NUM_CELLS);
    expect(u[18]).toBe(POPULATION_FLOOR);
  });

  it("keeps the grid compatible with the single-workgroup prefix scan", () => {
    expect(NUM_CELLS % 256).toBe(0);
  });
});
