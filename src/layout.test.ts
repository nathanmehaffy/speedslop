import { describe, expect, it } from "vitest";

import {
  GRID_DIM,
  HEADING_JITTER,
  HUE_DRIFT,
  MAX_AGENTS,
  POPULATION_AMPLITUDE,
  POPULATION_MID,
  POPULATION_OMEGA,
  STEP_DT,
  WORLD_SIZE,
} from "./config";
import { AGENT_BYTES, AGENT_F32, DENSE_BYTES, DRAW_INDIRECT_BYTES, SIM_PARAMS_BYTES } from "./layout";
import { buildSimulationParams } from "./simulation";

const NUM_CELLS = GRID_DIM * GRID_DIM;

describe("GPU layout contracts", () => {
  it("keeps buffer strides aligned with the WGSL structs", () => {
    expect(AGENT_F32).toBe(10);
    expect(AGENT_BYTES).toBe(40);
    expect(DENSE_BYTES).toBe(16);
    expect(DRAW_INDIRECT_BYTES).toBe(16);
  });

  it("packs simulation params at the offsets expected by WGSL", () => {
    const params = buildSimulationParams();
    expect(params.byteLength).toBe(SIM_PARAMS_BYTES);

    const f = new Float32Array(params);
    const u = new Uint32Array(params);
    expect(f[0]).toBeCloseTo(STEP_DT);
    expect(f[1]).toBeCloseTo(HEADING_JITTER);
    expect(f[2]).toBeCloseTo(HUE_DRIFT);
    expect(f[3]).toBeCloseTo(WORLD_SIZE);
    expect(f[4]).toBeCloseTo(POPULATION_MID);
    expect(f[5]).toBeCloseTo(POPULATION_AMPLITUDE);
    expect(f[6]).toBeCloseTo(POPULATION_OMEGA);
    expect(u[9]).toBe(MAX_AGENTS);
    expect(u[10]).toBe(GRID_DIM);
    expect(u[11]).toBe(NUM_CELLS);
  });

  it("keeps the grid compatible with the single-workgroup prefix scan", () => {
    expect(NUM_CELLS % 256).toBe(0);
  });
});
