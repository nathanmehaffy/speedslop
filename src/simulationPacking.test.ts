import { describe, expect, it } from "vitest";

import { AGENT_F32, DENSE_BYTES, LIFE_RECORD_U32, META_BYTES, SIM_PARAMS_BYTES } from "./layout";
import {
  LIFE_ORIGIN_INITIAL,
  META_U32_OFFSET,
  SIM_PARAM_F32,
  SIM_PARAM_U32,
  buildSimulationParams,
  initialAliveCount,
  writeInitialAgents,
  writeInitialDense,
  writeInitialLifeRecords,
  writeInitialMeta,
} from "./simulationPacking";
import { GRID_DIM, MAX_AGENTS, SENSOR_RADIUS, WORLD_SIZE } from "./config";
import { NUM_CELLS } from "./simulationPolicy";

describe("initial simulation packing", () => {
  it("builds initial dense entries from the seeded live slots", () => {
    const agents = new ArrayBuffer(MAX_AGENTS * AGENT_F32 * 4);
    const dense = new ArrayBuffer(MAX_AGENTS * DENSE_BYTES);

    writeInitialAgents(agents);
    writeInitialDense(dense, agents);

    const agentF = new Float32Array(agents);
    const denseF = new Float32Array(dense);
    const denseU = new Uint32Array(dense);
    const lastLive = initialAliveCount() - 1;

    expect(denseF[0]).toBe(agentF[0]);
    expect(denseF[1]).toBe(agentF[1]);
    expect(denseU[2]).toBe(0);
    expect(denseF[lastLive * 4]).toBe(agentF[lastLive * AGENT_F32]);
    expect(denseU[lastLive * 4 + 2]).toBe(lastLive);
  });

  it("builds initial life records from seeded agent IDs", () => {
    const agents = new ArrayBuffer(MAX_AGENTS * AGENT_F32 * 4);
    const records = new ArrayBuffer(MAX_AGENTS * LIFE_RECORD_U32 * 4);

    writeInitialAgents(agents);
    writeInitialLifeRecords(records, agents);

    const agentU = new Uint32Array(agents);
    const recordU = new Uint32Array(records);
    const lastLive = initialAliveCount() - 1;
    expect(recordU[0]).toBe(agentU[8]);
    expect(recordU[5]).toBe(LIFE_ORIGIN_INITIAL);
    expect(recordU[lastLive * LIFE_RECORD_U32]).toBe(agentU[lastLive * AGENT_F32 + 8]);
    expect(recordU[(lastLive + 1) * LIFE_RECORD_U32]).toBe(0);
  });

  it("initializes meta with the seeded live count", () => {
    const meta = new ArrayBuffer(META_BYTES);

    writeInitialMeta(meta);

    const u = new Uint32Array(meta);
    expect(u[META_U32_OFFSET.step]).toBe(0);
    expect(u[META_U32_OFFSET.liveCount]).toBe(initialAliveCount());
    expect(u[META_U32_OFFSET.birthTotal]).toBe(0);
  });

});

describe("simulation params packing", () => {
  it("uses named offsets for layout-sensitive params", () => {
    const params = buildSimulationParams();
    const f = new Float32Array(params);
    const u = new Uint32Array(params);

    expect(params.byteLength).toBe(SIM_PARAMS_BYTES);
    expect(f[SIM_PARAM_F32.worldSize]).toBeCloseTo(WORLD_SIZE);
    expect(f[SIM_PARAM_F32.sensorRadius]).toBeCloseTo(SENSOR_RADIUS);
    expect(u[SIM_PARAM_U32.maxAgents]).toBe(MAX_AGENTS);
    expect(u[SIM_PARAM_U32.gridDim]).toBe(GRID_DIM);
    expect(u[SIM_PARAM_U32.numCells]).toBe(NUM_CELLS);
  });
});
