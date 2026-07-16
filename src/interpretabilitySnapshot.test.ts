import { describe, expect, it } from "vitest";

import { AGENT_F32, LIFE_RECORD_U32, META_BYTES } from "./layout";
import { parseAgents, parseLifeRecords, parseMeta } from "./interpretabilitySnapshot";
import { META_U32_OFFSET } from "./simulationPacking";

describe("interpretability snapshot parsing", () => {
  it("parses fixed-slot agents and life records", () => {
    const agents = new ArrayBuffer(AGENT_F32 * 4);
    const agentF = new Float32Array(agents);
    const agentU = new Uint32Array(agents);
    agentF[0] = 1;
    agentF[1] = 2;
    agentF[2] = 3;
    agentF[3] = 4;
    agentF[4] = 0.5;
    agentU[7] = 1;
    agentU[8] = 42;

    const records = new ArrayBuffer(LIFE_RECORD_U32 * 4);
    const recordU = new Uint32Array(records);
    recordU[0] = 42;
    recordU[3] = 7;
    recordU[4] = 2;

    expect(parseAgents(agents, 1)[0]).toMatchObject({ x: 1, y: 2, alive: true, id: 42 });
    expect(parseLifeRecords(records, 1)[0]).toMatchObject({ lineageId: 42, birthStep: 7, childCount: 2 });
  });

  it("derives mean death age from cumulative meta counters", () => {
    const meta = new ArrayBuffer(META_BYTES);
    const u = new Uint32Array(meta);
    u[META_U32_OFFSET.step] = 10;
    u[META_U32_OFFSET.liveCount] = 5;
    u[META_U32_OFFSET.deathTotal] = 4;
    u[META_U32_OFFSET.deathAgeTotal] = 20;

    const parsed = parseMeta(meta, 100);

    expect(parsed.step).toBe(10);
    expect(parsed.liveCount).toBe(5);
    expect(parsed.meanDeathAge).toBe(5);
  });
});
