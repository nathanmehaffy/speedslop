import { bench, describe } from "vitest";

import { MAX_AGENTS } from "../config";
import { AGENT_BYTES, DENSE_BYTES } from "../layout";
import { BRAIN_BYTES, writeInitialAgents, writeInitialBrains, writeInitialDense } from "../simulationPacking";

describe("simulation packing", () => {
  bench("writeInitialAgents 10k slots", () => {
    const buffer = new ArrayBuffer(MAX_AGENTS * AGENT_BYTES);
    writeInitialAgents(buffer, MAX_AGENTS);
  });

  bench("writeInitialBrains 10k slots", () => {
    const buffer = new ArrayBuffer(MAX_AGENTS * BRAIN_BYTES);
    writeInitialBrains(buffer, MAX_AGENTS);
  });

  bench("writeInitialDense floor population", () => {
    const agents = new ArrayBuffer(MAX_AGENTS * AGENT_BYTES);
    const dense = new ArrayBuffer(MAX_AGENTS * DENSE_BYTES);
    writeInitialAgents(agents, MAX_AGENTS);
    writeInitialDense(dense, agents, MAX_AGENTS);
  });
});
