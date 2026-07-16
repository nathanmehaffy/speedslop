import { describe, expect, it } from "vitest";

import { AGENT_MAX_SPEED, AGENT_MIN_SPEED, BRAIN_WEIGHT_COUNT, NEURAL_INPUTS } from "./config";
import { evaluateNeuralStep, runProbeScenario, standardProbeScenarios } from "./neuralProbe";
import type { ProbeAgent } from "./interpretabilityTypes";

describe("CPU neural probe", () => {
  it("mirrors zero-weight network output as neutral turn and midpoint speed", () => {
    const focal = probeAgent();

    const trace = evaluateNeuralStep(focal, [], 0);

    expect(trace.inputs).toHaveLength(NEURAL_INPUTS);
    expect(trace.hidden.every((value) => value === 0)).toBe(true);
    expect(trace.nextDir).toBeCloseTo(focal.dir);
    expect(trace.nextVel).toBeCloseTo((AGENT_MIN_SPEED + AGENT_MAX_SPEED) / 2);
  });

  it("runs standard probe scenarios into position traces", () => {
    const scenario = standardProbeScenarios(probeAgent())[0];

    const trace = runProbeScenario(scenario);

    expect(trace.positions).toHaveLength(scenario.steps + 1);
    expect(trace.neural).toHaveLength(scenario.steps);
    expect(trace.collisionKinds).toHaveLength(scenario.steps);
  });
});

function probeAgent(): ProbeAgent {
  return {
    id: 123,
    x: 2,
    y: 2,
    dir: 0,
    vel: AGENT_MIN_SPEED,
    genome: new Float32Array(BRAIN_WEIGHT_COUNT),
  };
}
