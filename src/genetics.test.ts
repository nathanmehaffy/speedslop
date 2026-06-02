import { describe, expect, it } from "vitest";

import { BRAIN_WEIGHT_COUNT, MUTATION_SCALE, MUTATION_WEIGHT_LIMIT } from "./config";
import { assertGenomeContract, crossoverAndMutateWeight, genomeWeightCount } from "./genetics";

describe("genomeWeightCount", () => {
  it("matches the configured neural network shape", () => {
    expect(genomeWeightCount()).toBe(BRAIN_WEIGHT_COUNT);
    expect(() => assertGenomeContract()).not.toThrow();
  });
});

describe("crossoverAndMutateWeight", () => {
  it("chooses either parent before mutation", () => {
    expect(crossoverAndMutateWeight(-0.25, 0.75, 0.1, 1, 0.5)).toBeCloseTo(-0.25);
    expect(crossoverAndMutateWeight(-0.25, 0.75, 0.9, 1, 0.5)).toBeCloseTo(0.75);
  });

  it("applies bounded mutation", () => {
    expect(crossoverAndMutateWeight(0, 1, 0.1, 0, 1)).toBeCloseTo(MUTATION_SCALE);
    expect(crossoverAndMutateWeight(MUTATION_WEIGHT_LIMIT, 0, 0.1, 0, 1)).toBe(
      MUTATION_WEIGHT_LIMIT,
    );
  });
});
