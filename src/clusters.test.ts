import { describe, expect, it } from "vitest";

import { clusterGenomes, distance } from "./clusters";
import type { GenomeSample } from "./interpretabilityTypes";

describe("genetic clustering", () => {
  it("forms deterministic clusters from nearby genomes", () => {
    const genomes: GenomeSample[] = [
      genome(0, 1, [0, 0]),
      genome(1, 2, [0.1, 0.1]),
      genome(2, 3, [0.2, 0.2]),
      genome(3, 4, [0.3, 0.3]),
      genome(4, 5, [8, 8]),
      genome(5, 6, [8.1, 8.1]),
      genome(6, 7, [8.2, 8.2]),
      genome(7, 8, [8.3, 8.3]),
    ];

    const clustered = clusterGenomes(genomes);

    expect(clustered.summaries).toHaveLength(2);
    expect(clustered.summaries.map((cluster) => cluster.size)).toEqual([4, 4]);
    expect(clustered.assignments.filter((assignment) => assignment.clusterId > 0)).toHaveLength(8);
  });

  it("normalizes Euclidean distance by vector length", () => {
    expect(distance([0, 0, 0, 0], [1, 1, 1, 1])).toBe(1);
  });
});

function genome(slot: number, id: number, values: number[]): GenomeSample {
  return { slot, id, weights: Float32Array.from(values) };
}
