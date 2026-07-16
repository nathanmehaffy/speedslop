import { describe, expect, it } from "vitest";

import { liveGraphSeries } from "./liveGraphs";
import type { MetaSample } from "./interpretabilityTypes";

describe("liveGraphSeries", () => {
  it("derives population and counter rates from rolling meta samples", () => {
    const series = liveGraphSeries([
      sample(0, 5_000, 10, 4),
      sample(500, 5_050, 15, 6),
      sample(1_500, 4_900, 35, 16),
    ]);

    expect(series.population).toEqual([5_000, 5_050, 4_900]);
    expect(series.birthRate).toEqual([0, 10, 20]);
    expect(series.deathRate).toEqual([0, 4, 10]);
  });

  it("caps history to the requested point count", () => {
    const series = liveGraphSeries([
      sample(0, 1, 0, 0),
      sample(500, 2, 0, 0),
      sample(1000, 3, 0, 0),
    ], 2);

    expect(series.population).toEqual([2, 3]);
  });
});

function sample(recordedAtMs: number, liveCount: number, birthTotal: number, deathTotal: number): MetaSample {
  return {
    recordedAtMs,
    step: 0,
    liveCount,
    birthTotal,
    deathTotal,
    immigrantTotal: 0,
    overwriteBirthTotal: 0,
    deathAgeTotal: 0,
    meanDeathAge: 0,
  };
}
