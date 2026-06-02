import { describe, expect, it } from "vitest";

import { renderTelemetry } from "./telemetry";

describe("renderTelemetry", () => {
  it("shows fps, sim throughput, and demographic rates", () => {
    const text = renderTelemetry({
      elapsedMs: 1000,
      frames: 60,
      steps: 500_000,
      deaths: 1200,
      births: 40,
    });
    expect(text).toContain("60.0 fps");
    expect(text).toContain("500.0k sim steps/s");
    expect(text).toContain("1.2k deaths/s");
    expect(text).toContain("40 births/s");
    expect(text).not.toContain("ms frame");
    expect(text).not.toContain("ms gpu");
  });
});
