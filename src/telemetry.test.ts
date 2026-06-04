import { describe, expect, it } from "vitest";

import { renderStepTelemetry, renderTelemetry } from "./telemetry";

describe("renderTelemetry", () => {
  it("shows fps and sim throughput", () => {
    const text = renderTelemetry({
      elapsedMs: 1000,
      frames: 60,
      steps: 500_000,
    });
    expect(text).toContain("60.0 fps");
    expect(text).toContain("500.0k sim steps/s");
    expect(text).not.toContain("deaths/s");
    expect(text).not.toContain("births/s");
  });

  it("can show sim throughput without fps", () => {
    const text = renderStepTelemetry({
      elapsedMs: 1000,
      steps: 5_000_000,
    });
    expect(text).toBe("5.00M sim steps/s");
    expect(text).not.toContain("fps");
  });
});
