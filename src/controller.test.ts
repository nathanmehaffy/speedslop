import { describe, expect, it } from "vitest";

import { ThroughputController } from "./controller";

/** Helper: pass GPU-only cost when CPU encode is negligible in a test. */
function recordCost(controller: ThroughputController, gpuMs: number, ranSteps: number): void {
  controller.recordFrameCost(gpuMs, 0, ranSteps);
}

describe("ThroughputController", () => {
  it("spreads fractional rates across frames", () => {
    const controller = new ThroughputController({ rateInitial: 0.25 });

    expect(controller.nextSteps()).toBe(0);
    expect(controller.nextSteps()).toBe(0);
    expect(controller.nextSteps()).toBe(0);
    expect(controller.nextSteps()).toBe(1);
  });

  it("estimates refresh from a low percentile of recent deltas", () => {
    const controller = new ThroughputController({ refreshPercentile: 0.2 });

    controller.recordFrameDelta(16);
    controller.recordFrameDelta(17);
    controller.recordFrameDelta(50);

    expect(controller.refreshEstimateMs).toBe(16);
  });

  it("ignores invalid and long-pause deltas when estimating refresh", () => {
    const controller = new ThroughputController({ refreshPercentile: 0, maxPauseDeltaMs: 250 });

    controller.recordFrameDelta(16);
    controller.recordFrameDelta(Number.NaN);
    controller.recordFrameDelta(-5);
    controller.recordFrameDelta(4_000);

    expect(controller.refreshEstimateMs).toBe(16);
  });

  it("does not let a stall inflate the refresh estimate", () => {
    const controller = new ThroughputController({ refreshPercentile: 0 });
    for (let i = 0; i < 10; i += 1) {
      controller.recordFrameDelta(16);
    }
    expect(controller.refreshEstimateMs).toBe(16);

    // A burst of slow frames (e.g. a context menu or notification) must not
    // raise the estimate, which would inflate the budget and trap the loop.
    for (let i = 0; i < 80; i += 1) {
      controller.recordFrameDelta(80);
    }

    expect(controller.refreshEstimateMs).toBe(16);
  });

  it("does not treat frame-bound frames as refresh samples", () => {
    const controller = new ThroughputController({ refreshPercentile: 0 });
    controller.recordFrameDelta(16);

    // Frame work nearly fills the interval: the delta reflects our own load,
    // not the display refresh, so it must be rejected as a refresh sample.
    recordCost(controller, 20, 100);
    controller.recordFrameDelta(22);

    expect(controller.refreshEstimateMs).toBe(16);
  });

  it("rejects refresh samples when CPU encode dominates the frame", () => {
    const controller = new ThroughputController({ refreshPercentile: 0 });
    controller.recordFrameDelta(16);

    controller.recordFrameCost(2, 18, 100);
    controller.recordFrameDelta(22);

    expect(controller.refreshEstimateMs).toBe(16);
  });

  it("still adopts a faster refresh interval", () => {
    const controller = new ThroughputController({ refreshPercentile: 0 });
    controller.recordFrameDelta(16);

    for (let i = 0; i < 5; i += 1) {
      controller.recordFrameDelta(8);
    }

    expect(controller.refreshEstimateMs).toBe(8);
  });

  it("raises the rate when frame cost is under budget", () => {
    const controller = new ThroughputController({ rateInitial: 10, refreshPercentile: 0 });
    controller.recordFrameDelta(16); // refresh = 16 -> budget = 13.6 ms

    recordCost(controller, 6.8, 10); // half the budget

    expect(controller.currentRate).toBeGreaterThan(10);
    expect(controller.lastFrameCostMs).toBe(6.8);
  });

  it("lowers the rate when frame cost is over budget", () => {
    const controller = new ThroughputController({ rateInitial: 10, refreshPercentile: 0 });
    controller.recordFrameDelta(16);

    recordCost(controller, 27.2, 10); // double the budget

    expect(controller.currentRate).toBeLessThan(10);
  });

  it("lowers the rate when GPU is under budget but CPU pushes total over", () => {
    const controller = new ThroughputController({ rateInitial: 10, refreshPercentile: 0 });
    controller.recordFrameDelta(16); // budget = 13.6 ms

    controller.recordFrameCost(5, 20, 10); // gpu fine, total over budget

    expect(controller.currentRate).toBeLessThan(10);
  });

  it("holds the rate when frame cost matches the budget", () => {
    const controller = new ThroughputController({ rateInitial: 10, refreshPercentile: 0 });
    controller.recordFrameDelta(16);

    recordCost(controller, 16 * 0.85, 10); // exactly the budget

    expect(controller.currentRate).toBeCloseTo(10);
  });

  it("clamps a single sample to the maximum step factor", () => {
    const controller = new ThroughputController({
      rateInitial: 10,
      refreshPercentile: 0,
      smoothing: 1,
      maxStepFactor: 8,
    });
    controller.recordFrameDelta(16);

    recordCost(controller, 0.0001, 10); // wildly under budget

    expect(controller.currentRate).toBeCloseTo(80); // 10 * maxStepFactor, not larger
  });

  it("uses the integer step count that produced a frame-cost sample", () => {
    const controller = new ThroughputController({
      rateInitial: 0.25,
      refreshPercentile: 0,
      smoothing: 1,
      maxStepFactor: 100,
    });
    controller.recordFrameDelta(16);

    recordCost(controller, 16 * 0.85, 1);

    expect(controller.currentRate).toBeCloseTo(1);
  });

  it("converges toward the budget operating point over repeated samples", () => {
    const controller = new ThroughputController({ rateInitial: 1, refreshPercentile: 0 });
    controller.recordFrameDelta(16); // budget = 13.6 ms

    // Linear cost model: 0.5 ms fixed overhead + 0.05 ms per step. The budget
    // operating point is (13.6 - 0.5) / 0.05 = 262 steps/frame.
    for (let i = 0; i < 200; i += 1) {
      const rate = controller.currentRate;
      const gpuMs = 0.5 + 0.05 * rate;
      recordCost(controller, gpuMs, Math.max(1, Math.round(rate)));
    }

    expect(controller.currentRate).toBeGreaterThan(230);
    expect(controller.currentRate).toBeLessThan(290);
  });

  it("converges when CPU encode scales with step count", () => {
    const controller = new ThroughputController({ rateInitial: 1, refreshPercentile: 0 });
    controller.recordFrameDelta(16); // budget = 13.6 ms

    // GPU: 0.3 ms + 0.03 ms/step; CPU encode: 0.2 ms + 0.02 ms/step.
    for (let i = 0; i < 200; i += 1) {
      const rate = controller.currentRate;
      const steps = Math.max(1, Math.round(rate));
      controller.recordFrameCost(0.3 + 0.03 * steps, 0.2 + 0.02 * steps, steps);
    }

    expect(controller.currentRate).toBeGreaterThan(230);
    expect(controller.currentRate).toBeLessThan(290);
  });

  it("ignores non-positive frame-cost measurements", () => {
    const controller = new ThroughputController({ rateInitial: 10, refreshPercentile: 0 });
    controller.recordFrameDelta(16);

    recordCost(controller, 0, 10);
    recordCost(controller, -3, 10);

    expect(controller.currentRate).toBeCloseTo(10);
    expect(controller.lastFrameCostMs).toBeNull();
  });

  it("clamps the rate to the configured bounds", () => {
    const controller = new ThroughputController({
      rateInitial: 10,
      rateMax: 12,
      refreshPercentile: 0,
      smoothing: 1,
      maxStepFactor: 100,
    });
    controller.recordFrameDelta(16);

    recordCost(controller, 0.001, 10); // would explode without the bound

    expect(controller.currentRate).toBeCloseTo(12);
  });

  it("records render-only measurements without changing the rate", () => {
    const controller = new ThroughputController({ rateInitial: 0.25, refreshPercentile: 0 });
    controller.recordFrameDelta(16);

    recordCost(controller, 1, 0);

    expect(controller.currentRate).toBeCloseTo(0.25);
    expect(controller.lastFrameCostMs).toBe(1);
  });
});
