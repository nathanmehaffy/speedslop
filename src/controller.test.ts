import { describe, expect, it } from "vitest";

import { ThroughputController, type FrameDecision } from "./controller";

describe("ThroughputController", () => {
  it("spreads fractional rates across frames", () => {
    const controller = new ThroughputController({
      rateInitial: 0.25,
      acquireFactor: 1,
      warmupFrames: 0,
      probeFactor: 1,
    });

    expect(controller.bootstrap()).toBe(0);
    expect(controller.recordFrame(16).steps).toBe(0);
    expect(controller.recordFrame(16).steps).toBe(0);
    expect(controller.recordFrame(16).steps).toBe(1);
  });

  it("ignores startup spikes during warmup", () => {
    const controller = new ThroughputController({
      rateInitial: 10,
      acquireFactor: 1,
      acquireBackoff: 0.5,
      warmupFrames: 2,
      probeFactor: 1,
    });

    controller.bootstrap();
    const first = controller.recordFrame(80);
    const second = controller.recordFrame(80);

    expect(first.dropped).toBe(false);
    expect(second.dropped).toBe(false);
    expect(second.phase).toBe("acquire");
    expect(second.rate).toBeCloseTo(10);
  });

  it("ignores invalid deltas without advancing controller state", () => {
    const controller = new ThroughputController({
      rateInitial: 1,
      acquireFactor: 2,
      warmupFrames: 0,
      probeFactor: 1,
    });

    controller.bootstrap();
    const decision = controller.recordFrame(Number.NaN);

    expect(decision.rate).toBe(1);
    expect(decision.phase).toBe("acquire");
    expect(decision.dropped).toBe(false);
  });

  it("ignores long browser pauses without advancing controller state", () => {
    const controller = new ThroughputController({
      rateInitial: 1,
      acquireFactor: 2,
      warmupFrames: 0,
      probeFactor: 1,
      maxPauseDeltaMs: 250,
    });

    controller.bootstrap();
    const decision = controller.recordFrame(4_000);

    expect(decision.rate).toBe(1);
    expect(decision.phase).toBe("acquire");
    expect(decision.dropped).toBe(false);
    expect(controller.refreshEstimateMs).toBeCloseTo(16.6667);
  });

  it("keeps refresh estimation anchored to fast frame deltas", () => {
    const controller = new ThroughputController({
      refreshPercentile: 0.2,
      warmupFrames: 0,
      probeFactor: 1,
    });

    controller.recordFrame(16);
    controller.recordFrame(17);
    controller.recordFrame(50);

    expect(controller.refreshEstimateMs).toBe(16);
  });

  it("backs off and enters tracking when acquisition observes a drop", () => {
    const controller = new ThroughputController({
      rateInitial: 10,
      acquireFactor: 1,
      acquireBackoff: 0.5,
      warmupFrames: 0,
      refreshPercentile: 0,
      probeFactor: 1,
    });

    enterTrack(controller);
    const decision = controller.recordFrame(16);

    expect(decision.phase).toBe("track");
    expect(decision.rate).toBeCloseTo(5);
  });

  it("raises rate after clean tracking batches", () => {
    const controller = new ThroughputController({
      ...trackTestConfig(),
      targetDropRate: 0.1,
    });
    const baseRate = enterTrack(controller);

    const decision = runFrames(controller, 4, 16);

    expect(decision.phase).toBe("track");
    expect(decision.pHat).toBe(0);
    expect(decision.rate).toBeGreaterThan(baseRate);
  });

  it("lowers rate after high-drop tracking batches", () => {
    const controller = new ThroughputController({
      ...trackTestConfig(),
      targetDropRate: 0.1,
      overloadThreshold: 999,
    });
    const baseRate = enterTrack(controller);

    const decision = runFrames(controller, 4, 50);

    expect(decision.phase).toBe("track");
    expect(decision.pHat).toBe(1);
    expect(decision.rate).toBeLessThan(baseRate);
  });

  it("detects overload and enters recovery", () => {
    const controller = new ThroughputController({
      ...trackTestConfig(),
      targetDropRate: 0.01,
      overloadDropRate: 0.5,
      overloadThreshold: 2,
      crashFactor: 0.5,
      recoverFrames: 1,
    });
    const baseRate = enterTrack(controller);

    const overload = controller.recordFrame(50);
    const recovered = controller.recordFrame(16);

    expect(overload.regime).toBe("overload");
    expect(overload.phase).toBe("recover");
    expect(overload.rate).toBeCloseTo(baseRate * 0.5);
    expect(recovered.phase).toBe("track");
  });

  it("suppresses repeated overload crashes during the recovery dwell", () => {
    const controller = new ThroughputController({
      ...trackTestConfig(),
      targetDropRate: 0.01,
      overloadDropRate: 0.5,
      overloadThreshold: 2,
      crashFactor: 0.5,
      recoverFrames: 4,
    });
    const baseRate = enterTrack(controller);

    const overload = controller.recordFrame(50);
    const duringRecovery = runFrames(controller, 3, 50);

    expect(overload.regime).toBe("overload");
    expect(duringRecovery.regime).toBe("none");
    expect(duringRecovery.phase).toBe("recover");
    expect(duringRecovery.rate).toBeCloseTo(baseRate * 0.5);
  });

  it("regains rate quickly after clean recovery windows", () => {
    const controller = new ThroughputController({
      ...trackTestConfig(),
      targetDropRate: 0.01,
      trackWindowFrames: 2,
      trackGain: 0,
      trackGainFloor: 0,
      overloadDropRate: 0.5,
      overloadThreshold: 2,
      crashFactor: 0.5,
      recoverFrames: 1,
      recoveryGrowthFactor: 2,
    });
    const baseRate = enterTrack(controller);

    const overload = controller.recordFrame(50);
    controller.recordFrame(16);
    const recovered = runFrames(controller, 2, 16);

    expect(overload.rate).toBeCloseTo(baseRate * 0.5);
    expect(recovered.phase).toBe("track");
    expect(recovered.rate).toBeCloseTo(baseRate);
  });

  it("accepts a clean active headroom probe", () => {
    const controller = new ThroughputController(probeTestConfig());
    const baseRate = enterTrack(controller);

    const started = runFrames(controller, 2, 16);
    const accepted = runFrames(controller, 3, 16);

    expect(started.phase).toBe("probe");
    expect(started.probeStatus).toBe("running");
    expect(accepted.regime).toBe("headroom");
    expect(accepted.probeStatus).toBe("accepted");
    expect(accepted.phase).toBe("track");
    expect(accepted.rate).toBeGreaterThan(baseRate);
  });

  it("rejects a probe with a dropped frame and restores the previous rate", () => {
    const controller = new ThroughputController(probeTestConfig());
    const baseRate = enterTrack(controller);

    runFrames(controller, 2, 16);
    const rejected = controller.recordFrame(50);

    expect(rejected.probeStatus).toBe("rejected");
    expect(rejected.phase).toBe("track");
    expect(rejected.rate).toBeCloseTo(baseRate);
  });

  it("waits for the configured clean tracking-window streak before probing", () => {
    const controller = new ThroughputController({
      ...probeTestConfig(),
      probeCleanWindows: 2,
    });

    enterTrack(controller);
    const firstCleanWindow = runFrames(controller, 2, 16);
    const secondCleanWindow = runFrames(controller, 2, 16);

    expect(firstCleanWindow.phase).toBe("track");
    expect(firstCleanWindow.probeStatus).toBe("idle");
    expect(secondCleanWindow.phase).toBe("probe");
    expect(secondCleanWindow.probeStatus).toBe("running");
  });

  it("does not create passive headroom resets when probes are disabled", () => {
    const controller = new ThroughputController({
      ...trackTestConfig(),
      targetDropRate: 0.1,
      probeFactor: 1,
    });
    enterTrack(controller);

    for (let i = 0; i < 500; i += 1) {
      const decision = controller.recordFrame(16);
      expect(decision.regime).toBe("none");
      expect(decision.phase).toBe("track");
    }
  });
});

function trackTestConfig(): ConstructorParameters<typeof ThroughputController>[0] {
  return {
    rateInitial: 10,
    acquireFactor: 1,
    acquireBackoff: 1,
    warmupFrames: 0,
    refreshPercentile: 0,
    trackWindowFrames: 4,
    trackGain: 0.5,
    trackGainFloor: 0.5,
    probeFactor: 1,
  };
}

function probeTestConfig(): ConstructorParameters<typeof ThroughputController>[0] {
  return {
    ...trackTestConfig(),
    targetDropRate: 0.1,
    trackWindowFrames: 2,
    trackGain: 0,
    trackGainFloor: 0,
    overloadThreshold: 999,
    probeFactor: 1.1,
    probeFrames: 3,
    probeCooldownFrames: 0,
    probeMinTrackFrames: 2,
    probeDropMargin: 0.05,
    probeMaxOverloadScore: 999,
    probeCleanWindows: 1,
  };
}

function enterTrack(controller: ThroughputController): number {
  controller.bootstrap();
  controller.recordFrame(16);
  const decision = controller.recordFrame(50);
  expect(decision.phase).toBe("track");
  return decision.rate;
}

function runFrames(controller: ThroughputController, count: number, deltaMs: number): FrameDecision {
  let decision = controller.recordFrame(deltaMs);
  for (let i = 1; i < count; i += 1) {
    decision = controller.recordFrame(deltaMs);
  }
  return decision;
}
