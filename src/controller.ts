// Blind throughput controller.
//
// Regulates how many simulation steps to run per displayed frame so the render
// stays locked to the display refresh. Its only input is the wall-clock time
// between requestAnimationFrame callbacks; it performs no GPU readback. See
// ARCHITECTURE.md for the design rationale.
//
// This module is intentionally pure (numbers in, numbers out) so it can be
// tested without a GPU or a browser.

export interface ControllerConfig {
  /** Target steady-state frame-drop probability the regulator aims for. */
  targetDropRate: number;
  /** Initial batched stochastic-approximation gain. */
  trackGain: number;
  /** Floor on the batched gain so the loop keeps tracking a drifting edge. */
  trackGainFloor: number;
  /** Frames per Robbins-Monro tracking batch. */
  trackWindowFrames: number;
  /** A frame counts as dropped when its delta exceeds refresh * this factor. */
  dropFactor: number;
  /** Number of recent frame deltas kept for refresh-interval estimation. */
  refreshWindow: number;
  /** Low percentile of recent deltas used as the refresh estimate (0..1). */
  refreshPercentile: number;
  /** Valid startup deltas to observe before classifying drops. */
  warmupFrames: number;
  /** Bounds on the continuous steps-per-frame control variable. */
  rateMin: number;
  rateMax: number;
  /** Initial steps-per-frame at cold start. */
  rateInitial: number;
  /** Geometric growth per frame during acquisition. */
  acquireFactor: number;
  /** Multiplicative backoff applied when acquisition hits its first drop. */
  acquireBackoff: number;
  /** Bernoulli alternative hypothesis for overload detection. */
  overloadDropRate: number;
  /** Log-likelihood threshold for overload detection. */
  overloadThreshold: number;
  /** Multiplicative crash applied to the rate on a detected overload. */
  crashFactor: number;
  /** Frames spent in recover before re-entering tracking. */
  recoverFrames: number;
  /** Clean-window growth used to regain rate after a transient overload. */
  recoveryGrowthFactor: number;
  /** Long rAF pauses above this are ignored as browser/OS interruptions. */
  maxPauseDeltaMs: number;
  /** Multiplicative rate increase during an active upward probe. */
  probeFactor: number;
  /** Classified frames collected before accepting a clean probe. */
  probeFrames: number;
  /** Frames to wait after a probe or overload before probing again. */
  probeCooldownFrames: number;
  /** Minimum stable tracking frames before probing. */
  probeMinTrackFrames: number;
  /** Probe only when the last tracking window is this far below target. */
  probeDropMargin: number;
  /** Probe only when overload evidence is below this score. */
  probeMaxOverloadScore: number;
  /** Completed zero-drop tracking windows required before an upward probe. */
  probeCleanWindows: number;
}

export const DEFAULT_CONFIG: ControllerConfig = {
  targetDropRate: 0.003,
  trackGain: 0.04,
  trackGainFloor: 0.03,
  trackWindowFrames: 30,
  dropFactor: 1.5,
  refreshWindow: 120,
  refreshPercentile: 0.2,
  warmupFrames: 2,
  rateMin: 0.01,
  rateMax: 1024,
  rateInitial: 1,
  acquireFactor: 1.4,
  acquireBackoff: 0.7,
  overloadDropRate: 0.12,
  overloadThreshold: 12,
  crashFactor: 0.75,
  recoverFrames: 24,
  recoveryGrowthFactor: 1.08,
  maxPauseDeltaMs: 250,
  probeFactor: 1.02,
  probeFrames: 30,
  probeCooldownFrames: 120,
  probeMinTrackFrames: 90,
  probeDropMargin: 0.001,
  probeMaxOverloadScore: 0.5,
  probeCleanWindows: 3,
};

export type Phase = "warmup" | "acquire" | "track" | "probe" | "recover";
export type RegimeEvent = "none" | "overload" | "headroom";
export type ProbeStatus = "idle" | "running" | "accepted" | "rejected";

export interface FrameDecision {
  /** Raw rAF delta observed for this frame, in milliseconds. */
  deltaMs: number;
  /** Integer number of simulation steps to run this frame. */
  steps: number;
  /** Continuous control variable before this update. */
  rateBefore: number;
  /** Continuous control variable (steps per frame) after this update. */
  rate: number;
  /** Current controller phase. */
  phase: Phase;
  /** Current estimate of the display refresh interval, in milliseconds. */
  refreshMs: number;
  /** Current delta threshold used to classify a dropped frame. */
  dropThresholdMs: number;
  /** Whether this frame was still part of startup timing warmup. */
  warmingUp: boolean;
  /** Batched stochastic-approximation gain used on this frame, if any. */
  gain: number;
  /** Last completed tracking-window drop fraction. */
  pHat: number;
  /** Drops in the last completed tracking/probe window. */
  windowDrops: number;
  /** Frames in the last completed tracking/probe window. */
  windowFrames: number;
  /** One-sided overload log-likelihood score. */
  overloadScore: number;
  /** Current probe status for this frame. */
  probeStatus: ProbeStatus;
  /** Probe frames observed in the current or just-finished probe. */
  probeFrames: number;
  /** Probe drops observed in the current or just-finished probe. */
  probeDrops: number;
  /** Whether the just-observed frame was classified as dropped. */
  dropped: boolean;
  /** Regime change detected on this frame, if any. */
  regime: RegimeEvent;
}

interface FrameObservation {
  valid: boolean;
  dropped: boolean;
  dropThresholdMs: number;
}

interface ControllerUpdate {
  regime: RegimeEvent;
  gain: number;
  probeStatus: ProbeStatus;
}

export class ThroughputController {
  private readonly config: ControllerConfig;

  private logRate: number;
  private phase: Phase;
  private accumulator = 0;
  private batchCount = 1;

  private readonly deltas: number[] = [];
  private deltaCursor = 0;
  private refreshMs = 16.6667;
  private warmupFramesSeen = 0;

  private overloadScore = 0;
  private trackWindowFrames = 0;
  private trackWindowDrops = 0;
  private lastWindowFrames = 0;
  private lastWindowDrops = 0;
  private lastPHat = 0;
  private cleanTrackingWindows = 0;

  private trackFramesSinceReset = 0;
  private framesSinceProbe = 0;
  private recoverFramesRemaining = 0;
  private recoveryTargetLogRate: number | null = null;

  private preProbeLogRate = 0;
  private probeFramesSeen = 0;
  private probeDropsSeen = 0;

  constructor(config: Partial<ControllerConfig> = {}) {
    this.config = { ...DEFAULT_CONFIG, ...config };
    this.logRate = Math.log(clamp(this.config.rateInitial, this.config.rateMin, this.config.rateMax));
    this.phase = this.config.warmupFrames > 0 ? "warmup" : "acquire";
    this.framesSinceProbe = this.config.probeCooldownFrames;
  }

  /** Steps to run on the very first frame, before any delta is observed. */
  bootstrap(): number {
    return this.drawSteps();
  }

  /**
   * Record the wall-clock delta (ms) since the previous frame and decide how
   * many simulation steps to run on the upcoming frame.
   */
  recordFrame(deltaMs: number): FrameDecision {
    const rateBefore = this.currentRate;
    const observation = this.observeFrame(deltaMs);
    let update: ControllerUpdate = { regime: "none", gain: 0, probeStatus: "idle" };

    if (observation.valid) {
      update = this.advance(observation.dropped);
    }

    return {
      deltaMs,
      steps: this.drawSteps(),
      rateBefore,
      rate: this.currentRate,
      phase: this.phase,
      refreshMs: this.refreshMs,
      dropThresholdMs: observation.dropThresholdMs,
      warmingUp: this.phase === "warmup",
      gain: update.gain,
      pHat: this.lastPHat,
      windowDrops: this.lastWindowDrops,
      windowFrames: this.lastWindowFrames,
      overloadScore: this.overloadScore,
      probeStatus: update.probeStatus,
      probeFrames: this.probeFramesSeen,
      probeDrops: this.probeDropsSeen,
      dropped: observation.dropped,
      regime: update.regime,
    };
  }

  get currentRate(): number {
    return Math.exp(this.logRate);
  }

  get currentPhase(): Phase {
    return this.phase;
  }

  get refreshEstimateMs(): number {
    return this.refreshMs;
  }

  private advance(dropped: boolean): ControllerUpdate {
    if (this.phase === "warmup") {
      this.warmupFramesSeen += 1;
      if (this.warmupFramesSeen >= this.config.warmupFrames) {
        this.phase = "acquire";
      }
      return { regime: "none", gain: 0, probeStatus: "idle" };
    }

    if (this.phase === "acquire") {
      this.advanceAcquire(dropped);
      return { regime: "none", gain: 0, probeStatus: "idle" };
    }

    if (this.phase === "recover") {
      this.advanceRecover();
      return { regime: "none", gain: 0, probeStatus: "idle" };
    }

    if (this.phase === "probe") {
      return this.advanceProbe(dropped);
    }

    return this.advanceTrack(dropped);
  }

  private observeFrame(deltaMs: number): FrameObservation {
    if (!(deltaMs > 0) || !isFinite(deltaMs)) {
      return { valid: false, dropped: false, dropThresholdMs: this.refreshMs * this.config.dropFactor };
    }

    if (deltaMs > this.config.maxPauseDeltaMs) {
      return { valid: false, dropped: false, dropThresholdMs: this.refreshMs * this.config.dropFactor };
    }

    if (this.deltas.length < this.config.refreshWindow) {
      this.deltas.push(deltaMs);
    } else {
      this.deltas[this.deltaCursor] = deltaMs;
      this.deltaCursor = (this.deltaCursor + 1) % this.config.refreshWindow;
    }
    this.refreshMs = this.estimateRefresh();

    const dropThresholdMs = this.refreshMs * this.config.dropFactor;
    const classifiesDrops = this.phase !== "warmup";
    return { valid: true, dropped: classifiesDrops && deltaMs > dropThresholdMs, dropThresholdMs };
  }

  // The refresh interval is intentionally estimated from a low percentile of
  // recent deltas. Rendering cannot beat the display refresh, so the fastest
  // frames cluster at the true interval while dropped frames are larger
  // outliers.
  private estimateRefresh(): number {
    if (this.deltas.length === 0) {
      return this.refreshMs;
    }
    const sorted = [...this.deltas].sort((a, b) => a - b);
    const index = Math.min(
      sorted.length - 1,
      Math.floor(this.config.refreshPercentile * sorted.length),
    );
    return sorted[index];
  }

  // Acquisition climbs quickly until a reliable drop reveals the local ceiling,
  // then backs off and hands the discovered rate to the statistical tracker.
  private advanceAcquire(dropped: boolean): void {
    if (dropped) {
      this.logRate += Math.log(this.config.acquireBackoff);
      this.clampLogRate();
      this.enterTrack();
      return;
    }

    this.logRate += Math.log(this.config.acquireFactor);
    this.clampLogRate();
    if (this.currentRate >= this.config.rateMax) {
      this.enterTrack();
    }
  }

  private advanceTrack(dropped: boolean): ControllerUpdate {
    this.trackFramesSinceReset += 1;
    this.framesSinceProbe += 1;

    if (this.updateOverloadScore(dropped ? 1 : 0)) {
      this.enterRecover();
      return { regime: "overload", gain: 0, probeStatus: "idle" };
    }

    this.trackWindowFrames += 1;
    this.trackWindowDrops += dropped ? 1 : 0;

    let gain = 0;
    let completedWindow = false;
    if (this.trackWindowFrames >= this.config.trackWindowFrames) {
      gain = this.applyTrackingWindow();
      completedWindow = true;
    }

    if (completedWindow && this.shouldStartProbe()) {
      this.enterProbe();
      return { regime: "none", gain, probeStatus: "running" };
    }

    return { regime: "none", gain, probeStatus: "idle" };
  }

  private advanceProbe(dropped: boolean): ControllerUpdate {
    this.probeFramesSeen += 1;
    this.probeDropsSeen += dropped ? 1 : 0;

    if (this.probeDropsSeen > 0) {
      this.rejectProbe();
      return { regime: "none", gain: 0, probeStatus: "rejected" };
    }

    if (this.probeFramesSeen >= this.config.probeFrames) {
      this.acceptProbe();
      return { regime: "headroom", gain: 0, probeStatus: "accepted" };
    }

    return { regime: "none", gain: 0, probeStatus: "running" };
  }

  private advanceRecover(): void {
    this.recoverFramesRemaining -= 1;
    if (this.recoverFramesRemaining <= 0) {
      this.enterTrack();
    }
  }

  private applyTrackingWindow(): number {
    const frames = this.trackWindowFrames;
    const drops = this.trackWindowDrops;
    const pHat = drops / frames;
    const gain = Math.max(this.config.trackGain / this.batchCount, this.config.trackGainFloor);

    const expectedDrops = frames * this.config.targetDropRate;
    this.logRate -= gain * (drops - expectedDrops);
    this.clampLogRate();

    this.lastWindowFrames = frames;
    this.lastWindowDrops = drops;
    this.lastPHat = pHat;
    this.cleanTrackingWindows = drops === 0 ? this.cleanTrackingWindows + 1 : 0;
    this.updateFastRecovery(drops);
    this.batchCount += 1;

    this.trackWindowFrames = 0;
    this.trackWindowDrops = 0;

    return gain;
  }

  private updateOverloadScore(drop: number): boolean {
    const p0 = clampProbability(this.config.targetDropRate);
    const p1 = clampProbability(Math.max(this.config.overloadDropRate, p0 + 0.001));
    const llr = drop === 1 ? Math.log(p1 / p0) : Math.log((1 - p1) / (1 - p0));

    this.overloadScore = Math.max(0, this.overloadScore + llr);
    return this.overloadScore > this.config.overloadThreshold;
  }

  private shouldStartProbe(): boolean {
    if (this.config.probeFrames <= 0 || this.config.probeFactor <= 1) {
      return false;
    }
    if (this.trackFramesSinceReset < this.config.probeMinTrackFrames) {
      return false;
    }
    if (this.framesSinceProbe < this.config.probeCooldownFrames) {
      return false;
    }
    if (this.overloadScore > this.config.probeMaxOverloadScore) {
      return false;
    }
    if (this.recoveryTargetLogRate !== null) {
      return false;
    }
    if (this.cleanTrackingWindows < this.config.probeCleanWindows) {
      return false;
    }
    return this.lastPHat <= Math.max(0, this.config.targetDropRate - this.config.probeDropMargin);
  }

  private enterTrack(): void {
    this.phase = "track";
    this.batchCount = 1;
    this.trackFramesSinceReset = 0;
    this.cleanTrackingWindows = 0;
    this.resetTrackingWindow();
    this.resetOverloadScore();
  }

  private enterRecover(): void {
    this.recoveryTargetLogRate = Math.max(this.recoveryTargetLogRate ?? -Infinity, this.logRate);
    this.logRate += Math.log(this.config.crashFactor);
    this.clampLogRate();
    this.phase = "recover";
    this.recoverFramesRemaining = this.config.recoverFrames;
    this.framesSinceProbe = 0;
    this.resetTrackingWindow();
    this.resetOverloadScore();
  }

  private updateFastRecovery(drops: number): void {
    if (this.recoveryTargetLogRate === null) {
      return;
    }

    if (drops > 0) {
      this.recoveryTargetLogRate = null;
      return;
    }

    const recoveryTargetLogRate = Math.min(
      this.recoveryTargetLogRate,
      Math.log(this.config.rateMax),
    );
    if (this.logRate >= recoveryTargetLogRate) {
      this.recoveryTargetLogRate = null;
      return;
    }

    this.logRate = Math.min(this.logRate + Math.log(this.config.recoveryGrowthFactor), recoveryTargetLogRate);
    this.clampLogRate();
    if (this.logRate >= recoveryTargetLogRate) {
      this.recoveryTargetLogRate = null;
    }
  }

  private enterProbe(): void {
    this.phase = "probe";
    this.preProbeLogRate = this.logRate;
    this.logRate += Math.log(this.config.probeFactor);
    this.clampLogRate();
    this.probeFramesSeen = 0;
    this.probeDropsSeen = 0;
  }

  private acceptProbe(): void {
    this.phase = "track";
    this.framesSinceProbe = 0;
    this.cleanTrackingWindows = 0;
    this.resetTrackingWindow();
  }

  private rejectProbe(): void {
    this.logRate = this.preProbeLogRate;
    this.clampLogRate();
    this.phase = "track";
    this.framesSinceProbe = 0;
    this.cleanTrackingWindows = 0;
    this.resetTrackingWindow();
  }

  private resetTrackingWindow(): void {
    this.trackWindowFrames = 0;
    this.trackWindowDrops = 0;
  }

  private resetOverloadScore(): void {
    this.overloadScore = 0;
  }

  private clampLogRate(): void {
    this.logRate = Math.log(clamp(Math.exp(this.logRate), this.config.rateMin, this.config.rateMax));
  }

  // Convert the continuous rate into an integer step count, carrying the
  // fractional remainder so rates below 1 spread one step across many frames.
  private drawSteps(): number {
    this.accumulator += this.currentRate;
    const steps = Math.floor(this.accumulator);
    this.accumulator -= steps;
    return steps;
  }
}

function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value));
}

function clampProbability(value: number): number {
  return clamp(value, 0.000001, 0.999999);
}
