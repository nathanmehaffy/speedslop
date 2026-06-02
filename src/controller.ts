// Throughput controller (direct GPU timing).
//
// Picks how many simulation steps to run per displayed frame so the render
// stays locked to the display refresh. Unlike a blind timing-only loop, this
// controller is driven by the actual GPU time of each frame (measured with
// timestamp queries; see profiler.ts). With a continuous cost signal the law is
// simple: GPU time is monotone in steps, so nudging the rate by the ratio of
// the time budget to the measured time converges to the operating point where
// GPU time equals the budget.
//
// The module is pure (numbers in, numbers out) so it can be unit-tested without
// a GPU or a browser.

export interface ControllerConfig {
  /** Fraction of the display refresh interval budgeted for GPU work. */
  targetUtilization: number;
  /** EWMA factor (0..1) applied to log-rate updates; higher reacts faster. */
  smoothing: number;
  /** Clamp on the per-sample multiplicative rate change. */
  maxStepFactor: number;
  /** Bounds on the continuous steps-per-frame control variable. */
  rateMin: number;
  rateMax: number;
  /** Initial steps-per-frame before any GPU timing arrives. */
  rateInitial: number;
  /** Number of recent frame deltas kept for refresh-interval estimation. */
  refreshWindow: number;
  /** Low percentile of recent deltas used as the refresh estimate (0..1). */
  refreshPercentile: number;
  /** Long rAF pauses above this are ignored as browser/OS interruptions. */
  maxPauseDeltaMs: number;
  /** Implausibly short deltas (ms) are rejected as coalesced-callback glitches. */
  minRefreshMs: number;
  /** Reject deltas longer than this multiple of the current estimate (stalls/drops). */
  maxRefreshRatio: number;
  /** Reject a delta as a refresh sample if GPU time exceeded this fraction of it. */
  gpuBoundFraction: number;
}

export const DEFAULT_CONFIG: ControllerConfig = {
  targetUtilization: 0.85,
  smoothing: 0.4,
  maxStepFactor: 8,
  rateMin: 0.01,
  rateMax: 4096,
  rateInitial: 1,
  refreshWindow: 120,
  refreshPercentile: 0.2,
  maxPauseDeltaMs: 250,
  minRefreshMs: 2,
  maxRefreshRatio: 2.5,
  gpuBoundFraction: 0.9,
};

export class ThroughputController {
  private readonly config: ControllerConfig;

  private logRate: number;
  private accumulator = 0;

  private readonly deltas: number[] = [];
  private refreshMs = 16.6667;
  private lastGpuMs: number | null = null;

  constructor(config: Partial<ControllerConfig> = {}) {
    this.config = { ...DEFAULT_CONFIG, ...config };
    this.logRate = Math.log(clamp(this.config.rateInitial, this.config.rateMin, this.config.rateMax));
  }

  /**
   * Record the wall-clock delta (ms) since the previous frame. Used only to
   * estimate the display refresh interval (the GPU-time budget); it does not
   * directly move the rate.
   *
   * Only frames that look vsync-limited update the estimate. Slow frames - from
   * our own GPU load (GPU-bound), an external stall (right-click menu, lost
   * focus, a notification), or a dropped vsync - are rejected, because letting
   * them raise the refresh estimate would inflate the budget and trap the loop
   * at a degraded frame rate. Faster deltas are always allowed through (down to
   * a glitch floor) so a higher-refresh display is still discovered.
   */
  recordFrameDelta(deltaMs: number): void {
    if (!(deltaMs > 0) || !isFinite(deltaMs) || deltaMs > this.config.maxPauseDeltaMs) {
      return;
    }
    if (deltaMs < this.config.minRefreshMs) {
      return;
    }
    if (deltaMs > this.refreshMs * this.config.maxRefreshRatio) {
      return;
    }
    if (this.lastGpuMs !== null && this.lastGpuMs > deltaMs * this.config.gpuBoundFraction) {
      return;
    }
    this.deltas.push(deltaMs);
    if (this.deltas.length > this.config.refreshWindow) {
      this.deltas.shift();
    }
    this.refreshMs = this.estimateRefresh();
  }

  /**
   * Record the measured GPU time (ms) for a frame. Render-only frames are useful
   * telemetry, but they do not say how expensive a simulation step is.
   */
  recordGpuTime(gpuMs: number, ranSteps: number): void {
    if (!(gpuMs > 0) || !isFinite(gpuMs)) {
      return;
    }
    this.lastGpuMs = gpuMs;
    if (!(ranSteps > 0) || !isFinite(ranSteps)) {
      return;
    }

    const budgetMs = this.refreshMs * this.config.targetUtilization;
    // GPU time is monotone in the integer workload that actually ran, so this
    // ratio moves from that measured step count toward the budget. Clamp it so
    // one noisy sample cannot swing the continuous rate wildly.
    const factor = clamp(budgetMs / gpuMs, 1 / this.config.maxStepFactor, this.config.maxStepFactor);
    const targetLogRate = Math.log(ranSteps * factor);

    this.logRate += this.config.smoothing * (targetLogRate - this.logRate);
    this.clampLogRate();
  }

  get currentRate(): number {
    return Math.exp(this.logRate);
  }

  get refreshEstimateMs(): number {
    return this.refreshMs;
  }

  get lastGpuTimeMs(): number | null {
    return this.lastGpuMs;
  }

  // The refresh interval is estimated from a low percentile of recent deltas:
  // rendering cannot beat the display refresh, so the fastest frames cluster at
  // the true interval while dropped frames are larger outliers.
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

  /** Integer steps to run on the upcoming frame. */
  nextSteps(): number {
    return this.drawSteps();
  }
}

function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value));
}
