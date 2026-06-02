export interface TelemetrySample {
  elapsedMs: number;
  frames: number;
  steps: number;
}

export function renderTelemetry(sample: TelemetrySample): string {
  const fps = (sample.frames * 1000) / sample.elapsedMs;
  const stepsPerSecond = (sample.steps * 1000) / sample.elapsedMs;

  return [
    `${fps.toFixed(1)} fps`,
    `${formatCount(stepsPerSecond)} sim steps/s`,
  ].join("\n");
}

function formatCount(value: number): string {
  if (value >= 1_000_000) {
    return `${(value / 1_000_000).toFixed(2)}M`;
  }
  if (value >= 1_000) {
    return `${(value / 1_000).toFixed(1)}k`;
  }
  return value.toFixed(0);
}
