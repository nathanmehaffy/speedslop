export interface TelemetrySample {
  elapsedMs: number;
  frames: number;
  steps: number;
  deaths: number;
  births: number;
}

export function renderTelemetry(sample: TelemetrySample): string {
  const fps = (sample.frames * 1000) / sample.elapsedMs;
  const stepsPerSecond = (sample.steps * 1000) / sample.elapsedMs;
  const deathsPerSecond = (sample.deaths * 1000) / sample.elapsedMs;
  const birthsPerSecond = (sample.births * 1000) / sample.elapsedMs;

  return [
    `${fps.toFixed(1)} fps`,
    `${formatCount(stepsPerSecond)} sim steps/s`,
    `${formatCount(deathsPerSecond)} deaths/s`,
    `${formatCount(birthsPerSecond)} births/s`,
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
