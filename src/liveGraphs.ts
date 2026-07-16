import type { MetaSample } from "./interpretabilityTypes";

export interface LiveGraphSeries {
  population: number[];
  birthRate: number[];
  deathRate: number[];
}

export class LiveGraphs {
  private readonly populationCanvas: HTMLCanvasElement | null;
  private readonly birthCanvas: HTMLCanvasElement | null;
  private readonly deathCanvas: HTMLCanvasElement | null;
  private readonly populationValue: HTMLElement | null;
  private readonly birthValue: HTMLElement | null;
  private readonly deathValue: HTMLElement | null;

  constructor(private readonly root: HTMLElement | null) {
    this.populationCanvas = root?.querySelector<HTMLCanvasElement>("[data-live-graph='population']") ?? null;
    this.birthCanvas = root?.querySelector<HTMLCanvasElement>("[data-live-graph='birthrate']") ?? null;
    this.deathCanvas = root?.querySelector<HTMLCanvasElement>("[data-live-graph='deathrate']") ?? null;
    this.populationValue = root?.querySelector<HTMLElement>("[data-live-value='population']") ?? null;
    this.birthValue = root?.querySelector<HTMLElement>("[data-live-value='birthrate']") ?? null;
    this.deathValue = root?.querySelector<HTMLElement>("[data-live-value='deathrate']") ?? null;
  }

  render(samples: readonly MetaSample[]): void {
    if (!this.root) {
      return;
    }
    const series = liveGraphSeries(samples);
    const latest = samples.at(-1);
    this.populationValue && (this.populationValue.textContent = latest ? formatCount(latest.liveCount) : "--");
    this.birthValue && (this.birthValue.textContent = formatRate(series.birthRate.at(-1)));
    this.deathValue && (this.deathValue.textContent = formatRate(series.deathRate.at(-1)));
    drawSparkline(this.populationCanvas, series.population, "#9ecbff");
    drawSparkline(this.birthCanvas, series.birthRate, "#8ee6a8");
    drawSparkline(this.deathCanvas, series.deathRate, "#ff8f8f");
  }
}

export function liveGraphSeries(samples: readonly MetaSample[], maxPoints = 80): LiveGraphSeries {
  const tail = samples.slice(Math.max(0, samples.length - maxPoints));
  return {
    population: tail.map((sample) => sample.liveCount),
    birthRate: counterRate(tail, (sample) => sample.birthTotal),
    deathRate: counterRate(tail, (sample) => sample.deathTotal),
  };
}

function counterRate(samples: readonly MetaSample[], readCounter: (sample: MetaSample) => number): number[] {
  return samples.map((sample, index) => {
    if (index === 0) {
      return 0;
    }
    const previous = samples[index - 1];
    const seconds = (sample.recordedAtMs - previous.recordedAtMs) / 1000;
    if (!(seconds > 0)) {
      return 0;
    }
    return Math.max(0, (readCounter(sample) - readCounter(previous)) / seconds);
  });
}

function drawSparkline(canvas: HTMLCanvasElement | null, values: readonly number[], color: string): void {
  if (!canvas) {
    return;
  }
  const ctx = canvas.getContext("2d");
  if (!ctx) {
    return;
  }
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.fillStyle = "rgba(5, 8, 16, 0.7)";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  if (values.length < 2) {
    return;
  }
  const min = Math.min(...values);
  const max = Math.max(...values);
  const span = Math.max(1, max - min);
  ctx.strokeStyle = "rgba(255, 255, 255, 0.12)";
  ctx.beginPath();
  ctx.moveTo(0, canvas.height - 0.5);
  ctx.lineTo(canvas.width, canvas.height - 0.5);
  ctx.stroke();
  ctx.strokeStyle = color;
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  values.forEach((value, index) => {
    const x = (index / (values.length - 1)) * (canvas.width - 1);
    const y = canvas.height - 3 - ((value - min) / span) * (canvas.height - 6);
    if (index === 0) {
      ctx.moveTo(x, y);
    } else {
      ctx.lineTo(x, y);
    }
  });
  ctx.stroke();
}

function formatRate(value: number | undefined): string {
  if (value === undefined) {
    return "--/s";
  }
  return `${formatCount(value)}/s`;
}

function formatCount(value: number): string {
  if (value >= 1_000_000) {
    return `${(value / 1_000_000).toFixed(2)}M`;
  }
  if (value >= 10_000) {
    return `${Math.round(value / 1_000)}k`;
  }
  if (value >= 1_000) {
    return `${(value / 1_000).toFixed(1)}k`;
  }
  return String(Math.round(value));
}
