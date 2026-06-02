// Pure spatial/demographic math shared as a CPU oracle for the GPU shaders.
//
// These functions mirror small shader-side invariants. Keeping a tested
// TypeScript reference lets cell indexing, toroidal distance, and population
// target math be checked without a GPU.

import { GRID_DIM, WORLD_SIZE } from "./config";

/** Grid cell index for a world position, clamped into `[0, dim)` per axis. */
export function cellIndex(x: number, y: number, dim: number = GRID_DIM): number {
  const cx = clampCell(Math.floor((x / WORLD_SIZE) * dim), dim);
  const cy = clampCell(Math.floor((y / WORLD_SIZE) * dim), dim);
  return cy * dim + cx;
}

/** Shortest signed difference `b - a` on a wrapped axis of length `size`. */
export function toroidalDelta(a: number, b: number, size: number = WORLD_SIZE): number {
  let d = b - a;
  d -= size * Math.round(d / size);
  return d;
}

/** Squared shortest distance between two points on the torus. */
export function toroidalDistanceSq(
  ax: number,
  ay: number,
  bx: number,
  by: number,
  size: number = WORLD_SIZE,
): number {
  const dx = toroidalDelta(ax, bx, size);
  const dy = toroidalDelta(ay, by, size);
  return dx * dx + dy * dy;
}

function clampCell(c: number, dim: number): number {
  if (c < 0) {
    return 0;
  }
  if (c >= dim) {
    return dim - 1;
  }
  return c;
}
