import { describe, expect, it } from "vitest";

import { WORLD_SIZE } from "./config";
import { cellIndex, toroidalDelta, toroidalDistanceSq } from "./spatial";

describe("cellIndex", () => {
  it("maps the origin to cell 0", () => {
    expect(cellIndex(0, 0, 4)).toBe(0);
  });

  it("maps corners to the last row/column without overflow", () => {
    // Exactly at the far edge floors to dim, which must clamp back to dim-1.
    expect(cellIndex(WORLD_SIZE, WORLD_SIZE, 4)).toBe(4 * 4 - 1);
    expect(cellIndex(WORLD_SIZE - 1e-6, WORLD_SIZE - 1e-6, 4)).toBe(15);
  });

  it("packs row-major (y * dim + x)", () => {
    // Second column of the second row in a 4x4 grid.
    const x = WORLD_SIZE * (1.5 / 4);
    const y = WORLD_SIZE * (1.5 / 4);
    expect(cellIndex(x, y, 4)).toBe(1 * 4 + 1);
  });
});

describe("toroidalDelta", () => {
  it("returns the short way around the wrap", () => {
    expect(toroidalDelta(0.1, 0.9, 1)).toBeCloseTo(-0.2, 12);
    expect(toroidalDelta(0.9, 0.1, 1)).toBeCloseTo(0.2, 12);
  });

  it("is zero for identical points", () => {
    expect(toroidalDelta(0.42, 0.42, 1)).toBe(0);
  });

  it("never exceeds half the world size", () => {
    for (let i = 0; i <= 20; i += 1) {
      const d = Math.abs(toroidalDelta(0, i / 20, 1));
      expect(d).toBeLessThanOrEqual(0.5 + 1e-9);
    }
  });
});

describe("toroidalDistanceSq", () => {
  it("uses the wrapped distance on both axes", () => {
    // Opposite corners of the unit torus are only (0.1, 0.1) apart.
    expect(toroidalDistanceSq(0.05, 0.05, 0.95, 0.95, 1)).toBeCloseTo(0.02, 12);
  });
});

