import assert from "node:assert/strict";
import { describe, it } from "node:test";
import {
  DEFAULT_POPULATION,
  DEFAULT_WORLD_SIZE,
  FIXED_STEP_SECONDS,
  GENOME_LEN,
  INPUT_COUNT,
  MAX_MODE_STEPS_PER_FRAME,
  MAX_POPULATION,
  fastTanh,
  gridColsForWorld,
  normalizeSimRate,
  outputToColor,
  sanitizePopulation,
  sanitizeWorldSize,
  stepsDueForFrame,
  wrapDelta,
  wrapNear,
  wrapUnit,
} from "./simulation-helpers.ts";

describe("simulation helpers", () => {
  it("keeps the neural-network shape contract stable", () => {
    assert.equal(INPUT_COUNT, 59);
    assert.equal(GENOME_LEN, 525);
  });

  it("sanitizes construction inputs", () => {
    assert.equal(sanitizeWorldSize(Number.NaN), DEFAULT_WORLD_SIZE);
    assert.equal(sanitizeWorldSize(127.9), DEFAULT_WORLD_SIZE);
    assert.equal(sanitizeWorldSize(128), 128);
    assert.equal(sanitizePopulation(0), DEFAULT_POPULATION);
    assert.equal(sanitizePopulation(12.9), 12);
    assert.equal(sanitizePopulation(MAX_POPULATION + 1), MAX_POPULATION);
  });

  it("wraps toroidal values and deltas", () => {
    assert.equal(wrapUnit(1.25), 0.25);
    assert.equal(wrapUnit(-0.25), 0.75);
    assert.equal(wrapNear(1030, 1024), 6);
    assert.equal(wrapNear(-2, 1024), 1022);
    assert.equal(wrapDelta(900, 1024), -124);
    assert.equal(wrapDelta(-900, 1024), 124);
  });

  it("maps neural outputs into bounded display values", () => {
    assert.equal(outputToColor(-1), 0);
    assert.equal(outputToColor(0), 0.5);
    assert.equal(outputToColor(1), 1);
    assert.equal(outputToColor(3), 1);
    assert.ok(Math.abs(fastTanh(0.75) - Math.tanh(0.75)) < 0.03);
    assert.equal(fastTanh(-10), -1);
    assert.equal(fastTanh(10), 1);
  });

  it("derives grid and fixed-step scheduling values", () => {
    assert.equal(gridColsForWorld(8192), 128);
    assert.equal(gridColsForWorld(129), 3);
    assert.equal(normalizeSimRate(-1), 1);
    assert.equal(normalizeSimRate("max"), "max");

    const target = stepsDueForFrame(FIXED_STEP_SECONDS * 2.5, 1, 0);
    assert.equal(target.steps, 2);
    assert.ok(target.remainder > 0);

    const max = stepsDueForFrame(0, "max", 1);
    assert.equal(max.steps, MAX_MODE_STEPS_PER_FRAME);
    assert.equal(max.remainder, 0);

    const interactiveMax = stepsDueForFrame(0, "max", 1, 1);
    assert.equal(interactiveMax.steps, 1);
    assert.equal(interactiveMax.remainder, 0);
  });
});
