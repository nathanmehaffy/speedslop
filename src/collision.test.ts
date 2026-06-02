import { describe, expect, it } from "vitest";

import { AGENT_HIT_RADIUS } from "./config";
import { classifyCollision } from "./collision";

describe("classifyCollision", () => {
  it("ignores separated agents", () => {
    expect(classifyCollision(0.1, 0.1, 0, 0.5, 0.5, Math.PI).kind).toBe("none");
  });

  it("classifies reciprocal face-to-face overlaps as head-on", () => {
    const gap = AGENT_HIT_RADIUS * 1.5;
    const result = classifyCollision(0.5, 0.5, 0, 0.5 + gap, 0.5, Math.PI);
    expect(result.kind).toBe("head-on");
  });

  it("kills the agent whose heading points into a side/back collision", () => {
    const gap = AGENT_HIT_RADIUS * 1.5;
    const result = classifyCollision(0.5, 0.5, 0, 0.5 + gap, 0.5, 0);
    expect(result.kind).toBe("a-hits-b");
  });

  it("uses the torus wrap for collision distance", () => {
    const gap = AGENT_HIT_RADIUS * 1.5;
    const result = classifyCollision(1 - gap * 0.5, 0.5, 0, gap * 0.5, 0.5, Math.PI);
    expect(result.kind).toBe("head-on");
  });
});
