import { AGENT_HIT_RADIUS, CONTACT_DOT, HEAD_ON_DOT, WORLD_SIZE } from "./config";
import { toroidalDelta } from "./spatial";

export type CollisionKind = "none" | "head-on" | "a-hits-b" | "b-hits-a";

export interface CollisionResult {
  kind: CollisionKind;
  distanceSq: number;
  aImpact: number;
  bImpact: number;
}

export function classifyCollision(
  ax: number,
  ay: number,
  aDir: number,
  bx: number,
  by: number,
  bDir: number,
  radius: number = AGENT_HIT_RADIUS,
): CollisionResult {
  const dx = toroidalDelta(ax, bx, WORLD_SIZE);
  const dy = toroidalDelta(ay, by, WORLD_SIZE);
  const distanceSq = dx * dx + dy * dy;
  const collisionDistance = radius * 2;
  if (distanceSq > collisionDistance * collisionDistance) {
    return { kind: "none", distanceSq, aImpact: 0, bImpact: 0 };
  }

  const distance = Math.sqrt(distanceSq);
  if (distance <= 1e-9) {
    return { kind: "none", distanceSq, aImpact: 0, bImpact: 0 };
  }

  const nx = dx / distance;
  const ny = dy / distance;
  const aImpact = Math.cos(aDir) * nx + Math.sin(aDir) * ny;
  const bImpact = -(Math.cos(bDir) * nx + Math.sin(bDir) * ny);

  if (aImpact >= HEAD_ON_DOT && bImpact >= HEAD_ON_DOT) {
    return { kind: "head-on", distanceSq, aImpact, bImpact };
  }
  if (aImpact < CONTACT_DOT && bImpact < CONTACT_DOT) {
    return { kind: "none", distanceSq, aImpact, bImpact };
  }
  return {
    kind: aImpact >= bImpact ? "a-hits-b" : "b-hits-a",
    distanceSq,
    aImpact,
    bImpact,
  };
}
