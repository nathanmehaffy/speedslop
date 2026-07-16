import { SENSOR_RADIUS, WORLD_SIZE } from "./config";
import type { AgentSample, BehaviorDescriptor, LifeRecord } from "./interpretabilityTypes";
import { toroidalDelta, toroidalDistanceSq } from "./spatial";

export function describeBehaviors(
  agents: readonly AgentSample[],
  lifeRecords: readonly LifeRecord[],
  currentStep: number,
  previousAgents: readonly AgentSample[] = [],
): BehaviorDescriptor[] {
  const previousById = new Map(previousAgents.filter((agent) => agent.alive).map((agent) => [agent.id, agent]));
  const live = agents.filter((agent) => agent.alive);
  return live.map((agent) => {
    const neighbors = liveNeighbors(agent, live);
    const previous = previousById.get(agent.id);
    const forwardX = Math.cos(agent.dir);
    const forwardY = Math.sin(agent.dir);
    const approachBias = neighbors.length === 0
      ? 0
      : average(neighbors.map((neighbor) => {
        const dx = toroidalDelta(agent.x, neighbor.x, WORLD_SIZE);
        const dy = toroidalDelta(agent.y, neighbor.y, WORLD_SIZE);
        const distance = Math.max(Math.hypot(dx, dy), 1e-9);
        return (forwardX * dx + forwardY * dy) / distance;
      }));
    const persistence = previous
      ? displacementPersistence(previous, agent)
      : 0;
    const life = lifeRecords[agent.slot];
    return {
      slot: agent.slot,
      id: agent.id,
      speed: agent.vel,
      age: Math.max(0, currentStep - (life?.birthStep ?? currentStep)),
      childCount: life?.childCount ?? 0,
      localCrowding: neighbors.length,
      approachBias,
      persistence,
    };
  });
}

export function behaviorVector(descriptor: BehaviorDescriptor): number[] {
  return [
    descriptor.speed * 1_000,
    descriptor.age / 1_000,
    descriptor.childCount,
    descriptor.localCrowding,
    descriptor.approachBias,
    descriptor.persistence,
  ];
}

function liveNeighbors(agent: AgentSample, live: readonly AgentSample[]): AgentSample[] {
  const radiusSq = SENSOR_RADIUS * SENSOR_RADIUS;
  return live.filter((other) =>
    other.id !== agent.id &&
    toroidalDistanceSq(agent.x, agent.y, other.x, other.y, WORLD_SIZE) <= radiusSq
  );
}

function displacementPersistence(previous: AgentSample, current: AgentSample): number {
  const dx = toroidalDelta(previous.x, current.x, WORLD_SIZE);
  const dy = toroidalDelta(previous.y, current.y, WORLD_SIZE);
  const distance = Math.hypot(dx, dy);
  if (distance <= 1e-9) {
    return 0;
  }
  return (Math.cos(current.dir) * dx + Math.sin(current.dir) * dy) / distance;
}

function average(values: readonly number[]): number {
  return values.reduce((sum, value) => sum + value, 0) / values.length;
}
