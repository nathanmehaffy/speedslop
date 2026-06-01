import { DebugAgentState, DebugCollisionProbe, GpuSimulation } from "./gpu-simulation";
import { MAX_SPEED, MIN_SPEED, SimulationStats } from "./simulation-helpers";

type SelfCheckStep = {
  name: string;
  detail: string;
};

function assertCondition(condition: boolean, message: string): void {
  if (!condition) {
    throw new Error(message);
  }
}

function makeAgent(partial: Partial<DebugAgentState>): DebugAgentState {
  return {
    x: partial.x ?? 40,
    y: partial.y ?? 40,
    dirX: partial.dirX ?? 1,
    dirY: partial.dirY ?? 0,
    r: partial.r ?? 0.7,
    g: partial.g ?? 0.7,
    b: partial.b ?? 0.7,
    speed: partial.speed ?? MIN_SPEED,
    ageTicks: partial.ageTicks ?? 180,
    generation: partial.generation ?? 0,
    matePartnerPlusOne: partial.matePartnerPlusOne ?? 0,
    mateTimerTicks: partial.mateTimerTicks ?? 0,
    rngState: partial.rngState ?? 0x12345678,
  };
}

function encodeSteps(device: GPUDevice, simulation: GpuSimulation, steps: number): void {
  const encoder = device.createCommandEncoder({ label: "gpu-self-check-step-encoder" });
  simulation.encodeSteps(encoder, steps);
  device.queue.submit([encoder.finish()]);
}

async function stepAndReadStats(
  device: GPUDevice,
  simulation: GpuSimulation,
  steps: number,
): Promise<SimulationStats> {
  encodeSteps(device, simulation, steps);
  await device.queue.onSubmittedWorkDone();
  return simulation.readStatsAsync();
}

function assertStatsPopulation(stats: SimulationStats, population: number): void {
  assertCondition(stats.population === population, `Expected population ${population}, got ${stats.population}.`);
}

function sideCollisionGeometry(attacker: DebugAgentState, target: DebugAgentState): Record<string, number | boolean> {
  const headOffset = 5.5;
  const bodyBack = -3.3;
  const bodyFront = 1.1;
  const bodyHalfWidth = 2;
  const headX = attacker.x + attacker.dirX * headOffset;
  const headY = attacker.y + attacker.dirY * headOffset;
  const dx = headX - target.x;
  const dy = headY - target.y;
  const forward = dx * target.dirX + dy * target.dirY;
  const lateral = Math.abs(dx * -target.dirY + dy * target.dirX);
  const dot = attacker.dirX * target.dirX + attacker.dirY * target.dirY;

  return {
    headX,
    headY,
    dx,
    dy,
    forward,
    lateral,
    dot,
    shouldHitSide: forward >= bodyBack && forward <= bodyFront && lateral <= bodyHalfWidth,
    shouldBeHeadOn: dot <= -0.75,
  };
}

function occupiedCells(probe: DebugCollisionProbe): Array<{ cell: number; count: number; entries: number[] }> {
  const cells: Array<{ cell: number; count: number; entries: number[] }> = [];
  const empty = 0xffffffff;

  for (let cell = 0; cell < probe.heads.length; cell += 1) {
    let cursor = probe.heads[cell];
    if (cursor === empty) {
      continue;
    }

    const entries: number[] = [];
    for (let guard = 0; guard < probe.next.length && cursor !== empty; guard += 1) {
      entries.push(cursor);
      cursor = probe.next[cursor] ?? empty;
    }

    cells.push({
      cell,
      count: entries.length,
      entries,
    });
  }

  return cells;
}

function debugJson(value: unknown): string {
  return JSON.stringify(value, (_key, data) => {
    if (typeof data === "number" && Number.isFinite(data)) {
      return Math.round(data * 1000) / 1000;
    }
    return data;
  });
}

async function createSmallSimulation(
  device: GPUDevice,
  format: GPUTextureFormat,
  population: number,
  seed: number,
): Promise<GpuSimulation> {
  const simulation = await GpuSimulation.create(device, {
    worldSize: 128,
    population,
    seed,
    format,
  });
  await device.queue.onSubmittedWorkDone();
  return simulation;
}

async function checkSeededReset(device: GPUDevice, format: GPUTextureFormat): Promise<SelfCheckStep> {
  const a = await createSmallSimulation(device, format, 16, 99);
  const b = await createSmallSimulation(device, format, 16, 99);
  const [agentsA, agentsB] = await Promise.all([
    a.debugReadAgentsAsync(16),
    b.debugReadAgentsAsync(16),
  ]);

  for (let index = 0; index < agentsA.length; index += 1) {
    assertCondition(
      JSON.stringify(agentsA[index]) === JSON.stringify(agentsB[index]),
      `Seeded reset diverged at agent ${index}.`,
    );
  }

  a.destroy();
  b.destroy();
  return { name: "seeded reset", detail: "same seed produced identical initial GPU state" };
}

async function checkSideCollisionDeath(
  device: GPUDevice,
  format: GPUTextureFormat,
): Promise<SelfCheckStep> {
  const simulation = await createSmallSimulation(device, format, 2, 7);
  simulation.debugZeroGenomes();
  simulation.debugOverwriteAgents([
    makeAgent({ x: 44.25, y: 50, dirX: 1, dirY: 0, rngState: 1 }),
    makeAgent({ x: 50, y: 50, dirX: 0, dirY: 1, rngState: 2 }),
  ]);
  await simulation.debugFlushWritesAsync();

  const beforeAgents = await simulation.debugReadAgentsAsync(2);
  const probe = await simulation.debugProbeCollisionsAsync();
  const geometry = sideCollisionGeometry(beforeAgents[0], beforeAgents[1]);
  if (probe.deathFlags[0] === 0) {
    throw new Error(
      `GPU collision probe did not flag attacker. ${debugJson({
        deathFlags: probe.deathFlags.slice(0, 2),
        agents: beforeAgents,
        geometry,
        occupiedCells: occupiedCells(probe),
      })}`,
    );
  }

  const stats = await stepAndReadStats(device, simulation, 2);
  const afterAgents = await simulation.debugReadAgentsAsync(2);
  assertStatsPopulation(stats, 2);
  assertCondition(
    stats.deaths >= 1,
    `Expected a side collision death after probe flagged attacker, got ${stats.deaths}. ${debugJson({
      stats,
      beforeAgents,
      afterAgents,
      geometry,
    })}`,
  );

  simulation.destroy();
  return { name: "side collision", detail: "side/body impact replaced a dead agent" };
}

async function checkHeadOnNonlethal(
  device: GPUDevice,
  format: GPUTextureFormat,
): Promise<SelfCheckStep> {
  const simulation = await createSmallSimulation(device, format, 2, 11);
  simulation.debugZeroGenomes();
  simulation.debugOverwriteAgents([
    makeAgent({ x: 45, y: 50, dirX: 1, dirY: 0, rngState: 3 }),
    makeAgent({ x: 56, y: 50, dirX: -1, dirY: 0, rngState: 4 }),
  ]);

  const stats = await stepAndReadStats(device, simulation, 1);
  assertStatsPopulation(stats, 2);
  assertCondition(stats.deaths === 0, `Expected head-on contact to be nonlethal, got ${stats.deaths}.`);

  simulation.destroy();
  return { name: "head-on collision", detail: "opposing head contact did not kill either agent" };
}

async function checkBreeding(
  device: GPUDevice,
  format: GPUTextureFormat,
): Promise<SelfCheckStep> {
  const simulation = await createSmallSimulation(device, format, 2, 13);
  simulation.debugZeroGenomes();
  simulation.debugOverwriteAgents([
    makeAgent({ x: 40, y: 40, dirX: 1, dirY: 0, rngState: 5 }),
    makeAgent({ x: 45, y: 50, dirX: 1, dirY: 0, rngState: 6 }),
  ]);

  const stats = await stepAndReadStats(device, simulation, 48);
  assertStatsPopulation(stats, 2);
  assertCondition(stats.births >= 1, `Expected aligned mature pair to breed, got ${stats.births}.`);
  assertCondition(stats.generation >= 1, `Expected generation to advance, got ${stats.generation}.`);

  simulation.destroy();
  return { name: "breeding", detail: "aligned mature pair created a replacement child" };
}

async function checkFiniteState(device: GPUDevice, format: GPUTextureFormat): Promise<SelfCheckStep> {
  const simulation = await createSmallSimulation(device, format, 16, 21);
  const stats = await stepAndReadStats(device, simulation, 8);
  const agents = await simulation.debugReadAgentsAsync(16);

  assertStatsPopulation(stats, 16);
  for (const agent of agents) {
    assertCondition(Number.isFinite(agent.x) && agent.x >= 0 && agent.x <= 128, "Agent x is out of range.");
    assertCondition(Number.isFinite(agent.y) && agent.y >= 0 && agent.y <= 128, "Agent y is out of range.");
    assertCondition(Number.isFinite(agent.dirX) && Number.isFinite(agent.dirY), "Agent direction is not finite.");
    assertCondition(agent.speed >= MIN_SPEED && agent.speed <= MAX_SPEED, "Agent speed is out of range.");
    assertCondition(agent.r >= 0 && agent.r <= 1, "Agent red channel is out of range.");
    assertCondition(agent.g >= 0 && agent.g <= 1, "Agent green channel is out of range.");
    assertCondition(agent.b >= 0 && agent.b <= 1, "Agent blue channel is out of range.");
  }

  simulation.destroy();
  return { name: "finite state", detail: "debug-read state stayed finite and in expected ranges" };
}

export async function runGpuSelfCheck(
  device: GPUDevice,
  format: GPUTextureFormat,
): Promise<SelfCheckStep[]> {
  const checks = [
    checkSeededReset,
    checkSideCollisionDeath,
    checkHeadOnNonlethal,
    checkBreeding,
    checkFiniteState,
  ];
  const results: SelfCheckStep[] = [];

  for (const check of checks) {
    results.push(await check(device, format));
  }

  return results;
}
