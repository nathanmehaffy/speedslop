import type { Vec2 } from "./camera";

export interface AgentSample {
  slot: number;
  x: number;
  y: number;
  dir: number;
  vel: number;
  hue: number;
  sat: number;
  val: number;
  alive: boolean;
  id: number;
}

export interface LifeRecord {
  slot: number;
  lineageId: number;
  parentAId: number;
  parentBId: number;
  birthStep: number;
  childCount: number;
  originKind: number;
}

export interface MetaSample {
  recordedAtMs: number;
  step: number;
  liveCount: number;
  birthTotal: number;
  deathTotal: number;
  immigrantTotal: number;
  overwriteBirthTotal: number;
  deathAgeTotal: number;
  meanDeathAge: number;
}

export interface GenomeSample {
  slot: number;
  id: number;
  weights: Float32Array;
}

export interface BehaviorDescriptor {
  slot: number;
  id: number;
  speed: number;
  age: number;
  childCount: number;
  localCrowding: number;
  approachBias: number;
  persistence: number;
}

export interface ClusterSummary {
  id: number;
  size: number;
  centroidSlot: number;
  meanDistance: number;
  kind: "genetic" | "behavior";
}

export interface InterpretabilitySnapshot {
  capturedAtMs: number;
  step: number;
  agents: AgentSample[];
  lifeRecords: LifeRecord[];
  genomes: GenomeSample[];
  metaSamples: MetaSample[];
  agentHistory: AgentSample[][];
  camera: {
    center: Vec2;
    zoom: number;
  };
}

export interface ProbeAgent {
  id: number;
  x: number;
  y: number;
  dir: number;
  vel: number;
  genome: Float32Array;
}

export interface ProbeScenario {
  name: string;
  focal: ProbeAgent;
  neighbors: ProbeAgent[];
  steps: number;
}

export interface NeuralTrace {
  inputs: number[];
  hidden: number[];
  turnRaw: number;
  speedRaw: number;
  nextDir: number;
  nextVel: number;
}

export interface ProbeTrace {
  scenarioName: string;
  positions: Vec2[];
  neural: NeuralTrace[];
  collisionKinds: string[];
}
