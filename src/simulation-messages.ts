export type SimRate = number | "max";

export type SimulationStats = {
  population: number;
  births: number;
  deaths: number;
  generation: number;
  simSteps: number;
  stepsPerSecond: number;
};

export type WorkerInitMessage = {
  type: "init";
  worldSize: number;
  population: number;
  seed: number;
  epoch: number;
  paused: boolean;
  simRate: SimRate;
};

export type WorkerSetPausedMessage = {
  type: "setPaused";
  paused: boolean;
};

export type WorkerSetSimRateMessage = {
  type: "setSimRate";
  simRate: SimRate;
};

export type WorkerResetMessage = {
  type: "reset";
  seed: number;
  epoch: number;
};

export type WorkerReturnSnapshotBufferMessage = {
  type: "returnSnapshotBuffer";
  buffer: ArrayBuffer;
};

export type MainToWorkerMessage =
  | WorkerInitMessage
  | WorkerSetPausedMessage
  | WorkerSetSimRateMessage
  | WorkerResetMessage
  | WorkerReturnSnapshotBufferMessage;

export type WorkerReadyMessage = {
  type: "ready";
  epoch: number;
  agentF32Len: number;
  agentStrideF32: number;
  fixedStepSeconds: number;
  stats: SimulationStats;
};

export type WorkerSnapshotMessage = {
  type: "snapshot";
  epoch: number;
  buffer: ArrayBuffer;
  stats: SimulationStats;
};

export type WorkerStatsMessage = {
  type: "stats";
  epoch: number;
  stats: SimulationStats;
};

export type WorkerErrorMessage = {
  type: "error";
  message: string;
};

export type WorkerToMainMessage =
  | WorkerReadyMessage
  | WorkerSnapshotMessage
  | WorkerStatsMessage
  | WorkerErrorMessage;
