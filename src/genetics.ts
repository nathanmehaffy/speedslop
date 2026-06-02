import {
  BRAIN_WEIGHT_COUNT,
  MUTATION_RATE,
  MUTATION_SCALE,
  MUTATION_WEIGHT_LIMIT,
  NEURAL_HIDDEN,
  NEURAL_INPUTS,
  NEURAL_OUTPUTS,
} from "./config";

export function genomeWeightCount(
  inputs: number = NEURAL_INPUTS,
  hidden: number = NEURAL_HIDDEN,
  outputs: number = NEURAL_OUTPUTS,
): number {
  return inputs * hidden + hidden * outputs;
}

export function crossoverAndMutateWeight(
  parentA: number,
  parentB: number,
  crossoverRoll: number,
  mutationRoll: number,
  mutationUnit: number,
  mutationRate: number = MUTATION_RATE,
  mutationScale: number = MUTATION_SCALE,
  weightLimit: number = MUTATION_WEIGHT_LIMIT,
): number {
  const inherited = crossoverRoll < 0.5 ? parentA : parentB;
  const delta = mutationRoll < mutationRate ? (mutationUnit * 2 - 1) * mutationScale : 0;
  return clamp(inherited + delta, -weightLimit, weightLimit);
}

export function assertGenomeContract(): void {
  if (genomeWeightCount() !== BRAIN_WEIGHT_COUNT) {
    throw new Error("BRAIN_WEIGHT_COUNT must match the configured neural network shape");
  }
}

function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value));
}
