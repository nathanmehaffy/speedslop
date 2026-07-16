import type { BehaviorDescriptor, ClusterSummary, GenomeSample } from "./interpretabilityTypes";
import { behaviorVector } from "./behavior";

const GENETIC_RADIUS = 0.55;
const BEHAVIOR_RADIUS = 1.4;
const MIN_CLUSTER_SIZE = 4;

export interface ClusterAssignment {
  itemId: number;
  slot: number;
  clusterId: number;
}

interface ClusterWorkItem {
  id: number;
  slot: number;
  vector: number[];
}

export function clusterGenomes(genomes: readonly GenomeSample[]): {
  summaries: ClusterSummary[];
  assignments: ClusterAssignment[];
} {
  return clusterItems(
    genomes.map((genome) => ({
      id: genome.id,
      slot: genome.slot,
      vector: Array.from(genome.weights),
    })),
    GENETIC_RADIUS,
    "genetic",
  );
}

export function clusterBehaviors(descriptors: readonly BehaviorDescriptor[]): {
  summaries: ClusterSummary[];
  assignments: ClusterAssignment[];
} {
  return clusterItems(
    descriptors.map((descriptor) => ({
      id: descriptor.id,
      slot: descriptor.slot,
      vector: behaviorVector(descriptor),
    })),
    BEHAVIOR_RADIUS,
    "behavior",
  );
}

export function distance(a: readonly number[], b: readonly number[]): number {
  const n = Math.min(a.length, b.length);
  let sum = 0;
  for (let i = 0; i < n; i += 1) {
    const d = a[i] - b[i];
    sum += d * d;
  }
  return Math.sqrt(sum / Math.max(1, n));
}

function clusterItems(
  items: readonly ClusterWorkItem[],
  radius: number,
  kind: ClusterSummary["kind"],
): { summaries: ClusterSummary[]; assignments: ClusterAssignment[] } {
  const clusters: { centroid: ClusterWorkItem; members: ClusterWorkItem[] }[] = [];
  for (const item of items) {
    let bestIndex = -1;
    let bestDistance = Infinity;
    for (let i = 0; i < clusters.length; i += 1) {
      const d = distance(item.vector, clusters[i].centroid.vector);
      if (d < bestDistance) {
        bestDistance = d;
        bestIndex = i;
      }
    }
    if (bestIndex === -1 || bestDistance > radius) {
      clusters.push({ centroid: item, members: [item] });
    } else {
      clusters[bestIndex].members.push(item);
    }
  }

  const summaries: ClusterSummary[] = [];
  const assignments: ClusterAssignment[] = [];
  clusters.forEach((cluster, index) => {
    const clusterId = index + 1;
    const size = cluster.members.length;
    const meanDistance = cluster.members.reduce(
      (sum, member) => sum + distance(member.vector, cluster.centroid.vector),
      0,
    ) / Math.max(1, size);
    if (size >= MIN_CLUSTER_SIZE) {
      summaries.push({
        id: clusterId,
        size,
        centroidSlot: cluster.centroid.slot,
        meanDistance,
        kind,
      });
    }
    for (const member of cluster.members) {
      assignments.push({ itemId: member.id, slot: member.slot, clusterId: size >= MIN_CLUSTER_SIZE ? clusterId : 0 });
    }
  });
  summaries.sort((a, b) => b.size - a.size);
  return { summaries, assignments };
}
