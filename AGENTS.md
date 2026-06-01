# Implicit Evolutionary Simulation: Agent Instructions

This repo contains a high-performance browser simulation of agents controlled by
small neural networks. The aim is to create a simulation in which complex behavior
can evolve and emerge without hardcoding evolutionary outcomes. Breeding and death
are available pathways, and evolution emerges from those mechanics.

To avoid extinction, the population must be held at constant size, and all
interaction rules must maintain this invariant.

This is a new/greenfield project. Backwards compatibility, fallbacks, and old
runtime paths are not wanted. Keep the codebase small and direct, and remove obsolete
paths instead of preserving unused alternatives.

All code pathways should be extensively tested, especially simulation contracts and
GPU data layout assumptions.

## Onboarding Summary

SpeedSlop is a browser-based evolutionary simulation with a WebGPU simulation core
and a WebGPU live renderer. Agent state, genomes, spatial grid data, interaction
flags, aggregate stats, and rendering data live on one `GPUDevice`; the CPU only
schedules steps, writes small control uniforms, reads tiny aggregate HUD stats, and
records render passes.

## Repository Structure

- `src/main.ts`: Browser entrypoint. Initializes WebGPU, creates `GpuSimulation`,
  handles pause/reset/speed controls and camera input, and runs the animation loop.
- `src/gpu-simulation.ts`: WebGPU simulation engine. Owns buffers, compute pipelines,
  render pipelines, reset, stepping, tiny stats readback, and destruction.
- `src/simulation-helpers.ts`: Shared constants, public stats types, deterministic
  helper math, and fixed-step scheduling.
- `src/simulation-helpers.test.ts`: Node-run TypeScript tests for helper contracts.
- `src/gpu-self-check.ts`: Browser-run GPU self-check scenarios for small scripted
  simulations.
- `src/style.css`: Fullscreen canvas and compact HUD styling.
- `index.html`: Canvas, HUD metrics, controls, and Vite script entry.
- `package.json`: Build, dev, check, test, preview, and browser benchmark scripts.
- `dist/`: Generated production frontend bundle.

## Runtime Architecture

The `GpuSimulation` owns all authoritative state in GPU buffers:

- positions, directions, speed, color, age ticks, and generation
- neural-network genomes and temporary network commands
- breeding state, death flags, per-agent RNG state, and spatial grid buffers
- aggregate stats and the current oldest-agent highlight index

The main thread records ordered WebGPU compute dispatches for fixed simulation steps,
then records render passes that read the same GPU-resident buffers. The normal frame
path must not copy per-agent state to or from the CPU. Tiny aggregate stats readback
for the HUD is acceptable.

## Simulation Core

The world is toroidal: positions wrap at the world bounds, and local distance checks
use wrapped deltas. The spatial grid is rebuilt on GPU as a linked-cell grid: every
agent is inserted exactly once, with no fixed bucket cap and no dropped occupants.

Each fixed simulation step currently does this:

1. Rebuild the spatial grid.
2. Write sensory inputs and evaluate each agent's neural network.
3. Apply turn, acceleration, movement, color, and age updates.
4. Rebuild the grid.
5. Resolve lethal side/body collisions by immediately replacing dead agents with
   randomized agents.
6. Rebuild the grid.
7. Resolve mating. Successful breeding replaces one parent with a mutated child
   genome.
8. Update aggregate stats and the oldest-agent highlight index.

The constant-population invariant is central. Death is implemented as replacement,
and birth is implemented as replacing an existing agent, so population must remain
unchanged across all interactions.

## Neural Agents

Agents use a small one-hidden-layer neural network:

- `INPUT_COUNT`: sensory and self-state values.
- `HIDDEN_COUNT`: hidden layer width.
- `OUTPUT_COUNT`: turn, acceleration, and RGB color targets.
- `GENOME_LEN`: derived from the network shape and tested in TypeScript.

Vision is ray-based, with a forward fan of rays spanning -90 to +90 degrees. Each
ray reports whether another agent was seen, normalized distance, observed RGB color,
and the seen agent's heading relative to the observer. Self inputs include normalized
speed, capped age, and the agent's own RGB color.

## Development Commands

Run from the repository root:

```powershell
npm install
npm run dev
npm run build
npm run check
npm run bench
```

`npm run bench` opens the app at `/?bench=1`, where the HUD reports GPU steps/s and
confirms that the normal runtime does not read per-agent state back to the CPU.
Open `/?selfcheck=1` to run small GPU self-check scenarios before the normal app
starts.

## Testing Expectations

Keep simulation contracts heavily tested. Tests should cover invariants and behavior,
especially:

- population remains constant after ticks, deaths, and breeding
- toroidal wrapping and wrapped distance logic
- deterministic seeded helper logic
- genome/network shape contracts
- mutation bounds
- collision rules
- breeding eligibility and replacement semantics
- render/state buffer layout and value ranges

`npm run check` must pass for ordinary changes. `npm run build` should pass for
frontend or shader changes. Do not use the Codex browser tool for browser testing in
this repo; the developer will run all browser testing manually in his own browser.

## Engineering Preferences

Prefer direct, small, readable code over compatibility layers or defensive leftovers.
When changing architecture, remove obsolete paths instead of preserving unused
alternatives.

Keep high-performance GPU data flow in mind. Preserve the no per-agent CPU transfer
model unless there is a strong reason to change it. Debug-only explicit readbacks are
acceptable, but the normal live simulation/rendering path must remain GPU-resident.
