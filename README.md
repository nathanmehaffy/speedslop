# SpeedSlop

SpeedSlop is a browser-based evolutionary simulation. The simulation and renderer
both run on WebGPU: agent state, neural genomes, the spatial grid, fixed-step
updates, interaction rules, aggregate stats, and instanced rendering stay resident
on one `GPUDevice`.

The default simulation keeps 10,000 neural agents in a constant-size toroidal world.
Agents sense neighbors with nine rays, steer with a tiny neural network, breed by
proximity and alignment, die on side/body collisions after a short grace period, and
are immediately replaced so the population never changes.

## Setup

Install Node.js and dependencies:

```powershell
npm install
```

## Commands

```powershell
npm run dev      # start Vite on 127.0.0.1
npm run build    # type-check and bundle production assets
npm run check    # run helper tests and type-check TypeScript
npm run bench    # open the app in GPU benchmark mode
```

Benchmark mode is also available manually at `/?bench=1`. It runs the GPU simulation
at max speed and reports aggregate GPU steps/s in the HUD without reading agent data
back to the CPU.

GPU self-check mode is available at `/?selfcheck=1`. It runs small debug simulations
with explicit tiny readbacks before starting the normal app.

## Architecture

- `src/main.ts`: Browser entrypoint. Creates WebGPU, handles HUD controls and camera
  input, schedules fixed simulation steps, and records render passes.
- `src/gpu-simulation.ts`: GPU simulation engine. Owns all simulation buffers,
  compute pipelines, render pipelines, reset logic, stepping, tiny stats readback,
  and destruction.
- `src/simulation-helpers.ts`: Shared constants, helper math, fixed-step scheduling,
  and public stats types.
- `src/simulation-helpers.test.ts`: Node-run TypeScript tests for deterministic helper
  contracts.
- `src/gpu-self-check.ts`: Browser-run GPU self-check scenarios for small scripted
  simulations.
- `src/style.css`: Fullscreen canvas and compact HUD styling.

The spatial grid is a full linked-cell grid: every agent is inserted exactly once,
with no fixed bucket cap and no per-frame prefix scan. The normal frame path performs
no per-agent CPU transfer. The CPU writes small control uniforms, submits compute/render
commands, and reads a tiny stats buffer a few times per second for the HUD.
