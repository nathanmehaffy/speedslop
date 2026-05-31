# SpeedSlop

High-performance browser evolutionary simulation using Rust, WebAssembly, Vite,
TypeScript, and WebGPU.

The simulation state is owned by WASM. The browser app drives the animation
clock, reads a stable packed agent buffer from WASM, uploads it to WebGPU, and
draws arrow-shaped agents with instanced triangles.

## Prerequisites

- Node.js and npm
- Rust and Cargo
- The `wasm32-unknown-unknown` target
- `wasm-pack`

Install the Rust pieces with:

```powershell
rustup target add wasm32-unknown-unknown
cargo install wasm-pack
```

## Setup

```powershell
npm install
npm run dev
```

Then open the local Vite URL printed by the dev server.

## Scripts

- `npm run build:wasm` builds the Rust simulation crate with `wasm-pack`.
- `npm run dev` builds WASM and starts the Vite dev server.
- `npm run build` builds WASM, type-checks TypeScript, and creates a production
  frontend bundle.
- `npm run check` builds WASM, type-checks TypeScript, and runs `cargo check`
  for the WASM target.
- `npm run preview` serves the production bundle after `npm run build`.

## Simulation API

The Rust crate exports a `Simulation` class through `wasm-bindgen`:

- `new(world_size, population, seed)` creates a deterministic toroidal world.
- `tick(dt_seconds)` advances the fixed-step simulation.
- `reset(seed)` reseeds the population and clears counters.
- `world_size()` and `population()` expose world metadata.
- `births()`, `deaths()`, `sim_steps()`, and `generation()` expose HUD stats.
- `agent_ptr()`, `agent_f32_len()`, and `agent_stride_f32()` expose the stable
  WASM-owned render buffer.

The TypeScript app imports `Simulation` from the generated `sim/pkg` package.
Each agent occupies eight `f32` values:

```text
[x_norm, y_norm, dir_x, dir_y, r, g, b, speed_norm]
```

The v1 simulation uses 10,000 neural-network-driven agents in a 4096 x 4096
toroidal world. Each agent has seven forward vision rays, a small
one-hidden-layer neural network, color output, proximity/alignment breeding,
side/body collision deaths, and immediate random replacement for dead agents.
