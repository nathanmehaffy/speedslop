# SpeedSlop

High-performance browser simulation boilerplate using Rust, WebAssembly, Vite,
TypeScript, and WebGPU.

The simulation state and framebuffer are owned by WASM. The browser app drives
the animation clock, reads the stable WASM framebuffer, uploads it to a WebGPU
texture, and presents it to a full-window canvas.

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

- `new(width, height)` creates a deterministic 2D field simulation.
- `tick(dt_seconds)` advances the simulation and refreshes the RGBA framebuffer.
- `reset()` returns the simulation to time zero.
- `width()` and `height()` expose the framebuffer dimensions.
- `frame_ptr()` and `frame_len()` expose the stable WASM-owned RGBA buffer.

The TypeScript app imports `Simulation` from the generated `sim/pkg` package and
uses WebGPU to upload the framebuffer as an `rgba8unorm` texture every animation
frame.
