# SpeedSlop

SpeedSlop is a browser-based evolutionary simulation. A Rust simulation core owns the
agent state, compiles to WebAssembly, and runs in a web worker. The TypeScript frontend
streams packed agent snapshots to WebGPU and renders the population as instanced
triangles with a small HUD for speed, reset, camera, and stats.

The current simulation keeps 10,000 neural agents in a constant-size toroidal world.
Agents sense neighbors with nine rays, steer with a tiny neural network, breed by
proximity and alignment, die on side/body collisions after a short grace period, and
are immediately replaced so the population never changes.

## Setup

Install Node.js, Rust, `wasm-pack`, and the `wasm32-unknown-unknown` target:

```powershell
rustup target add wasm32-unknown-unknown
cargo install wasm-pack
npm install
```

## Commands

```powershell
npm run dev      # build WASM and start Vite on 127.0.0.1
npm run build    # build WASM, type-check, and bundle production assets
npm run check    # build WASM, type-check, and cargo-check the WASM target
npm run bench    # run the worker-oriented simulation benchmark
cargo test --manifest-path sim/Cargo.toml
```

The Rust render buffer layout is eight `f32` values per agent:

```text
[x_norm, y_norm, dir_x, dir_y, r, g, b, speed_norm]
```
