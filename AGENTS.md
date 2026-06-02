# SpeedSlop: Agent Instructions

This repo is a browser-based evolutionary simulation in early development. The aim is to create a simulation in which complex behavior can evolve and emerge without hardcoding evolutionary outcomes. Breeding and death will be available pathways, and evolution should emerge from those mechanics without artificial selection or explicit generations.

This is a greenfield project. Backwards compatibility, fallbacks, and old runtime paths are not wanted. Keep the codebase small and direct, and remove obsolete paths instead of preserving unused alternatives.

Critical code pathways should be extensively tested as the simulation grows, especially invariants and GPU data layout assumptions.

## Project Structure

See `ARCHITECTURE.md` for the controller and GPU pipeline rationale. In the current implementation:

- `src/main.ts` handles DOM lookup and fatal-error display.
- `src/app.ts` owns the `requestAnimationFrame` loop, wires together GPU setup, simulation, rendering, throughput control, and the fps/steps monitor, and handles pan/zoom input.
- `src/gpu.ts` initializes WebGPU (requiring the `timestamp-query` feature), resizes the canvas, and installs GPU error handlers.
- `src/simulation.ts` is the torus agent simulation: fixed-capacity agent slots, random rainbow movement, GPU-side sine-waved births/deaths, and a per-step counting-sort that builds the cell-sorted neighbor index (`dense` + `cellStart`) used for rendering and future k-nearest queries.
- `src/renderer.ts` draws agents as direction-facing HSV triangles via an indirect draw, tiled across the visible viewport to show torus wrapping with grey edge borders.
- `src/layout.ts` centralizes GPU buffer-layout constants and shared WGSL structs used by both simulation and rendering.
- `src/spatial.ts` is pure, tested cell-index / toroidal-distance / population-target math that mirrors small shader-side invariants.
- `src/camera.ts` is the pure, tested pan/zoom camera (world<->screen transform and visible-tile range).
- `src/profiler.ts` wraps each frame in a `timestamp-query` pair, pairs it with a
  synchronous CPU encode measurement, and reads GPU time back through a
  non-blocking pipelined buffer pool.
- `src/controller.ts` is the pure, tested throughput controller driven by measured
  frame cost (GPU + CPU encode).
- `src/telemetry.ts` formats the lightweight on-screen monitor.
- `src/config.ts` centralizes small runtime constants.

## Development Commands

Run from the repository root:

```powershell
npm install
npm run dev
npm run build
npm run check
```

Unless specifically asked, do not run `npm run dev` yourself to start a dev server for the developer; the developer will do this manually in their own terminal window.

`npm run check` must pass for ordinary changes. `npm run build` should pass for frontend or shader changes. The developer always runs in-browser testing manually, and agents should not use browser control tools to attempt their own in-browser testing, unless specifically using a headless browser for scripted correctness or performance tests.

## Engineering Preferences

Prefer direct, small, readable code over compatibility layers or defensive leftovers.
When changing architecture, remove obsolete paths instead of preserving unused
alternatives.

Keep high-performance GPU data flow in mind as the simulation is built out.
