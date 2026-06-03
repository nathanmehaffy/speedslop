# SpeedSlop

A work-in-progress, browser-based evolutionary simulation. Agents live on a
wrapped (toroidal) world, move under small evolved neural networks, and breed or
die through collisions — evolution is meant to emerge from those mechanics rather
than from hardcoded selection or explicit generations.

The whole simulation runs on the GPU via WebGPU. A throughput controller, driven
by measured GPU + CPU frame cost, runs as many simulation steps per displayed
frame as it can without dropping the render below the display refresh rate.

See `ARCHITECTURE.md` for the controller and GPU pipeline rationale, and
`AGENTS.md` for repository/contributor instructions.

## Requirements

A WebGPU-capable browser exposing the `timestamp-query` feature (recent Chrome or
Edge; Firefox Nightly with WebGPU enabled).

## Development

Run from the repository root:

```powershell
npm install
npm run dev     # start the Vite dev server
npm run build   # type-check and production build
npm run check   # type-check and run the unit tests
npm run bench   # run CPU microbenchmarks and the browser WebGPU benchmark
```

## Benchmarking

Optimization work should use the fixed-workload benchmark scripts rather than
the live app's adaptive throughput controller.

```powershell
npm run bench:cpu
npm run bench:gpu
npm run bench:gpu -- --samples=50 --warmup=10 --steps=1,8,32,128
```

`bench:gpu` starts a temporary Vite server, launches a Playwright-controlled
browser, opens `benchmark.html`, and reports:

- `sim-only`: fixed batches of GPU simulation steps.
- `render-only`: fixed camera and canvas render cost.
- `sim-plus-render`: fixed simulation batches followed by rendering.

The primary optimization signal is `sim-only` median GPU time per fixed step
batch, with `sim-plus-render` used to catch whole-frame regressions. Results are
also written to `benchmark-results.json` by default. The GPU benchmark requires a
browser exposing WebGPU `timestamp-query`; if Playwright has no browser
installed, run `npx playwright install chromium`.
