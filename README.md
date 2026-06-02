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
```
