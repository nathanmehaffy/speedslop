# SpeedSlop: Agent Instructions

This repo is a browser-based evolutionary simulation in early development. The aim
is to create a simulation in which complex behavior can evolve and emerge without
hardcoding evolutionary outcomes. Breeding and death will be available pathways,
and evolution should emerge from those mechanics.

To avoid extinction, the population must be held at constant size, and all
interaction rules must maintain this invariant.

This is a greenfield project. Backwards compatibility, fallbacks, and old runtime
paths are not wanted. Keep the codebase small and direct, and remove obsolete paths
instead of preserving unused alternatives.

Code pathways should be tested as the simulation grows, especially invariants and
GPU data layout assumptions.

## Development Commands

Run from the repository root:

```powershell
npm install
npm run dev
npm run build
npm run check
```

`npm run check` must pass for ordinary changes. `npm run build` should pass for
frontend or shader changes. The developer runs browser testing manually.

## Engineering Preferences

Prefer direct, small, readable code over compatibility layers or defensive leftovers.
When changing architecture, remove obsolete paths instead of preserving unused
alternatives.

Keep high-performance GPU data flow in mind as the simulation is built out.
