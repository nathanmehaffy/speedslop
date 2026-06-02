# SpeedSlop Architecture

This document describes the high-level architecture for running a simulation as
fast as the GPU allows while rendering a live view locked to the display refresh
rate. It is an architecture/planning reference, not an implementation spec. The
behavioral *content* of the simulation is deliberately out of scope here.

## Goals

- Run the simulation on the GPU at maximum throughput.
- Render a live view at the display refresh rate, kept smooth.
- Run as many simulation steps per displayed frame as possible **without** driving
  the render rate below the display refresh.
- Keep CPU<->GPU interop at an absolute minimum.
- Single code path that works on any device, with no device-specific features.

## Core constraints that shape everything

1. **WebGPU exposes one queue per device.** Compute (sim) and render share one GPU
   timeline; they do not run in true parallel. Each frame we encode sim work and
   render work into one ordered submission. The design problem is choosing *how
   much* sim work to attach to each frame so render never starves.

2. **There is no cheap synchronous way to measure GPU time.** Anything synchronous
   (`mapAsync` awaited in-frame, `onSubmittedWorkDone`) stalls the CPU on the GPU,
   which is exactly what we want to avoid. We therefore do **not** read GPU
   timings at all.

3. **vsync clamps the only timing signal we keep.** `requestAnimationFrame`
   deltas sit at the refresh interval whether the GPU is 40% or 98% utilized, then
   jump when a frame is dropped. So the frame-time signal carries no continuous
   "headroom" information below the cap — the only information is in the *drop
   events* themselves. This is a deliberate accepted tradeoff (see Controller).

## High-level shape

- **One `GPUDevice`, one queue, one `requestAnimationFrame` loop**, all on the
  main thread.
- **Zero GPU->CPU readback.** The controller's only input is CPU-side rAF timing.
  No buffer maps, no timestamp queries.
- **Simple sim-then-render ordering** within a single submission per frame. Because
  the queue is ordered, the render pass encoded after the sim passes observes the
  finished sim state with no explicit synchronization and effectively no added
  latency. No double-buffered snapshot is required unless later profiling demands
  it.
- **Simulation world space is fully decoupled from canvas/render space.** Canvas
  resize affects only rendering; it is never a simulation event.

## Current implementation map

- `src/main.ts` performs DOM lookup, starts the app, and surfaces fatal WebGPU
  errors.
- `src/app.ts` owns the `requestAnimationFrame` loop and wires together GPU
  initialization, the demo simulation, renderer, throughput controller, and
  lightweight telemetry.
- `src/gpu.ts` contains WebGPU adapter/device/context setup, canvas resizing, and
  device-loss/error hooks.
- `src/simulation.ts` is the torus agent simulation and its GPU buffers/pipelines:
  fixed-capacity agent slots, a per-step counting-sort spatial grid that compacts
  live agents into a dense cell-sorted array for N-closest queries, an N-nearest
  sensory gather pass (placeholder brain input), random rainbow movement, and
  GPU-side births/deaths that sine-wave population around half capacity.
- `src/renderer.ts` renders agents as direction-facing HSV triangles into the
  swapchain texture, tiled across the viewport to visualize torus wrapping.
- `src/spatial.ts` and `src/camera.ts` are pure, unit-tested math modules (grid /
  toroidal-distance / population target, and the pan/zoom camera) with no GPU
  dependencies.
- `src/controller.ts` is a pure, unit-tested throughput controller with no browser
  or GPU dependencies.
- `src/telemetry.ts` formats the small on-screen fps and sim-steps/sec monitor.
- `src/config.ts` holds shared demo/runtime constants.

## The frame loop

Each `requestAnimationFrame(t)` callback:

1. **Sense.** Compare the rAF delta against the running estimate of the refresh
   interval to decide whether the previous frame dropped. Feed that single
   boolean into the controller.
2. **Decide.** Update the control variable `rate` (simulation steps per displayed
   frame, a positive real). Convert it to an integer step count for this frame via
   the step accumulator (below).
3. **Encode** one command buffer:
   - the chosen number of simulation steps, back to back (each step is its own
     sequence of dependent compute passes), with no CPU work in between;
   - the render pass into the swapchain texture.
4. **Submit once.**
5. `requestAnimationFrame(next)`.

There is no per-frame readback step. The loop never blocks on the GPU.

## The throughput controller (blind)

The controller regulates a single continuous variable and is driven solely by
frame-drop observations.

### Control variable: steps-per-frame as a real number

`rate` is steps per displayed frame and may be fractional or below 1. A fractional
accumulator converts it to an integer count each frame:

```
acc  += rate
steps = floor(acc)
acc  -= steps
```

- `rate = 4.0` -> 4 steps every frame.
- `rate = 0.25` -> one step roughly every 4 frames (most frames run zero).

This unifies "many steps per frame" and "one step every K frames" into one path,
so the system degrades gracefully when even a single step per frame is too
expensive to hold refresh.

### Drop detection (the entire sensor)

- The **refresh interval is unknown a priori** (60/120/144 Hz, and it can change if
  the window moves to another monitor). Estimate it continuously as a running
  median/mode of recent rAF deltas. Never hard-code it.
- A frame is counted as dropped when its rAF delta exceeds the estimated interval
  by a margin (e.g. >= ~1.5x).
- The compositor drops frames for reasons unrelated to our load. The detector must
  accumulate evidence rather than react to a single long delta.

### Controller phases

The controller is an explicit finite-state machine:

- **Warmup:** collect a few valid rAF deltas before classifying drops, so shader
  compilation and first-frame startup noise do not set the operating point.
- **Acquire:** grow `rate` geometrically until the first reliable dropped frame
  reveals the local ceiling, then back off below it and hand control to tracking.
- **Track:** use batched stochastic approximation to keep the observed drop rate
  near a small nonzero target.
- **Probe:** occasionally run a short, explicit upward test to learn whether more
  headroom is available.
- **Recover:** after statistically significant overload, crash the rate down,
  pause long enough for a transient rAF burst to pass, reset controller state, and
  resume tracking from the safer base.

A floored-gain regulator targeting a *small* drop rate moves upward only slowly,
so it cannot traverse the orders of magnitude between a cold-start `rate` of 1
and an operating point of hundreds-to-thousands of steps. Acquisition handles
that initial climb quickly and deliberately costs at most a small number of
dropped frames while finding the ceiling.

### Steady-state regulator: batched stochastic approximation

The workload and framerate are expected to be roughly constant, so steady-state
adaptation can be slow. Model drops as `Bernoulli(p)` and treat operating-point
selection as stochastic root-finding (Robbins-Monro) toward a small target drop
rate `p*`:

```
log(rate) <- log(rate) - a_n * (drops - frames * p*)
```

- `p_hat` is measured over a small tracking batch, not from a single frame. This
  damps compositor noise and avoids overreacting to isolated long deltas.
- The update uses the batch's drop-count error rather than the raw probability
  error, so one missed vsync in a short batch produces a meaningful correction
  instead of being divided down to near zero.
- Log-space updates make rate changes multiplicative, so the same control law
  works from sub-1 rates to hundreds of steps per frame.
- Diminishing gain `a_n` gives convergence (settling), unlike constant-step AIMD,
  which tends to produce a perpetual sawtooth.
- Because the true operating point drifts slowly, the gain is **floored**
  (`a_n = max(a / n, a_min)`) so the loop never goes fully deaf. The cost is a
  small residual jitter band around the moving edge rather than a perfectly flat
  line — an accepted, fundamental convergence-vs-tracking tradeoff.
- Operate with a margin so steady-state drops stay rare.

### Overload detection: one-sided sequential test

Smooth drift is handled by the regulator; **sudden regime changes** (e.g. tab
foreground/background, GPU up/down-clock, thermal throttle, another app contending
for the GPU) are handled by a **one-sided Bernoulli CUSUM / SPRT** on the drop
stream:

- Capacity loss is detected quickly from a burst of drops by comparing the normal
  hypothesis `p0 = p*` against an overload hypothesis `p1` (for example 10-20%
  drops).
- On a detected **overload**, crash `rate` by a multiplicative factor (overshoot
  is the expensive failure; undershooting a few frames is cheap), reset the
  detector, and resume tracking from the safer base. The recovery dwell suppresses
  repeated crashes from one short focus-switch or compositor burst.
- The controller remembers the pre-crash rate as a temporary recovery target. If
  subsequent tracking windows are clean, it climbs back toward that target faster
  than normal probes; if drops continue, the target is discarded and the lower
  operating point is treated as real. Acquisition is reserved for cold start and
  first ceiling discovery so overload recovery cannot immediately recreate the
  same overshoot.

No explicit OS/visibility/refresh-rate signals are used — detection remains
purely statistical, keeping a single device-agnostic code path.

### Headroom detection: active probes

With only a clamped frame-time signal, all information is in drop events.
Capacity *losses* are detected fast; capacity *gains* are detected more slowly.
Passive absence of drops is not enough evidence for a fast upward reset: long
no-drop runs also occur during normal stable operation at a low target drop
rate. The controller therefore learns headroom through **active probes**:

- After a stable tracking dwell, when several recent tracking windows are clean
  and overload evidence is near zero, temporarily raise `rate` by a small
  multiplicative factor.
- Run the probe for a short fixed window.
- Accept the higher rate if the probe is clean (or statistically consistent with
  the target); reject and restore the previous rate if it produces drops.
- Apply a cooldown after either result so probes do not create visible jitter.

This makes upward movement explicit and observable, rather than treating ordinary
no-drop periods as a hidden headroom event.

### Degenerate case

If even one step per frame overruns the budget, `rate` falls below 1 and the
render rate drops below refresh. No architecture can prevent this when the device
cannot run a single step within a refresh interval; the accumulator makes the
degradation smooth (steps spread across multiple frames).

## Simulation data model

- **Structure-of-arrays (SoA):** one storage buffer per attribute, for coalesced
  GPU access. Chosen independently of the births/deaths scheme; a flat win.
- **Fixed-capacity buffers:** population capacity is fixed (no GPU allocation
  churn), which keeps per-step cost stable and predictable for the controller.
  Births beyond capacity are dropped (clamp policy).
- **Persistent agent identity:** every agent carries a stable 64-bit ID distinct
  from its slot index. A child's ID is derived **deterministically** from its
  parent (e.g. `hash(parent_id, step)`), not from a global atomic counter.
- **Counter-based RNG:** stateless PRNG (PCG/Philox-style) keyed by
  `(agent_id, step)`. No stored RNG state to evolve, and deterministic regardless
  of thread scheduling.
- **GPU-resident step counter:** the current step index lives in a small storage
  buffer and is incremented **on the GPU** (a one-thread bump, or folded into the
  first pass of each step). Sim passes derive time and RNG seeds from it. Within a
  compute pass, successive dispatches observe prior dispatches' storage writes, so
  the counter stays correct across a whole batch with zero per-step CPU work and no
  uniform-offset alignment waste.

### Reproducibility

Per-step reproducibility is preferred (not strictly required) and is naturally
satisfied: a step is the same deterministic kernel regardless of how many steps
are batched into a frame, so the machine-dependent step count never affects
outcomes. Deterministic child IDs + ID-keyed RNG keep *behavior* reproducible even
when physical slot/append order is nondeterministic. The remaining caveat is
non-associative float reductions / float atomics, whose completion order can vary;
any pass where exact reproducibility matters should use fixed-order or
integer/fixed-point accumulation.

## Per-step simulation pipeline

Agents interact only via spatial proximity, so each step rebuilds a spatial index.
Births/deaths use **stream compaction folded into that spatial index build**
(the highest-performance option for this workload):

1. **(Step counter bump.)**
2. **Spatial index build = compaction (counting sort):** histogram live agents per
   grid cell, prefix-sum, scatter into a dense, cell-sorted output buffer. Dead
   agents are simply not counted, so the output is compacted *and* spatially
   ordered in a single pass. Newborns pending from the previous step are included
   here.
3. **Interaction / integration:** each agent looks up neighbors via the grid,
   updates its state, may flag itself dead, and may append newborns to a pending
   births buffer (atomic append; nondeterministic slot, deterministic ID).
4. Deaths flagged this step and births appended this step are resolved by the next
   step's compaction.

Why this scheme for this workload:

- Spatial interaction means **no persistent agent-to-agent references**, so moving
  slots during compaction is harmless.
- A spatial index is rebuilt every step anyway, so **compaction is nearly free** —
  it rides the counting sort already being paid for.
- **High birth/death churn** is exactly the case that punishes atomic free-lists;
  compaction avoids per-birth global-allocator contention.
- Counting-sort compaction is **deterministic** (stable order), aiding
  reproducibility of layout.

The sim may be designed to keep occupancy (population / capacity) relatively high
and stable, which keeps the dense array efficient and per-step cost predictable.

## Render path

- Reads simulation state directly after the sim passes in the same submission.
- Exact render representation (draw agents directly vs. a derived field/instance
  buffer) is **deferred**; it slots into the render portion of the frame without
  changing the loop or the controller, since its cost is just part of the measured
  frame.
- Render/world coordinate spaces are separate; resize affects rendering only.

## Deferred / open items

- **Render representation details** (#5 from discussion): decided later with sim
  content.
- **CPU command-encoding cost:** recording N steps per frame is CPU work that
  scales with N, and WebGPU command buffers are single-use (no record-once/replay).
  The blind controller absorbs this transparently (a longer frame simply lowers
  `rate`), so it is never incorrect, but it can cap useful throughput. Left simple
  for now; revisit with indirect dispatch / fewer passes per step only if profiling
  shows CPU-bound encoding.
- **Simulation behavioral content:** entirely out of scope for this document.

## Robustness / ops

- **Device loss:** handle `device.lost` with re-initialization; a long-running
  flat-out GPU loop is a realistic trigger for driver resets.
- **Observability:** keep the default UI minimal: fps and simulation steps per
  second. When deeper controller analysis is needed, prefer temporary targeted
  diagnostics that can be removed after tuning.

## Tunable parameters (summary)

- `p*` — target steady-state drop rate.
- Tracking batch size, `a`, `a_min` — stochastic-approximation gain and its
  floor.
- Acquisition growth factor and post-drop backoff.
- Drop-detection threshold (multiple of estimated refresh interval) and
  refresh-estimation window.
- Long-pause cutoff for ignoring tab-backgrounding, devtools pauses, and other
  browser/OS interruptions that are not useful load signals.
- Overload hypothesis `p1`, CUSUM/SPRT threshold, crash factor, and recovery
  dwell/growth.
- Probe factor, probe window size, probe cooldown, clean-window gate, probe
  dwell, and probe acceptance policy.
- `rate` bounds (`rate_max`, minimum implied by the accumulator floor).
- Population capacity, grid/cell sizing, over-capacity (birth-drop) policy.
