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
- Keep CPU<->GPU interop minimal: the only readback is a tiny, pipelined GPU
  timestamp, never a synchronous stall.
- Single code path. The WebGPU `timestamp-query` feature is **required** (there is
  no blind fallback); the app refuses to start without it.
- The neural simulation compute bind group requires at least 11 storage buffers
  per shader stage, so device creation requests `maxStorageBuffersPerShaderStage`
  above the WebGPU default limit.

## Core constraints that shape everything

1. **WebGPU exposes one queue per device.** Compute (sim) and render share one GPU
   timeline; they do not run in true parallel. Each frame we encode sim work and
   render work into one ordered submission. The design problem is choosing *how
   much* sim work to attach to each frame so render never starves.

2. **GPU time is measured directly, but never synchronously.** Synchronous
   measurement (`mapAsync` awaited in-frame, `onSubmittedWorkDone`) would stall the
   CPU on the GPU. Instead we wrap the frame's work in a `timestamp-query` pair and
   read the elapsed time back **asynchronously**, pipelined through a small pool of
   mappable buffers, so a sample taken this frame is consumed a frame or two later.
   The loop never blocks.

3. **GPU time is a continuous control signal.** Unlike vsync-clamped
   `requestAnimationFrame` deltas (which sit flat at the refresh interval whether
   the GPU is 40% or 98% utilized), measured GPU time varies smoothly with the
   amount of sim work. That continuous headroom signal is what makes the controller
   simple. rAF deltas are still used, but only to estimate the display refresh
   interval (the budget).

## High-level shape

- **One `GPUDevice`, one queue, one `requestAnimationFrame` loop**, all on the
  main thread.
- **One small pipelined readback:** a two-entry timestamp query per frame, resolved
  and copied into a rotating pool of mappable buffers and read back without
  blocking. No other GPU->CPU traffic.
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
- `src/simulation.ts` owns the torus simulation GPU buffers/pipelines and
  compute-pass ordering.
- `src/simulationShader.ts` contains the WGSL kernels for fixed-capacity agent
  slots, neural-network movement, circular hitbox collision deaths, head-on
  breeding, genetic crossover/mutation, random immigrants, and the counting-sort
  neighbor index (`dense` + `cellStart`) used by sensing and collision.
- `src/simulationPacking.ts` owns CPU-side parameter and initial-buffer packing.
- `src/simulationPolicy.ts` holds pure, tested policy invariants for sensing,
  collision broadphase bounds, and demographic slot allocation.
- `src/renderer.ts` renders fixed agent slots as direction-facing HSV triangles
  into the swapchain texture, with dead slots discarded in the vertex shader and
  torus tile copies drawn through instancing.
- `src/layout.ts` centralizes GPU buffer-layout constants and WGSL structs shared
  by simulation and rendering.
- `src/spatial.ts`, `src/collision.ts`, `src/genetics.ts`, and `src/camera.ts`
  are pure, unit-tested modules for grid/toroidal math, collision
  classification, genome contracts, and the pan/zoom camera/view model with no
  GPU dependencies.
- `src/profiler.ts` owns the `timestamp-query` set, resolves the per-frame GPU
  timestamps, and reads the elapsed time back through a non-blocking pipelined
  pool of mappable buffers.
- `src/controller.ts` is a pure, unit-tested throughput controller driven by the
  measured GPU time; it has no browser or GPU dependencies.
- `src/telemetry.ts` formats the small on-screen fps / sim-steps-per-sec
  monitor.
- `src/config.ts` holds shared demo/runtime constants.

## The frame loop

Each `requestAnimationFrame(t)` callback:

1. **Sense.** Feed the rAF delta to the controller's refresh-interval estimator
   (the budget). Consume the most recent completed frame-cost sample (GPU +
   paired CPU encode time), if any.
2. **Decide.** Update the control variable `rate` (simulation steps per displayed
   frame, a positive real) from the measured frame cost. Convert it to an integer
   step count for this frame via the step accumulator (below).
3. **Encode** one command buffer:
   - the chosen number of simulation steps, back to back (each step is its own
     sequence of dependent compute passes), with no CPU work in between, with the
     **start** timestamp written at the first pass;
   - the render pass into the swapchain texture, with the **end** timestamp;
   - bracket main-thread encode work with `performance.now()` and tag the elapsed
     CPU time onto the same pipelined sample as the GPU timestamps;
   - `resolveQuerySet` + a copy into a mappable readback buffer.
4. **Submit once,** then kick the async readback (no await).
5. `requestAnimationFrame(next)`.

The only readback is the pipelined timestamp copy; the loop never blocks on the
GPU.

## The throughput controller

The controller regulates a single continuous variable, driven by the measured
frame cost (GPU timestamp span plus synchronous CPU encode time) of each frame.

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

### The budget: estimated refresh interval

- The **refresh interval is unknown a priori** (60/120/144 Hz, and it can change if
  the window moves to another monitor). Estimate it continuously as a low
  percentile of recent rAF deltas (rendering cannot beat the display refresh, so
  the fastest frames cluster at the true interval). Never hard-code it.
- The **frame-work budget** is a fraction of that interval (`targetUtilization`,
  e.g. 0.85), leaving margin for compositor/present jitter after submit.

### Control law: nudge toward the budget

Frame cost is monotonically increasing in the number of simulation steps, so
finding the operating point is one-dimensional root-finding with a measured
signal. Each completed sample `w = g + c` (GPU time `g`, CPU encode time `c`, for
the integer step count `s` that actually ran) drives a multiplicative update in log
space:

```
factor     = clamp(budget / w, 1/maxStepFactor, maxStepFactor)
target     = s * factor
log(rate) += smoothing * (log(target) - log(rate))
```

- The fixed point is `w = budget`: when measured cost equals the budget, `factor`
  is 1 and the rate holds. Because `w(r)` is monotone, the iteration converges to
  that point — and the linear cost offset (fixed render/overhead time) is absorbed
  automatically, since the law only cares that `w` hits the budget.
- `clamp(..., maxStepFactor)` bounds how far one noisy sample can move the rate
  (e.g. an overhead-only frame at a sub-1 rate), and `smoothing` is an EWMA that
  damps measurement noise. Together they give fast acquisition (a few samples to
  climb orders of magnitude) without oscillation at the operating point.
- Log-space (multiplicative) updates make the same law work from sub-1 rates to
  thousands of steps per frame.
- Responsiveness is symmetric and fast in **both** directions: a sudden cost
  increase (thermal throttle, zoom, population swing, another app contending) shows
  up immediately as larger `w` and the rate drops within a sample or two; freed-up
  headroom shows up as smaller `w` and the rate climbs just as quickly. There is no
  blind probing asymmetry, because headroom is observed directly rather than
  inferred from the absence of dropped frames.

### Pipelined measurement (no stall)

GPU timestamps are resolved every frame and copied into a rotating pool of
mappable buffers; each entry also carries the synchronous CPU encode duration from
the same frame. `mapAsync` completions are consumed FIFO a frame or two later.
The controller simply uses the most recent completed sample, so the one-to-two-frame
read latency is harmless for a slowly varying operating point.

### Remaining unmeasured slice

Time from `queue.submit()` through compositor present is still outside the control
loop. `targetUtilization` leaves headroom for that jitter.

### Degenerate case

If even one step per frame overruns the budget, `rate` falls below 1 and the
render rate drops below refresh. No architecture can prevent this when the device
cannot run a single step within a refresh interval; the accumulator makes the
degradation smooth (steps spread across multiple frames).

## Simulation data model

- **Compact agent slots:** one fixed-size `Agent` storage buffer holds the small
  per-agent state used by both simulation and rendering. Neural genomes, planned
  movement, collision marks, and birth events live in separate fixed-size storage
  buffers. Shared WGSL snippets and layout tests keep the CPU/GPU contract
  explicit.
- **Fixed-capacity buffers:** population capacity is fixed (no GPU allocation
  churn), which keeps per-step cost stable and predictable for the controller.
  If a head-on birth happens while no free slot exists, one random parent slot is
  overwritten by the child.
- **Agent identity:** every agent carries a stable 32-bit ID distinct from its
  slot index. Initial IDs and newborn IDs are deterministic from slot/step
  inputs; child IDs change when a parent slot is reused.
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

Per-step reproducibility is preferred, but not promised yet. A step is the same
deterministic kernel regardless of how many steps are batched into a frame, and
RNG is keyed from agent ID plus step. Atomic scatter/free-list ordering and future
floating-point reductions can still introduce nondeterminism, so any pass where
exact reproducibility matters should use fixed-order or integer/fixed-point
accumulation.

## Per-step simulation pipeline

Each simulation step begins with a full cell histogram, prefix scan, and scatter
that build a query-ready neighbor index (`cellStart` offsets plus cell-sorted
`dense` with positions). Neural sensing and collision broadphase both read that
index.

1. **Grid/event reset and step bump:** clear cell counters, collision marks,
   mate targets, birth-event count, and the free-slot counter; increment the GPU
   step counter.
2. **Population scan:** histogram live agents by cell, gather dead slots into the
   free list, prefix-sum counts into `cellStart`, scatter into `dense`.
3. **Neural movement planning:** each live agent scans nearby cells for a fixed
   number of nearest neighbors, evaluates its compact neural genome, and writes a
   planned next position/direction/speed.
4. **Collision choice:** agents compare planned circular hitboxes in nearby
   cells. Side/back impacts mark the hitter for death; reciprocal head-on choices
   record mate targets.
5. **Contact commit:** reciprocal mate targets become birth events, killed agents
   are cleared and appended directly to the free list, and survivors commit
   planned movement.
6. **Childbirth:** birth events consume free-list slots already known from the
   population scan plus same-step deaths, or randomly overwrite one parent if
   capacity is full.
7. **Population replenishment:** random immigrants consume remaining free-list
   slots until the population floor is reached.

The step batch does not rebuild a final render index. Rendering reads fixed
agent slots directly, so the last simulation step's post-birth/post-death state
is visible without a post-batch counting sort.

## Render path

- Reads fixed-slot simulation state directly after the sim passes in the same
  submission.
- Draws one instanced border call and one instanced agent call for all visible
  torus tiles. Dead agent slots emit degenerate offscreen triangles in the vertex
  shader, avoiding a live-only render compaction pass.
- Render/world coordinate spaces are separate; resize affects rendering only.

## Deferred / open items

- **Indirect dispatch / fewer passes per step:** optional future optimization if
  CPU encoding becomes the dominant cost again at much higher step counts.
- **Simulation behavioral content:** entirely out of scope for this document.

## Robustness / ops

- **Device loss:** currently fatal. `device.lost` stops the app and surfaces the
  error through the fatal-error path; browser/manual testing remains the runtime
  target for now.
- **Observability:** keep the default UI minimal: fps and simulation steps per
  second. When deeper controller analysis is needed, prefer temporary targeted
  diagnostics that can be removed after tuning.

## Tunable parameters (summary)

- `targetUtilization` — fraction of the estimated refresh interval budgeted for
  measured frame work (the controller's set point).
- `smoothing` — EWMA factor on the log-rate update (noise damping vs reactivity).
- `maxStepFactor` — clamp on how far a single frame-cost sample can move the rate.
- Refresh estimation window/percentile and long-pause cutoff for ignoring
  tab-backgrounding, devtools pauses, and other browser/OS interruptions.
- `rate` bounds (`rateMax`, `rateMin`) and initial rate.
- Population capacity, grid/cell sizing, over-capacity (birth-drop) policy.
