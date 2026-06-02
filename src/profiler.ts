// GPU frame-time profiler.
//
// Wraps the per-frame submission in a pair of WebGPU timestamps (start of the
// simulation work, end of the render work) and reads the elapsed GPU time back
// asynchronously. Each sample is paired with a synchronous CPU measurement from
// the same frame (main-thread work through command encoding). The readback is
// pipelined through a small pool of mappable buffers so the CPU never blocks on
// the GPU: a sample taken on one frame is consumed a frame or two later.
// Requires the device to have been created with the "timestamp-query" feature
// (see gpu.ts).

const TIMESTAMP_COUNT = 2;
const TIMESTAMP_BYTES = TIMESTAMP_COUNT * 8; // two u64 nanosecond stamps
// Extra headroom so a slow mapAsync does not drop samples and blind the controller.
const READBACK_POOL = 3;

export interface TimingSample {
  /** Measured GPU time for the frame, in milliseconds. */
  gpuMs: number;
  /** Main-thread work through command encoding for the frame, in milliseconds. */
  cpuMs: number;
  /** Integer simulation steps encoded for the frame. */
  steps: number;
}

interface PendingReadback {
  buffer: GPUBuffer;
  steps: number;
  cpuMs: number;
  mapping: boolean;
}

export class GpuProfiler {
  readonly querySet: GPUQuerySet;

  private readonly device: GPUDevice;
  private readonly resolveBuffer: GPUBuffer;
  private readonly freeBuffers: GPUBuffer[] = [];
  private readonly pending: PendingReadback[] = [];
  private latest: TimingSample | null = null;
  private fresh = false;

  constructor(device: GPUDevice) {
    this.device = device;
    this.querySet = device.createQuerySet({ type: "timestamp", count: TIMESTAMP_COUNT });
    this.resolveBuffer = device.createBuffer({
      size: TIMESTAMP_BYTES,
      usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC,
      label: "timestamp-resolve",
    });
    for (let i = 0; i < READBACK_POOL; i += 1) {
      this.freeBuffers.push(
        device.createBuffer({
          size: TIMESTAMP_BYTES,
          usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
          label: `timestamp-readback-${i}`,
        }),
      );
    }
  }

  /** Timestamp writes for the simulation compute pass (start of GPU work). */
  computePassWrites(): GPUComputePassTimestampWrites {
    return { querySet: this.querySet, beginningOfPassWriteIndex: 0 };
  }

  /**
   * Timestamp writes for the render pass. When no simulation pass runs this
   * frame, the render pass carries the start stamp as well so the measured span
   * still covers the whole submission.
   */
  renderPassWrites(hasSimPass: boolean): GPURenderPassTimestampWrites {
    return hasSimPass
      ? { querySet: this.querySet, endOfPassWriteIndex: 1 }
      : { querySet: this.querySet, beginningOfPassWriteIndex: 0, endOfPassWriteIndex: 1 };
  }

  /**
   * Resolve this frame's timestamps and queue a copy into a readback buffer.
   * Call once, after the render pass has been encoded. `steps` and `cpuMs` are
   * tagged onto the sample so the controller knows what workload produced it.
   */
  resolve(encoder: GPUCommandEncoder, steps: number, cpuMs: number): void {
    encoder.resolveQuerySet(this.querySet, 0, TIMESTAMP_COUNT, this.resolveBuffer, 0);
    const buffer = this.freeBuffers.pop();
    if (!buffer) {
      return; // All readback buffers in flight; drop this sample.
    }
    encoder.copyBufferToBuffer(this.resolveBuffer, 0, buffer, 0, TIMESTAMP_BYTES);
    this.pending.push({ buffer, steps, cpuMs, mapping: false });
  }

  /** Kick async readback of the oldest pending sample (strict single-in-flight FIFO). */
  poll(): void {
    const front = this.pending[0];
    if (!front || front.mapping) {
      return;
    }
    front.mapping = true;
    front.buffer
      .mapAsync(GPUMapMode.READ)
      .then(() => {
        const stamps = new BigUint64Array(front.buffer.getMappedRange());
        const deltaNs = stamps[1] - stamps[0];
        front.buffer.unmap();
        this.recycle(front);
        if (deltaNs > 0n) {
          this.latest = {
            gpuMs: Number(deltaNs) / 1e6,
            cpuMs: front.cpuMs,
            steps: front.steps,
          };
          this.fresh = true;
        }
      })
      .catch(() => {
        this.recycle(front);
      });
  }

  /** Most recent completed timing sample, or null if none is new since last call. */
  takeSample(): TimingSample | null {
    if (!this.fresh) {
      return null;
    }
    this.fresh = false;
    return this.latest;
  }

  private recycle(entry: PendingReadback): void {
    const index = this.pending.indexOf(entry);
    if (index !== -1) {
      this.pending.splice(index, 1);
    }
    this.freeBuffers.push(entry.buffer);
  }
}
