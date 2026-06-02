import { describe, expect, it } from "vitest";

import { GpuProfiler } from "./profiler";

Object.defineProperty(globalThis, "GPUBufferUsage", {
  configurable: true,
  value: {
    COPY_DST: 1,
    COPY_SRC: 2,
    MAP_READ: 4,
    QUERY_RESOLVE: 8,
  },
});

Object.defineProperty(globalThis, "GPUMapMode", {
  configurable: true,
  value: { READ: 1 },
});

class FakeBuffer {
  deltaNs = 0n;
  private mapped = new ArrayBuffer(16);

  constructor(readonly label: string) {}

  async mapAsync(): Promise<void> {
    const stamps = new BigUint64Array(this.mapped);
    stamps[0] = 0n;
    stamps[1] = this.deltaNs;
  }

  getMappedRange(): ArrayBuffer {
    return this.mapped;
  }

  unmap(): void {}
}

class FakeDevice {
  readonly buffers: FakeBuffer[] = [];

  createQuerySet(): GPUQuerySet {
    return {} as GPUQuerySet;
  }

  createBuffer(descriptor: GPUBufferDescriptor): GPUBuffer {
    const buffer = new FakeBuffer(descriptor.label ?? "");
    this.buffers.push(buffer);
    return buffer as unknown as GPUBuffer;
  }
}

class FakeEncoder {
  nextDeltaNs = 0n;

  resolveQuerySet(): void {}

  copyBufferToBuffer(
    _source: GPUBuffer,
    _sourceOffset: GPUSize64,
    destination: GPUBuffer,
    _destinationOffset: GPUSize64,
    _size: GPUSize64,
  ): void {
    (destination as unknown as FakeBuffer).deltaNs = this.nextDeltaNs;
  }
}

async function flushMicrotasks(): Promise<void> {
  await Promise.resolve();
  await Promise.resolve();
}

describe("GpuProfiler", () => {
  it("returns completed samples in FIFO order", async () => {
    const profiler = new GpuProfiler(new FakeDevice() as unknown as GPUDevice);
    const encoder = new FakeEncoder();

    encoder.nextDeltaNs = 3_000_000n;
    profiler.resolve(encoder as unknown as GPUCommandEncoder, 7, 0.25);
    profiler.poll();

    expect(profiler.takeSample()).toBeNull();
    await flushMicrotasks();

    expect(profiler.takeSample()).toEqual({ gpuMs: 3, cpuMs: 0.25, steps: 7 });
    expect(profiler.takeSample()).toBeNull();
  });

  it("drops samples when the readback pool is exhausted without corrupting later samples", async () => {
    const profiler = new GpuProfiler(new FakeDevice() as unknown as GPUDevice);
    const encoder = new FakeEncoder();
    const submit = (deltaNs: bigint, steps: number): void => {
      encoder.nextDeltaNs = deltaNs;
      profiler.resolve(encoder as unknown as GPUCommandEncoder, steps, 0);
    };

    submit(1_000_000n, 1);
    submit(2_000_000n, 2);
    submit(3_000_000n, 3);
    submit(4_000_000n, 4); // dropped: all three readback buffers are in flight

    const samples: number[] = [];
    for (const expectedSteps of [1, 2, 3]) {
      profiler.poll();
      await flushMicrotasks();
      samples.push(profiler.takeSample()?.steps ?? -1);
      if (expectedSteps === 1) {
        submit(5_000_000n, 5);
      }
    }

    profiler.poll();
    await flushMicrotasks();
    samples.push(profiler.takeSample()?.steps ?? -1);

    expect(samples).toEqual([1, 2, 3, 5]);
  });
});
