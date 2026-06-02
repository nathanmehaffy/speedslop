// WebGPU device + canvas context initialization.

export interface GpuContext {
  device: GPUDevice;
  context: GPUCanvasContext;
  format: GPUTextureFormat;
}

const REQUIRED_STORAGE_BUFFERS_PER_SHADER_STAGE = 11;

export async function initGpu(canvas: HTMLCanvasElement): Promise<GpuContext> {
  if (!navigator.gpu) {
    throw new Error(
      "WebGPU is not available in this browser. SpeedSlop needs a WebGPU-capable browser (recent Chrome, Edge, or Firefox Nightly).",
    );
  }

  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) {
    throw new Error("No WebGPU adapter found. SpeedSlop requires a working GPU.");
  }

  // The throughput controller is driven entirely by GPU timestamp readings, so
  // the feature is mandatory: there is no blind/timing-free fallback path.
  if (!adapter.features.has("timestamp-query")) {
    throw new Error(
      'This GPU/browser does not expose the WebGPU "timestamp-query" feature, which SpeedSlop requires to measure GPU frame time. Try a recent Chrome or Edge build (enable "Unsafe WebGPU" if needed).',
    );
  }

  if (adapter.limits.maxStorageBuffersPerShaderStage < REQUIRED_STORAGE_BUFFERS_PER_SHADER_STAGE) {
    throw new Error(
      `This GPU/browser supports only ${adapter.limits.maxStorageBuffersPerShaderStage} storage buffers per shader stage; SpeedSlop requires ${REQUIRED_STORAGE_BUFFERS_PER_SHADER_STAGE} for the neural simulation pipeline.`,
    );
  }

  const device = await adapter.requestDevice({
    requiredFeatures: ["timestamp-query"],
    requiredLimits: {
      maxStorageBuffersPerShaderStage: REQUIRED_STORAGE_BUFFERS_PER_SHADER_STAGE,
    },
  });

  const context = canvas.getContext("webgpu");
  if (!context) {
    throw new Error("Failed to get WebGPU canvas context");
  }

  const format = navigator.gpu.getPreferredCanvasFormat();
  context.configure({ device, format, alphaMode: "opaque" });

  return { device, context, format };
}

export function resizeCanvas(
  canvas: HTMLCanvasElement,
  maxDevicePixelRatio: number,
): { width: number; height: number } {
  const dpr = Math.min(window.devicePixelRatio, maxDevicePixelRatio);
  const width = Math.max(1, Math.floor(canvas.clientWidth * dpr));
  const height = Math.max(1, Math.floor(canvas.clientHeight * dpr));
  if (canvas.width !== width || canvas.height !== height) {
    canvas.width = width;
    canvas.height = height;
  }
  return { width, height };
}

export function installGpuErrorHandlers(device: GPUDevice, onError: (error: Error) => void): () => void {
  let active = true;

  const handleUncapturedError = (event: Event): void => {
    const gpuEvent = event as GPUUncapturedErrorEvent;
    onError(new Error(`Uncaptured WebGPU error: ${gpuEvent.error.message}`));
  };

  device.addEventListener("uncapturederror", handleUncapturedError);

  void device.lost.then((info) => {
    if (!active) {
      return;
    }
    const detail = info.message ? `: ${info.message}` : "";
    onError(new Error(`WebGPU device lost (${info.reason})${detail}`));
  });

  return () => {
    active = false;
    device.removeEventListener("uncapturederror", handleUncapturedError);
  };
}
