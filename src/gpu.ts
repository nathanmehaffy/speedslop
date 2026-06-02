// WebGPU device + canvas context initialization.

export interface GpuContext {
  device: GPUDevice;
  context: GPUCanvasContext;
  format: GPUTextureFormat;
}

export async function initGpu(canvas: HTMLCanvasElement): Promise<GpuContext> {
  if (!navigator.gpu) {
    throw new Error("WebGPU is not available in this browser");
  }

  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) {
    throw new Error("No WebGPU adapter found");
  }

  const device = await adapter.requestDevice();

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
