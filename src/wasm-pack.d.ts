declare module "*speedslop.js" {
  export class Simulation {
    constructor(width: number, height: number);
    tick(dt_seconds: number): void;
    reset(): void;
    width(): number;
    height(): number;
    frame_ptr(): number;
    frame_len(): number;
  }

  export type InitInput =
    | RequestInfo
    | URL
    | Response
    | BufferSource
    | WebAssembly.Module;

  export interface InitOutput {
    readonly memory: WebAssembly.Memory;
  }

  export default function initWasm(
    moduleOrPath?: InitInput | Promise<InitInput> | { module_or_path: InitInput | Promise<InitInput> },
  ): Promise<InitOutput>;
}
