declare module "*speedslop.js" {
  export class Simulation {
    constructor(world_size: number, population: number, seed: number);
    tick(dt_seconds: number): void;
    reset(seed: number): void;
    world_size(): number;
    population(): number;
    births(): number;
    deaths(): number;
    sim_steps(): number;
    generation(): number;
    agent_ptr(): number;
    agent_f32_len(): number;
    agent_stride_f32(): number;
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
