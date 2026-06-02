// Shared GPU buffer layout contracts used by simulation, renderer, and tests.

export const AGENT_F32 = 10;
export const AGENT_BYTES = AGENT_F32 * 4;
export const DENSE_BYTES = 16;
export const DRAW_INDIRECT_BYTES = 16;
export const SIM_PARAMS_BYTES = 64;

export const AGENT_STRUCT_WGSL = /* wgsl */ `
struct Agent {
  pos: vec2f,
  dir: f32,
  vel: f32,
  hue: f32,
  sat: f32,
  val: f32,
  alive: u32,
  id: u32,
}
`;

export const DENSE_STRUCT_WGSL = /* wgsl */ `
struct Dense {
  pos: vec2f,
  slot: u32,
  pad: u32,
}
`;
