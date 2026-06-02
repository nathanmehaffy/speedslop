// Shared runtime constants for the torus agent simulation and its renderer.

export const TELEMETRY_SAMPLE_MS = 500;
export const MAX_DEVICE_PIXEL_RATIO = 2;

export const CONTROLLER_WARMUP_FRAMES = 2;

// Simulation ---------------------------------------------------------------

// Fixed agent capacity. Every buffer is sized for this many slots; population
// floats below it (see population sine params).
export const MAX_AGENTS = 10_000;

// Number of closest neighbours each agent senses. Drives the sensory buffer
// stride and the future brain input layout.
export const N_NEIGHBORS = 8;

// Spatial grid resolution per axis. GRID_DIM^2 cells tile the unit world; sized
// so each cell holds a few agents on average for cheap k-nearest queries.
export const GRID_DIM = 64;

// The world is the unit square, wrapped at the edges (a torus).
export const WORLD_SIZE = 1;

// Population sine wave: live count hovers around POPULATION_MID and swings by
// POPULATION_AMPLITUDE with angular frequency POPULATION_OMEGA (radians/step).
export const POPULATION_MID = MAX_AGENTS / 2;
export const POPULATION_AMPLITUDE = MAX_AGENTS * 0.25;
export const POPULATION_OMEGA = 0.001;

// Per-step random-walk magnitudes for the placeholder (no brain yet).
export const HEADING_JITTER = 0.15; // radians of random turn per step
export const HUE_DRIFT = 0.002; // hue increment per step (rainbow cycling)
export const STEP_DT = 1.0;

// Rendering ----------------------------------------------------------------

// Agent triangle size in world units (so it scales with zoom).
export const AGENT_TRIANGLE_SIZE = 0.004;

// Grey border drawn around each tiled copy of the world square.
export const BORDER_COLOR: [number, number, number, number] = [0.5, 0.5, 0.5, 1];
export const CLEAR_COLOR = { r: 0.02, g: 0.03, b: 0.06, a: 1 };

// Camera zoom limits in pixels-per-world-unit and wheel sensitivity.
export const MIN_ZOOM = 32;
export const MAX_ZOOM = 100_000;
export const ZOOM_SENSITIVITY = 0.0015;
