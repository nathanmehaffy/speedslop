// Shared runtime constants for the torus agent simulation and its renderer.

export const TELEMETRY_SAMPLE_MS = 500;
export const MAX_DEVICE_PIXEL_RATIO = 2;

// Simulation ---------------------------------------------------------------

// Fixed agent capacity. Every buffer is sized for this many slots; population
// floats below it through collision deaths and childbirth.
export const MAX_AGENTS = 10_000;

// Spatial grid resolution per axis. GRID_DIM^2 cells tile the world; sized so
// cell width stays stable when WORLD_SIZE changes.
export const GRID_DIM = 256;

// Torus world extent per axis (16x the original unit-square area).
export const WORLD_SIZE = 4;

// Minimum population before collision deaths are offset by random immigrants.
export const POPULATION_FLOOR = MAX_AGENTS / 2;

// Initial random population, seeded at the floor.
export const INITIAL_AGENTS = POPULATION_FLOOR;
export const STEP_DT = 1.0;

// Collision ---------------------------------------------------------------

// Hitboxes are circles in world space. The radius and max speed are sized so a
// one-cell broadphase neighborhood is enough at the default grid resolution.
export const AGENT_HIT_RADIUS = 0.003;
export const CONTACT_DOT = 0.15;
export const HEAD_ON_DOT = 0.65;

// Neural control -----------------------------------------------------------

export const NEURAL_NEIGHBORS = 4;
export const NEURAL_NEIGHBOR_INPUTS = 6;
export const NEURAL_SELF_INPUTS = 3;
export const NEURAL_INPUTS = NEURAL_SELF_INPUTS + NEURAL_NEIGHBORS * NEURAL_NEIGHBOR_INPUTS;
export const NEURAL_HIDDEN = 8;
export const NEURAL_OUTPUTS = 2;
export const BRAIN_WEIGHT_COUNT = NEURAL_INPUTS * NEURAL_HIDDEN + NEURAL_HIDDEN * NEURAL_OUTPUTS;

export const AGENT_MIN_SPEED = 0.0005;
export const AGENT_MAX_SPEED = 0.004;
export const AGENT_MAX_TURN = 0.2;
export const SENSOR_RADIUS = 0.035;

// Genetic mutation tuning.
export const MUTATION_RATE = 0.03;
export const MUTATION_SCALE = 0.12;
export const MUTATION_WEIGHT_LIMIT = 2.5;
export const SPEED_MUTATION_SCALE = 0.0004;
export const HUE_MUTATION_SCALE = 0.04;

// Rendering ----------------------------------------------------------------

// Agent triangle size in world units (so it scales with zoom).
export const AGENT_TRIANGLE_SIZE = 0.004;

// Grey border drawn around each tiled copy of the world square.
export const BORDER_COLOR: [number, number, number, number] = [0.5, 0.5, 0.5, 1];
export const CLEAR_COLOR = { r: 0.02, g: 0.03, b: 0.06, a: 1 };

// Camera zoom limits relative to the fit-world reference (see Camera.fitWorld).
export const ZOOM_OUT_LIMIT = 2; // max zoom-out as a multiple of the fit zoom
export const ZOOM_IN_LIMIT = 20; // max zoom-in as a multiple of the fit zoom
export const ZOOM_SENSITIVITY = 0.0015;
