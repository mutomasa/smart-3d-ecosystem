export interface BoidsWeights {
  separation: number;
  alignment: number;
  cohesion: number;
  boundary: number;
  predator: number;
  food: number;
  ml: number;
}

export interface SimConfig {
  count: number;
  seed: number;
  /** Fixed simulation time step in seconds. */
  dt: number;
  /** World is the box [-h, h] on each axis. */
  halfExtents: [number, number, number];
  perceptionRadius: number;
  separationRadius: number;
  collisionRadius: number;
  maxSpeed: number;
  minSpeed: number;
  maxForce: number;
  /** Maximum heading change in radians per second. */
  maxTurnRate: number;
  boundaryMargin: number;
  maxNeighbors: number;
  groupCount: number;
  weights: BoidsWeights;
}

export const DEFAULT_WEIGHTS: BoidsWeights = {
  separation: 1.6,
  alignment: 1.0,
  cohesion: 0.8,
  boundary: 2.5,
  predator: 3.0,
  food: 1.0,
  ml: 1.0,
};

export const DEFAULT_CONFIG: SimConfig = {
  count: 1000,
  seed: 42,
  dt: 1 / 60,
  halfExtents: [50, 25, 50],
  perceptionRadius: 4,
  separationRadius: 1.6,
  collisionRadius: 0.3,
  maxSpeed: 10,
  minSpeed: 3,
  maxForce: 25,
  maxTurnRate: Math.PI * 1.5,
  boundaryMargin: 8,
  maxNeighbors: 48,
  groupCount: 16,
  weights: { ...DEFAULT_WEIGHTS },
};

export function makeConfig(overrides: Partial<SimConfig> = {}): SimConfig {
  return {
    ...DEFAULT_CONFIG,
    ...overrides,
    weights: { ...DEFAULT_CONFIG.weights, ...(overrides.weights ?? {}) },
  };
}
