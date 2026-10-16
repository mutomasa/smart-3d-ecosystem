export interface HudMetrics {
  survivalRate: number;
  foodCollected: number;
  meanEnergy: number;
  dispersion: number;
}

export interface HudState {
  fps: number;
  agents: number;
  alive: number;
  mode: string;
  stepMs: number;
  dt: number;
  metrics?: HudMetrics;
  extra?: string[];
}

/** Formats the HUD text; kept pure for testing. */
export function formatHud(s: HudState): string {
  const lines = [
    "SMART 3D ECOSYSTEM",
    `FPS ${s.fps.toFixed(0).padStart(3)} | Agents ${s.alive}/${s.agents} | Mode ${s.mode}`,
    `Step ${s.stepMs.toFixed(2)} ms | dt 1/${Math.round(1 / s.dt)} s`,
    ...(s.metrics ? [formatMetrics(s.metrics)] : []),
    ...(s.extra ?? []),
  ];
  return lines.join("\n");
}

export class Hud {
  constructor(private readonly el: HTMLElement) {}

  update(s: HudState): void {
    this.el.textContent = formatHud(s);
  }
}

function formatMetrics(m: HudMetrics): string {
  return `Survival ${(m.survivalRate * 100).toFixed(1)}% | Food ${m.foodCollected.toFixed(1)} | Energy ${m.meanEnergy.toFixed(2)} | Disp ${m.dispersion.toFixed(1)}`;
}
