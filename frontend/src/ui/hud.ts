export interface HudState {
  fps: number;
  agents: number;
  alive: number;
  mode: string;
  stepMs: number;
  dt: number;
  extra?: string[];
}

/** Formats the HUD text; kept pure for testing. */
export function formatHud(s: HudState): string {
  const lines = [
    "SMART 3D ECOSYSTEM",
    `FPS ${s.fps.toFixed(0).padStart(3)} | Agents ${s.alive}/${s.agents} | Mode ${s.mode}`,
    `Step ${s.stepMs.toFixed(2)} ms | dt 1/${Math.round(1 / s.dt)} s`,
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
