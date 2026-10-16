export interface PolicyInfo {
  name: string;
  available: boolean;
  kind: string;
  model_version: string;
  label: string;
}

/** Fetches the human-readable label of the server-side ML policy (shown in the HUD). */
export async function fetchMlLabel(fetcher: typeof fetch = fetch): Promise<string> {
  try {
    const policies = (await (await fetcher("api/policies")).json()) as PolicyInfo[];
    const ml = policies.find((p) => p.name === "ml");
    return ml?.available ? ml.label : "ML (not loaded)";
  } catch {
    return "ML (backend unreachable)";
  }
}
