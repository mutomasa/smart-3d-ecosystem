import { describe, expect, it } from "vitest";
import cases from "../../shared/fixtures/rules_cases.json";
import { rulesAction } from "../src/sim/policies";
import { GroupStrategies } from "../src/sim/strategy";

/** The TS rules must match the Python RulesPolicy (backend/policy.py) on the shared fixture. */
describe("rules parity with Python", () => {
  for (const c of cases) {
    it(c.name, () => {
      const [dx, dy, dz, ...w] = rulesAction(c.features);
      const s = new GroupStrategies(1);
      s.setTarget(0, dx, dy, dz, w[0], w[1], w[2]);
      const got = [...s.targetDir, ...s.targetW];
      got.forEach((v, k) => expect(v).toBeCloseTo(c.expected[k], 5));
    });
  }
});
