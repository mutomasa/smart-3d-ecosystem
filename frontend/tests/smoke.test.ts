import { describe, expect, it } from "vitest";
import schema from "../../shared/schemas/policy.json";

describe("shared schema", () => {
  it("defines 20 observation features and 6 action outputs", () => {
    expect(schema.observation.features).toHaveLength(20);
    expect(schema.action.outputs).toHaveLength(6);
  });
});
