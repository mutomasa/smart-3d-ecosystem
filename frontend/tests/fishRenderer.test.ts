import { describe, expect, it } from "vitest";
import * as THREE from "three";
import { FishRenderer } from "../src/scene/FishRenderer";
import { FISH_STRIDE } from "../src/sim/frame";

function frame(count: number, deadIndex = -1): Float32Array {
  const f = new Float32Array(count * FISH_STRIDE);
  for (let i = 0; i < count; i++) {
    f.set([i, 0, 0, 1, 0, 0, i === deadIndex ? 0 : 1], i * FISH_STRIDE);
  }
  return f;
}

describe("FishRenderer", () => {
  it("grows the InstancedMesh when the population exceeds capacity", () => {
    const scene = new THREE.Scene();
    const r = new FishRenderer(scene, 10);
    r.update(frame(50), 50);
    expect(r.capacity).toBeGreaterThanOrEqual(50);
    expect(r.mesh.count).toBe(50);
    expect(scene.children.filter((c) => c instanceof THREE.InstancedMesh)).toHaveLength(1);
  });

  it("shrinks the drawn count without reallocating", () => {
    const r = new FishRenderer(new THREE.Scene(), 100);
    r.update(frame(100), 100);
    const mesh = r.mesh;
    r.update(frame(20), 20);
    expect(r.mesh).toBe(mesh);
    expect(r.mesh.count).toBe(20);
  });

  it("hides dead fish with a zero-scale matrix and orients live fish along velocity", () => {
    const r = new FishRenderer(new THREE.Scene(), 4);
    r.update(frame(4, 2), 4);
    const m = new THREE.Matrix4();
    const s = new THREE.Vector3(), p = new THREE.Vector3(), q = new THREE.Quaternion();
    r.mesh.getMatrixAt(2, m);
    expect(m.determinant()).toBe(0); // zero-scale => invisible (decompose() reports 1 for singular matrices)
    expect(m.elements[0] + m.elements[5] + m.elements[10]).toBe(0);
    r.mesh.getMatrixAt(1, m);
    m.decompose(p, q, s);
    expect(p.x).toBeCloseTo(1);
    const fwd = new THREE.Vector3(0, 0, 1).applyQuaternion(q);
    expect(fwd.x).toBeCloseTo(1);
  });
});
