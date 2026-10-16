import * as THREE from "three";
import { MAX_FOOD } from "../sim/food";
import { MAX_PREDATORS } from "../sim/predator";
import { createFishGeometry } from "./fishGeometry";

const FORWARD = new THREE.Vector3(0, 0, 1);

/** Large red predators (stride 6: x,y,z,vx,vy,vz). */
export class PredatorRenderer {
  readonly mesh: THREE.InstancedMesh;
  private readonly m = new THREE.Matrix4();
  private readonly p = new THREE.Vector3();
  private readonly d = new THREE.Vector3();
  private readonly q = new THREE.Quaternion();
  private readonly s = new THREE.Vector3(1, 1, 1);

  constructor(scene: THREE.Scene) {
    const material = new THREE.MeshStandardMaterial({ color: 0xff4d4d, emissive: 0x5a0000, roughness: 0.4 });
    this.mesh = new THREE.InstancedMesh(createFishGeometry(4.5), material, MAX_PREDATORS);
    this.mesh.count = 0;
    this.mesh.frustumCulled = false;
    scene.add(this.mesh);
  }

  update(data: Float32Array): void {
    const n = Math.min(MAX_PREDATORS, Math.floor(data.length / 6));
    for (let k = 0; k < n; k++) {
      this.p.fromArray(data, k * 6);
      this.d.fromArray(data, k * 6 + 3);
      if (this.d.lengthSq() > 1e-8) this.q.setFromUnitVectors(FORWARD, this.d.normalize());
      this.mesh.setMatrixAt(k, this.m.compose(this.p, this.q, this.s));
    }
    this.mesh.count = n;
    this.mesh.instanceMatrix.needsUpdate = true;
  }
}

/** Glowing food patches (stride 4: x,y,z,amount); size follows the remaining amount. */
export class FoodRenderer {
  readonly mesh: THREE.InstancedMesh;
  private readonly m = new THREE.Matrix4();
  private readonly p = new THREE.Vector3();
  private readonly q = new THREE.Quaternion();
  private readonly s = new THREE.Vector3();

  constructor(scene: THREE.Scene, private readonly fullAmount = 40) {
    const material = new THREE.MeshBasicMaterial({ color: 0xc8ff6a, transparent: true, opacity: 0.85 });
    this.mesh = new THREE.InstancedMesh(new THREE.IcosahedronGeometry(1.6, 1), material, MAX_FOOD);
    this.mesh.count = 0;
    this.mesh.frustumCulled = false;
    scene.add(this.mesh);
  }

  update(data: Float32Array): void {
    const n = Math.min(MAX_FOOD, Math.floor(data.length / 4));
    for (let k = 0; k < n; k++) {
      this.p.fromArray(data, k * 4);
      const r = data[k * 4 + 3] > 0 ? 0.35 + Math.sqrt(data[k * 4 + 3] / this.fullAmount) : 0;
      this.s.setScalar(r);
      this.mesh.setMatrixAt(k, this.m.compose(this.p, this.q, this.s));
    }
    this.mesh.count = n;
    this.mesh.instanceMatrix.needsUpdate = true;
  }
}
