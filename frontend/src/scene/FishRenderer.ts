import * as THREE from "three";
import { FISH_STRIDE } from "../sim/frame";
import { createFishGeometry } from "./fishGeometry";

const FORWARD = new THREE.Vector3(0, 0, 1);
const GROUP_PALETTE = [0x4fd1ff, 0x7ef0c8, 0x9ab8ff, 0x5ce1e6, 0xa0f0ff, 0x6fa8ff, 0x8ae6b0, 0xb5d8ff];

/**
 * Draws every fish with one InstancedMesh. Slots map 1:1 to simulation slots; eaten fish are
 * hidden with a zero-scale matrix so instance colours never need to be re-packed.
 */
export class FishRenderer {
  mesh: THREE.InstancedMesh;
  private readonly geometry = createFishGeometry();
  private readonly material = new THREE.MeshStandardMaterial({ roughness: 0.5, metalness: 0.1 });
  // Scratch objects reused every frame (no per-frame allocation).
  private readonly dir = new THREE.Vector3();
  private readonly pos = new THREE.Vector3();
  private readonly quat = new THREE.Quaternion();
  private readonly scale = new THREE.Vector3(1, 1, 1);
  private readonly hidden = new THREE.Matrix4().makeScale(0, 0, 0);
  private readonly matrix = new THREE.Matrix4();
  private coloredSlots = 0;

  constructor(private readonly scene: THREE.Scene, capacity: number, private readonly groupCount = 16) {
    this.mesh = this.createMesh(capacity);
  }

  get capacity(): number {
    return this.mesh.instanceMatrix.count;
  }

  /** Updates instance matrices from a packed frame (see `FISH_STRIDE`). */
  update(frame: Float32Array, count: number): void {
    this.ensureCapacity(count);
    this.ensureColors(count);
    for (let i = 0; i < count; i++) this.writeInstance(frame, i);
    this.mesh.count = count;
    this.mesh.instanceMatrix.needsUpdate = true;
  }

  private writeInstance(frame: Float32Array, i: number): void {
    const f = i * FISH_STRIDE;
    if (frame[f + 6] === 0) {
      this.mesh.setMatrixAt(i, this.hidden);
      return;
    }
    this.pos.set(frame[f], frame[f + 1], frame[f + 2]);
    this.dir.set(frame[f + 3], frame[f + 4], frame[f + 5]);
    if (this.dir.lengthSq() > 1e-8) this.quat.setFromUnitVectors(FORWARD, this.dir.normalize());
    this.matrix.compose(this.pos, this.quat, this.scale);
    this.mesh.setMatrixAt(i, this.matrix);
  }

  private ensureCapacity(count: number): void {
    if (count <= this.capacity) return;
    this.scene.remove(this.mesh);
    this.mesh.dispose();
    this.mesh = this.createMesh(Math.ceil(count * 1.5));
    this.coloredSlots = 0;
  }

  /** Colours are written only for newly used slots (group = slot % groupCount, as in the sim). */
  private ensureColors(count: number): void {
    if (count <= this.coloredSlots) return;
    const color = new THREE.Color();
    for (let i = this.coloredSlots; i < count; i++) {
      color.setHex(GROUP_PALETTE[(i % this.groupCount) % GROUP_PALETTE.length]);
      this.mesh.setColorAt(i, color);
    }
    this.coloredSlots = count;
    if (this.mesh.instanceColor) this.mesh.instanceColor.needsUpdate = true;
  }

  private createMesh(capacity: number): THREE.InstancedMesh {
    const mesh = new THREE.InstancedMesh(this.geometry, this.material, capacity);
    mesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
    mesh.frustumCulled = false; // instances span the whole world
    mesh.count = 0;
    this.scene.add(mesh);
    return mesh;
  }
}
