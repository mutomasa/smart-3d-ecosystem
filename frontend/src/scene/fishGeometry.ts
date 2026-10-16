import * as THREE from "three";

/** Low-poly fish pointing along +Z (no GLTF dependency): diamond body + forked tail. */
export function createFishGeometry(length = 1.2): THREE.BufferGeometry {
  const L = length, W = length * 0.18, H = length * 0.28;
  const nose: V = [0, 0, L * 0.5];
  const tailRoot: V = [0, 0, -L * 0.3];
  const top: V = [0, H, 0.05 * L];
  const bottom: V = [0, -H * 0.8, 0.05 * L];
  const left: V = [-W, 0, 0.05 * L];
  const right: V = [W, 0, 0.05 * L];
  const tailTop: V = [0, H * 0.9, -L * 0.55];
  const tailBottom: V = [0, -H * 0.9, -L * 0.55];
  const tris: V[] = [
    nose, top, right, nose, right, bottom, nose, bottom, left, nose, left, top,
    tailRoot, right, top, tailRoot, bottom, right, tailRoot, left, bottom, tailRoot, top, left,
    tailRoot, tailTop, tailBottom, tailRoot, tailBottom, tailTop,
  ];
  const geo = new THREE.BufferGeometry();
  geo.setAttribute("position", new THREE.Float32BufferAttribute(tris.flat(), 3));
  geo.computeVertexNormals();
  return geo;
}

type V = [number, number, number];
