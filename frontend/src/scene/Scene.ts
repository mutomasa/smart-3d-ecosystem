import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

const BACKGROUND = 0x030b1a;

/** Underwater scene: renderer, camera, fog, lights, orbit controls and the world bounds. */
export class AquariumScene {
  readonly scene = new THREE.Scene();
  readonly camera: THREE.PerspectiveCamera;
  readonly renderer: THREE.WebGLRenderer;
  readonly controls: OrbitControls;

  constructor(container: HTMLElement, halfExtents: [number, number, number]) {
    this.renderer = createRenderer(container);
    this.camera = new THREE.PerspectiveCamera(55, 1, 0.1, 1000);
    this.camera.position.set(0, 35, 120);
    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.setupEnvironment(halfExtents);
    this.resize(container);
    window.addEventListener("resize", () => this.resize(container));
  }

  render(): void {
    this.controls.update();
    this.renderer.render(this.scene, this.camera);
  }

  /** Reports which GPU the browser renders with (WebGL runs on the viewer's machine). */
  gpuName(): string {
    const gl = this.renderer.getContext();
    const ext = gl.getExtension("WEBGL_debug_renderer_info");
    return ext ? String(gl.getParameter(ext.UNMASKED_RENDERER_WEBGL)) : "unknown";
  }

  private setupEnvironment(halfExtents: [number, number, number]): void {
    this.scene.background = new THREE.Color(BACKGROUND);
    this.scene.fog = new THREE.Fog(BACKGROUND, 60, 260);
    this.scene.add(new THREE.HemisphereLight(0x6fc8ff, 0x02122b, 1.1));
    const sun = new THREE.DirectionalLight(0x9fe7ff, 1.4);
    sun.position.set(20, 80, 30);
    this.scene.add(sun);
    this.scene.add(boundsBox(halfExtents));
  }

  private resize(container: HTMLElement): void {
    const w = container.clientWidth || window.innerWidth;
    const h = container.clientHeight || window.innerHeight;
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(w, h);
  }
}

function createRenderer(container: HTMLElement): THREE.WebGLRenderer {
  const renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: "high-performance" });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  container.appendChild(renderer.domElement);
  return renderer;
}

function boundsBox([hx, hy, hz]: [number, number, number]): THREE.LineSegments {
  const geo = new THREE.EdgesGeometry(new THREE.BoxGeometry(hx * 2, hy * 2, hz * 2));
  const mat = new THREE.LineBasicMaterial({ color: 0x1b4f7a, transparent: true, opacity: 0.45 });
  return new THREE.LineSegments(geo, mat);
}
