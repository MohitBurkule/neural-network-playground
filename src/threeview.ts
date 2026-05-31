// THREE is loaded as a global via <script> tag in the HTML page.
declare const THREE: any;

export interface Point3D {
  x: number;
  y: number;
  z: number;
  label: number;
}

export class ThreeView {
  private renderer: any;
  private scene: any;
  private camera: any;
  private rotGroup: any;
  private dataPointsMesh: any = null;
  private boundaryMesh: any = null;

  private rotX = 0.3;
  private rotY = 0.5;
  private isDragging = false;
  private prevMouse = { x: 0, y: 0 };

  private controlsEnabled = false;

  // Bound event handlers stored so they can be removed on dispose
  private _onMouseDown: (e: MouseEvent) => void;
  private _onMouseUp: () => void;
  private _onMouseMove: (e: MouseEvent) => void;
  private _onTouchStart: (e: TouchEvent) => void;
  private _onTouchEnd: () => void;
  private _onTouchMove: (e: TouchEvent) => void;
  private _onWheel: (e: WheelEvent) => void;

  private static readonly POS_COLOR = { r: 1.0, g: 0.302, b: 0.427 }; // 0xff4d6d pink
  private static readonly NEG_COLOR = { r: 0.302, g: 0.624, b: 1.0 };  // 0x4d9fff blue

  constructor(container: HTMLElement, width: number, height: number) {
    // Renderer
    this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    this.renderer.setPixelRatio(window.devicePixelRatio);
    this.renderer.setSize(width, height);
    container.appendChild(this.renderer.domElement);

    // Scene
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0x1a1a2e);

    // Camera
    this.camera = new THREE.PerspectiveCamera(50, width / height, 0.1, 100);
    this.camera.position.set(0, 0, 14);

    // Lights
    const ambient = new THREE.AmbientLight(0xffffff, 0.6);
    this.scene.add(ambient);
    const dirLight = new THREE.DirectionalLight(0xffffff, 0.8);
    dirLight.position.set(5, 10, 7);
    this.scene.add(dirLight);

    // Grid
    const gridHelper = new THREE.GridHelper(12, 12, 0x333355, 0x222244);
    this.scene.add(gridHelper);

    // Rotation group
    this.rotGroup = new THREE.Group();
    this.scene.add(this.rotGroup);
    this._applyRotation();

    // Pre-bind handlers
    this._onMouseDown = (e: MouseEvent) => {
      this.isDragging = true;
      this.prevMouse = { x: e.clientX, y: e.clientY };
    };
    this._onMouseUp = () => { this.isDragging = false; };
    this._onMouseMove = (e: MouseEvent) => {
      if (!this.isDragging) return;
      this.rotY += (e.clientX - this.prevMouse.x) * 0.01;
      this.rotX += (e.clientY - this.prevMouse.y) * 0.01;
      this._applyRotation();
      this.prevMouse = { x: e.clientX, y: e.clientY };
    };
    this._onTouchStart = (e: TouchEvent) => {
      this.isDragging = true;
      this.prevMouse = { x: e.touches[0].clientX, y: e.touches[0].clientY };
    };
    this._onTouchEnd = () => { this.isDragging = false; };
    this._onTouchMove = (e: TouchEvent) => {
      if (!this.isDragging) return;
      this.rotY += (e.touches[0].clientX - this.prevMouse.x) * 0.01;
      this.rotX += (e.touches[0].clientY - this.prevMouse.y) * 0.01;
      this._applyRotation();
      this.prevMouse = { x: e.touches[0].clientX, y: e.touches[0].clientY };
    };
    this._onWheel = (e: WheelEvent) => {
      this.camera.position.z = Math.max(4, Math.min(30, this.camera.position.z + e.deltaY * 0.02));
      e.preventDefault();
    };
  }

  private _applyRotation(): void {
    this.rotGroup.rotation.x = this.rotX;
    this.rotGroup.rotation.y = this.rotY;
  }

  setDataPoints(points: Point3D[]): void {
    if (this.dataPointsMesh) {
      this.rotGroup.remove(this.dataPointsMesh);
      this.dataPointsMesh.geometry.dispose();
      this.dataPointsMesh.material.dispose();
      this.dataPointsMesh = null;
    }
    if (points.length === 0) return;

    const positions: number[] = [];
    const colors: number[] = [];
    const pos = ThreeView.POS_COLOR;
    const neg = ThreeView.NEG_COLOR;

    for (const pt of points) {
      positions.push(pt.x, pt.y, pt.z);
      const c = pt.label >= 0 ? pos : neg;
      colors.push(c.r, c.g, c.b);
    }

    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
    geo.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));

    const mat = new THREE.PointsMaterial({
      size: 0.18,
      vertexColors: true,
      sizeAttenuation: true
    });

    this.dataPointsMesh = new THREE.Points(geo, mat);
    this.rotGroup.add(this.dataPointsMesh);
  }

  setBoundary(voxels: { x: number; y: number; z: number; value: number }[]): void {
    if (this.boundaryMesh) {
      this.rotGroup.remove(this.boundaryMesh);
      this.boundaryMesh.geometry.dispose();
      this.boundaryMesh.material.dispose();
      this.boundaryMesh = null;
    }
    if (voxels.length === 0) return;

    const positions: number[] = [];
    const colors: number[] = [];
    const opacities: number[] = [];
    const pos = ThreeView.POS_COLOR;
    const neg = ThreeView.NEG_COLOR;

    for (const v of voxels) {
      positions.push(v.x, v.y, v.z);
      // Interpolate color by sign
      const t = (v.value + 1) / 2; // 0..1
      const r = pos.r * t + neg.r * (1 - t);
      const g = pos.g * t + neg.g * (1 - t);
      const b = pos.b * t + neg.b * (1 - t);
      colors.push(r, g, b);
      // Fade opacity by |value| so boundary (value near 0) is most visible
      const conf = Math.abs(v.value);
      opacities.push(Math.max(0.08, 0.55 * (1 - conf)));
    }

    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
    geo.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));

    // PointsMaterial doesn't support per-point opacity; use a moderate global opacity
    // weighted toward the boundary (host should pre-filter if needed).
    const mat = new THREE.PointsMaterial({
      size: 0.30,
      vertexColors: true,
      sizeAttenuation: true,
      transparent: true,
      opacity: 0.45,
      depthWrite: false
    });

    this.boundaryMesh = new THREE.Points(geo, mat);
    this.rotGroup.add(this.boundaryMesh);
  }

  render(): void {
    this.renderer.render(this.scene, this.camera);
  }

  enableControls(): void {
    if (this.controlsEnabled) return;
    this.controlsEnabled = true;
    const canvas = this.renderer.domElement as HTMLCanvasElement;
    canvas.addEventListener('mousedown', this._onMouseDown);
    window.addEventListener('mouseup', this._onMouseUp);
    window.addEventListener('mousemove', this._onMouseMove);
    canvas.addEventListener('touchstart', this._onTouchStart);
    window.addEventListener('touchend', this._onTouchEnd);
    window.addEventListener('touchmove', this._onTouchMove);
    canvas.addEventListener('wheel', this._onWheel, { passive: false });
  }

  resize(width: number, height: number): void {
    this.camera.aspect = width / height;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(width, height);
  }

  dispose(): void {
    if (this.controlsEnabled) {
      const canvas = this.renderer.domElement as HTMLCanvasElement;
      canvas.removeEventListener('mousedown', this._onMouseDown);
      window.removeEventListener('mouseup', this._onMouseUp);
      window.removeEventListener('mousemove', this._onMouseMove);
      canvas.removeEventListener('touchstart', this._onTouchStart);
      window.removeEventListener('touchend', this._onTouchEnd);
      window.removeEventListener('touchmove', this._onTouchMove);
      canvas.removeEventListener('wheel', this._onWheel);
      this.controlsEnabled = false;
    }

    if (this.dataPointsMesh) {
      this.rotGroup.remove(this.dataPointsMesh);
      this.dataPointsMesh.geometry.dispose();
      this.dataPointsMesh.material.dispose();
      this.dataPointsMesh = null;
    }
    if (this.boundaryMesh) {
      this.rotGroup.remove(this.boundaryMesh);
      this.boundaryMesh.geometry.dispose();
      this.boundaryMesh.material.dispose();
      this.boundaryMesh = null;
    }

    this.renderer.dispose();
    const canvas = this.renderer.domElement as HTMLCanvasElement;
    if (canvas.parentNode) {
      canvas.parentNode.removeChild(canvas);
    }
  }
}
