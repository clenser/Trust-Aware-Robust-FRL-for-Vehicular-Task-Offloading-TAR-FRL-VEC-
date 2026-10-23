/**
 * Ultra-Smooth 60 FPS 3D WebGL VANET Simulation Engine (Three.js)
 * Highlights:
 * 1. 100% Fluid Real-World Highway Motion:
 *    - Continuous frame-by-frame kinematics (60 FPS) with zero snapping or rubberbanding
 *    - Vehicles cruise smoothly along lanes with realistic speed and wheel rotation
 * 2. High-Visibility Task Offloading Demonstration:
 *    - V2I (Vehicle to Tower): Bright Cyan/Emerald laser beams connecting car roof to RSU dish
 *    - V2V (Vehicle to Vehicle): Luminous Amber/Gold laser beams connecting peer cars directly
 *    - Dynamic tracking: Beams follow moving vehicles in real-time
 *    - Animated photon packet streams traveling along each beam
 *    - Floating 3D HUD badges on beams displaying: "[V2I] RSU-1 (35ms)" or "[V2V] V-4 Peer"
 *    - Generous 1.5s visible duration so offloading is clearly seen and understood
 * 3. 2400m Extended Highway with Collision-Free IDM Spacing
 * 4. 1-Click Day / Night atmosphere toggle with realistic lighting
 */

class VANET3DSimulation {
  constructor(containerId) {
    this.container = document.getElementById(containerId);
    this.scene = null;
    this.camera = null;
    this.renderer = null;
    this.controls = null;

    // Extended 3200m Highway: 1600 Three.js units (-800 to +800)
    this.highwayLengthM = 3200.0;
    this.scaleFactor = 0.5;
    this.roadLength = this.highwayLengthM * this.scaleFactor; // 1600.0 units
    this.laneWidth = 3.75 * 2.2;                 // 8.25 units
    this.numLanes = 5;
    this.totalRoadWidth = this.numLanes * this.laneWidth;

    this.vehiclesMap = new Map();
    this.rsusMap = new Map();
    this.activeBeams = [];
    this.activeShockwaves = [];

    this.cameraMode = 'orbit'; // 'orbit', 'chase', 'cockpit', 'drone', 'top'
    this.chaseTargetId = 1;
    this.droneProgress = 0;

    this.isDayMode = true;
    this.sunLight = null;
    this.ambientLight = null;
    this.hemiLight = null;
    this.lampMeshes = [];
    this.lightPuddles = [];

    this.shared = {};
    this.clock = new THREE.Clock();

    this.init();
    this.animate();
  }

  init() {
    this.scene = new THREE.Scene();

    this.camera = new THREE.PerspectiveCamera(
      48,
      this.container.clientWidth / this.container.clientHeight,
      0.5,
      3500
    );
    this.camera.position.set(-80, 95, 180);

    this.renderer = new THREE.WebGLRenderer({
      antialias: true,
      powerPreference: 'high-performance'
    });
    this.renderer.setSize(this.container.clientWidth, this.container.clientHeight);
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 1.5));
    this.renderer.shadowMap.enabled = true;
    this.renderer.shadowMap.type = THREE.BasicShadowMap;
    this.container.appendChild(this.renderer.domElement);

    if (typeof THREE.OrbitControls !== 'undefined') {
      this.controls = new THREE.OrbitControls(this.camera, this.renderer.domElement);
      this.controls.enableDamping = true;
      this.controls.dampingFactor = 0.08;
      this.controls.maxPolarAngle = Math.PI / 2.05;
      this.controls.target.set(0, 0, 0);
    }

    this.initSharedAssets();
    this.setupLighting();
    this.applyEnvironmentMode(true); // Default: Daylight

    this.buildTerrain();
    this.buildDistantCity();
    this.buildHighway();
    this.buildStreetlights();
    this.buildHighwayGantries();
    this.buildRSUs();

    window.addEventListener('resize', () => this.onWindowResize());
  }

  initSharedAssets() {
    this.shared.asphaltTexture = this.createAsphaltTexture();
    this.shared.barrierTexture = this.createConcreteBarrierTexture();
    this.shared.lightPuddleTexture = this.createLightPuddleTexture();

    // Wheels
    this.shared.tireGeoCar = new THREE.CylinderGeometry(0.48, 0.48, 0.45, 14);
    this.shared.tireGeoCar.rotateX(Math.PI / 2);
    this.shared.rimGeoCar = new THREE.CylinderGeometry(0.32, 0.32, 0.48, 8);
    this.shared.rimGeoCar.rotateX(Math.PI / 2);

    this.shared.tireGeoTruck = new THREE.CylinderGeometry(0.72, 0.72, 0.5, 14);
    this.shared.tireGeoTruck.rotateX(Math.PI / 2);
    this.shared.rimGeoTruck = new THREE.CylinderGeometry(0.48, 0.48, 0.52, 8);
    this.shared.rimGeoTruck.rotateX(Math.PI / 2);

    this.shared.tireMat = new THREE.MeshLambertMaterial({ color: 0x111827 });
    this.shared.rimMat = new THREE.MeshLambertMaterial({ color: 0xd1d5db });
    this.shared.glassMat = new THREE.MeshLambertMaterial({ color: 0x0a1128, transparent: true, opacity: 0.85 });
    this.shared.headlightLensMat = new THREE.MeshBasicMaterial({ color: 0xffffff });
    this.shared.taillightLensMat = new THREE.MeshBasicMaterial({ color: 0xff1744 });

    // Holograms
    this.shared.outerRingGeo = new THREE.RingGeometry(1.4, 1.6, 24);
    this.shared.innerRingGeo = new THREE.RingGeometry(0.9, 1.1, 6);
  }

  setupLighting() {
    this.ambientLight = new THREE.AmbientLight(0x1a2639, 1.0);
    this.scene.add(this.ambientLight);

    this.hemiLight = new THREE.HemisphereLight(0x7090b0, 0x141e28, 0.8);
    this.scene.add(this.hemiLight);

    this.sunLight = new THREE.DirectionalLight(0xaadcff, 1.4);
    this.sunLight.position.set(100, 220, 80);
    this.sunLight.castShadow = true;
    this.sunLight.shadow.mapSize.width = 1024;
    this.sunLight.shadow.mapSize.height = 1024;
    this.sunLight.shadow.camera.near = 10;
    this.sunLight.shadow.camera.far = 600;
    this.sunLight.shadow.camera.left = -400;
    this.sunLight.shadow.camera.right = 400;
    this.sunLight.shadow.camera.top = 200;
    this.sunLight.shadow.camera.bottom = -200;
    this.scene.add(this.sunLight);
  }

  applyEnvironmentMode(isDay) {
    this.isDayMode = isDay;

    if (isDay) {
      this.scene.background = new THREE.Color(0x87ceeb);
      this.scene.fog = new THREE.FogExp2(0x87ceeb, 0.0004);
      this.ambientLight.color.setHex(0xffffff);
      this.ambientLight.intensity = 1.3;
      this.hemiLight.color.setHex(0x9bd4f5);
      this.hemiLight.groundColor.setHex(0x3e5229);
      this.hemiLight.intensity = 1.0;
      this.sunLight.color.setHex(0xfff6db);
      this.sunLight.intensity = 2.2;
      this.sunLight.position.set(160, 280, 120);

      this.lightPuddles.forEach(p => p.visible = false);
      this.lampMeshes.forEach(l => l.material.color.setHex(0x64748b));
    } else {
      this.scene.background = new THREE.Color(0x060914);
      this.scene.fog = new THREE.FogExp2(0x060914, 0.0012);
      this.ambientLight.color.setHex(0x1a2639);
      this.ambientLight.intensity = 0.8;
      this.hemiLight.color.setHex(0x1e3a5f);
      this.hemiLight.groundColor.setHex(0x0a0e17);
      this.hemiLight.intensity = 0.5;
      this.sunLight.color.setHex(0x6a8eb8);
      this.sunLight.intensity = 1.0;
      this.sunLight.position.set(80, 180, 70);

      this.lightPuddles.forEach(p => p.visible = true);
      this.lampMeshes.forEach(l => l.material.color.setHex(0xfffae0));
    }
  }

  toggleDayNight() {
    this.applyEnvironmentMode(!this.isDayMode);
    return this.isDayMode;
  }

  createAsphaltTexture() {
    const canvas = document.createElement('canvas');
    canvas.width = 512;
    canvas.height = 512;
    const ctx = canvas.getContext('2d');
    ctx.fillStyle = '#161920';
    ctx.fillRect(0, 0, 512, 512);

    for (let i = 0; i < 20000; i++) {
      const x = Math.random() * 512;
      const y = Math.random() * 512;
      const shade = Math.floor(Math.random() * 30 + 25);
      ctx.fillStyle = `rgb(${shade}, ${shade}, ${shade})`;
      ctx.fillRect(x, y, 2, 2);
    }
    const tex = new THREE.CanvasTexture(canvas);
    tex.wrapS = THREE.RepeatWrapping;
    tex.wrapT = THREE.RepeatWrapping;
    tex.repeat.set(24, 1);
    return tex;
  }

  createConcreteBarrierTexture() {
    const canvas = document.createElement('canvas');
    canvas.width = 128;
    canvas.height = 32;
    const ctx = canvas.getContext('2d');
    ctx.fillStyle = '#475569';
    ctx.fillRect(0, 0, 128, 32);

    ctx.fillStyle = '#f59e0b';
    for (let x = -32; x < 128; x += 24) {
      ctx.beginPath();
      ctx.moveTo(x, 20);
      ctx.lineTo(x + 10, 20);
      ctx.lineTo(x + 20, 32);
      ctx.lineTo(x + 10, 32);
      ctx.closePath();
      ctx.fill();
    }
    const tex = new THREE.CanvasTexture(canvas);
    tex.wrapS = THREE.RepeatWrapping;
    tex.repeat.set(36, 1);
    return tex;
  }

  createLightPuddleTexture() {
    const canvas = document.createElement('canvas');
    canvas.width = 128;
    canvas.height = 128;
    const ctx = canvas.getContext('2d');
    const grad = ctx.createRadialGradient(64, 64, 0, 64, 64, 64);
    grad.addColorStop(0, 'rgba(160, 220, 255, 0.45)');
    grad.addColorStop(0.5, 'rgba(100, 180, 255, 0.20)');
    grad.addColorStop(1, 'rgba(0, 0, 0, 0)');
    ctx.fillStyle = grad;
    ctx.fillRect(0, 0, 128, 128);
    return new THREE.CanvasTexture(canvas);
  }

  buildTerrain() {
    const groundGeo = new THREE.PlaneGeometry(4800, 4800);
    const groundMat = new THREE.MeshLambertMaterial({ color: 0x0a101d });
    const ground = new THREE.Mesh(groundGeo, groundMat);
    ground.rotation.x = -Math.PI / 2;
    ground.position.y = -0.3;
    ground.receiveShadow = true;
    this.scene.add(ground);

    const grid = new THREE.GridHelper(4800, 160, 0x1e3a5f, 0x0c1726);
    grid.position.y = -0.25;
    this.scene.add(grid);
  }

  buildDistantCity() {
    const cityGroup = new THREE.Group();
    const bMat = new THREE.MeshLambertMaterial({ color: 0x0c1527 });
    const bGeo = new THREE.BoxGeometry(1, 1, 1);

    for (let i = 0; i < 90; i++) {
      const w = Math.random() * 30 + 15;
      const d = Math.random() * 30 + 15;
      const h = Math.random() * 120 + 40;
      const bMesh = new THREE.Mesh(bGeo, bMat);
      bMesh.scale.set(w, h, d);

      const side = Math.random() < 0.5 ? 1 : -1;
      const posX = (Math.random() - 0.5) * 3600;
      const posZ = side * (Math.random() * 320 + 200);
      bMesh.position.set(posX, h / 2, posZ);
      cityGroup.add(bMesh);
    }
    this.scene.add(cityGroup);
  }

  buildHighway() {
    const roadGeo = new THREE.PlaneGeometry(this.roadLength, this.totalRoadWidth);
    const roadMat = new THREE.MeshLambertMaterial({
      map: this.shared.asphaltTexture,
      roughness: 0.8
    });
    const road = new THREE.Mesh(roadGeo, roadMat);
    road.rotation.x = -Math.PI / 2;
    road.position.y = 0.0;
    road.receiveShadow = true;
    this.scene.add(road);

    const edgeGeo = new THREE.PlaneGeometry(this.roadLength, 0.4);
    const yellowMat = new THREE.MeshBasicMaterial({ color: 0xffb703 });
    const whiteMat = new THREE.MeshBasicMaterial({ color: 0xffffff });

    const innerEdge = new THREE.Mesh(edgeGeo, yellowMat);
    innerEdge.rotation.x = -Math.PI / 2;
    innerEdge.position.set(0, 0.04, this.totalRoadWidth / 2 - 0.4);
    this.scene.add(innerEdge);

    const outerEdge = new THREE.Mesh(edgeGeo, whiteMat);
    outerEdge.rotation.x = -Math.PI / 2;
    outerEdge.position.set(0, 0.04, -this.totalRoadWidth / 2 + 0.4);
    this.scene.add(outerEdge);

    const dashMat = new THREE.MeshBasicMaterial({ color: 0x00f2fe, transparent: true, opacity: 0.85 });
    for (let lane = 1; lane < this.numLanes; lane++) {
      const zPos = -this.totalRoadWidth / 2 + lane * this.laneWidth;
      const dashLen = 7.0;
      const gap = 9.0;
      const count = Math.floor(this.roadLength / (dashLen + gap));

      for (let i = 0; i < count; i++) {
        const xPos = -this.roadLength / 2 + i * (dashLen + gap) + dashLen / 2;
        const dashGeo = new THREE.PlaneGeometry(dashLen, 0.35);
        const dash = new THREE.Mesh(dashGeo, dashMat);
        dash.rotation.x = -Math.PI / 2;
        dash.position.set(xPos, 0.04, zPos);
        this.scene.add(dash);
      }
    }

    const barrierGeo = new THREE.BoxGeometry(this.roadLength, 1.8, 1.0);
    const barrierMat = new THREE.MeshLambertMaterial({ map: this.shared.barrierTexture });

    const northBarrier = new THREE.Mesh(barrierGeo, barrierMat);
    northBarrier.position.set(0, 0.9, this.totalRoadWidth / 2 + 0.8);
    northBarrier.castShadow = true;
    this.scene.add(northBarrier);

    const southBarrier = new THREE.Mesh(barrierGeo, barrierMat);
    southBarrier.position.set(0, 0.9, -this.totalRoadWidth / 2 - 0.8);
    southBarrier.castShadow = true;
    this.scene.add(southBarrier);
  }

  buildStreetlights() {
    const spacing = 75.0;
    const numPoles = Math.floor(this.roadLength / spacing);
    const poleMat = new THREE.MeshLambertMaterial({ color: 0x475569 });

    for (let i = 0; i <= numPoles; i++) {
      const xPos = -this.roadLength / 2 + i * spacing;
      [-1, 1].forEach(side => {
        const zPos = side * (this.totalRoadWidth / 2 + 3.0);
        const poleGroup = new THREE.Group();
        poleGroup.position.set(xPos, 0, zPos);

        const mastGeo = new THREE.CylinderGeometry(0.25, 0.45, 18, 8);
        const mast = new THREE.Mesh(mastGeo, poleMat);
        mast.position.y = 9.0;
        poleGroup.add(mast);

        const armLen = 5.0;
        const armGeo = new THREE.CylinderGeometry(0.18, 0.18, armLen, 8);
        armGeo.rotateZ(Math.PI / 2);
        const arm = new THREE.Mesh(armGeo, poleMat);
        arm.rotation.y = (side > 0) ? -Math.PI / 2 : Math.PI / 2;
        arm.position.set(0, 18.0, side * -(armLen / 2));
        poleGroup.add(arm);

        const braceGeo = new THREE.CylinderGeometry(0.12, 0.12, 3.0, 6);
        const brace = new THREE.Mesh(braceGeo, poleMat);
        brace.rotation.x = side * (Math.PI / 4);
        brace.position.set(0, 16.5, side * -1.2);
        poleGroup.add(brace);

        const fixture = new THREE.Mesh(new THREE.BoxGeometry(0.8, 0.35, 1.8), poleMat);
        fixture.position.set(0, 18.0, side * -armLen);
        poleGroup.add(fixture);

        const lampMat = new THREE.MeshBasicMaterial({ color: 0xfffae0 });
        const lampMesh = new THREE.Mesh(new THREE.BoxGeometry(0.6, 0.1, 1.4), lampMat);
        lampMesh.position.set(0, 17.8, side * -armLen);
        poleGroup.add(lampMesh);
        this.lampMeshes.push(lampMesh);


        const puddleGeo = new THREE.PlaneGeometry(28, 28);
        const puddleMat = new THREE.MeshBasicMaterial({
          map: this.shared.lightPuddleTexture,
          transparent: true,
          opacity: 0.5,
          depthWrite: false,
          blending: THREE.AdditiveBlending
        });
        const puddle = new THREE.Mesh(puddleGeo, puddleMat);
        puddle.rotation.x = -Math.PI / 2;
        puddle.position.set(xPos, 0.05, zPos + side * -6.0);
        this.scene.add(puddle);
        this.lightPuddles.push(puddle);

        this.scene.add(poleGroup);
      });
    }
  }

  buildHighwayGantries() {
    [-480, 0, 480].forEach((xPos, idx) => {
      const gantryGroup = new THREE.Group();
      gantryGroup.position.set(xPos, 0, 0);

      const legGeo = new THREE.CylinderGeometry(0.5, 0.6, 22, 6);
      const legMat = new THREE.MeshLambertMaterial({ color: 0x334155 });
      
      const leg1 = new THREE.Mesh(legGeo, legMat);
      leg1.position.set(0, 11, this.totalRoadWidth / 2 + 2.0);
      gantryGroup.add(leg1);

      const leg2 = new THREE.Mesh(legGeo, legMat);
      leg2.position.set(0, 11, -this.totalRoadWidth / 2 - 2.0);
      gantryGroup.add(leg2);

      const trussGeo = new THREE.BoxGeometry(2.0, 2.0, this.totalRoadWidth + 5.0);
      const truss = new THREE.Mesh(trussGeo, legMat);
      truss.position.set(0, 20, 0);
      gantryGroup.add(truss);

      const signCanvas = document.createElement('canvas');
      signCanvas.width = 256;
      signCanvas.height = 64;
      const sCtx = signCanvas.getContext('2d');
      sCtx.fillStyle = '#060913';
      sCtx.fillRect(0, 0, 256, 64);
      sCtx.strokeStyle = '#00f2fe';
      sCtx.lineWidth = 3;
      sCtx.strokeRect(2, 2, 252, 60);
      sCtx.font = 'bold 18px monospace';
      sCtx.fillStyle = '#00ff87';
      sCtx.fillText('5G-NR V2X HIGHWAY', 18, 26);
      sCtx.font = '13px monospace';
      sCtx.fillStyle = '#ffb300';
      const gantryLabels = ['ROADSIDE 5G CELL ZONE', 'APEX 5G ROOFTOP MEC', 'LOGISTICS 5G EDGE CLOUD'];
      sCtx.fillText(gantryLabels[idx], 14, 48);

      const signTex = new THREE.CanvasTexture(signCanvas);
      const signMesh = new THREE.Mesh(
        new THREE.PlaneGeometry(16, 3.0),
        new THREE.MeshBasicMaterial({ map: signTex, transparent: true })
      );
      signMesh.rotation.y = -Math.PI / 2;
      signMesh.position.set(-0.6, 20, 0);
      gantryGroup.add(signMesh);

      this.scene.add(gantryGroup);
    });
  }

  buildRSUs() {
    // Heterogeneous 5G-NR Infrastructure across 3200m Highway:
    // Scale: 0.5 (X_3D = (x_m - 1600.0) * 0.5)
    // 1. RSU_1: Roadside 5G Pole gNodeB at x = 420m -> X = (420 - 1600) * 0.5 = -590.0
    // 2. RSU_2: Rooftop 5G Macro gNodeB atop Apex Tech Center at x = 1480m -> X = -60.0
    // 3. RSU_3: Rooftop 5G Micro gNodeB atop Logistics Cloud Hub at x = 2460m -> X = +430.0
    const rsuDefinitions = [
      {
        id: "RSU_1",
        name: "Roadside 5G Pole gNodeB",
        x: -590.0,
        z: this.totalRoadWidth / 2 + 14.0,
        dishY: 26.0,
        coverageRadius: 240, // 480m coverage in 3D units
        isRooftop: false
      },
      {
        id: "RSU_2",
        name: "Apex Tech Center (Rooftop 5G Macro)",
        x: -60.0,
        z: -(this.totalRoadWidth / 2 + 38.0),
        dishY: 60.0,
        coverageRadius: 325, // 650m coverage in 3D units
        isRooftop: true,
        building: {
          width: 40,
          depth: 34,
          height: 42,
          roofY: 42,
          color: 0x0f172a,
          sign: "APEX 5G EDGE",
          signColor: "#00f2fe"
        }
      },
      {
        id: "RSU_3",
        name: "Logistics Cloud Hub (Rooftop 5G Micro)",
        x: 430.0,
        z: this.totalRoadWidth / 2 + 38.0,
        dishY: 44.0,
        coverageRadius: 275, // 550m coverage in 3D units
        isRooftop: true,
        building: {
          width: 46,
          depth: 32,
          height: 28,
          roofY: 28,
          color: 0x1e293b,
          sign: "5G LOGISTICS CLOUD",
          signColor: "#00ff87"
        }
      }
    ];

    rsuDefinitions.forEach(def => {
      const rsuGroup = new THREE.Group();
      rsuGroup.position.set(def.x, 0, def.z);

      let mastBaseY = 0;

      if (def.isRooftop && def.building) {
        // --- 1. Architectural Building Base ---
        const b = def.building;
        const bGeo = new THREE.BoxGeometry(b.width, b.height, b.depth);
        const bMat = new THREE.MeshLambertMaterial({ color: b.color });
        const bMesh = new THREE.Mesh(bGeo, bMat);
        bMesh.position.y = b.height / 2;
        bMesh.castShadow = true;
        bMesh.receiveShadow = true;
        rsuGroup.add(bMesh);

        // Building Facade Window Ribbons
        const numFloors = Math.floor(b.height / 8);
        for (let fl = 1; fl <= numFloors; fl++) {
          const ribbonGeo = new THREE.BoxGeometry(b.width + 0.4, 1.8, b.depth + 0.4);
          const ribbonMat = new THREE.MeshBasicMaterial({
            color: (fl % 2 === 0) ? 0x00f2fe : 0x38bdf8,
            transparent: true,
            opacity: 0.4
          });
          const ribbon = new THREE.Mesh(ribbonGeo, ribbonMat);
          ribbon.position.y = fl * 7.5;
          rsuGroup.add(ribbon);
        }

        // Rooftop Parapet Border
        const parapetGeo = new THREE.BoxGeometry(b.width + 0.8, 1.4, b.depth + 0.8);
        const parapetMat = new THREE.MeshLambertMaterial({ color: 0x334155 });
        const parapet = new THREE.Mesh(parapetGeo, parapetMat);
        parapet.position.y = b.roofY + 0.7;
        rsuGroup.add(parapet);

        // Rooftop Equipment / HVAC Shelter
        const shelterGeo = new THREE.BoxGeometry(14, 5, 12);
        const shelterMat = new THREE.MeshLambertMaterial({ color: 0x475569 });
        const shelter = new THREE.Mesh(shelterGeo, shelterMat);
        shelter.position.set(4, b.roofY + 2.5, 3);
        rsuGroup.add(shelter);

        // Illuminated Building Signboard facing highway
        const signCanvas = document.createElement('canvas');
        signCanvas.width = 320;
        signCanvas.height = 64;
        const sCtx = signCanvas.getContext('2d');
        sCtx.fillStyle = '#050a14';
        sCtx.fillRect(0, 0, 320, 64);
        sCtx.strokeStyle = b.signColor;
        sCtx.lineWidth = 3;
        sCtx.strokeRect(3, 3, 314, 58);
        sCtx.font = 'bold 20px monospace';
        sCtx.fillStyle = b.signColor;
        sCtx.fillText(b.sign, 20, 40);

        const signTex = new THREE.CanvasTexture(signCanvas);
        const signMesh = new THREE.Mesh(
          new THREE.PlaneGeometry(22, 4.4),
          new THREE.MeshBasicMaterial({ map: signTex, transparent: true })
        );
        // Face the highway
        const faceRoadSign = (def.z > 0) ? -1 : 1;
        signMesh.position.set(0, b.height - 4, faceRoadSign * (b.depth / 2 + 0.3));
        if (def.z > 0) signMesh.rotation.y = Math.PI;
        rsuGroup.add(signMesh);

        mastBaseY = b.roofY;
      } else {
        // Roadside ground equipment cabinet
        const baseGeo = new THREE.BoxGeometry(6, 4, 6);
        const baseMat = new THREE.MeshLambertMaterial({ color: 0x1e293b });
        const base = new THREE.Mesh(baseGeo, baseMat);
        base.position.y = 2.0;
        base.castShadow = true;
        rsuGroup.add(base);
        mastBaseY = 4.0;
      }

      // --- 2. 5G Cellular Tower Mast ---
      const mastHeight = def.dishY - mastBaseY;
      const mastGeo = new THREE.CylinderGeometry(0.6, 1.2, mastHeight, 8);
      const mastMat = new THREE.MeshLambertMaterial({ color: 0x64748b });
      const mast = new THREE.Mesh(mastGeo, mastMat);
      mast.position.y = mastBaseY + mastHeight / 2;
      rsuGroup.add(mast);

      // --- 3. 5G MIMO Sector Antenna Array ---
      // 3 Directional Sector Panels angled across highway
      const panelGeo = new THREE.BoxGeometry(0.5, 3.2, 1.2);
      const panelMat = new THREE.MeshBasicMaterial({ color: 0xe2e8f0 });
      for (let angle = 0; angle < 3; angle++) {
        const panel = new THREE.Mesh(panelGeo, panelMat);
        const rotY = (angle * (Math.PI * 2 / 3));
        panel.position.set(Math.cos(rotY) * 1.4, def.dishY - 1.0, Math.sin(rotY) * 1.4);
        panel.rotation.y = rotY;
        rsuGroup.add(panel);
      }

      // --- 4. Microwave Backhaul Drum Dish ---
      const dishGeo = new THREE.CylinderGeometry(2.8, 0.4, 1.0, 14, 1, true);
      const dishMat = new THREE.MeshLambertMaterial({ color: 0x94a3b8 });
      const dish = new THREE.Mesh(dishGeo, dishMat);
      dish.position.set(0, def.dishY, 0);
      dish.rotation.x = Math.PI / 3;
      rsuGroup.add(dish);

      // --- 5. Red Pulsating Aviation Warning Beacon ---
      const beaconMat = new THREE.MeshBasicMaterial({ color: 0xff1744 });
      const beacon = new THREE.Mesh(new THREE.SphereGeometry(0.8, 10, 10), beaconMat);
      beacon.position.y = def.dishY + 2.5;
      rsuGroup.add(beacon);

      // --- 6. 5G Cell Coverage Dome Overlay ---
      const covGeo = new THREE.SphereGeometry(def.coverageRadius, 24, 12, 0, Math.PI * 2, 0, Math.PI / 2);
      const covMat = new THREE.MeshBasicMaterial({
        color: def.isRooftop ? 0x00ff87 : 0x00f2fe,
        transparent: true,
        opacity: 0.032,
        wireframe: true
      });
      const dome = new THREE.Mesh(covGeo, covMat);
      rsuGroup.add(dome);

      this.scene.add(rsuGroup);
      this.rsusMap.set(def.id, {
        group: rsuGroup,
        x: def.x,
        y: def.dishY,
        z: def.z,
        dome: dome,
        beacon: beacon,
        isRooftop: def.isRooftop,
        name: def.name
      });
    });
  }

  createVehicleMesh(vData) {
    const vGroup = new THREE.Group();
    const wheels = [];

    const isTruck = (vData.vclass === 'Truck');
    const isSUV = (vData.vclass === 'SUV');
    const isEV = (vData.vclass === 'Autonomous_EV');

    let bodyColor = 0x2563eb;
    if (isEV) bodyColor = 0x059669;
    if (isSUV) bodyColor = 0xd97706;
    if (isTruck) bodyColor = 0xdc2626;
    // NOTE: this colour reflects the SIMULATED GROUND-TRUTH role of the
    // vehicle (a controlled-simulation property), not a detector decision.
    // Detector decisions are surfaced separately as trust scores and as
    // "detector alert" events.
    if (vData.role === 'poisoning') bodyColor = 0x991b1b;

    const bodyMat = new THREE.MeshLambertMaterial({ color: bodyColor });

    if (isTruck) {
      const cab = new THREE.Mesh(new THREE.BoxGeometry(3.8, 3.2, 2.7), bodyMat);
      cab.position.set(3.4, 2.3, 0);
      cab.castShadow = true;
      vGroup.add(cab);

      const trailerMat = new THREE.MeshLambertMaterial({ color: 0xe2e8f0 });
      const trailer = new THREE.Mesh(new THREE.BoxGeometry(9.0, 3.8, 2.8), trailerMat);
      trailer.position.set(-3.2, 2.6, 0);
      trailer.castShadow = true;
      vGroup.add(trailer);

      const wheelPos = [
        { x: 4.0, y: 0.72, z: 1.35, isFront: true },
        { x: 4.0, y: 0.72, z: -1.35, isFront: true },
        { x: -5.5, y: 0.72, z: 1.35, isFront: false },
        { x: -5.5, y: 0.72, z: -1.35, isFront: false }
      ];

      wheelPos.forEach(wp => {
        const w = this.buildSharedWheel(true);
        w.position.set(wp.x, wp.y, wp.z);
        vGroup.add(w);
        wheels.push({ group: w, isFront: wp.isFront, radius: 0.72 });
      });
    } else {
      const len = isSUV ? 5.0 : 4.6;
      const h = isSUV ? 1.8 : 1.45;
      const w = 2.1;

      const lower = new THREE.Mesh(new THREE.BoxGeometry(len, h * 0.55, w), bodyMat);
      lower.position.y = 0.8;
      lower.castShadow = true;
      vGroup.add(lower);

      const cabin = new THREE.Mesh(new THREE.BoxGeometry(len * 0.52, h * 0.48, w * 0.85), this.shared.glassMat);
      cabin.position.set(-0.2, 0.8 + h * 0.5, 0);
      vGroup.add(cabin);

      const halfL = len * 0.32;
      const halfW = w * 0.52;
      const wheelPos = [
        { x: halfL, y: 0.48, z: halfW, isFront: true },
        { x: halfL, y: 0.48, z: -halfW, isFront: true },
        { x: -halfL, y: 0.48, z: halfW, isFront: false },
        { x: -halfL, y: 0.48, z: -halfW, isFront: false }
      ];

      wheelPos.forEach(wp => {
        const wMesh = this.buildSharedWheel(false);
        wMesh.position.set(wp.x, wp.y, wp.z);
        vGroup.add(wMesh);
        wheels.push({ group: wMesh, isFront: wp.isFront, radius: 0.48 });
      });
    }

    const lensL = new THREE.Mesh(new THREE.BoxGeometry(0.12, 0.28, 0.4), this.shared.headlightLensMat);
    lensL.position.set(2.35, 0.9, 0.7);
    vGroup.add(lensL);
    const lensR = new THREE.Mesh(new THREE.BoxGeometry(0.12, 0.28, 0.4), this.shared.headlightLensMat);
    lensR.position.set(2.35, 0.9, -0.7);
    vGroup.add(lensR);

    const tail = new THREE.Mesh(new THREE.BoxGeometry(0.12, 0.22, 1.7), this.shared.taillightLensMat);
    tail.position.set(-2.35, 1.05, 0);
    vGroup.add(tail);

    const ringGroup = new THREE.Group();
    const initialTrust = vData.trust_score || 0.75;
    let initialHalo = 0x00ff87;
    if (initialTrust < 0.4) initialHalo = 0xff3366;
    else if (initialTrust < 0.6) initialHalo = 0xffb300;
    else if (initialTrust < 0.8) initialHalo = 0x00f2fe;

    const ringMat = new THREE.MeshBasicMaterial({
      color: initialHalo,
      side: THREE.DoubleSide,
      transparent: true,
      opacity: 0.85
    });

    const outerRing = new THREE.Mesh(this.shared.outerRingGeo, ringMat);
    outerRing.rotation.x = Math.PI / 2;
    ringGroup.add(outerRing);
    const innerRing = new THREE.Mesh(this.shared.innerRingGeo, ringMat);
    innerRing.rotation.x = Math.PI / 2;
    ringGroup.add(innerRing);
    vGroup.add(ringGroup);

    const billboard = this.createVehicleBillboard(vData);
    billboard.position.y = 5.5;
    vGroup.add(billboard);

    this.scene.add(vGroup);

    const initProgress = parseFloat(vData.x) || 0.0;
    const initZ = (vData.lane - 3.0) * this.laneWidth;
    const initX = (initProgress - this.highwayLengthM / 2) * this.scaleFactor;
    vGroup.position.set(initX, 0, initZ);

    return {
      group: vGroup,
      wheels: wheels,
      ringGroup: ringGroup,
      ringMat: ringMat,
      billboard: billboard,
      cachedTrust: vData.trust_score || 0.75,
      cachedRole: vData.role || 'honest',
      progressMeters: initProgress,
      serverX: initProgress,
      lane: vData.lane || 1,
      currentLaneZ: initZ,
      targetLaneZ: initZ,
      speed: vData.speed || 24.0
    };
  }

  buildSharedWheel(isTruck) {
    const wGroup = new THREE.Group();
    const tireMesh = new THREE.Mesh(
      isTruck ? this.shared.tireGeoTruck : this.shared.tireGeoCar,
      this.shared.tireMat
    );
    wGroup.add(tireMesh);

    const rimMesh = new THREE.Mesh(
      isTruck ? this.shared.rimGeoTruck : this.shared.rimGeoCar,
      this.shared.rimMat
    );
    wGroup.add(rimMesh);
    return wGroup;
  }

  createVehicleBillboard(vData) {
    const canvas = document.createElement('canvas');
    canvas.width = 256;
    canvas.height = 96;
    const ctx = canvas.getContext('2d');
    this.renderBillboardCanvas(ctx, vData);

    const texture = new THREE.CanvasTexture(canvas);
    const spriteMat = new THREE.SpriteMaterial({ map: texture, transparent: true });
    const sprite = new THREE.Sprite(spriteMat);
    sprite.scale.set(4.5, 1.7, 1.0);
    sprite.userData = { canvas: canvas, ctx: ctx, texture: texture };
    return sprite;
  }

  renderBillboardCanvas(ctx, vData) {
    ctx.clearRect(0, 0, 256, 96);
    ctx.fillStyle = 'rgba(6, 12, 24, 0.88)';
    ctx.strokeStyle = '#00f2fe';
    ctx.lineWidth = 3;
    ctx.beginPath();
    ctx.roundRect(4, 4, 248, 88, 8);
    ctx.fill();
    ctx.stroke();

    const trust = vData.trust_score || 0.75;
    let trustColor = '#00ff87';
    if (trust < 0.4) trustColor = '#ff3366';
    else if (trust < 0.6) trustColor = '#ffb300';
    else if (trust < 0.8) trustColor = '#00f2fe';

    ctx.font = 'bold 22px monospace';
    ctx.fillStyle = '#ffffff';
    ctx.fillText(`V-${vData.vehicle_id}  ${vData.vclass || 'Car'}`, 16, 32);

    ctx.font = 'bold 20px monospace';
    ctx.fillStyle = trustColor;
    ctx.fillText(`TRUST: ${(trust * 100).toFixed(0)}%`, 16, 58);

    ctx.font = '16px monospace';
    ctx.fillStyle = (vData.role === 'honest') ? '#94a3b8' : '#ff3366';
    ctx.fillText(`ROLE: ${(vData.role || 'HONEST').toUpperCase()}`, 16, 80);
  }

  updateVehicles(vehiclesData) {
    const currentIds = new Set();

    vehiclesData.forEach(v => {
      // A vehicle that is not present in the trace observation window (it has
      // not arrived yet, or it has left) is not rendered on the road.
      if (v.active === false) {
        const absent = this.vehiclesMap.get(v.vehicle_id);
        if (absent) {
          absent.group.visible = false;
        }
        return;
      }

      currentIds.add(v.vehicle_id);
      let vEntry = this.vehiclesMap.get(v.vehicle_id);

      if (!vEntry) {
        vEntry = this.createVehicleMesh(v);
        this.vehiclesMap.set(v.vehicle_id, vEntry);
      }
      vEntry.group.visible = true;

      // Store server reference for soft natural drift correction
      vEntry.serverX = parseFloat(v.x);
      vEntry.speed = v.speed || 24.0;
      vEntry.lane = v.lane;
      vEntry.targetLaneZ = (v.lane - 3.0) * this.laneWidth;

      // Update trust & role
      const trust = v.trust_score || 0.75;
      if (Math.abs(trust - vEntry.cachedTrust) > 0.02 || v.role !== vEntry.cachedRole) {
        vEntry.cachedTrust = trust;
        vEntry.cachedRole = v.role;

        let haloHex = 0x00ff87;
        if (trust < 0.4) haloHex = 0xff3366;
        else if (trust < 0.6) haloHex = 0xffb300;
        else if (trust < 0.8) haloHex = 0x00f2fe;
        vEntry.ringMat.color.setHex(haloHex);

        if (vEntry.billboard.userData) {
          this.renderBillboardCanvas(vEntry.billboard.userData.ctx, v);
          vEntry.billboard.userData.texture.needsUpdate = true;
        }
      }
    });

    for (let [vid, entry] of this.vehiclesMap.entries()) {
      if (!currentIds.has(vid)) {
        this.scene.remove(entry.group);
        this.vehiclesMap.delete(vid);
      }
    }
  }

  updateCamera() {
    if (this.cameraMode === 'chase' && this.chaseTargetId) {
      const target = this.vehiclesMap.get(this.chaseTargetId);
      if (target) {
        const targetPos = target.group.position;
        const idealOffset = new THREE.Vector3(targetPos.x - 32, targetPos.y + 12, targetPos.z);
        this.camera.position.lerp(idealOffset, 0.12);
        this.camera.lookAt(targetPos.x + 25, targetPos.y + 2, targetPos.z);
      }
    } else if (this.cameraMode === 'cockpit' && this.chaseTargetId) {
      const target = this.vehiclesMap.get(this.chaseTargetId);
      if (target) {
        const targetPos = target.group.position;
        this.camera.position.set(targetPos.x + 0.4, targetPos.y + 1.7, targetPos.z);
        this.camera.lookAt(targetPos.x + 120, targetPos.y + 1.7, targetPos.z);
      }
    } else if (this.cameraMode === 'drone') {
      this.droneProgress += 0.0012;
      const xPos = Math.sin(this.droneProgress) * 350;
      const zPos = Math.cos(this.droneProgress) * 90;
      this.camera.position.set(xPos, 85, zPos + 120);
      this.camera.lookAt(xPos * 0.4, 0, 0);
    }
  }

  triggerOffloadingBeam(ev) {
    if (this.beamMode === 'none') return;
    if (this.beamMode === 'focused' && ev.vid !== this.chaseTargetId) return;

    // Strict throttle: at most 2 simultaneous clean beams on screen
    if (this.activeBeams.length >= 2) return;

    const now = performance.now();
    if (this.lastBeamSpawn && (now - this.lastBeamSpawn < 400)) return;
    this.lastBeamSpawn = now;

    const sourceEntry = this.vehiclesMap.get(ev.vid);
    if (!sourceEntry) return;

    let targetCoords = null;
    let isV2V = false;
    let beamColor = 0x00f2fe; // V2I Cyan
    let targetLabel = "";


    // 1. Identify Target Node & Color strictly matching the Legend
    if (ev.target.startsWith('RSU')) {
      const rsu = this.rsusMap.get(ev.target);
      if (rsu) {
        targetCoords = new THREE.Vector3(rsu.x, rsu.y, rsu.z);
        const tag = rsu.isRooftop ? (ev.target === 'RSU_2' ? 'APEX 5G ROOFTOP' : 'LOGISTICS 5G') : 'ROADSIDE 5G';
        targetLabel = `⚡ [5G-V2I] V-${ev.vid} -> ${ev.target} [${tag}] (${ev.delay_ms}ms)`;
      }
    } else if (ev.target.startsWith('V')) {
      isV2V = true;
      const peerId = parseInt(ev.target.replace('V', ''));
      const peerEntry = this.vehiclesMap.get(peerId);
      if (peerEntry) {
        targetCoords = new THREE.Vector3(peerEntry.group.position.x, 3.5, peerEntry.group.position.z);
        targetLabel = `⚡ [V2V] V-${ev.vid} -> PEER V-${peerId} (${ev.delay_ms}ms)`;
      }
    }

    if (!targetCoords) return;

    // Legend color logic:
    // - 🚨 Detector Alert: Crimson Red (0xff3366)
    //   Shown ONLY when the trust engine actually flagged the target node
    //   (detector_alert). A failing task is NOT an attack.
    // - ✗ Task Failed (no detector alert): neutral grey (0x8892b0)
    // - ⚡ V2V: Vehicle -> Peer Car: Golden Amber (0xffb703)
    // - ⚡ V2I: Vehicle -> Tower: Electric Cyan (0x00f2fe)
    if (ev.detector_alert) {
      beamColor = 0xff3366;
      targetLabel = `🚨 [DETECTOR ALERT] V-${ev.vid} -> ${ev.target}`;
    } else if (!ev.success) {
      beamColor = 0x8892b0;
      targetLabel = `✗ [TASK FAILED] V-${ev.vid} -> ${ev.target}`;
    } else if (isV2V) {
      beamColor = 0xffb703;
    } else {
      beamColor = 0x00f2fe;
    }

    const startPos = new THREE.Vector3(sourceEntry.group.position.x, 3.8, sourceEntry.group.position.z);

    // 2. Thick, Vibrant Visible Laser Beam (Radius 0.35)
    const distance = startPos.distanceTo(targetCoords);
    const beamGeo = new THREE.CylinderGeometry(0.35, 0.35, distance, 12);
    const beamMat = new THREE.MeshBasicMaterial({
      color: beamColor,
      transparent: true,
      opacity: 0.95
    });
    const beam = new THREE.Mesh(beamGeo, beamMat);
    const mid = new THREE.Vector3().addVectors(startPos, targetCoords).multiplyScalar(0.5);
    beam.position.copy(mid);
    beam.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), targetCoords.clone().sub(startPos).normalize());
    this.scene.add(beam);

    // 3. Floating 3D Label Tag Sprite in center of beam
    const tagSprite = this.createBeamLabel(targetLabel, beamColor);
    tagSprite.position.copy(mid);
    tagSprite.position.y += 3.0;
    this.scene.add(tagSprite);

    // 4. Animated Traveling Photon Packets along the beam
    const packets = [];
    for (let p = 0; p < 4; p++) {
      const pGeo = new THREE.SphereGeometry(0.65, 8, 8);
      const pMat = new THREE.MeshBasicMaterial({ color: 0xffffff });
      const pMesh = new THREE.Mesh(pGeo, pMat);
      this.scene.add(pMesh);
      packets.push({ mesh: pMesh, progress: p * 0.25 });
    }

    // Clean, crisp 50 frames lifetime (~0.85s) for uncluttered observation
    this.activeBeams.push({
      mesh: beam,
      tag: tagSprite,
      packets: packets,
      sourceVid: ev.vid,
      targetVid: isV2V ? parseInt(ev.target.replace('V', '')) : null,
      targetRsuId: ev.target.startsWith('RSU') ? ev.target : null,
      targetCoords: targetCoords,
      life: 50,
      maxLife: 50,
      startPos: startPos
    });

    // Shockwave ring at target
    this.createShockwave(targetCoords, beamColor);
  }

  createBeamLabel(text, colorHex) {
    const canvas = document.createElement('canvas');
    canvas.width = 340;
    canvas.height = 70;
    const ctx = canvas.getContext('2d');
    ctx.fillStyle = 'rgba(6, 12, 24, 0.92)';
    ctx.strokeStyle = '#' + colorHex.toString(16).padStart(6, '0');
    ctx.lineWidth = 3;
    ctx.beginPath();
    ctx.roundRect(4, 4, 332, 62, 8);
    ctx.fill();
    ctx.stroke();

    ctx.font = 'bold 20px monospace';
    ctx.fillStyle = '#' + colorHex.toString(16).padStart(6, '0');
    ctx.fillText(text, 14, 42);

    const texture = new THREE.CanvasTexture(canvas);
    const spriteMat = new THREE.SpriteMaterial({ map: texture, transparent: true });
    const sprite = new THREE.Sprite(spriteMat);
    sprite.scale.set(15.0, 3.2, 1.0);
    return sprite;
  }


  createShockwave(position, colorHex) {
    const ringGeo = new THREE.RingGeometry(0.5, 1.5, 20);
    const ringMat = new THREE.MeshBasicMaterial({
      color: colorHex,
      side: THREE.DoubleSide,
      transparent: true,
      opacity: 0.95
    });
    const ring = new THREE.Mesh(ringGeo, ringMat);
    ring.rotation.x = Math.PI / 2;
    ring.position.copy(position);
    this.scene.add(ring);
    this.activeShockwaves.push({ mesh: ring, scale: 1.0, maxScale: 20.0, life: 25 });
  }

  animate() {
    requestAnimationFrame(() => this.animate());

    const dt = 0.016; // 60 FPS delta

    // 1. 100% FLUID CONTINUOUS REAL-WORLD VEHICLE MOTION
    const halfLen = this.highwayLengthM / 2.0; // 1600.0m

    for (let [vid, vEntry] of this.vehiclesMap.entries()) {
      // Continuous highway progress at constant speed
      vEntry.progressMeters = (vEntry.progressMeters + vEntry.speed * dt) % this.highwayLengthM;

      // Gentle natural synchronization with server simulation (soft drift correction without jerking)
      if (vEntry.serverX !== undefined) {
        let diff = vEntry.serverX - vEntry.progressMeters;
        if (diff > halfLen) diff -= this.highwayLengthM;
        if (diff < -halfLen) diff += this.highwayLengthM;
        vEntry.progressMeters += diff * 0.025;
      }

      // Map progress to 3D road coordinates
      const worldX = (vEntry.progressMeters - halfLen) * this.scaleFactor;
      vEntry.currentLaneZ = THREE.MathUtils.lerp(vEntry.currentLaneZ, vEntry.targetLaneZ, 0.04);

      vEntry.group.position.x = worldX;
      vEntry.group.position.z = vEntry.currentLaneZ;

      // Wheel rotation
      const wheelDelta = (vEntry.speed * dt) / 0.48;
      vEntry.wheels.forEach(w => {
        w.group.rotation.z -= wheelDelta;
      });

      // Hologram ring spin
      vEntry.ringGroup.children[0].rotation.z += 0.04;
      vEntry.ringGroup.children[1].rotation.z -= 0.06;
    }

    // Animate 5G Tower Hazard Strobe Lights
    const beaconPulse = Math.sin(Date.now() * 0.006) > 0.15 ? 1.0 : 0.25;
    this.rsusMap.forEach(r => {
      if (r.beacon && r.beacon.material) {
        r.beacon.material.opacity = beaconPulse;
        r.beacon.material.transparent = true;
      }
    });

    // 2. Camera Update
    this.updateCamera();

    // 3. Dynamic Offloading Beams Tracking & Packet Animation
    for (let i = this.activeBeams.length - 1; i >= 0; i--) {
      const b = this.activeBeams[i];
      b.life -= 1;

      // Dynamic Real-time Tracking: update beam start position to follow the moving car!
      const src = this.vehiclesMap.get(b.sourceVid);
      if (src) {
        const curStart = new THREE.Vector3(src.group.position.x, 3.8, src.group.position.z);
        let curTarget = b.targetCoords;
        if (b.targetVid) {
          const peer = this.vehiclesMap.get(b.targetVid);
          if (peer) {
            curTarget = new THREE.Vector3(peer.group.position.x, 3.5, peer.group.position.z);
          }
        } else if (b.targetRsuId) {
          const rsu = this.rsusMap.get(b.targetRsuId);
          if (rsu) {
            curTarget = new THREE.Vector3(rsu.x, rsu.y, rsu.z);
          }
        }
        const dist = Math.max(1.0, curStart.distanceTo(curTarget));
        b.mesh.scale.set(1, dist / Math.max(1, b.mesh.geometry.parameters.height), 1);
        const mid = new THREE.Vector3().addVectors(curStart, curTarget).multiplyScalar(0.5);
        b.mesh.position.copy(mid);
        b.mesh.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), curTarget.clone().sub(curStart).normalize());
        
        b.tag.position.copy(mid);
        b.tag.position.y += 3.5;

        // Animate traveling photon packets along beam
        b.packets.forEach(p => {
          p.progress = (p.progress + 0.04) % 1.0;
          p.mesh.position.lerpVectors(curStart, curTarget, p.progress);
        });
      }


      const alpha = b.life / b.maxLife;
      b.mesh.material.opacity = alpha * 0.95;
      b.tag.material.opacity = alpha;
      b.packets.forEach(p => p.mesh.material.opacity = alpha);

      if (b.life <= 0) {
        this.scene.remove(b.mesh);
        this.scene.remove(b.tag);
        b.packets.forEach(p => this.scene.remove(p.mesh));
        this.activeBeams.splice(i, 1);
      }
    }

    // 4. Expand Shockwaves
    for (let i = this.activeShockwaves.length - 1; i >= 0; i--) {
      const sw = this.activeShockwaves[i];
      sw.scale += (sw.maxScale - sw.scale) * 0.12;
      sw.life -= 1;
      sw.mesh.scale.set(sw.scale, sw.scale, sw.scale);
      sw.mesh.material.opacity = (sw.life / 25.0) * 0.95;
      if (sw.life <= 0) {
        this.scene.remove(sw.mesh);
        this.activeShockwaves.splice(i, 1);
      }
    }

    if (this.controls && this.cameraMode === 'orbit') {
      this.controls.update();
    }

    this.renderer.render(this.scene, this.camera);
  }

  setCameraMode(mode, targetId = null) {
    this.cameraMode = mode;
    if (targetId) this.chaseTargetId = targetId;

    if (mode === 'top') {
      this.camera.position.set(0, 480, 0);
      this.camera.lookAt(0, 0, 0);
      if (this.controls) this.controls.enabled = false;
    } else if (mode === 'orbit') {
      this.camera.position.set(-80, 95, 180);
      if (this.controls) {
        this.controls.enabled = true;
        this.controls.target.set(0, 0, 0);
      }
    } else {
      if (this.controls) this.controls.enabled = false;
    }
  }

  onWindowResize() {
    if (!this.container) return;
    this.camera.aspect = this.container.clientWidth / this.container.clientHeight;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(this.container.clientWidth, this.container.clientHeight);
  }
}

window.VANET3DSimulation = VANET3DSimulation;
