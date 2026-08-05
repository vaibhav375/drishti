/* DRISHTI — 3D orbit viewer (three.js, vendored, offline).
   Reads real Keplerian elements from #orbit3d-data and renders the
   object's orbit(s) around a wireframe Earth. Rotatable, auto-orbiting,
   reduced-motion aware. Geometry is real (a, e, i, RAAN, argp); the
   moving marker is schematic (uniform phase), which the caption states. */
(function () {
  var el = document.getElementById("orbit3d");
  var dataEl = document.getElementById("orbit3d-data");
  if (!el || !dataEl || typeof THREE === "undefined") return;

  var cfg = JSON.parse(dataEl.textContent);
  var R = cfg.earth_radius_km || 6378.137;
  var reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  var W = el.clientWidth, H = el.clientHeight || 380;
  var scene = new THREE.Scene();
  var camera = new THREE.PerspectiveCamera(42, W / H, 0.01, 2000);
  var renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.setSize(W, H);
  el.appendChild(renderer.domElement);

  // --- Earth: a solid dark sphere + a cyan wireframe shell (instrument globe)
  var earthR = 1.0;
  scene.add(new THREE.Mesh(
    new THREE.SphereGeometry(earthR, 48, 48),
    new THREE.MeshBasicMaterial({ color: 0x0e2036 })
  ));
  var wire = new THREE.Mesh(
    new THREE.SphereGeometry(earthR * 1.004, 24, 18),
    new THREE.MeshBasicMaterial({ color: 0x2f6fb0, wireframe: true, transparent: true, opacity: 0.55 })
  );
  scene.add(wire);
  // equatorial ring for reference
  var eq = new THREE.Mesh(
    new THREE.RingGeometry(earthR * 1.35, earthR * 1.36, 96),
    new THREE.MeshBasicMaterial({ color: 0x3a5060, side: THREE.DoubleSide, transparent: true, opacity: 0.5 })
  );
  eq.rotation.x = Math.PI / 2;
  scene.add(eq);

  function rotZ(x, y, a) { var c = Math.cos(a), s = Math.sin(a); return [x * c - y * s, x * s + y * c]; }
  function rotX(y, z, a) { var c = Math.cos(a), s = Math.sin(a); return [y * c - z * s, y * s + z * c]; }

  function orbitCurve(o, N) {
    var i = o.i * Math.PI / 180, raan = o.raan * Math.PI / 180, argp = o.argp * Math.PI / 180;
    var a = o.a / R, e = o.e, pts = [];
    for (var k = 0; k <= N; k++) {
      var nu = 2 * Math.PI * k / N;
      var r = a * (1 - e * e) / (1 + e * Math.cos(nu));
      var x = r * Math.cos(nu), y = r * Math.sin(nu), z = 0;
      var p;
      p = rotZ(x, y, argp); x = p[0]; y = p[1];
      p = rotX(y, z, i); y = p[0]; z = p[1];
      p = rotZ(x, y, raan); x = p[0]; y = p[1];
      pts.push(new THREE.Vector3(x, z, y)); // ECI z (north) -> three y (up)
    }
    return pts;
  }

  var maxR = earthR * 1.4;
  var markers = [];
  cfg.orbits.forEach(function (o) {
    var pts = orbitCurve(o, 256);
    pts.forEach(function (v) { maxR = Math.max(maxR, v.length()); });
    var geo = new THREE.BufferGeometry().setFromPoints(pts);
    scene.add(new THREE.Line(geo, new THREE.LineBasicMaterial({ color: o.color, transparent: true, opacity: 0.9 })));
    // satellite marker
    var m = new THREE.Mesh(new THREE.SphereGeometry(maxR * 0.02, 12, 12),
      new THREE.MeshBasicMaterial({ color: o.color }));
    scene.add(m);
    markers.push({ mesh: m, pts: pts, phase: Math.random() });
  });

  // faint starfield
  var starGeo = new THREE.BufferGeometry();
  var sv = [];
  for (var s = 0; s < 300; s++) {
    var rr = 60 + Math.random() * 60, th = Math.random() * Math.PI * 2, ph = Math.acos(2 * Math.random() - 1);
    sv.push(rr * Math.sin(ph) * Math.cos(th), rr * Math.cos(ph), rr * Math.sin(ph) * Math.sin(th));
  }
  starGeo.setAttribute("position", new THREE.Float32BufferAttribute(sv, 3));
  scene.add(new THREE.Points(starGeo, new THREE.PointsMaterial({ color: 0x8aa0b8, size: 0.15 })));

  camera.position.set(maxR * 1.9, maxR * 1.3, maxR * 1.9);
  var controls = new THREE.OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.08;
  controls.autoRotate = !reduce;
  controls.autoRotateSpeed = 0.6;
  controls.minDistance = earthR * 1.3;
  controls.maxDistance = maxR * 6;

  var t = 0;
  function animate() {
    requestAnimationFrame(animate);
    if (!reduce) {
      t += 0.0016;
      markers.forEach(function (mk) {
        var n = mk.pts.length - 1;
        var f = ((mk.phase + t) % 1) * n;
        var i0 = Math.floor(f), i1 = (i0 + 1) % n, frac = f - i0;
        mk.mesh.position.lerpVectors(mk.pts[i0], mk.pts[i1], frac);
      });
    } else {
      markers.forEach(function (mk) { mk.mesh.position.copy(mk.pts[0]); });
    }
    controls.update();
    renderer.render(scene, camera);
  }
  animate();

  window.addEventListener("resize", function () {
    W = el.clientWidth; H = el.clientHeight || 380;
    camera.aspect = W / H; camera.updateProjectionMatrix(); renderer.setSize(W, H);
  });
})();
