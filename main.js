import cloudSource from './shaders/cloud.wgsl?raw';
import noiseSource from './shaders/noise.wgsl?raw';

const shaderSource = noiseSource + cloudSource;

// ============================================================
// Matrix math helpers (column-major, matches WebGPU std140)
// ============================================================

function mat4Perspective(fovY, aspect, near, far) {
  const f = 1.0 / Math.tan(fovY / 2);
  const nf = 1.0 / (near - far);
  // Column-major flat array
  return new Float32Array([
    f / aspect, 0, 0, 0,
    0, f, 0, 0,
    0, 0, far * nf, -1,
    0, 0, far * near * nf, 0,
  ]);
}

function mat4LookAt(eye, target, up) {
  const zx = eye[0] - target[0], zy = eye[1] - target[1], zz = eye[2] - target[2];
  let len = Math.hypot(zx, zy, zz);
  const z = [zx / len, zy / len, zz / len];

  const xx = up[1] * z[2] - up[2] * z[1];
  const xy = up[2] * z[0] - up[0] * z[2];
  const xz = up[0] * z[1] - up[1] * z[0];
  len = Math.hypot(xx, xy, xz);
  const x = [xx / len, xy / len, xz / len];

  const y = [
    z[1] * x[2] - z[2] * x[1],
    z[2] * x[0] - z[0] * x[2],
    z[0] * x[1] - z[1] * x[0],
  ];

  return new Float32Array([
    x[0], y[0], z[0], 0,
    x[1], y[1], z[1], 0,
    x[2], y[2], z[2], 0,
    -(x[0] * eye[0] + x[1] * eye[1] + x[2] * eye[2]),
    -(y[0] * eye[0] + y[1] * eye[1] + y[2] * eye[2]),
    -(z[0] * eye[0] + z[1] * eye[1] + z[2] * eye[2]),
    1,
  ]);
}

function mat4Multiply(a, b) {
  const out = new Float32Array(16);
  for (let i = 0; i < 4; i++) {
    for (let j = 0; j < 4; j++) {
      out[j * 4 + i] =
        a[0 * 4 + i] * b[j * 4 + 0] +
        a[1 * 4 + i] * b[j * 4 + 1] +
        a[2 * 4 + i] * b[j * 4 + 2] +
        a[3 * 4 + i] * b[j * 4 + 3];
    }
  }
  return out;
}

function mat4Invert(m) {
  const inv = new Float32Array(16);
  inv[0]  =  m[5]*m[10]*m[15] - m[5]*m[11]*m[14] - m[9]*m[6]*m[15] + m[9]*m[7]*m[14] + m[13]*m[6]*m[11] - m[13]*m[7]*m[10];
  inv[4]  = -m[4]*m[10]*m[15] + m[4]*m[11]*m[14] + m[8]*m[6]*m[15] - m[8]*m[7]*m[14] - m[12]*m[6]*m[11] + m[12]*m[7]*m[10];
  inv[8]  =  m[4]*m[9]*m[15]  - m[4]*m[11]*m[13] - m[8]*m[5]*m[15] + m[8]*m[7]*m[13] + m[12]*m[5]*m[11] - m[12]*m[7]*m[9];
  inv[12] = -m[4]*m[9]*m[14]  + m[4]*m[10]*m[13] + m[8]*m[5]*m[14] - m[8]*m[6]*m[13] - m[12]*m[5]*m[10] + m[12]*m[6]*m[9];
  inv[1]  = -m[1]*m[10]*m[15] + m[1]*m[11]*m[14] + m[9]*m[2]*m[15] - m[9]*m[3]*m[14] - m[13]*m[2]*m[11] + m[13]*m[3]*m[10];
  inv[5]  =  m[0]*m[10]*m[15] - m[0]*m[11]*m[14] - m[8]*m[2]*m[15] + m[8]*m[3]*m[14] + m[12]*m[2]*m[11] - m[12]*m[3]*m[10];
  inv[9]  = -m[0]*m[9]*m[15]  + m[0]*m[11]*m[13] + m[8]*m[1]*m[15] - m[8]*m[3]*m[13] - m[12]*m[1]*m[11] + m[12]*m[3]*m[9];
  inv[13] =  m[0]*m[9]*m[14]  - m[0]*m[10]*m[13] - m[8]*m[1]*m[14] + m[8]*m[2]*m[13] + m[12]*m[1]*m[10] - m[12]*m[2]*m[9];
  inv[2]  =  m[1]*m[6]*m[15]  - m[1]*m[7]*m[14]  - m[5]*m[2]*m[15] + m[5]*m[3]*m[14] + m[13]*m[2]*m[7]  - m[13]*m[3]*m[6];
  inv[6]  = -m[0]*m[6]*m[15]  + m[0]*m[7]*m[14]  + m[4]*m[2]*m[15] - m[4]*m[3]*m[14] - m[12]*m[2]*m[7]  + m[12]*m[3]*m[6];
  inv[10] =  m[0]*m[5]*m[15]  - m[0]*m[7]*m[13]  - m[4]*m[1]*m[15] + m[4]*m[3]*m[13] + m[12]*m[1]*m[7]  - m[12]*m[3]*m[5];
  inv[14] = -m[0]*m[5]*m[14]  + m[0]*m[6]*m[13]  + m[4]*m[1]*m[14] - m[4]*m[2]*m[13] - m[12]*m[1]*m[6]  + m[12]*m[2]*m[5];
  inv[3]  = -m[1]*m[6]*m[11]  + m[1]*m[7]*m[10]  + m[5]*m[2]*m[11] - m[5]*m[3]*m[10] - m[9]*m[2]*m[7]   + m[9]*m[3]*m[6];
  inv[7]  =  m[0]*m[6]*m[11]  - m[0]*m[7]*m[10]  - m[4]*m[2]*m[11] + m[4]*m[3]*m[10] + m[8]*m[2]*m[7]   - m[8]*m[3]*m[6];
  inv[11] = -m[0]*m[5]*m[11]  + m[0]*m[7]*m[9]   + m[4]*m[1]*m[11] - m[4]*m[3]*m[9]  - m[8]*m[1]*m[7]   + m[8]*m[3]*m[5];
  inv[15] =  m[0]*m[5]*m[10]  - m[0]*m[6]*m[9]   - m[4]*m[1]*m[10] + m[4]*m[2]*m[9]  + m[8]*m[1]*m[6]   - m[8]*m[2]*m[5];

  const det = m[0]*inv[0] + m[1]*inv[4] + m[2]*inv[8] + m[3]*inv[12];
  if (Math.abs(det) < 1e-10) return new Float32Array(16); // singular
  const invDet = 1.0 / det;
  for (let i = 0; i < 16; i++) inv[i] *= invDet;
  return inv;
}

// ============================================================
// Main
// ============================================================

async function initWebGPU() {
  if (!navigator.gpu) {
    document.body.innerHTML = '<p style="color:white;padding:2rem;">WebGPU is not supported in this browser.</p>';
    throw new Error('WebGPU not supported');
  }

  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) {
    throw new Error('No appropriate GPUAdapter found');
  }

  const device = await adapter.requestDevice();

  const canvas = document.getElementById('canvas');
  const context = canvas.getContext('webgpu');

  const format = navigator.gpu.getPreferredCanvasFormat();
  context.configure({ device, format });

  // --- Shader module ---
  const shaderModule = device.createShaderModule({ code: shaderSource });

  // --- Pipeline ---
  const pipeline = device.createRenderPipeline({
    layout: 'auto',
    vertex: {
      module: shaderModule,
      entryPoint: 'vs',
    },
    fragment: {
      module: shaderModule,
      entryPoint: 'fs',
      targets: [{ format }],
    },
    primitive: {
      topology: 'triangle-list',
    },
  });

  // --- Camera uniform buffer ---
  // Layout: mat4x4f (64 bytes) + vec3f (12 bytes) + f32 pad (4 bytes) = 80 bytes
  const cameraBuffer = device.createBuffer({
    size: 80,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });

  // --- Params uniform buffer ---
  // Layout: 4 x vec4f = 64 bytes
  const paramsBuffer = device.createBuffer({
    size: 64,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });

  const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: cameraBuffer } },
      { binding: 1, resource: { buffer: paramsBuffer } },
    ],
  });

  // --- Camera setup ---
  let camTheta = Math.PI / 4;
  let camPhi = 0.5;
  let camDist = 10.0;
  const target = [0.0, 0.5, 0.0];
  const up = [0.0, 1.0, 0.0];

  // Mouse interaction
  let isDragging = false;
  let lastMouse = [0, 0];

  canvas.addEventListener('pointerdown', e => {
    isDragging = true;
    lastMouse = [e.clientX, e.clientY];
    canvas.setPointerCapture(e.pointerId);
  });

  canvas.addEventListener('pointermove', e => {
    if (!isDragging) return;
    const dx = e.clientX - lastMouse[0];
    const dy = e.clientY - lastMouse[1];
    camTheta -= dx * 0.005;
    camPhi = Math.max(0.1, Math.min(1.4, camPhi + dy * 0.005));
    lastMouse = [e.clientX, e.clientY];
  });

  canvas.addEventListener('pointerup', e => {
    isDragging = false;
    canvas.releasePointerCapture(e.pointerId);
  });

  canvas.addEventListener('wheel', e => {
    camDist = Math.max(2.0, Math.min(20.0, camDist + e.deltaY * 0.005));
    e.preventDefault();
  }, { passive: false });

  function resize() {
    const dpr = 1.0; 
    canvas.width = canvas.clientWidth * dpr;
    canvas.height = canvas.clientHeight * dpr;
  }

  window.addEventListener('resize', resize);
  resize();

  // --- UI Controls ---
  const controls = {
    density: document.getElementById('density'),
    coverage: document.getElementById('coverage'),
    scale: document.getElementById('scale'),
    altitude: document.getElementById('altitude'),
    detail: document.getElementById('detail'),
    windSpeed: document.getElementById('windSpeed'),
  };

  const values = {
    density: document.getElementById('v-density'),
    coverage: document.getElementById('v-coverage'),
    scale: document.getElementById('v-scale'),
    altitude: document.getElementById('v-altitude'),
    detail: document.getElementById('v-detail'),
    windSpeed: document.getElementById('v-windSpeed'),
  };

  // Set better defaults on sliders
  controls.density.value = 1.0;
  controls.coverage.value = 0.8;
  controls.scale.value = 3.0;
  controls.altitude.value = 0.5;
  controls.detail.value = 1.0;
  controls.windSpeed.value = 0.0;

  Object.keys(controls).forEach(key => {
    controls[key].addEventListener('input', () => {
      values[key].textContent = parseFloat(controls[key].value).toFixed(2);
    });
    values[key].textContent = parseFloat(controls[key].value).toFixed(2);
  });

  // --- Render loop ---
  const startTime = performance.now();

  function frame() {
    resize();

    const elapsed = (performance.now() - startTime) / 1000.0;
    const windSpeed = parseFloat(controls.windSpeed.value);

    // Orbit camera
    const eye = [
      camDist * Math.cos(camPhi) * Math.sin(camTheta),
      camDist * Math.sin(camPhi),
      camDist * Math.cos(camPhi) * Math.cos(camTheta)
    ];

    const aspect = canvas.width / canvas.height;
    const proj = mat4Perspective(Math.PI / 4, aspect, 0.1, 100.0);
    const view = mat4LookAt(eye, target, up);
    const viewProj = mat4Multiply(proj, view);
    const invViewProj = mat4Invert(viewProj);

    // Write camera uniform
    const cameraData = new Float32Array(20);
    cameraData.set(invViewProj, 0);
    cameraData.set(eye, 16);
    device.queue.writeBuffer(cameraBuffer, 0, cameraData);

    // Sync params with UI
    const time = elapsed * windSpeed;
    const density = parseFloat(controls.density.value);
    const altitude = parseFloat(controls.altitude.value);
    const factorMacro = parseFloat(controls.coverage.value);
    const scale = parseFloat(controls.scale.value);
    const detail = parseFloat(controls.detail.value);

    const paramsData = new Float32Array([
      time, time, time, density,
      0.2, altitude, factorMacro, 1.0,
      1.0, scale, scale, scale,
      scale, detail, 0.0, 0.0
    ]);
    device.queue.writeBuffer(paramsBuffer, 0, paramsData);

    const commandEncoder = device.createCommandEncoder();
    const textureView = context.getCurrentTexture().createView();

    const renderPass = commandEncoder.beginRenderPass({
      colorAttachments: [
        {
          view: textureView,
          loadOp: 'clear',
          clearValue: { r: 0.075, g: 0.145, b: 0.25, a: 1.0 },
          storeOp: 'store',
        },
      ],
    });

    renderPass.setPipeline(pipeline);
    renderPass.setBindGroup(0, bindGroup);
    renderPass.draw(3); // full-screen triangle
    renderPass.end();

    device.queue.submit([commandEncoder.finish()]);
    requestAnimationFrame(frame);
  }

  requestAnimationFrame(frame);
}

initWebGPU();
