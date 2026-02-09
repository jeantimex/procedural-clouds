import cloudSource from './shaders/cloud.wgsl?raw';
import noiseSource from './shaders/noise.wgsl?raw';
import GUI from 'lil-gui';
import Stats from 'stats.js';

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

  // --- Compute pipeline (density cache) ---
  const computePipeline = device.createComputePipeline({
    layout: 'auto',
    compute: {
      module: shaderModule,
      entryPoint: 'cs',
    },
  });

  // --- Camera uniform buffer ---
  // Layout: mat4x4f (64 bytes) + vec3f (12 bytes) + f32 pad (4 bytes) = 80 bytes
  const cameraBuffer = device.createBuffer({
    size: 80,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });

  // --- Params uniform buffer ---
  // Layout: 6 x vec4f = 96 bytes
  const paramsBuffer = device.createBuffer({
    size: 96,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });

  const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: cameraBuffer } },
      { binding: 1, resource: { buffer: paramsBuffer } },
    ],
  });

  // --- Density cache texture ---
  const linearSampler = device.createSampler({
    magFilter: 'linear',
    minFilter: 'linear',
    addressModeU: 'clamp-to-edge',
    addressModeV: 'clamp-to-edge',
    addressModeW: 'clamp-to-edge',
  });
  let densityRes = 96;
  let densityTextures = [null, null];
  let densitySampleBindGroup = null;
  let densityStoreBindGroup = null;

  function createDensityResources(res) {
    for (const t of densityTextures) {
      if (t) t.destroy();
    }
    densityTextures = [0, 1].map(() => device.createTexture({
      size: [res, res, res],
      dimension: '3d',
      format: 'rgba16float',
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
    }));

    densityStoreBindGroup = device.createBindGroup({
      layout: computePipeline.getBindGroupLayout(2),
      entries: [
        { binding: 0, resource: densityTextures[0].createView({ dimension: '3d' }) },
      ],
    });
    densitySampleBindGroup = device.createBindGroup({
      layout: pipeline.getBindGroupLayout(1),
      entries: [
        { binding: 0, resource: linearSampler },
        { binding: 1, resource: densityTextures[0].createView({ dimension: '3d' }) },
        { binding: 2, resource: densityTextures[1].createView({ dimension: '3d' }) },
      ],
    });
  }

  createDensityResources(densityRes);

  const computeBindGroup = device.createBindGroup({
    layout: computePipeline.getBindGroupLayout(0),
    entries: [
      { binding: 1, resource: { buffer: paramsBuffer } },
    ],
  });

  // --- Camera setup ---
  let camTheta = Math.PI / 4;
  let camPhi = 0.5;
  let camDist = 10.0;
  let targetTheta = camTheta;
  let targetPhi = camPhi;
  let targetDist = camDist;
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
    targetTheta -= dx * 0.005;
    targetPhi = Math.max(0.1, Math.min(1.4, targetPhi + dy * 0.005));
    lastMouse = [e.clientX, e.clientY];
  });

  canvas.addEventListener('pointerup', e => {
    isDragging = false;
    canvas.releasePointerCapture(e.pointerId);
  });

  canvas.addEventListener('wheel', e => {
    targetDist = Math.max(2.0, Math.min(20.0, targetDist + e.deltaY * 0.005));
    e.preventDefault();
  }, { passive: false });

  function resize() {
    const dpr = window.devicePixelRatio || 1; 
    canvas.width = canvas.clientWidth * dpr;
    canvas.height = canvas.clientHeight * dpr;
  }

  window.addEventListener('resize', resize);
  resize();

  // --- UI Controls (lil-gui) ---
  const params = {
    density: 1.0,
    coverage: 0.8,
    scale: 3.75,
    altitude: 0.5,
    detail: 1.0,
    windSpeed: 0.05,
    skipLight: false,
    rayMarchSteps: 48,
    lightMarchSteps: 4,
    shadowDarkness: 5,
    sunIntensity: 17,
    cloudHeight: 1.5,
    cacheResolution: 96,
    cacheUpdateRate: 2,
    cacheSmooth: 0,
  };

  const gui = new GUI({ title: 'Cloud Parameters' });
  gui.add(params, 'density', 0.1, 4.0, 0.05);
  gui.add(params, 'coverage', 0.0, 1.0, 0.01);
  gui.add(params, 'scale', 0.2, 15.0, 0.05);
  gui.add(params, 'altitude', 0.1, 1.0, 0.01);
  gui.add(params, 'detail', 0.0, 15.0, 0.5);
  gui.add(params, 'windSpeed', 0.0, 2.0, 0.05);
  gui.add(params, 'skipLight').name('Skip Light March');
  gui.add(params, 'rayMarchSteps', 16, 64, 1).name('Ray Steps');
  gui.add(params, 'lightMarchSteps', 1, 8, 1).name('Light Steps');
  gui.add(params, 'shadowDarkness', 0.5, 20.0, 0.1).name('Shadow Dark');
  gui.add(params, 'sunIntensity', 0.5, 20.0, 0.1).name('Sun Intensity');
  gui.add(params, 'cloudHeight', 0.5, 5.0, 0.1).name('Cloud Height');
  gui.add(params, 'cacheResolution', 32, 128, 1).name('Cache Res').onFinishChange(v => {
    const next = Math.max(32, Math.min(128, Math.round(v)));
    params.cacheResolution = next;
    densityRes = next;
    createDensityResources(densityRes);
  });
  gui.add(params, 'cacheUpdateRate', 1, 4, 1).name('Cache Update');
  gui.add(params, 'cacheSmooth', 0.0, 0.95, 0.01).name('Cache Smooth');

  // --- Render loop ---
  const startTime = performance.now();

  let frameIndex = 0;
  let cacheIndex = 0;
  let prevCacheTime = 0.0;
  let nextCacheTime = 0.0;

  // Pre-allocated buffers to avoid GC pressure
  const paramsData = new Float32Array(24);
  const cameraData = new Float32Array(20);

  function buildParams(time, cacheBlend) {
    const density = params.density;
    const altitude = params.altitude;
    const factorMacro = params.coverage;
    const scale = params.scale;
    const detail = params.detail;

    paramsData[0] = time;
    paramsData[1] = time;
    paramsData[2] = time;
    paramsData[3] = density;
    paramsData[4] = 0.2;
    paramsData[5] = altitude;
    paramsData[6] = factorMacro;
    paramsData[7] = 1.0;
    paramsData[8] = 1.0;
    paramsData[9] = scale;
    paramsData[10] = scale;
    paramsData[11] = scale;
    paramsData[12] = scale;
    paramsData[13] = detail;
    paramsData[14] = params.rayMarchSteps;
    paramsData[15] = params.skipLight ? 1.0 : 0.0;
    paramsData[16] = cacheBlend;
    paramsData[17] = params.lightMarchSteps;
    paramsData[18] = params.shadowDarkness;
    paramsData[19] = params.sunIntensity;
    paramsData[20] = params.cloudHeight;
    paramsData[21] = 1.0;
    paramsData[22] = 0.0;
    paramsData[23] = 0.0;
    return paramsData;
  }
  const stats = new Stats();
  stats.showPanel(0);
  document.body.appendChild(stats.dom);

  function frame() {
    stats.begin();
    frameIndex++;

    const elapsed = (performance.now() - startTime) / 1000.0;
    const windSpeed = params.windSpeed;

    // Smooth camera inertia
    camTheta += (targetTheta - camTheta) * 0.12;
    camPhi += (targetPhi - camPhi) * 0.12;
    camDist += (targetDist - camDist) * 0.12;

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
    cameraData.set(invViewProj, 0);
    cameraData[16] = eye[0];
    cameraData[17] = eye[1];
    cameraData[18] = eye[2];
    device.queue.writeBuffer(cameraBuffer, 0, cameraData);

    // Sync params with UI
    const time = elapsed * windSpeed;
    const blendDenom = Math.max(1e-5, nextCacheTime - prevCacheTime);
    const linearBlend = Math.min(1.0, Math.max(0.0, (time - prevCacheTime) / blendDenom));
    let cacheBlend = linearBlend;
    if (params.cacheSmooth > 0.0) {
      cacheBlend = Math.pow(linearBlend, 1.0 / (1.0 + params.cacheSmooth * 4.0));
    }

    device.queue.writeBuffer(paramsBuffer, 0, buildParams(time, cacheBlend));

    const commandEncoder = device.createCommandEncoder();

    // --- Compute density cache ---
    if (frameIndex % params.cacheUpdateRate === 0) {
      prevCacheTime = nextCacheTime;
      nextCacheTime = time;
      cacheIndex = 1 - cacheIndex;

      densityStoreBindGroup = device.createBindGroup({
        layout: computePipeline.getBindGroupLayout(2),
        entries: [
          { binding: 0, resource: densityTextures[cacheIndex].createView({ dimension: '3d' }) },
        ],
      });

      const pass = commandEncoder.beginComputePass();
      pass.setPipeline(computePipeline);
      pass.setBindGroup(0, computeBindGroup);
      pass.setBindGroup(2, densityStoreBindGroup);
      pass.dispatchWorkgroups(Math.ceil(densityRes / 8), Math.ceil(densityRes / 8), Math.ceil(densityRes / 4));
      pass.end();
    }
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
    renderPass.setBindGroup(1, densitySampleBindGroup);
    renderPass.draw(3); // full-screen triangle
    renderPass.end();

    device.queue.submit([commandEncoder.finish()]);
    stats.end();
    requestAnimationFrame(frame);
  }

  requestAnimationFrame(frame);
}

initWebGPU();
