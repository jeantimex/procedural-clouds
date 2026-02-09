// ============================================================
// Uniforms
// ============================================================

struct Camera {
  invViewProj : mat4x4f,
  position    : vec3f,
  _pad        : f32,
};

struct Params {
  time_pack   : vec4f, // timeNoise, timeVoronoi1, timeVoronoi2, density
  alt_pack    : vec4f, // lowAltDensity, altitude, factorMacro, factorDetail
  scale_pack  : vec4f, // factorShaper, scaleAlt, scaleNoise, scaleVoronoi1
  extra_pack  : vec4f, // scaleVoronoi2, detail, fastMode, skipLight
};

@group(0) @binding(0) var<uniform> camera : Camera;
@group(0) @binding(1) var<uniform> params : Params;
@group(1) @binding(0) var densitySampler : sampler;
@group(1) @binding(1) var densityTex : texture_3d<f32>;
@group(2) @binding(0) var densityStore : texture_storage_3d<rgba16float, write>;

// ============================================================
// Vertex
// ============================================================

struct VSOut {
  @builtin(position) pos : vec4f,
  @location(0)       uv  : vec2f,
};

@vertex
fn vs(@builtin(vertex_index) vi : u32) -> VSOut {
  let pos = array<vec2f, 3>(vec2f(-1.0, -1.0), vec2f(3.0, -1.0), vec2f(-1.0, 3.0));
  var out : VSOut;
  out.pos = vec4f(pos[vi], 0.0, 1.0);
  out.uv  = pos[vi];
  return out;
}

// ============================================================
// Helpers
// ============================================================

fn mapRange(value : f32, fromMin : f32, fromMax : f32, toMin : f32, toMax : f32) -> f32 {
  if (abs(fromMax - fromMin) < 1e-5) { return toMin; }
  let t = (value - fromMin) / (fromMax - fromMin);
  return clamp(mix(toMin, toMax, t), min(toMin, toMax), max(toMin, toMax));
}

fn clamp01(v: f32) -> f32 {
  return clamp(v, 0.0, 1.0);
}

fn sampleDensity(pos: vec3f) -> f32 {
  let uvw = (pos - BOX_MIN) / (BOX_MAX - BOX_MIN);
  if (any(uvw < vec3f(0.0)) || any(uvw > vec3f(1.0))) {
    return 0.0;
  }
  return textureSampleLevel(densityTex, densitySampler, uvw, 0.0).r;
}

// ------------------------------------------------------------
// Cloud Density (100% Blender Node Graph Match)
// ------------------------------------------------------------

fn cloudDensity(pos : vec3f, is_cheap : bool) -> f32 {
  let _unused = is_cheap;
  let timeNoise     = params.time_pack.x;
  let timeVoronoi1  = params.time_pack.y;
  let timeVoronoi2  = params.time_pack.z;
  let densityParam  = params.time_pack.w;

  let lowAltDens    = params.alt_pack.x;
  let altitude      = params.alt_pack.y;
  let factorMacro   = params.alt_pack.z;
  let factorDetail  = params.alt_pack.w;

  let factorShaper  = params.scale_pack.x;
  let scaleAlt      = params.scale_pack.y;
  let scaleNoise    = params.scale_pack.z;
  let scaleVoronoi1 = params.scale_pack.w;

  let scaleVoronoi2 = params.extra_pack.x;
  let detail        = params.extra_pack.y;
  let fastMode      = params.extra_pack.z > 0.5;

  // Blender "Object" coordinates for a cloud layer (Z-up).
  // World Y is treated as Blender Z.
  let objPos = vec3f(pos.x, pos.z, pos.y);
  let zNorm = (pos.y - BOX_MIN.y) / (BOX_MAX.y - BOX_MIN.y);
  let Z = 1.0 - clamp(zNorm, 0.0, 1.0);

  // --- STAGE 1: Altitude Mask ---
  // Map Range.010: Z from [0, Altitude/5] -> [1 - LowAlt, 1]
  let altFromMax = altitude / 5.0;
  let altToMin = 1.0 - lowAltDens;
  let altMaskRamp = mapRange(Z, 0.0, altFromMax, altToMin, 1.0);
  // Noise Texture: 4D, Scale 2.0, Detail 0.0 (FBM normalized)
  let noiseCoord = objPos / scaleNoise;
  let stage1Noise = node_noise_texture_4d_value(
    noiseCoord, timeNoise, 2.0, 0.0, 0.0, 0.0, 0.0, 1.0);
  // Math.008: Multiply (clamped)
  let altitudeMask = clamp01(altMaskRamp * stage1Noise);

  // --- STAGE 2: Macro Voronoi ---
  let v1Coord = objPos / scaleVoronoi1;
  let v1Detail = select(detail, min(detail, 2.0), fastMode);
  let v1dist = select(
    node_tex_voronoi_f1_4d_distance(v1Coord, timeVoronoi1, 5.0, v1Detail, 0.5, 3.0, 1.0, 0.5, 1.0, 0.0, 1.0),
    node_tex_voronoi_f1_4d_distance_fast(v1Coord, timeVoronoi1, 5.0, v1Detail, 0.5, 3.0, 1.0, 0.5, 1.0, 0.0, 1.0),
    fastMode
  );
  let v1mapped = mapRange(v1dist, 0.0, 0.75, factorMacro * -0.4, factorMacro);
  let v1scaled = clamp01(v1mapped * 0.5); // Math.012
  let stage2 = clamp01(altitudeMask + v1scaled); // Math.003

  // --- STAGE 3: Medium Voronoi Detail ---
  let v2Coord = objPos / scaleVoronoi2;
  let v2Detail = select(detail * 5.0, min(detail * 5.0, 3.0), fastMode);
  let v2dist = select(
    node_tex_voronoi_f1_4d_distance(v2Coord, timeVoronoi2, 2.0, v2Detail, 0.75, 2.5, 1.0, 0.5, 1.0, 0.0, 1.0),
    node_tex_voronoi_f1_4d_distance_fast(v2Coord, timeVoronoi2, 2.0, v2Detail, 0.75, 2.5, 1.0, 0.5, 1.0, 0.0, 1.0),
    fastMode
  );
  let v2mapped = mapRange(v2dist, 0.0, 1.0, factorDetail * -0.25, factorDetail);
  let stage3 = clamp01(stage2 + v2mapped); // Math.004

  // --- STAGE 4: Upper Altitude Cutoff ---
  let cutoffFromMin = altitude * scaleAlt;
  let cutoff = mapRange(Z, cutoffFromMin, 0.0, 0.0, 1.0); // Map Range.008 (Blender)
  let shaped = clamp01(stage3 - cutoff); // Math.020
  let finalShaped = clamp01(shaped - (1.0 - factorShaper)); // Math.005

  // --- STAGE 5: Final Multipliers ---
  let falloff = mapRange(Z, 0.0, altitude, 0.0, 1.0); // Map Range.009
  let densityScale = densityParam * 5.0; // Tune for WebGPU raymarching
  return finalShaped * falloff * densityScale; // Math.016
}

// ============================================================
// Ray Marching
// ============================================================

const BOX_MIN = vec3f(-4.5, 0.0, -4.5); // Reduced bounds
const BOX_MAX = vec3f( 4.5, 1.2,  4.5);

struct HitInfo {
  hit   : bool,
  tNear : f32,
  tFar  : f32,
};

fn intersectBox(ro : vec3f, rd : vec3f) -> HitInfo {
  let invRd = 1.0 / rd;
  let t0 = (BOX_MIN - ro) * invRd;
  let t1 = (BOX_MAX - ro) * invRd;
  let tmin = min(t0, t1);
  let tmax = max(t0, t1);
  let tNear = max(tmin.x, max(tmin.y, tmin.z));
  let tFar  = min(tmax.x, min(tmax.y, tmax.z));
  return HitInfo(tFar >= max(tNear, 0.0), tNear, tFar);
}

const SUN_DIR   = vec3f(0.189, 0.943, 0.283); 
const SUN_COLOR = vec3f(1.0, 1.0, 1.0);
const AMBIENT   = vec3f(0.26, 0.30, 0.42);
const BG_COLOR  = vec3f(0.045, 0.10, 0.18);
const NUM_STEPS_ACCURATE = 48;
const NUM_STEPS_FAST = 16;

fn hgPhase(cosTheta: f32, g: f32) -> f32 {
    let g2 = g * g;
    return (1.0 - g2) / (4.0 * 3.14159 * pow(1.0 + g2 - 2.0 * g * cosTheta, 1.5));
}

fn lightMarch(pos : vec3f) -> f32 {
  var shadow = 0.0;
  let steps = 4;
  let stepSize = 0.15;
  for (var i = 1; i <= steps; i++) {
    let p = pos + SUN_DIR * (f32(i) * stepSize);
    shadow += sampleDensity(p) * stepSize;
  }
  return exp(-shadow * 1.5); 
}

fn interleavedGradientNoise(uv: vec2f) -> f32 {
    let magic = vec3f(0.06711056, 0.00583715, 52.9829189);
    return fract(magic.z * fract(dot(uv, magic.xy)));
}

@fragment
fn fs(@builtin(position) fragCoord : vec4f, @location(0) uv : vec2f) -> @location(0) vec4f {
  let fastMode = params.extra_pack.z > 0.5;
  let skipLight = params.extra_pack.w > 0.5;
  let numSteps = select(NUM_STEPS_ACCURATE, NUM_STEPS_FAST, fastMode);
  let world_near = camera.invViewProj * vec4f(uv, 0.0, 1.0);
  let world_far  = camera.invViewProj * vec4f(uv, 1.0, 1.0);
  let ro = camera.position;
  let rd = normalize(world_far.xyz/world_far.w - world_near.xyz/world_near.w);

  let hit = intersectBox(ro, rd);

  let sky = mix(BG_COLOR, vec3f(0.1, 0.2, 0.4), clamp(rd.y * 0.5 + 0.5, 0.0, 1.0));
  let sunTheta = dot(rd, SUN_DIR);
  let finalSky = sky + pow(max(sunTheta, 0.0), 64.0) * SUN_COLOR * 0.8;

  var outColor = finalSky;

  if (hit.hit) {
    let tEntry = max(hit.tNear, 0.0);
    let tExit  = hit.tFar;
    let stepSize = (tExit - tEntry) / f32(numSteps);
    let dither = interleavedGradientNoise(fragCoord.xy);
    
    var pos = ro + rd * (tEntry + stepSize * dither);
    var transmittance = 1.0;
    var color = vec3f(0.0);
    let phase = mix(1.0, hgPhase(sunTheta, 0.45), 0.6);

    for (var i = 0; i < numSteps; i++) {
      let d = sampleDensity(pos);
      if (d > 0.01) {
        let step_trans = exp(-d * stepSize);
        let shadow = select(lightMarch(pos), 1.0, skipLight);
        let scattering = shadow * phase * (1.0 - exp(-d * 1.0));
        let litColor = SUN_COLOR * scattering * 3.5 + AMBIENT * 0.5;

        color += transmittance * (1.0 - step_trans) * litColor;
        transmittance *= step_trans;
        let cutoff = select(0.01, 0.05, fastMode);
        if (transmittance < cutoff) { break; }
      }
      pos += rd * stepSize;
    }
    outColor = color + transmittance * finalSky;
  }
    
  outColor = outColor / (outColor + vec3f(1.0));
  outColor = pow(outColor, vec3f(1.0 / 2.2));
  return vec4f(outColor, 1.0);
}

// ============================================================
// Density Cache Compute
// ============================================================

@compute @workgroup_size(4, 4, 4)
fn cs(@builtin(global_invocation_id) gid : vec3u) {
  let dims = textureDimensions(densityStore);
  if (gid.x >= dims.x || gid.y >= dims.y || gid.z >= dims.z) { return; }

  let uvw = (vec3f(gid) + 0.5) / vec3f(dims);
  let pos = mix(BOX_MIN, BOX_MAX, uvw);
  let d = cloudDensity(pos, false);
  textureStore(densityStore, vec3i(gid), vec4f(d, 0.0, 0.0, 1.0));
}
