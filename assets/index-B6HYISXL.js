(function(){const e=document.createElement("link").relList;if(e&&e.supports&&e.supports("modulepreload"))return;for(const s of document.querySelectorAll('link[rel="modulepreload"]'))n(s);new MutationObserver(s=>{for(const o of s)if(o.type==="childList")for(const a of o.addedNodes)a.tagName==="LINK"&&a.rel==="modulepreload"&&n(a)}).observe(document,{childList:!0,subtree:!0});function i(s){const o={};return s.integrity&&(o.integrity=s.integrity),s.referrerPolicy&&(o.referrerPolicy=s.referrerPolicy),s.crossOrigin==="use-credentials"?o.credentials="include":s.crossOrigin==="anonymous"?o.credentials="omit":o.credentials="same-origin",o}function n(s){if(s.ep)return;s.ep=!0;const o=i(s);fetch(s.href,o)}})();const dt=`// ============================================================
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
  extra_pack  : vec4f, // scaleVoronoi2, detail, rayMarchSteps, skipLight
  cache_pack  : vec4f, // cacheBlend, lightMarchSteps, shadowDarkness, sunIntensity
  bounds_pack : vec4f, // cloudHeight, _pad1, _pad2, _pad3
};

@group(0) @binding(0) var<uniform> camera : Camera;
@group(0) @binding(1) var<uniform> params : Params;
@group(1) @binding(0) var densitySampler : sampler;
@group(1) @binding(1) var densityTex0 : texture_3d<f32>;
@group(1) @binding(2) var densityTex1 : texture_3d<f32>;
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
  let uvw = (pos - BOX_MIN) / (getBoxMax() - BOX_MIN);
  if (any(uvw < vec3f(0.0)) || any(uvw > vec3f(1.0))) {
    return 0.0;
  }
  let a = textureSampleLevel(densityTex0, densitySampler, uvw, 0.0).r;
  let b = textureSampleLevel(densityTex1, densitySampler, uvw, 0.0).r;
  let blend = clamp(params.cache_pack.x, 0.0, 1.0);
  let density = mix(a, b, blend);
  return density;
}

// ------------------------------------------------------------
// Cloud Density (100% Blender Node Graph Match)
// ------------------------------------------------------------

fn cloudDensity(pos : vec3f) -> f32 {
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

  // Blender "Object" coordinates for a cloud layer (Z-up).
  // World Y is treated as Blender Z.
  let objPos = vec3f(pos.x, pos.z, pos.y);
  let zNorm = (pos.y - BOX_MIN.y) / (getBoxMax().y - BOX_MIN.y);
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
  let v1dist = node_tex_voronoi_f1_4d_distance(v1Coord, timeVoronoi1, 5.0, detail, 0.5, 3.0, 1.0, 0.5, 1.0, 0.0, 1.0);
  let v1mapped = mapRange(v1dist, 0.0, 0.75, factorMacro * -0.4, factorMacro);
  let v1scaled = clamp01(v1mapped * 0.5); // Math.012
  let stage2 = clamp01(altitudeMask + v1scaled); // Math.003

  // --- STAGE 3: Medium Voronoi Detail ---
  let v2Coord = objPos / scaleVoronoi2;
  let v2dist = node_tex_voronoi_f1_4d_distance(v2Coord, timeVoronoi2, 2.0, detail * 5.0, 0.75, 2.5, 1.0, 0.5, 1.0, 0.0, 1.0);
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
const BOX_MAX_XZ = 4.5;

fn getBoxMax() -> vec3f {
  return vec3f(BOX_MAX_XZ, params.bounds_pack.x, BOX_MAX_XZ);
}

struct HitInfo {
  hit   : bool,
  tNear : f32,
  tFar  : f32,
};

fn intersectBox(ro : vec3f, rd : vec3f) -> HitInfo {
  let invRd = 1.0 / rd;
  let t0 = (BOX_MIN - ro) * invRd;
  let t1 = (getBoxMax() - ro) * invRd;
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

fn hgPhase(cosTheta: f32, g: f32) -> f32 {
    let g2 = g * g;
    return (1.0 - g2) / (4.0 * 3.14159 * pow(1.0 + g2 - 2.0 * g * cosTheta, 1.5));
}

fn lightMarch(pos : vec3f) -> f32 {
  var shadow = 0.0;
  let steps = i32(params.cache_pack.y);
  let stepSize = 0.15;
  for (var i = 1; i <= steps; i++) {
    let p = pos + SUN_DIR * (f32(i) * stepSize);
    shadow += sampleDensity(p) * stepSize;
  }
  return exp(-shadow * params.cache_pack.z); 
}

fn interleavedGradientNoise(uv: vec2f) -> f32 {
    let magic = vec3f(0.06711056, 0.00583715, 52.9829189);
    return fract(magic.z * fract(dot(uv, magic.xy)));
}

@fragment
fn fs(@builtin(position) fragCoord : vec4f, @location(0) uv : vec2f) -> @location(0) vec4f {
  let skipLight = params.extra_pack.w > 0.5;
  let numSteps = i32(params.extra_pack.z);
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
        let litColor = SUN_COLOR * scattering * params.cache_pack.w + AMBIENT * 0.5;

        color += transmittance * (1.0 - step_trans) * litColor;
        transmittance *= step_trans;
        let cutoff = 0.01;
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

@compute @workgroup_size(8, 8, 4)
fn cs(@builtin(global_invocation_id) gid : vec3u) {
  let dims = textureDimensions(densityStore);
  if (gid.x >= dims.x || gid.y >= dims.y || gid.z >= dims.z) { return; }

  let uvw = (vec3f(gid) + 0.5) / vec3f(dims);
  let pos = mix(BOX_MIN, getBoxMax(), uvw);
  let d = cloudDensity(pos);
  textureStore(densityStore, vec3i(gid), vec4f(d, 0.0, 0.0, 1.0));
}
`,ht=`// ============================================================
// Blender-Compatible 4D Noise & Voronoi (WGSL Port)
// Based on Blender GPU shader sources in ref/
// ============================================================

// ------------------------------------------------------------
// HASH FUNCTIONS (Jenkins Lookup3 - for Noise Texture)
// ------------------------------------------------------------

fn rot_u32(x: u32, k: u32) -> u32 {
    return (x << k) | (x >> (32u - k));
}

fn hash_uint(kx: u32) -> u32 {
    var a = 0xdeadbeefu + (1u << 2u) + 13u;
    var b = a;
    var c = a;
    a += kx;
    c ^= b; c -= rot_u32(b, 14u);
    a ^= c; a -= rot_u32(c, 11u);
    b ^= a; b -= rot_u32(a, 25u);
    c ^= b; c -= rot_u32(b, 16u);
    a ^= c; a -= rot_u32(c, 4u);
    b ^= a; b -= rot_u32(a, 14u);
    c ^= b; c -= rot_u32(b, 24u);
    return c;
}

fn hash_uint2(kx: u32, ky: u32) -> u32 {
    var a = 0xdeadbeefu + (2u << 2u) + 13u;
    var b = a;
    var c = a;
    a += kx;
    b += ky;
    c ^= b; c -= rot_u32(b, 14u);
    a ^= c; a -= rot_u32(c, 11u);
    b ^= a; b -= rot_u32(a, 25u);
    c ^= b; c -= rot_u32(b, 16u);
    a ^= c; a -= rot_u32(c, 4u);
    b ^= a; b -= rot_u32(a, 14u);
    c ^= b; c -= rot_u32(b, 24u);
    return c;
}

fn hash_uint3(kx: u32, ky: u32, kz: u32) -> u32 {
    var a = 0xdeadbeefu + (3u << 2u) + 13u;
    var b = a;
    var c = a;
    a += kx;
    b += ky;
    c += kz;
    c ^= b; c -= rot_u32(b, 14u);
    a ^= c; a -= rot_u32(c, 11u);
    b ^= a; b -= rot_u32(a, 25u);
    c ^= b; c -= rot_u32(b, 16u);
    a ^= c; a -= rot_u32(c, 4u);
    b ^= a; b -= rot_u32(a, 14u);
    c ^= b; c -= rot_u32(b, 24u);
    return c;
}

fn hash_uint4(kx: u32, ky: u32, kz: u32, kw: u32) -> u32 {
    var a = 0xdeadbeefu + (4u << 2u) + 13u;
    var b = a;
    var c = a;
    a += kx;
    b += ky;
    a -= c; a ^= rot_u32(c, 4u);  c += b;
    b -= a; b ^= rot_u32(a, 6u);  a += c;
    c -= b; c ^= rot_u32(b, 8u);  b += a;
    a -= c; a ^= rot_u32(c, 16u); c += b;
    b -= a; b ^= rot_u32(a, 19u); a += c;
    c -= b; c ^= rot_u32(b, 4u);  b += a;
    a += kz;
    b += kw;
    c ^= b; c -= rot_u32(b, 14u);
    a ^= c; a -= rot_u32(c, 11u);
    b ^= a; b -= rot_u32(a, 25u);
    c ^= b; c -= rot_u32(b, 16u);
    a ^= c; a -= rot_u32(c, 4u);
    b ^= a; b -= rot_u32(a, 14u);
    c ^= b; c -= rot_u32(b, 24u);
    return c;
}

fn hash_uint4_to_float(kx: u32, ky: u32, kz: u32, kw: u32) -> f32 {
    return f32(hash_uint4(kx, ky, kz, kw)) / f32(0xFFFFFFFFu);
}

fn hash_vec4_to_vec4(k: vec4f) -> vec4f {
    return vec4f(
        hash_uint4_to_float(bitcast<u32>(k.x), bitcast<u32>(k.y), bitcast<u32>(k.z), bitcast<u32>(k.w)),
        hash_uint4_to_float(bitcast<u32>(k.w), bitcast<u32>(k.x), bitcast<u32>(k.y), bitcast<u32>(k.z)),
        hash_uint4_to_float(bitcast<u32>(k.z), bitcast<u32>(k.w), bitcast<u32>(k.x), bitcast<u32>(k.y)),
        hash_uint4_to_float(bitcast<u32>(k.y), bitcast<u32>(k.z), bitcast<u32>(k.w), bitcast<u32>(k.x))
    );
}

// ------------------------------------------------------------
// PERLIN NOISE HELPERS
// ------------------------------------------------------------

fn noise_fade(t: f32) -> f32 {
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0);
}

fn tri_mix(v0: f32, v1: f32, v2: f32, v3: f32,
           v4: f32, v5: f32, v6: f32, v7: f32,
           x: f32, y: f32, z: f32) -> f32 {
    let x1 = 1.0 - x;
    let y1 = 1.0 - y;
    let z1 = 1.0 - z;
    return z1 * (y1 * (v0 * x1 + v1 * x) + y * (v2 * x1 + v3 * x)) +
           z * (y1 * (v4 * x1 + v5 * x) + y * (v6 * x1 + v7 * x));
}

fn quad_mix(v0: f32, v1: f32, v2: f32, v3: f32,
            v4: f32, v5: f32, v6: f32, v7: f32,
            v8: f32, v9: f32, v10: f32, v11: f32,
            v12: f32, v13: f32, v14: f32, v15: f32,
            x: f32, y: f32, z: f32, w: f32) -> f32 {
    return mix(tri_mix(v0, v1, v2, v3, v4, v5, v6, v7, x, y, z),
               tri_mix(v8, v9, v10, v11, v12, v13, v14, v15, x, y, z),
               w);
}

fn noiseg_4d(hash: u32, x: f32, y: f32, z: f32, w: f32) -> f32 {
    let h = hash & 31u;
    let u = select(x, y, h >= 24u);
    let v = select(y, z, h >= 16u);
    let s = select(z, w, h >= 8u);
    let res = select(u, -u, (h & 1u) != 0u);
    let res_v = select(v, -v, (h & 2u) != 0u);
    let res_s = select(s, -s, (h & 4u) != 0u);
    return res + res_v + res_s;
}

fn perlin_noise_4d(position: vec4f) -> f32 {
    let pf = floor(position);
    let X = i32(pf.x);
    let Y = i32(pf.y);
    let Z = i32(pf.z);
    let W = i32(pf.w);

    let fx = position.x - pf.x;
    let fy = position.y - pf.y;
    let fz = position.z - pf.z;
    let fw = position.w - pf.w;

    let u = noise_fade(fx);
    let v = noise_fade(fy);
    let t = noise_fade(fz);
    let s = noise_fade(fw);

    return quad_mix(
        noiseg_4d(hash_uint4(u32(X),   u32(Y),   u32(Z),   u32(W)),   fx,       fy,       fz,       fw),
        noiseg_4d(hash_uint4(u32(X+1), u32(Y),   u32(Z),   u32(W)),   fx-1.0,   fy,       fz,       fw),
        noiseg_4d(hash_uint4(u32(X),   u32(Y+1), u32(Z),   u32(W)),   fx,       fy-1.0,   fz,       fw),
        noiseg_4d(hash_uint4(u32(X+1), u32(Y+1), u32(Z),   u32(W)),   fx-1.0,   fy-1.0,   fz,       fw),
        noiseg_4d(hash_uint4(u32(X),   u32(Y),   u32(Z+1), u32(W)),   fx,       fy,       fz-1.0,   fw),
        noiseg_4d(hash_uint4(u32(X+1), u32(Y),   u32(Z+1), u32(W)),   fx-1.0,   fy,       fz-1.0,   fw),
        noiseg_4d(hash_uint4(u32(X),   u32(Y+1), u32(Z+1), u32(W)),   fx,       fy-1.0,   fz-1.0,   fw),
        noiseg_4d(hash_uint4(u32(X+1), u32(Y+1), u32(Z+1), u32(W)),   fx-1.0,   fy-1.0,   fz-1.0,   fw),
        noiseg_4d(hash_uint4(u32(X),   u32(Y),   u32(Z),   u32(W+1)), fx,       fy,       fz,       fw-1.0),
        noiseg_4d(hash_uint4(u32(X+1), u32(Y),   u32(Z),   u32(W+1)), fx-1.0,   fy,       fz,       fw-1.0),
        noiseg_4d(hash_uint4(u32(X),   u32(Y+1), u32(Z),   u32(W+1)), fx,       fy-1.0,   fz,       fw-1.0),
        noiseg_4d(hash_uint4(u32(X+1), u32(Y+1), u32(Z),   u32(W+1)), fx-1.0,   fy-1.0,   fz,       fw-1.0),
        noiseg_4d(hash_uint4(u32(X),   u32(Y),   u32(Z+1), u32(W+1)), fx,       fy,       fz-1.0,   fw-1.0),
        noiseg_4d(hash_uint4(u32(X+1), u32(Y),   u32(Z+1), u32(W+1)), fx-1.0,   fy,       fz-1.0,   fw-1.0),
        noiseg_4d(hash_uint4(u32(X),   u32(Y+1), u32(Z+1), u32(W+1)), fx,       fy-1.0,   fz-1.0,   fw-1.0),
        noiseg_4d(hash_uint4(u32(X+1), u32(Y+1), u32(Z+1), u32(W+1)), fx-1.0,   fy-1.0,   fz-1.0,   fw-1.0),
        u, v, t, s);
}

fn noise_fbm(p: vec4f, detail: f32, roughness: f32, lacunarity: f32, normalize: bool) -> f32 {
    var fscale = 1.0;
    var amp = 1.0;
    var maxamp = 0.0;
    var sum = 0.0;

    let d_int = i32(detail);
    for (var i = 0; i <= d_int; i++) {
        let t = perlin_noise_4d(fscale * p);
        sum += t * amp;
        maxamp += amp;
        amp *= roughness;
        fscale *= lacunarity;
    }

    let rmd = detail - floor(detail);
    if (rmd != 0.0) {
        let t = perlin_noise_4d(fscale * p);
        let sum2 = sum + t * amp;
        return select(
            mix(sum, sum2, rmd),
            mix(0.5 + 0.5 * (sum / maxamp), 0.5 + 0.5 * (sum2 / (maxamp + amp)), rmd),
            normalize
        );
    }

    return select(sum, 0.5 + 0.5 * (sum / maxamp), normalize);
}

fn random_vec4_offset(seed: f32) -> vec4f {
    return hash_vec4_to_vec4(vec4f(seed, seed * 1.37, seed * 2.23, seed * 3.11));
}

fn node_noise_texture_4d_value(
    co: vec3f,
    w: f32,
    scale: f32,
    detail: f32,
    roughness: f32,
    lacunarity: f32,
    distortion: f32,
    normalize: f32
) -> f32 {
    var p = vec4f(co, w) * scale;
    if (distortion != 0.0) {
        p += vec4f(
            perlin_noise_4d(p + random_vec4_offset(0.0)) * distortion,
            perlin_noise_4d(p + random_vec4_offset(1.0)) * distortion,
            perlin_noise_4d(p + random_vec4_offset(2.0)) * distortion,
            perlin_noise_4d(p + random_vec4_offset(3.0)) * distortion
        );
    }
    return noise_fbm(p, detail, roughness, lacunarity, normalize != 0.0);
}

// ------------------------------------------------------------
// VORONOI (Blender exact path for F1)
// ------------------------------------------------------------

fn hash_pcg4d_i(v_in: vec4i) -> vec4i {
    var v = v_in * 1664525 + 1013904223;
    v.x += v.y * v.w;
    v.y += v.z * v.x;
    v.z += v.x * v.y;
    v.w += v.y * v.z;
    v = v ^ (v >> vec4u(16u));
    v.x += v.y * v.w;
    v.y += v.z * v.x;
    v.z += v.x * v.y;
    v.w += v.y * v.z;
    return v;
}

fn hash_int4_to_vec4(k: vec4i) -> vec4f {
    let h = hash_pcg4d_i(k);
    return vec4f(h & vec4i(0x7fffffff)) * (1.0 / f32(0x7fffffff));
}

fn hash_int4_to_vec3(k: vec4i) -> vec3f {
    return hash_int4_to_vec4(k).xyz;
}

const SHD_VORONOI_EUCLIDEAN = 0;
const SHD_VORONOI_MANHATTAN = 1;
const SHD_VORONOI_CHEBYCHEV = 2;
const SHD_VORONOI_MINKOWSKI = 3;

const SHD_VORONOI_F1 = 0;

struct VoronoiParams {
    scale: f32,
    detail: f32,
    roughness: f32,
    lacunarity: f32,
    smoothness: f32,
    exponent: f32,
    randomness: f32,
    max_distance: f32,
    normalize: bool,
    feature: i32,
    metric: i32,
};

struct VoronoiOutput {
    Distance: f32,
    Color: vec3f,
    Position: vec4f,
};

fn voronoi_distance(a: vec4f, b: vec4f, params: VoronoiParams) -> f32 {
    if (params.metric == SHD_VORONOI_EUCLIDEAN) {
        return distance(a, b);
    } else if (params.metric == SHD_VORONOI_MANHATTAN) {
        return abs(a.x - b.x) + abs(a.y - b.y) + abs(a.z - b.z) + abs(a.w - b.w);
    } else if (params.metric == SHD_VORONOI_CHEBYCHEV) {
        return max(abs(a.x - b.x), max(abs(a.y - b.y), max(abs(a.z - b.z), abs(a.w - b.w))));
    } else if (params.metric == SHD_VORONOI_MINKOWSKI) {
        return pow(
            pow(abs(a.x - b.x), params.exponent) +
            pow(abs(a.y - b.y), params.exponent) +
            pow(abs(a.z - b.z), params.exponent) +
            pow(abs(a.w - b.w), params.exponent),
            1.0 / params.exponent
        );
    }
    return 0.0;
}

fn voronoi_f1(params: VoronoiParams, coord: vec4f) -> VoronoiOutput {
    let cellPosition_f = floor(coord);
    let localPosition = coord - cellPosition_f;
    let cellPosition = vec4i(cellPosition_f);

    var minDistance = 3.402823466e+38;
    var targetOffset = vec4i(0);
    var targetPosition = vec4f(0.0);

    for (var u = -1; u <= 1; u++) {
        for (var k = -1; k <= 1; k++) {
            for (var j = -1; j <= 1; j++) {
                for (var i = -1; i <= 1; i++) {
                    let cellOffset = vec4i(i, j, k, u);
                    let pointPosition = vec4f(cellOffset) + hash_int4_to_vec4(cellPosition + cellOffset) * params.randomness;
                    let distanceToPoint = voronoi_distance(pointPosition, localPosition, params);
                    if (distanceToPoint < minDistance) {
                        targetOffset = cellOffset;
                        minDistance = distanceToPoint;
                        targetPosition = pointPosition;
                    }
                }
            }
        }
    }

    var octave: VoronoiOutput;
    octave.Distance = minDistance;
    octave.Color = hash_int4_to_vec3(cellPosition + targetOffset);
    octave.Position = targetPosition + cellPosition_f;
    return octave;
}

fn fractal_voronoi_x_fx(params: VoronoiParams, coord: vec4f) -> VoronoiOutput {
    var amplitude = 1.0;
    var max_amplitude = 0.0;
    var scale = 1.0;

    var Output: VoronoiOutput;
    Output.Distance = 0.0;
    Output.Color = vec3f(0.0);
    Output.Position = vec4f(0.0);

    let zero_input = params.detail == 0.0 || params.roughness == 0.0;
    let max_i = i32(ceil(params.detail));

    for (var i = 0; i <= max_i; i++) {
        let octave = voronoi_f1(params, coord * scale);
        if (zero_input) {
            max_amplitude = 1.0;
            Output = octave;
            break;
        } else if (f32(i) <= params.detail) {
            max_amplitude += amplitude;
            Output.Distance += octave.Distance * amplitude;
            Output.Color += octave.Color * amplitude;
            Output.Position = mix(Output.Position, octave.Position / scale, amplitude);
            scale *= params.lacunarity;
            amplitude *= params.roughness;
        } else {
            let remainder = params.detail - floor(params.detail);
            if (remainder != 0.0) {
                max_amplitude = mix(max_amplitude, max_amplitude + amplitude, remainder);
                Output.Distance = mix(Output.Distance, Output.Distance + octave.Distance * amplitude, remainder);
                Output.Color = mix(Output.Color, Output.Color + octave.Color * amplitude, remainder);
                Output.Position = mix(Output.Position, mix(Output.Position, octave.Position / scale, amplitude), remainder);
            }
        }
    }

    if (params.normalize) {
        Output.Distance /= max_amplitude * params.max_distance;
        Output.Color /= max_amplitude;
    }

    Output.Position = Output.Position / params.scale;
    return Output;
}

fn node_tex_voronoi_f1_4d_distance(
    coord: vec3f,
    w: f32,
    scale: f32,
    detail: f32,
    roughness: f32,
    lacunarity: f32,
    smoothness: f32,
    exponent: f32,
    randomness: f32,
    metric: f32,
    normalize: f32
) -> f32 {
    var params: VoronoiParams;
    params.feature = SHD_VORONOI_F1;
    params.metric = i32(metric);
    params.scale = scale;
    params.detail = clamp(detail, 0.0, 15.0);
    params.roughness = clamp(roughness, 0.0, 1.0);
    params.lacunarity = lacunarity;
    params.smoothness = clamp(smoothness / 2.0, 0.0, 0.5);
    params.exponent = exponent;
    params.randomness = clamp(randomness, 0.0, 1.0);
    params.max_distance = 0.0;
    params.normalize = normalize != 0.0;

    let w_scaled = w * scale;
    let coord_scaled = coord * scale;
    params.max_distance = voronoi_distance(vec4f(0.0), vec4f(0.5 + 0.5 * params.randomness), params);
    let Output = fractal_voronoi_x_fx(params, vec4f(coord_scaled, w_scaled));
    return Output.Distance;
}

`;/**
 * lil-gui
 * https://lil-gui.georgealways.com
 * @version 0.21.0
 * @author George Michael Brower
 * @license MIT
 */class M{constructor(e,i,n,s,o="div"){this.parent=e,this.object=i,this.property=n,this._disabled=!1,this._hidden=!1,this.initialValue=this.getValue(),this.domElement=document.createElement(o),this.domElement.classList.add("lil-controller"),this.domElement.classList.add(s),this.$name=document.createElement("div"),this.$name.classList.add("lil-name"),M.nextNameID=M.nextNameID||0,this.$name.id=`lil-gui-name-${++M.nextNameID}`,this.$widget=document.createElement("div"),this.$widget.classList.add("lil-widget"),this.$disable=this.$widget,this.domElement.appendChild(this.$name),this.domElement.appendChild(this.$widget),this.domElement.addEventListener("keydown",a=>a.stopPropagation()),this.domElement.addEventListener("keyup",a=>a.stopPropagation()),this.parent.children.push(this),this.parent.controllers.push(this),this.parent.$children.appendChild(this.domElement),this._listenCallback=this._listenCallback.bind(this),this.name(n)}name(e){return this._name=e,this.$name.textContent=e,this}onChange(e){return this._onChange=e,this}_callOnChange(){this.parent._callOnChange(this),this._onChange!==void 0&&this._onChange.call(this,this.getValue()),this._changed=!0}onFinishChange(e){return this._onFinishChange=e,this}_callOnFinishChange(){this._changed&&(this.parent._callOnFinishChange(this),this._onFinishChange!==void 0&&this._onFinishChange.call(this,this.getValue())),this._changed=!1}reset(){return this.setValue(this.initialValue),this._callOnFinishChange(),this}enable(e=!0){return this.disable(!e)}disable(e=!0){return e===this._disabled?this:(this._disabled=e,this.domElement.classList.toggle("lil-disabled",e),this.$disable.toggleAttribute("disabled",e),this)}show(e=!0){return this._hidden=!e,this.domElement.style.display=this._hidden?"none":"",this}hide(){return this.show(!1)}options(e){const i=this.parent.add(this.object,this.property,e);return i.name(this._name),this.destroy(),i}min(e){return this}max(e){return this}step(e){return this}decimals(e){return this}listen(e=!0){return this._listening=e,this._listenCallbackID!==void 0&&(cancelAnimationFrame(this._listenCallbackID),this._listenCallbackID=void 0),this._listening&&this._listenCallback(),this}_listenCallback(){this._listenCallbackID=requestAnimationFrame(this._listenCallback);const e=this.save();e!==this._listenPrevValue&&this.updateDisplay(),this._listenPrevValue=e}getValue(){return this.object[this.property]}setValue(e){return this.getValue()!==e&&(this.object[this.property]=e,this._callOnChange(),this.updateDisplay()),this}updateDisplay(){return this}load(e){return this.setValue(e),this._callOnFinishChange(),this}save(){return this.getValue()}destroy(){this.listen(!1),this.parent.children.splice(this.parent.children.indexOf(this),1),this.parent.controllers.splice(this.parent.controllers.indexOf(this),1),this.parent.$children.removeChild(this.domElement)}}class pt extends M{constructor(e,i,n){super(e,i,n,"lil-boolean","label"),this.$input=document.createElement("input"),this.$input.setAttribute("type","checkbox"),this.$input.setAttribute("aria-labelledby",this.$name.id),this.$widget.appendChild(this.$input),this.$input.addEventListener("change",()=>{this.setValue(this.$input.checked),this._callOnFinishChange()}),this.$disable=this.$input,this.updateDisplay()}updateDisplay(){return this.$input.checked=this.getValue(),this}}function Z(t){let e,i;return(e=t.match(/(#|0x)?([a-f0-9]{6})/i))?i=e[2]:(e=t.match(/rgb\(\s*(\d*)\s*,\s*(\d*)\s*,\s*(\d*)\s*\)/))?i=parseInt(e[1]).toString(16).padStart(2,0)+parseInt(e[2]).toString(16).padStart(2,0)+parseInt(e[3]).toString(16).padStart(2,0):(e=t.match(/^#?([a-f0-9])([a-f0-9])([a-f0-9])$/i))&&(i=e[1]+e[1]+e[2]+e[2]+e[3]+e[3]),i?"#"+i:!1}const ft={isPrimitive:!0,match:t=>typeof t=="string",fromHexString:Z,toHexString:Z},P={isPrimitive:!0,match:t=>typeof t=="number",fromHexString:t=>parseInt(t.substring(1),16),toHexString:t=>"#"+t.toString(16).padStart(6,0)},gt={isPrimitive:!1,match:t=>Array.isArray(t)||ArrayBuffer.isView(t),fromHexString(t,e,i=1){const n=P.fromHexString(t);e[0]=(n>>16&255)/255*i,e[1]=(n>>8&255)/255*i,e[2]=(n&255)/255*i},toHexString([t,e,i],n=1){n=255/n;const s=t*n<<16^e*n<<8^i*n<<0;return P.toHexString(s)}},vt={isPrimitive:!1,match:t=>Object(t)===t,fromHexString(t,e,i=1){const n=P.fromHexString(t);e.r=(n>>16&255)/255*i,e.g=(n>>8&255)/255*i,e.b=(n&255)/255*i},toHexString({r:t,g:e,b:i},n=1){n=255/n;const s=t*n<<16^e*n<<8^i*n<<0;return P.toHexString(s)}},mt=[ft,P,gt,vt];function _t(t){return mt.find(e=>e.match(t))}class xt extends M{constructor(e,i,n,s){super(e,i,n,"lil-color"),this.$input=document.createElement("input"),this.$input.setAttribute("type","color"),this.$input.setAttribute("tabindex",-1),this.$input.setAttribute("aria-labelledby",this.$name.id),this.$text=document.createElement("input"),this.$text.setAttribute("type","text"),this.$text.setAttribute("spellcheck","false"),this.$text.setAttribute("aria-labelledby",this.$name.id),this.$display=document.createElement("div"),this.$display.classList.add("lil-display"),this.$display.appendChild(this.$input),this.$widget.appendChild(this.$display),this.$widget.appendChild(this.$text),this._format=_t(this.initialValue),this._rgbScale=s,this._initialValueHexString=this.save(),this._textFocused=!1,this.$input.addEventListener("input",()=>{this._setValueFromHexString(this.$input.value)}),this.$input.addEventListener("blur",()=>{this._callOnFinishChange()}),this.$text.addEventListener("input",()=>{const o=Z(this.$text.value);o&&this._setValueFromHexString(o)}),this.$text.addEventListener("focus",()=>{this._textFocused=!0,this.$text.select()}),this.$text.addEventListener("blur",()=>{this._textFocused=!1,this.updateDisplay(),this._callOnFinishChange()}),this.$disable=this.$text,this.updateDisplay()}reset(){return this._setValueFromHexString(this._initialValueHexString),this}_setValueFromHexString(e){if(this._format.isPrimitive){const i=this._format.fromHexString(e);this.setValue(i)}else this._format.fromHexString(e,this.getValue(),this._rgbScale),this._callOnChange(),this.updateDisplay()}save(){return this._format.toHexString(this.getValue(),this._rgbScale)}load(e){return this._setValueFromHexString(e),this._callOnFinishChange(),this}updateDisplay(){return this.$input.value=this._format.toHexString(this.getValue(),this._rgbScale),this._textFocused||(this.$text.value=this.$input.value.substring(1)),this.$display.style.backgroundColor=this.$input.value,this}}class W extends M{constructor(e,i,n){super(e,i,n,"lil-function"),this.$button=document.createElement("button"),this.$button.appendChild(this.$name),this.$widget.appendChild(this.$button),this.$button.addEventListener("click",s=>{s.preventDefault(),this.getValue().call(this.object),this._callOnChange()}),this.$button.addEventListener("touchstart",()=>{},{passive:!0}),this.$disable=this.$button}}class bt extends M{constructor(e,i,n,s,o,a){super(e,i,n,"lil-number"),this._initInput(),this.min(s),this.max(o);const l=a!==void 0;this.step(l?a:this._getImplicitStep(),l),this.updateDisplay()}decimals(e){return this._decimals=e,this.updateDisplay(),this}min(e){return this._min=e,this._onUpdateMinMax(),this}max(e){return this._max=e,this._onUpdateMinMax(),this}step(e,i=!0){return this._step=e,this._stepExplicit=i,this}updateDisplay(){const e=this.getValue();if(this._hasSlider){let i=(e-this._min)/(this._max-this._min);i=Math.max(0,Math.min(i,1)),this.$fill.style.width=i*100+"%"}return this._inputFocused||(this.$input.value=this._decimals===void 0?e:e.toFixed(this._decimals)),this}_initInput(){this.$input=document.createElement("input"),this.$input.setAttribute("type","text"),this.$input.setAttribute("aria-labelledby",this.$name.id),window.matchMedia("(pointer: coarse)").matches&&(this.$input.setAttribute("type","number"),this.$input.setAttribute("step","any")),this.$widget.appendChild(this.$input),this.$disable=this.$input;const i=()=>{let r=parseFloat(this.$input.value);isNaN(r)||(this._stepExplicit&&(r=this._snap(r)),this.setValue(this._clamp(r)))},n=r=>{const c=parseFloat(this.$input.value);isNaN(c)||(this._snapClampSetValue(c+r),this.$input.value=this.getValue())},s=r=>{r.key==="Enter"&&this.$input.blur(),r.code==="ArrowUp"&&(r.preventDefault(),n(this._step*this._arrowKeyMultiplier(r))),r.code==="ArrowDown"&&(r.preventDefault(),n(this._step*this._arrowKeyMultiplier(r)*-1))},o=r=>{this._inputFocused&&(r.preventDefault(),n(this._step*this._normalizeMouseWheel(r)))};let a=!1,l,_,d,b,h;const v=5,g=r=>{l=r.clientX,_=d=r.clientY,a=!0,b=this.getValue(),h=0,window.addEventListener("mousemove",x),window.addEventListener("mouseup",y)},x=r=>{if(a){const c=r.clientX-l,w=r.clientY-_;Math.abs(w)>v?(r.preventDefault(),this.$input.blur(),a=!1,this._setDraggingStyle(!0,"vertical")):Math.abs(c)>v&&y()}if(!a){const c=r.clientY-d;h-=c*this._step*this._arrowKeyMultiplier(r),b+h>this._max?h=this._max-b:b+h<this._min&&(h=this._min-b),this._snapClampSetValue(b+h)}d=r.clientY},y=()=>{this._setDraggingStyle(!1,"vertical"),this._callOnFinishChange(),window.removeEventListener("mousemove",x),window.removeEventListener("mouseup",y)},C=()=>{this._inputFocused=!0},u=()=>{this._inputFocused=!1,this.updateDisplay(),this._callOnFinishChange()};this.$input.addEventListener("input",i),this.$input.addEventListener("keydown",s),this.$input.addEventListener("wheel",o,{passive:!1}),this.$input.addEventListener("mousedown",g),this.$input.addEventListener("focus",C),this.$input.addEventListener("blur",u)}_initSlider(){this._hasSlider=!0,this.$slider=document.createElement("div"),this.$slider.classList.add("lil-slider"),this.$fill=document.createElement("div"),this.$fill.classList.add("lil-fill"),this.$slider.appendChild(this.$fill),this.$widget.insertBefore(this.$slider,this.$input),this.domElement.classList.add("lil-has-slider");const e=(u,r,c,w,O)=>(u-r)/(c-r)*(O-w)+w,i=u=>{const r=this.$slider.getBoundingClientRect();let c=e(u,r.left,r.right,this._min,this._max);this._snapClampSetValue(c)},n=u=>{this._setDraggingStyle(!0),i(u.clientX),window.addEventListener("mousemove",s),window.addEventListener("mouseup",o)},s=u=>{i(u.clientX)},o=()=>{this._callOnFinishChange(),this._setDraggingStyle(!1),window.removeEventListener("mousemove",s),window.removeEventListener("mouseup",o)};let a=!1,l,_;const d=u=>{u.preventDefault(),this._setDraggingStyle(!0),i(u.touches[0].clientX),a=!1},b=u=>{u.touches.length>1||(this._hasScrollBar?(l=u.touches[0].clientX,_=u.touches[0].clientY,a=!0):d(u),window.addEventListener("touchmove",h,{passive:!1}),window.addEventListener("touchend",v))},h=u=>{if(a){const r=u.touches[0].clientX-l,c=u.touches[0].clientY-_;Math.abs(r)>Math.abs(c)?d(u):(window.removeEventListener("touchmove",h),window.removeEventListener("touchend",v))}else u.preventDefault(),i(u.touches[0].clientX)},v=()=>{this._callOnFinishChange(),this._setDraggingStyle(!1),window.removeEventListener("touchmove",h),window.removeEventListener("touchend",v)},g=this._callOnFinishChange.bind(this),x=400;let y;const C=u=>{if(Math.abs(u.deltaX)<Math.abs(u.deltaY)&&this._hasScrollBar)return;u.preventDefault();const c=this._normalizeMouseWheel(u)*this._step;this._snapClampSetValue(this.getValue()+c),this.$input.value=this.getValue(),clearTimeout(y),y=setTimeout(g,x)};this.$slider.addEventListener("mousedown",n),this.$slider.addEventListener("touchstart",b,{passive:!1}),this.$slider.addEventListener("wheel",C,{passive:!1})}_setDraggingStyle(e,i="horizontal"){this.$slider&&this.$slider.classList.toggle("lil-active",e),document.body.classList.toggle("lil-dragging",e),document.body.classList.toggle(`lil-${i}`,e)}_getImplicitStep(){return this._hasMin&&this._hasMax?(this._max-this._min)/1e3:.1}_onUpdateMinMax(){!this._hasSlider&&this._hasMin&&this._hasMax&&(this._stepExplicit||this.step(this._getImplicitStep(),!1),this._initSlider(),this.updateDisplay())}_normalizeMouseWheel(e){let{deltaX:i,deltaY:n}=e;return Math.floor(e.deltaY)!==e.deltaY&&e.wheelDelta&&(i=0,n=-e.wheelDelta/120,n*=this._stepExplicit?1:10),i+-n}_arrowKeyMultiplier(e){let i=this._stepExplicit?1:10;return e.shiftKey?i*=10:e.altKey&&(i/=10),i}_snap(e){let i=0;return this._hasMin?i=this._min:this._hasMax&&(i=this._max),e-=i,e=Math.round(e/this._step)*this._step,e+=i,e=parseFloat(e.toPrecision(15)),e}_clamp(e){return e<this._min&&(e=this._min),e>this._max&&(e=this._max),e}_snapClampSetValue(e){this.setValue(this._clamp(this._snap(e)))}get _hasScrollBar(){const e=this.parent.root.$children;return e.scrollHeight>e.clientHeight}get _hasMin(){return this._min!==void 0}get _hasMax(){return this._max!==void 0}}class yt extends M{constructor(e,i,n,s){super(e,i,n,"lil-option"),this.$select=document.createElement("select"),this.$select.setAttribute("aria-labelledby",this.$name.id),this.$display=document.createElement("div"),this.$display.classList.add("lil-display"),this.$select.addEventListener("change",()=>{this.setValue(this._values[this.$select.selectedIndex]),this._callOnFinishChange()}),this.$select.addEventListener("focus",()=>{this.$display.classList.add("lil-focus")}),this.$select.addEventListener("blur",()=>{this.$display.classList.remove("lil-focus")}),this.$widget.appendChild(this.$select),this.$widget.appendChild(this.$display),this.$disable=this.$select,this.options(s)}options(e){return this._values=Array.isArray(e)?e:Object.values(e),this._names=Array.isArray(e)?e:Object.keys(e),this.$select.replaceChildren(),this._names.forEach(i=>{const n=document.createElement("option");n.textContent=i,this.$select.appendChild(n)}),this.updateDisplay(),this}updateDisplay(){const e=this.getValue(),i=this._values.indexOf(e);return this.$select.selectedIndex=i,this.$display.textContent=i===-1?e:this._names[i],this}}class wt extends M{constructor(e,i,n){super(e,i,n,"lil-string"),this.$input=document.createElement("input"),this.$input.setAttribute("type","text"),this.$input.setAttribute("spellcheck","false"),this.$input.setAttribute("aria-labelledby",this.$name.id),this.$input.addEventListener("input",()=>{this.setValue(this.$input.value)}),this.$input.addEventListener("keydown",s=>{s.code==="Enter"&&this.$input.blur()}),this.$input.addEventListener("blur",()=>{this._callOnFinishChange()}),this.$widget.appendChild(this.$input),this.$disable=this.$input,this.updateDisplay()}updateDisplay(){return this.$input.value=this.getValue(),this}}var kt=`.lil-gui {
  font-family: var(--font-family);
  font-size: var(--font-size);
  line-height: 1;
  font-weight: normal;
  font-style: normal;
  text-align: left;
  color: var(--text-color);
  user-select: none;
  -webkit-user-select: none;
  touch-action: manipulation;
  --background-color: #1f1f1f;
  --text-color: #ebebeb;
  --title-background-color: #111111;
  --title-text-color: #ebebeb;
  --widget-color: #424242;
  --hover-color: #4f4f4f;
  --focus-color: #595959;
  --number-color: #2cc9ff;
  --string-color: #a2db3c;
  --font-size: 11px;
  --input-font-size: 11px;
  --font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Arial, sans-serif;
  --font-family-mono: Menlo, Monaco, Consolas, "Droid Sans Mono", monospace;
  --padding: 4px;
  --spacing: 4px;
  --widget-height: 20px;
  --title-height: calc(var(--widget-height) + var(--spacing) * 1.25);
  --name-width: 45%;
  --slider-knob-width: 2px;
  --slider-input-width: 27%;
  --color-input-width: 27%;
  --slider-input-min-width: 45px;
  --color-input-min-width: 45px;
  --folder-indent: 7px;
  --widget-padding: 0 0 0 3px;
  --widget-border-radius: 2px;
  --checkbox-size: calc(0.75 * var(--widget-height));
  --scrollbar-width: 5px;
}
.lil-gui, .lil-gui * {
  box-sizing: border-box;
  margin: 0;
  padding: 0;
}
.lil-gui.lil-root {
  width: var(--width, 245px);
  display: flex;
  flex-direction: column;
  background: var(--background-color);
}
.lil-gui.lil-root > .lil-title {
  background: var(--title-background-color);
  color: var(--title-text-color);
}
.lil-gui.lil-root > .lil-children {
  overflow-x: hidden;
  overflow-y: auto;
}
.lil-gui.lil-root > .lil-children::-webkit-scrollbar {
  width: var(--scrollbar-width);
  height: var(--scrollbar-width);
  background: var(--background-color);
}
.lil-gui.lil-root > .lil-children::-webkit-scrollbar-thumb {
  border-radius: var(--scrollbar-width);
  background: var(--focus-color);
}
@media (pointer: coarse) {
  .lil-gui.lil-allow-touch-styles, .lil-gui.lil-allow-touch-styles .lil-gui {
    --widget-height: 28px;
    --padding: 6px;
    --spacing: 6px;
    --font-size: 13px;
    --input-font-size: 16px;
    --folder-indent: 10px;
    --scrollbar-width: 7px;
    --slider-input-min-width: 50px;
    --color-input-min-width: 65px;
  }
}
.lil-gui.lil-force-touch-styles, .lil-gui.lil-force-touch-styles .lil-gui {
  --widget-height: 28px;
  --padding: 6px;
  --spacing: 6px;
  --font-size: 13px;
  --input-font-size: 16px;
  --folder-indent: 10px;
  --scrollbar-width: 7px;
  --slider-input-min-width: 50px;
  --color-input-min-width: 65px;
}
.lil-gui.lil-auto-place, .lil-gui.autoPlace {
  max-height: 100%;
  position: fixed;
  top: 0;
  right: 15px;
  z-index: 1001;
}

.lil-controller {
  display: flex;
  align-items: center;
  padding: 0 var(--padding);
  margin: var(--spacing) 0;
}
.lil-controller.lil-disabled {
  opacity: 0.5;
}
.lil-controller.lil-disabled, .lil-controller.lil-disabled * {
  pointer-events: none !important;
}
.lil-controller > .lil-name {
  min-width: var(--name-width);
  flex-shrink: 0;
  white-space: pre;
  padding-right: var(--spacing);
  line-height: var(--widget-height);
}
.lil-controller .lil-widget {
  position: relative;
  display: flex;
  align-items: center;
  width: 100%;
  min-height: var(--widget-height);
}
.lil-controller.lil-string input {
  color: var(--string-color);
}
.lil-controller.lil-boolean {
  cursor: pointer;
}
.lil-controller.lil-color .lil-display {
  width: 100%;
  height: var(--widget-height);
  border-radius: var(--widget-border-radius);
  position: relative;
}
@media (hover: hover) {
  .lil-controller.lil-color .lil-display:hover:before {
    content: " ";
    display: block;
    position: absolute;
    border-radius: var(--widget-border-radius);
    border: 1px solid #fff9;
    top: 0;
    right: 0;
    bottom: 0;
    left: 0;
  }
}
.lil-controller.lil-color input[type=color] {
  opacity: 0;
  width: 100%;
  height: 100%;
  cursor: pointer;
}
.lil-controller.lil-color input[type=text] {
  margin-left: var(--spacing);
  font-family: var(--font-family-mono);
  min-width: var(--color-input-min-width);
  width: var(--color-input-width);
  flex-shrink: 0;
}
.lil-controller.lil-option select {
  opacity: 0;
  position: absolute;
  width: 100%;
  max-width: 100%;
}
.lil-controller.lil-option .lil-display {
  position: relative;
  pointer-events: none;
  border-radius: var(--widget-border-radius);
  height: var(--widget-height);
  line-height: var(--widget-height);
  max-width: 100%;
  overflow: hidden;
  word-break: break-all;
  padding-left: 0.55em;
  padding-right: 1.75em;
  background: var(--widget-color);
}
@media (hover: hover) {
  .lil-controller.lil-option .lil-display.lil-focus {
    background: var(--focus-color);
  }
}
.lil-controller.lil-option .lil-display.lil-active {
  background: var(--focus-color);
}
.lil-controller.lil-option .lil-display:after {
  font-family: "lil-gui";
  content: "↕";
  position: absolute;
  top: 0;
  right: 0;
  bottom: 0;
  padding-right: 0.375em;
}
.lil-controller.lil-option .lil-widget,
.lil-controller.lil-option select {
  cursor: pointer;
}
@media (hover: hover) {
  .lil-controller.lil-option .lil-widget:hover .lil-display {
    background: var(--hover-color);
  }
}
.lil-controller.lil-number input {
  color: var(--number-color);
}
.lil-controller.lil-number.lil-has-slider input {
  margin-left: var(--spacing);
  width: var(--slider-input-width);
  min-width: var(--slider-input-min-width);
  flex-shrink: 0;
}
.lil-controller.lil-number .lil-slider {
  width: 100%;
  height: var(--widget-height);
  background: var(--widget-color);
  border-radius: var(--widget-border-radius);
  padding-right: var(--slider-knob-width);
  overflow: hidden;
  cursor: ew-resize;
  touch-action: pan-y;
}
@media (hover: hover) {
  .lil-controller.lil-number .lil-slider:hover {
    background: var(--hover-color);
  }
}
.lil-controller.lil-number .lil-slider.lil-active {
  background: var(--focus-color);
}
.lil-controller.lil-number .lil-slider.lil-active .lil-fill {
  opacity: 0.95;
}
.lil-controller.lil-number .lil-fill {
  height: 100%;
  border-right: var(--slider-knob-width) solid var(--number-color);
  box-sizing: content-box;
}

.lil-dragging .lil-gui {
  --hover-color: var(--widget-color);
}
.lil-dragging * {
  cursor: ew-resize !important;
}
.lil-dragging.lil-vertical * {
  cursor: ns-resize !important;
}

.lil-gui .lil-title {
  height: var(--title-height);
  font-weight: 600;
  padding: 0 var(--padding);
  width: 100%;
  text-align: left;
  background: none;
  text-decoration-skip: objects;
}
.lil-gui .lil-title:before {
  font-family: "lil-gui";
  content: "▾";
  padding-right: 2px;
  display: inline-block;
}
.lil-gui .lil-title:active {
  background: var(--title-background-color);
  opacity: 0.75;
}
@media (hover: hover) {
  body:not(.lil-dragging) .lil-gui .lil-title:hover {
    background: var(--title-background-color);
    opacity: 0.85;
  }
  .lil-gui .lil-title:focus {
    text-decoration: underline var(--focus-color);
  }
}
.lil-gui.lil-root > .lil-title:focus {
  text-decoration: none !important;
}
.lil-gui.lil-closed > .lil-title:before {
  content: "▸";
}
.lil-gui.lil-closed > .lil-children {
  transform: translateY(-7px);
  opacity: 0;
}
.lil-gui.lil-closed:not(.lil-transition) > .lil-children {
  display: none;
}
.lil-gui.lil-transition > .lil-children {
  transition-duration: 300ms;
  transition-property: height, opacity, transform;
  transition-timing-function: cubic-bezier(0.2, 0.6, 0.35, 1);
  overflow: hidden;
  pointer-events: none;
}
.lil-gui .lil-children:empty:before {
  content: "Empty";
  padding: 0 var(--padding);
  margin: var(--spacing) 0;
  display: block;
  height: var(--widget-height);
  font-style: italic;
  line-height: var(--widget-height);
  opacity: 0.5;
}
.lil-gui.lil-root > .lil-children > .lil-gui > .lil-title {
  border: 0 solid var(--widget-color);
  border-width: 1px 0;
  transition: border-color 300ms;
}
.lil-gui.lil-root > .lil-children > .lil-gui.lil-closed > .lil-title {
  border-bottom-color: transparent;
}
.lil-gui + .lil-controller {
  border-top: 1px solid var(--widget-color);
  margin-top: 0;
  padding-top: var(--spacing);
}
.lil-gui .lil-gui .lil-gui > .lil-title {
  border: none;
}
.lil-gui .lil-gui .lil-gui > .lil-children {
  border: none;
  margin-left: var(--folder-indent);
  border-left: 2px solid var(--widget-color);
}
.lil-gui .lil-gui .lil-controller {
  border: none;
}

.lil-gui label, .lil-gui input, .lil-gui button {
  -webkit-tap-highlight-color: transparent;
}
.lil-gui input {
  border: 0;
  outline: none;
  font-family: var(--font-family);
  font-size: var(--input-font-size);
  border-radius: var(--widget-border-radius);
  height: var(--widget-height);
  background: var(--widget-color);
  color: var(--text-color);
  width: 100%;
}
@media (hover: hover) {
  .lil-gui input:hover {
    background: var(--hover-color);
  }
  .lil-gui input:active {
    background: var(--focus-color);
  }
}
.lil-gui input:disabled {
  opacity: 1;
}
.lil-gui input[type=text],
.lil-gui input[type=number] {
  padding: var(--widget-padding);
  -moz-appearance: textfield;
}
.lil-gui input[type=text]:focus,
.lil-gui input[type=number]:focus {
  background: var(--focus-color);
}
.lil-gui input[type=checkbox] {
  appearance: none;
  width: var(--checkbox-size);
  height: var(--checkbox-size);
  border-radius: var(--widget-border-radius);
  text-align: center;
  cursor: pointer;
}
.lil-gui input[type=checkbox]:checked:before {
  font-family: "lil-gui";
  content: "✓";
  font-size: var(--checkbox-size);
  line-height: var(--checkbox-size);
}
@media (hover: hover) {
  .lil-gui input[type=checkbox]:focus {
    box-shadow: inset 0 0 0 1px var(--focus-color);
  }
}
.lil-gui button {
  outline: none;
  cursor: pointer;
  font-family: var(--font-family);
  font-size: var(--font-size);
  color: var(--text-color);
  width: 100%;
  border: none;
}
.lil-gui .lil-controller button {
  height: var(--widget-height);
  text-transform: none;
  background: var(--widget-color);
  border-radius: var(--widget-border-radius);
}
@media (hover: hover) {
  .lil-gui .lil-controller button:hover {
    background: var(--hover-color);
  }
  .lil-gui .lil-controller button:focus {
    box-shadow: inset 0 0 0 1px var(--focus-color);
  }
}
.lil-gui .lil-controller button:active {
  background: var(--focus-color);
}

@font-face {
  font-family: "lil-gui";
  src: url("data:application/font-woff2;charset=utf-8;base64,d09GMgABAAAAAALkAAsAAAAABtQAAAKVAAEAAAAAAAAAAAAAAAAAAAAAAAAAAAAAHFQGYACDMgqBBIEbATYCJAMUCwwABCAFhAoHgQQbHAbIDiUFEYVARAAAYQTVWNmz9MxhEgodq49wYRUFKE8GWNiUBxI2LBRaVnc51U83Gmhs0Q7JXWMiz5eteLwrKwuxHO8VFxUX9UpZBs6pa5ABRwHA+t3UxUnH20EvVknRerzQgX6xC/GH6ZUvTcAjAv122dF28OTqCXrPuyaDER30YBA1xnkVutDDo4oCi71Ca7rrV9xS8dZHbPHefsuwIyCpmT7j+MnjAH5X3984UZoFFuJ0yiZ4XEJFxjagEBeqs+e1iyK8Xf/nOuwF+vVK0ur765+vf7txotUi0m3N0m/84RGSrBCNrh8Ee5GjODjF4gnWP+dJrH/Lk9k4oT6d+gr6g/wssA2j64JJGP6cmx554vUZnpZfn6ZfX2bMwPPrlANsB86/DiHjhl0OP+c87+gaJo/gY084s3HoYL/ZkWHTRfBXvvoHnnkHvngKun4KBE/ede7tvq3/vQOxDXB1/fdNz6XbPdcr0Vhpojj9dG+owuSKFsslCi1tgEjirjXdwMiov2EioadxmqTHUCIwo8NgQaeIasAi0fTYSPTbSmwbMOFduyh9wvBrESGY0MtgRjtgQR8Q1bRPohn2UoCRZf9wyYANMXFeJTysqAe0I4mrherOekFdKMrYvJjLvOIUM9SuwYB5DVZUwwVjJJOaUnZCmcEkIZZrKqNvRGRMvmFZsmhP4VMKCSXBhSqUBxgMS7h0cZvEd71AWkEhGWaeMFcNnpqyJkyXgYL7PQ1MoSq0wDAkRtJIijkZSmqYTiSImfLiSWXIZwhRh3Rug2X0kk1Dgj+Iu43u5p98ghopcpSo0Uyc8SnjlYX59WUeaMoDqmVD2TOWD9a4pCRAzf2ECgwGcrHjPOWY9bNxq/OL3I/QjwEAAAA=") format("woff2");
}`;function Ct(t){const e=document.createElement("style");e.innerHTML=t;const i=document.querySelector("head link[rel=stylesheet], head style");i?document.head.insertBefore(e,i):document.head.appendChild(e)}let it=!1;class q{constructor({parent:e,autoPlace:i=e===void 0,container:n,width:s,title:o="Controls",closeFolders:a=!1,injectStyles:l=!0,touchStyles:_=!0}={}){if(this.parent=e,this.root=e?e.root:this,this.children=[],this.controllers=[],this.folders=[],this._closed=!1,this._hidden=!1,this.domElement=document.createElement("div"),this.domElement.classList.add("lil-gui"),this.$title=document.createElement("button"),this.$title.classList.add("lil-title"),this.$title.setAttribute("aria-expanded",!0),this.$title.addEventListener("click",()=>this.openAnimated(this._closed)),this.$title.addEventListener("touchstart",()=>{},{passive:!0}),this.$children=document.createElement("div"),this.$children.classList.add("lil-children"),this.domElement.appendChild(this.$title),this.domElement.appendChild(this.$children),this.title(o),this.parent){this.parent.children.push(this),this.parent.folders.push(this),this.parent.$children.appendChild(this.domElement);return}this.domElement.classList.add("lil-root"),_&&this.domElement.classList.add("lil-allow-touch-styles"),!it&&l&&(Ct(kt),it=!0),n?n.appendChild(this.domElement):i&&(this.domElement.classList.add("lil-auto-place","autoPlace"),document.body.appendChild(this.domElement)),s&&this.domElement.style.setProperty("--width",s+"px"),this._closeFolders=a}add(e,i,n,s,o){if(Object(n)===n)return new yt(this,e,i,n);const a=e[i];switch(typeof a){case"number":return new bt(this,e,i,n,s,o);case"boolean":return new pt(this,e,i);case"string":return new wt(this,e,i);case"function":return new W(this,e,i)}console.error(`gui.add failed
	property:`,i,`
	object:`,e,`
	value:`,a)}addColor(e,i,n=1){return new xt(this,e,i,n)}addFolder(e){const i=new q({parent:this,title:e});return this.root._closeFolders&&i.close(),i}load(e,i=!0){return e.controllers&&this.controllers.forEach(n=>{n instanceof W||n._name in e.controllers&&n.load(e.controllers[n._name])}),i&&e.folders&&this.folders.forEach(n=>{n._title in e.folders&&n.load(e.folders[n._title])}),this}save(e=!0){const i={controllers:{},folders:{}};return this.controllers.forEach(n=>{if(!(n instanceof W)){if(n._name in i.controllers)throw new Error(`Cannot save GUI with duplicate property "${n._name}"`);i.controllers[n._name]=n.save()}}),e&&this.folders.forEach(n=>{if(n._title in i.folders)throw new Error(`Cannot save GUI with duplicate folder "${n._title}"`);i.folders[n._title]=n.save()}),i}open(e=!0){return this._setClosed(!e),this.$title.setAttribute("aria-expanded",!this._closed),this.domElement.classList.toggle("lil-closed",this._closed),this}close(){return this.open(!1)}_setClosed(e){this._closed!==e&&(this._closed=e,this._callOnOpenClose(this))}show(e=!0){return this._hidden=!e,this.domElement.style.display=this._hidden?"none":"",this}hide(){return this.show(!1)}openAnimated(e=!0){return this._setClosed(!e),this.$title.setAttribute("aria-expanded",!this._closed),requestAnimationFrame(()=>{const i=this.$children.clientHeight;this.$children.style.height=i+"px",this.domElement.classList.add("lil-transition");const n=o=>{o.target===this.$children&&(this.$children.style.height="",this.domElement.classList.remove("lil-transition"),this.$children.removeEventListener("transitionend",n))};this.$children.addEventListener("transitionend",n);const s=e?this.$children.scrollHeight:0;this.domElement.classList.toggle("lil-closed",!e),requestAnimationFrame(()=>{this.$children.style.height=s+"px"})}),this}title(e){return this._title=e,this.$title.textContent=e,this}reset(e=!0){return(e?this.controllersRecursive():this.controllers).forEach(n=>n.reset()),this}onChange(e){return this._onChange=e,this}_callOnChange(e){this.parent&&this.parent._callOnChange(e),this._onChange!==void 0&&this._onChange.call(this,{object:e.object,property:e.property,value:e.getValue(),controller:e})}onFinishChange(e){return this._onFinishChange=e,this}_callOnFinishChange(e){this.parent&&this.parent._callOnFinishChange(e),this._onFinishChange!==void 0&&this._onFinishChange.call(this,{object:e.object,property:e.property,value:e.getValue(),controller:e})}onOpenClose(e){return this._onOpenClose=e,this}_callOnOpenClose(e){this.parent&&this.parent._callOnOpenClose(e),this._onOpenClose!==void 0&&this._onOpenClose.call(this,e)}destroy(){this.parent&&(this.parent.children.splice(this.parent.children.indexOf(this),1),this.parent.folders.splice(this.parent.folders.indexOf(this),1)),this.domElement.parentElement&&this.domElement.parentElement.removeChild(this.domElement),Array.from(this.children).forEach(e=>e.destroy())}controllersRecursive(){let e=Array.from(this.controllers);return this.folders.forEach(i=>{e=e.concat(i.controllersRecursive())}),e}foldersRecursive(){let e=Array.from(this.folders);return this.folders.forEach(i=>{e=e.concat(i.foldersRecursive())}),e}}function Et(t){return t&&t.__esModule&&Object.prototype.hasOwnProperty.call(t,"default")?t.default:t}var F={exports:{}},Mt=F.exports,nt;function St(){return nt||(nt=1,(function(t,e){(function(i,n){t.exports=n()})(Mt,function(){var i=function(){function n(g){return a.appendChild(g.dom),g}function s(g){for(var x=0;x<a.children.length;x++)a.children[x].style.display=x===g?"block":"none";o=g}var o=0,a=document.createElement("div");a.style.cssText="position:fixed;top:0;left:0;cursor:pointer;opacity:0.9;z-index:10000",a.addEventListener("click",function(g){g.preventDefault(),s(++o%a.children.length)},!1);var l=(performance||Date).now(),_=l,d=0,b=n(new i.Panel("FPS","#0ff","#002")),h=n(new i.Panel("MS","#0f0","#020"));if(self.performance&&self.performance.memory)var v=n(new i.Panel("MB","#f08","#201"));return s(0),{REVISION:16,dom:a,addPanel:n,showPanel:s,begin:function(){l=(performance||Date).now()},end:function(){d++;var g=(performance||Date).now();if(h.update(g-l,200),g>_+1e3&&(b.update(1e3*d/(g-_),100),_=g,d=0,v)){var x=performance.memory;v.update(x.usedJSHeapSize/1048576,x.jsHeapSizeLimit/1048576)}return g},update:function(){l=this.end()},domElement:a,setMode:s}};return i.Panel=function(n,s,o){var a=1/0,l=0,_=Math.round,d=_(window.devicePixelRatio||1),b=80*d,h=48*d,v=3*d,g=2*d,x=3*d,y=15*d,C=74*d,u=30*d,r=document.createElement("canvas");r.width=b,r.height=h,r.style.cssText="width:80px;height:48px";var c=r.getContext("2d");return c.font="bold "+9*d+"px Helvetica,Arial,sans-serif",c.textBaseline="top",c.fillStyle=o,c.fillRect(0,0,b,h),c.fillStyle=s,c.fillText(n,v,g),c.fillRect(x,y,C,u),c.fillStyle=o,c.globalAlpha=.9,c.fillRect(x,y,C,u),{dom:r,update:function(w,O){a=Math.min(a,w),l=Math.max(l,w),c.fillStyle=o,c.globalAlpha=1,c.fillRect(0,0,b,y),c.fillStyle=s,c.fillText(_(w)+" "+n+" ("+_(a)+"-"+_(l)+")",v,g),c.drawImage(r,x+d,y,C-d,u,x,y,C-d,u),c.fillRect(x+C-d,y,d,u),c.fillStyle=o,c.globalAlpha=.9,c.fillRect(x+C-d,y,d,_((1-w/O)*u))}}},i})})(F)),F.exports}var Ot=St();const At=Et(Ot),$t=ht+dt;function zt(t,e,i,n){const s=1/Math.tan(t/2),o=1/(i-n);return new Float32Array([s/e,0,0,0,0,s,0,0,0,0,n*o,-1,0,0,n*i*o,0])}function Dt(t,e,i){const n=t[0]-e[0],s=t[1]-e[1],o=t[2]-e[2];let a=Math.hypot(n,s,o);const l=[n/a,s/a,o/a],_=i[1]*l[2]-i[2]*l[1],d=i[2]*l[0]-i[0]*l[2],b=i[0]*l[1]-i[1]*l[0];a=Math.hypot(_,d,b);const h=[_/a,d/a,b/a],v=[l[1]*h[2]-l[2]*h[1],l[2]*h[0]-l[0]*h[2],l[0]*h[1]-l[1]*h[0]];return new Float32Array([h[0],v[0],l[0],0,h[1],v[1],l[1],0,h[2],v[2],l[2],0,-(h[0]*t[0]+h[1]*t[1]+h[2]*t[2]),-(v[0]*t[0]+v[1]*t[1]+v[2]*t[2]),-(l[0]*t[0]+l[1]*t[1]+l[2]*t[2]),1])}function Pt(t,e){const i=new Float32Array(16);for(let n=0;n<4;n++)for(let s=0;s<4;s++)i[s*4+n]=t[0+n]*e[s*4+0]+t[4+n]*e[s*4+1]+t[8+n]*e[s*4+2]+t[12+n]*e[s*4+3];return i}function Vt(t){const e=new Float32Array(16);e[0]=t[5]*t[10]*t[15]-t[5]*t[11]*t[14]-t[9]*t[6]*t[15]+t[9]*t[7]*t[14]+t[13]*t[6]*t[11]-t[13]*t[7]*t[10],e[4]=-t[4]*t[10]*t[15]+t[4]*t[11]*t[14]+t[8]*t[6]*t[15]-t[8]*t[7]*t[14]-t[12]*t[6]*t[11]+t[12]*t[7]*t[10],e[8]=t[4]*t[9]*t[15]-t[4]*t[11]*t[13]-t[8]*t[5]*t[15]+t[8]*t[7]*t[13]+t[12]*t[5]*t[11]-t[12]*t[7]*t[9],e[12]=-t[4]*t[9]*t[14]+t[4]*t[10]*t[13]+t[8]*t[5]*t[14]-t[8]*t[6]*t[13]-t[12]*t[5]*t[10]+t[12]*t[6]*t[9],e[1]=-t[1]*t[10]*t[15]+t[1]*t[11]*t[14]+t[9]*t[2]*t[15]-t[9]*t[3]*t[14]-t[13]*t[2]*t[11]+t[13]*t[3]*t[10],e[5]=t[0]*t[10]*t[15]-t[0]*t[11]*t[14]-t[8]*t[2]*t[15]+t[8]*t[3]*t[14]+t[12]*t[2]*t[11]-t[12]*t[3]*t[10],e[9]=-t[0]*t[9]*t[15]+t[0]*t[11]*t[13]+t[8]*t[1]*t[15]-t[8]*t[3]*t[13]-t[12]*t[1]*t[11]+t[12]*t[3]*t[9],e[13]=t[0]*t[9]*t[14]-t[0]*t[10]*t[13]-t[8]*t[1]*t[14]+t[8]*t[2]*t[13]+t[12]*t[1]*t[10]-t[12]*t[2]*t[9],e[2]=t[1]*t[6]*t[15]-t[1]*t[7]*t[14]-t[5]*t[2]*t[15]+t[5]*t[3]*t[14]+t[13]*t[2]*t[7]-t[13]*t[3]*t[6],e[6]=-t[0]*t[6]*t[15]+t[0]*t[7]*t[14]+t[4]*t[2]*t[15]-t[4]*t[3]*t[14]-t[12]*t[2]*t[7]+t[12]*t[3]*t[6],e[10]=t[0]*t[5]*t[15]-t[0]*t[7]*t[13]-t[4]*t[1]*t[15]+t[4]*t[3]*t[13]+t[12]*t[1]*t[7]-t[12]*t[3]*t[5],e[14]=-t[0]*t[5]*t[14]+t[0]*t[6]*t[13]+t[4]*t[1]*t[14]-t[4]*t[2]*t[13]-t[12]*t[1]*t[6]+t[12]*t[2]*t[5],e[3]=-t[1]*t[6]*t[11]+t[1]*t[7]*t[10]+t[5]*t[2]*t[11]-t[5]*t[3]*t[10]-t[9]*t[2]*t[7]+t[9]*t[3]*t[6],e[7]=t[0]*t[6]*t[11]-t[0]*t[7]*t[10]-t[4]*t[2]*t[11]+t[4]*t[3]*t[10]+t[8]*t[2]*t[7]-t[8]*t[3]*t[6],e[11]=-t[0]*t[5]*t[11]+t[0]*t[7]*t[9]+t[4]*t[1]*t[11]-t[4]*t[3]*t[9]-t[8]*t[1]*t[7]+t[8]*t[3]*t[5],e[15]=t[0]*t[5]*t[10]-t[0]*t[6]*t[9]-t[4]*t[1]*t[10]+t[4]*t[2]*t[9]+t[8]*t[1]*t[6]-t[8]*t[2]*t[5];const i=t[0]*e[0]+t[1]*e[4]+t[2]*e[8]+t[3]*e[12];if(Math.abs(i)<1e-10)return new Float32Array(16);const n=1/i;for(let s=0;s<16;s++)e[s]*=n;return e}async function Lt(){if(!navigator.gpu)throw document.body.innerHTML='<p style="color:white;padding:2rem;">WebGPU is not supported in this browser.</p>',new Error("WebGPU not supported");const t=await navigator.gpu.requestAdapter();if(!t)throw new Error("No appropriate GPUAdapter found");const e=await t.requestDevice(),i=document.getElementById("canvas"),n=i.getContext("webgpu"),s=navigator.gpu.getPreferredCanvasFormat();n.configure({device:e,format:s});const o=e.createShaderModule({code:$t}),a=e.createRenderPipeline({layout:"auto",vertex:{module:o,entryPoint:"vs"},fragment:{module:o,entryPoint:"fs",targets:[{format:s}]},primitive:{topology:"triangle-list"}}),l=e.createComputePipeline({layout:"auto",compute:{module:o,entryPoint:"cs"}}),_=e.createBuffer({size:80,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST}),d=e.createBuffer({size:96,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST}),b=e.createBindGroup({layout:a.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:_}},{binding:1,resource:{buffer:d}}]}),h=e.createSampler({magFilter:"linear",minFilter:"linear",addressModeU:"clamp-to-edge",addressModeV:"clamp-to-edge",addressModeW:"clamp-to-edge"});let v=96,g=[null,null],x=null,y=null;function C(f){for(const E of g)E&&E.destroy();g=[0,1].map(()=>e.createTexture({size:[f,f,f],dimension:"3d",format:"rgba16float",usage:GPUTextureUsage.STORAGE_BINDING|GPUTextureUsage.TEXTURE_BINDING})),y=e.createBindGroup({layout:l.getBindGroupLayout(2),entries:[{binding:0,resource:g[0].createView({dimension:"3d"})}]}),x=e.createBindGroup({layout:a.getBindGroupLayout(1),entries:[{binding:0,resource:h},{binding:1,resource:g[0].createView({dimension:"3d"})},{binding:2,resource:g[1].createView({dimension:"3d"})}]})}C(v);const u=e.createBindGroup({layout:l.getBindGroupLayout(0),entries:[{binding:1,resource:{buffer:d}}]});let r=Math.PI/4,c=.5,w=10,O=r,B=c,I=w;const st=[0,.5,0],at=[0,1,0];let R=!1,V=[0,0];i.addEventListener("pointerdown",f=>{R=!0,V=[f.clientX,f.clientY],i.setPointerCapture(f.pointerId)}),i.addEventListener("pointermove",f=>{if(!R)return;const E=f.clientX-V[0],S=f.clientY-V[1];O-=E*.005,B=Math.max(.1,Math.min(1.4,B+S*.005)),V=[f.clientX,f.clientY]}),i.addEventListener("pointerup",f=>{R=!1,i.releasePointerCapture(f.pointerId)}),i.addEventListener("wheel",f=>{I=Math.max(2,Math.min(20,I+f.deltaY*.005)),f.preventDefault()},{passive:!1});function K(){const f=window.devicePixelRatio||1;i.width=i.clientWidth*f,i.height=i.clientHeight*f}window.addEventListener("resize",K),K();const p={density:1,coverage:.8,scale:3.75,altitude:.5,detail:1,windSpeed:.05,skipLight:!1,rayMarchSteps:48,lightMarchSteps:4,shadowDarkness:5,sunIntensity:17,cloudHeight:1.5,cacheResolution:96,cacheUpdateRate:2,cacheSmooth:0},k=new q({title:"Cloud Parameters"});k.add(p,"density",.1,4,.05),k.add(p,"coverage",0,1,.01),k.add(p,"scale",.2,15,.05),k.add(p,"altitude",.1,1,.01),k.add(p,"detail",0,15,.5),k.add(p,"windSpeed",0,2,.05),k.add(p,"skipLight").name("Skip Light March"),k.add(p,"rayMarchSteps",16,64,1).name("Ray Steps"),k.add(p,"lightMarchSteps",1,8,1).name("Light Steps"),k.add(p,"shadowDarkness",.5,20,.1).name("Shadow Dark"),k.add(p,"sunIntensity",.5,20,.1).name("Sun Intensity"),k.add(p,"cloudHeight",.5,5,.1).name("Cloud Height"),k.add(p,"cacheResolution",32,128,1).name("Cache Res").onFinishChange(f=>{const E=Math.max(32,Math.min(128,Math.round(f)));p.cacheResolution=E,v=E,C(v)}),k.add(p,"cacheUpdateRate",1,4,1).name("Cache Update"),k.add(p,"cacheSmooth",0,.95,.01).name("Cache Smooth");const rt=performance.now();let J=0,N=0,H=0,T=0;const m=new Float32Array(24),$=new Float32Array(20);function ot(f,E){const S=p.density,X=p.altitude,U=p.coverage,A=p.scale,G=p.detail;return m[0]=f,m[1]=f,m[2]=f,m[3]=S,m[4]=.2,m[5]=X,m[6]=U,m[7]=1,m[8]=1,m[9]=A,m[10]=A,m[11]=A,m[12]=A,m[13]=G,m[14]=p.rayMarchSteps,m[15]=p.skipLight?1:0,m[16]=E,m[17]=p.lightMarchSteps,m[18]=p.shadowDarkness,m[19]=p.sunIntensity,m[20]=p.cloudHeight,m[21]=1,m[22]=0,m[23]=0,m}const L=new At;L.showPanel(0),document.body.appendChild(L.dom);function Q(){L.begin(),J++;const f=(performance.now()-rt)/1e3,E=p.windSpeed;r+=(O-r)*.12,c+=(B-c)*.12,w+=(I-w)*.12;const S=[w*Math.cos(c)*Math.sin(r),w*Math.sin(c),w*Math.cos(c)*Math.cos(r)],X=i.width/i.height,U=zt(Math.PI/4,X,.1,100),A=Dt(S,st,at),G=Pt(U,A),lt=Vt(G);$.set(lt,0),$[16]=S[0],$[17]=S[1],$[18]=S[2],e.queue.writeBuffer(_,0,$);const Y=f*E,ct=Math.max(1e-5,T-H),tt=Math.min(1,Math.max(0,(Y-H)/ct));let et=tt;p.cacheSmooth>0&&(et=Math.pow(tt,1/(1+p.cacheSmooth*4))),e.queue.writeBuffer(d,0,ot(Y,et));const j=e.createCommandEncoder();if(J%p.cacheUpdateRate===0){H=T,T=Y,N=1-N,y=e.createBindGroup({layout:l.getBindGroupLayout(2),entries:[{binding:0,resource:g[N].createView({dimension:"3d"})}]});const D=j.beginComputePass();D.setPipeline(l),D.setBindGroup(0,u),D.setBindGroup(2,y),D.dispatchWorkgroups(Math.ceil(v/8),Math.ceil(v/8),Math.ceil(v/4)),D.end()}const ut=n.getCurrentTexture().createView(),z=j.beginRenderPass({colorAttachments:[{view:ut,loadOp:"clear",clearValue:{r:.075,g:.145,b:.25,a:1},storeOp:"store"}]});z.setPipeline(a),z.setBindGroup(0,b),z.setBindGroup(1,x),z.draw(3),z.end(),e.queue.submit([j.finish()]),L.end(),requestAnimationFrame(Q)}requestAnimationFrame(Q)}Lt();
