// ============================================================
// Uniforms
// ============================================================

struct Camera {
  invViewProj : mat4x4f,
  position    : vec3f,
  _pad        : f32,
};

struct Params {
  time            : f32,
  density         : f32,
  lowAltDensity   : f32,
  altitude        : f32,
  factor          : f32,
  scale           : f32,
  detail          : f32,
  _pad            : f32,
};

@group(0) @binding(0) var<uniform> camera : Camera;
@group(0) @binding(1) var<uniform> params : Params;

// ============================================================
// Vertex — full-screen triangle (3 vertices, no buffers)
// ============================================================

struct VSOut {
  @builtin(position) pos : vec4f,
  @location(0)       uv  : vec2f,
};

@vertex
fn vs(@builtin(vertex_index) vi : u32) -> VSOut {
  let x = f32(i32(vi & 1u) * 4 - 1);
  let y = f32(i32(vi >> 1u) * 4 - 1);
  var out : VSOut;
  out.pos = vec4f(x, y, 0.0, 1.0);
  out.uv  = vec2f(x, y);
  return out;
}

// ============================================================
// Hash functions — Jenkins Lookup3 (for Perlin noise)
// ============================================================

fn rot32(x : u32, k : u32) -> u32 {
  return (x << k) | (x >> (32u - k));
}

fn jenkins_final(a_in : u32, b_in : u32, c_in : u32) -> vec3u {
  var a = a_in; var b = b_in; var c = c_in;
  c ^= b; c -= rot32(b, 14u);
  a ^= c; a -= rot32(c, 11u);
  b ^= a; b -= rot32(a, 25u);
  c ^= b; c -= rot32(b, 16u);
  a ^= c; a -= rot32(c, 4u);
  b ^= a; b -= rot32(a, 14u);
  c ^= b; c -= rot32(b, 24u);
  return vec3u(a, b, c);
}

fn jenkins_mix(a_in : u32, b_in : u32, c_in : u32) -> vec3u {
  var a = a_in; var b = b_in; var c = c_in;
  a -= c; a ^= rot32(c, 4u);  c += b;
  b -= a; b ^= rot32(a, 6u);  a += c;
  c -= b; c ^= rot32(b, 8u);  b += a;
  a -= c; a ^= rot32(c, 16u); c += b;
  b -= a; b ^= rot32(a, 19u); a += c;
  c -= b; c ^= rot32(b, 4u);  b += a;
  return vec3u(a, b, c);
}

fn hash_uint4(kx : u32, ky : u32, kz : u32, kw : u32) -> u32 {
  let init = 0xdeadbeefu + (4u << 2u) + 13u;
  let m = jenkins_mix(init + kx, init + ky, init);
  let f = jenkins_final(m.x + kz, m.y + kw, m.z);
  return f.z;
}

fn hash_uint4_to_float(kx : u32, ky : u32, kz : u32, kw : u32) -> f32 {
  return f32(hash_uint4(kx, ky, kz, kw)) / f32(0xFFFFFFFFu);
}

// ============================================================
// Hash functions — PCG4D (for Voronoi)
// ============================================================

fn hash_pcg4d(v_in : vec4i) -> vec4i {
  var v = v_in * 1664525 + 1013904223;
  v.x += v.y * v.w;
  v.y += v.z * v.x;
  v.z += v.x * v.y;
  v.w += v.y * v.z;
  v = v ^ (v >> vec4u(16));
  v.x += v.y * v.w;
  v.y += v.z * v.x;
  v.z += v.x * v.y;
  v.w += v.y * v.z;
  return v;
}

fn hash_int4_to_vec4(k : vec4i) -> vec4f {
  let h = hash_pcg4d(k);
  return vec4f(h & vec4i(0x7fffffff)) * (1.0 / f32(0x7fffffff));
}

fn hash_int4_to_vec3(k : vec4i) -> vec3f {
  return hash_int4_to_vec4(k).xyz;
}

// ============================================================
// 4D Perlin Noise
// ============================================================

fn fade(t : f32) -> f32 {
  return t * t * t * (t * (t * 6.0 - 15.0) + 10.0);
}

fn floorfrac(x : f32) -> vec2f {
  let x_floor = floor(x);
  return vec2f(x_floor, x - x_floor);
}

fn noiseg_4d(hash_val : u32, x : f32, y : f32, z : f32, w : f32) -> f32 {
  let h = hash_val & 31u;
  let u = select(y, x, h < 24u);
  let v = select(z, y, h < 16u);
  let s = select(w, z, h < 8u);
  return select(u, -u, (h & 1u) != 0u) +
         select(v, -v, (h & 2u) != 0u) +
         select(s, -s, (h & 4u) != 0u);
}

fn tri_mix(v0 : f32, v1 : f32, v2 : f32, v3 : f32,
           v4 : f32, v5 : f32, v6 : f32, v7 : f32,
           x : f32, y : f32, z : f32) -> f32 {
  let x1 = 1.0 - x;
  let y1 = 1.0 - y;
  let z1 = 1.0 - z;
  return z1 * (y1 * (v0 * x1 + v1 * x) + y * (v2 * x1 + v3 * x)) +
         z  * (y1 * (v4 * x1 + v5 * x) + y * (v6 * x1 + v7 * x));
}

fn perlin_noise_4d(pos : vec4f) -> f32 {
  let ffx = floorfrac(pos.x); let X = i32(ffx.x); let fx = ffx.y;
  let ffy = floorfrac(pos.y); let Y = i32(ffy.x); let fy = ffy.y;
  let ffz = floorfrac(pos.z); let Z = i32(ffz.x); let fz = ffz.y;
  let ffw = floorfrac(pos.w); let W = i32(ffw.x); let fw = ffw.y;

  let u = fade(fx);
  let v = fade(fy);
  let t = fade(fz);
  let s = fade(fw);

  let uX = u32(X); let uY = u32(Y); let uZ = u32(Z); let uW = u32(W);

  let lo = tri_mix(
    noiseg_4d(hash_uint4(uX,     uY,     uZ,     uW),     fx,       fy,       fz,       fw),
    noiseg_4d(hash_uint4(uX+1u,  uY,     uZ,     uW),     fx-1.0,   fy,       fz,       fw),
    noiseg_4d(hash_uint4(uX,     uY+1u,  uZ,     uW),     fx,       fy-1.0,   fz,       fw),
    noiseg_4d(hash_uint4(uX+1u,  uY+1u,  uZ,     uW),     fx-1.0,   fy-1.0,   fz,       fw),
    noiseg_4d(hash_uint4(uX,     uY,     uZ+1u,  uW),     fx,       fy,       fz-1.0,   fw),
    noiseg_4d(hash_uint4(uX+1u,  uY,     uZ+1u,  uW),     fx-1.0,   fy,       fz-1.0,   fw),
    noiseg_4d(hash_uint4(uX,     uY+1u,  uZ+1u,  uW),     fx,       fy-1.0,   fz-1.0,   fw),
    noiseg_4d(hash_uint4(uX+1u,  uY+1u,  uZ+1u,  uW),     fx-1.0,   fy-1.0,   fz-1.0,   fw),
    u, v, t);

  let hi = tri_mix(
    noiseg_4d(hash_uint4(uX,     uY,     uZ,     uW+1u),  fx,       fy,       fz,       fw-1.0),
    noiseg_4d(hash_uint4(uX+1u,  uY,     uZ,     uW+1u),  fx-1.0,   fy,       fz,       fw-1.0),
    noiseg_4d(hash_uint4(uX,     uY+1u,  uZ,     uW+1u),  fx,       fy-1.0,   fz,       fw-1.0),
    noiseg_4d(hash_uint4(uX+1u,  uY+1u,  uZ,     uW+1u),  fx-1.0,   fy-1.0,   fz,       fw-1.0),
    noiseg_4d(hash_uint4(uX,     uY,     uZ+1u,  uW+1u),  fx,       fy,       fz-1.0,   fw-1.0),
    noiseg_4d(hash_uint4(uX+1u,  uY,     uZ+1u,  uW+1u),  fx-1.0,   fy,       fz-1.0,   fw-1.0),
    noiseg_4d(hash_uint4(uX,     uY+1u,  uZ+1u,  uW+1u),  fx,       fy-1.0,   fz-1.0,   fw-1.0),
    noiseg_4d(hash_uint4(uX+1u,  uY+1u,  uZ+1u,  uW+1u),  fx-1.0,   fy-1.0,   fz-1.0,   fw-1.0),
    u, v, t);

  return mix(lo, hi, s);
}

// ============================================================
// Fractal Brownian Motion (fBM) wrapping Perlin noise
// ============================================================

fn noise_fbm_4d(p : vec4f, detail : f32, roughness : f32, lacunarity : f32) -> f32 {
  var fscale = 1.0;
  var amp = 1.0;
  var maxamp = 0.0;
  var sum = 0.0;
  let d = max(detail, 0.0);

  for (var i = 0; i <= i32(d); i++) {
    let t = perlin_noise_4d(fscale * p);
    sum += t * amp;
    maxamp += amp;
    amp *= roughness;
    fscale *= lacunarity;
  }

  let rmd = d - floor(d);
  if (rmd != 0.0) {
    let t = perlin_noise_4d(fscale * p);
    let sum2 = sum + t * amp;
    return mix(0.5 + 0.5 * (sum / maxamp), 0.5 + 0.5 * (sum2 / (maxamp + amp)), rmd);
  }

  return 0.5 + 0.5 * (sum / maxamp);
}

// ============================================================
// 4D Voronoi F1 (Euclidean) — single octave
// ============================================================

struct VoronoiResult {
  distance : f32,
  color    : vec3f,
};

fn voronoi_f1_4d(coord : vec4f, randomness : f32) -> VoronoiResult {
  let cellPos_f = floor(coord);
  let localPos  = coord - cellPos_f;
  let cellPos   = vec4i(cellPos_f);

  var minDist = 1e20;
  var targetOffset = vec4i(0);

  for (var u = -1; u <= 1; u++) {
    for (var k = -1; k <= 1; k++) {
      for (var j = -1; j <= 1; j++) {
        for (var i = -1; i <= 1; i++) {
          let offset = vec4i(i, j, k, u);
          let pointPos = vec4f(offset) +
                         hash_int4_to_vec4(cellPos + offset) * randomness;
          let d = distance(pointPos, localPos);
          if (d < minDist) {
            minDist = d;
            targetOffset = offset;
          }
        }
      }
    }
  }

  var result : VoronoiResult;
  result.distance = minDist;
  result.color = hash_int4_to_vec3(cellPos + targetOffset);
  return result;
}

// ============================================================
// Fractal Voronoi (multi-octave wrapper)
// ============================================================

fn fractal_voronoi_f1_4d(coord : vec4f, scale : f32, detail : f32,
                         roughness : f32, lacunarity : f32,
                         randomness : f32, do_normalize : bool) -> f32 {
  let rand = clamp(randomness, 0.0, 1.0);
  let rough = clamp(roughness, 0.0, 1.0);
  let det = clamp(detail, 0.0, 15.0);

  let scaled_coord = coord * scale;

  // max_distance for Euclidean normalization
  let max_distance = distance(vec4f(0.0), vec4f(0.5 + 0.5 * rand));

  let zero_input = det == 0.0 || rough == 0.0;
  var amplitude = 1.0;
  var max_amplitude = 0.0;
  var octave_scale = 1.0;
  var total_dist = 0.0;

  for (var i = 0; i <= i32(ceil(det)); i++) {
    let octave = voronoi_f1_4d(scaled_coord * octave_scale, rand);

    if (zero_input) {
      max_amplitude = 1.0;
      total_dist = octave.distance;
      break;
    } else if (f32(i) <= det) {
      max_amplitude += amplitude;
      total_dist += octave.distance * amplitude;
      octave_scale *= lacunarity;
      amplitude *= rough;
    } else {
      let remainder = det - floor(det);
      if (remainder != 0.0) {
        max_amplitude = mix(max_amplitude, max_amplitude + amplitude, remainder);
        total_dist = mix(total_dist, total_dist + octave.distance * amplitude, remainder);
      }
    }
  }

  if (do_normalize) {
    total_dist /= max_amplitude * max_distance;
  }

  return total_dist;
}

// ============================================================
// Utility: clamped linear map range (matches Blender Map Range)
// ============================================================

fn mapRange(value : f32, fromMin : f32, fromMax : f32, toMin : f32, toMax : f32) -> f32 {
  let t = (value - fromMin) / (fromMax - fromMin);
  return clamp(mix(toMin, toMax, t), min(toMin, toMax), max(toMin, toMax));
}

// ============================================================
// 5-Stage Cloud Density Function
// Ported from Blender "Procedural Clouds Shader" node group
//
// In Blender Z is up. In our box Y is up (box: XZ [-1,1], Y [0,1]).
// All "Object" coordinates = pos in world space.
// Blender Z → our Y for altitude calculations.
// ============================================================

fn cloudDensity(pos : vec3f) -> f32 {
  let altitude     = params.altitude;
  let factor       = params.factor;
  let scale        = params.scale;
  let time         = params.time;
  let detail       = params.detail;
  let lowAltDens   = params.lowAltDensity;
  let densityParam = params.density;

  // Y is our vertical axis (0..1 in the box)
  let Z = pos.y;

  // ----------------------------------------------------------
  // Stage 1: Altitude mask modulated by low-frequency noise
  // ----------------------------------------------------------

  // Math.009: altitude / 5.0
  let altDiv5 = altitude / 5.0;

  // Math.010: lowAltDens * (-1) + 1 = 1 - lowAltDens
  let altToMin = 1.0 - lowAltDens;

  // Map Range.010: Z ∈ [0, altDiv5] → [1-lowAltDens, 1.0], clamped
  let altitudeProfile = mapRange(Z, 0.0, altDiv5, altToMin, 1.0);

  // Noise Texture (4D): Object/Scale, W=time, scale=2, detail=0
  let noiseCoord = vec4f(pos / scale, time) * 2.0;
  let noiseFac = noise_fbm_4d(noiseCoord, 0.0, 0.0, 0.0);

  // Math.008: altitude_profile * noise, clamped
  let altitudeMask = clamp(altitudeProfile * noiseFac, 0.0, 1.0);

  // ----------------------------------------------------------
  // Stage 2: Large-scale Voronoi cells (scale=5, detail=input)
  // ----------------------------------------------------------

  // Voronoi Texture.004: Object/Scale, W=time
  let voronoi1Coord = vec4f(pos / scale, time);
  let v1dist = fractal_voronoi_f1_4d(voronoi1Coord, 5.0, detail, 0.5, 3.0, 1.0, true);

  // Map Range.001: voronoi [0, 0.75] → [Factor*(-0.4), Factor], clamped
  let v1mapped = mapRange(v1dist, 0.0, 0.75, factor * (-0.4), factor);

  // Math.012: v1mapped * 0.5, clamped
  let v1scaled = clamp(v1mapped * 0.5, 0.0, 1.0);

  // Math.003: altitudeMask + v1scaled, clamped
  let accumulated1 = clamp(altitudeMask + v1scaled, 0.0, 1.0);

  // ----------------------------------------------------------
  // Stage 3: Medium-detail Voronoi (scale=2, detail=input*5)
  // ----------------------------------------------------------

  // Voronoi Texture.003: Object/Scale, W=time
  let voronoi2Coord = vec4f(pos / scale, time);
  let v2detail = detail * 5.0;
  let v2dist = fractal_voronoi_f1_4d(voronoi2Coord, 2.0, v2detail, 0.75, 2.5, 1.0, true);

  // Map Range.002: voronoi [0, 1] → [Factor*(-0.25), Factor], clamped
  let v2mapped = mapRange(v2dist, 0.0, 1.0, factor * (-0.25), factor);

  // Math.004: accumulated1 + v2mapped, clamped
  let accumulated2 = clamp(accumulated1 + v2mapped, 0.0, 1.0);

  // ----------------------------------------------------------
  // Stage 4: Lower-altitude cutoff
  // ----------------------------------------------------------

  // Map Range.008: Z from [altitude*scale, 0.0] → [0, 1], clamped
  // This creates a mask that is 1 at bottom and 0 at altitude*scale
  let lowerMask = mapRange(Z, altitude * scale, 0.0, 0.0, 1.0);

  // Math.020: accumulated2 - lowerMask, clamped
  let shaped = clamp(accumulated2 - lowerMask, 0.0, 1.0);

  // Math.002: factor * (-1) + 1 = 1 - factor
  let factorComplement = 1.0 - factor;

  // Math.005: shaped - factorComplement, clamped
  let finalShaped = clamp(shaped - factorComplement, 0.0, 1.0);

  // ----------------------------------------------------------
  // Stage 5: Final density scaling + max-altitude rolloff
  // ----------------------------------------------------------

  // Map Range.009: Z ∈ [0, altitude] → [0, 1], clamped
  let altitudeRamp = mapRange(Z, 0.0, altitude, 0.0, 1.0);

  // Math.011: densityParam * 500.0
  // Math.001: (densityParam * 500) * altitudeRamp
  // Math.016: finalShaped * (densityParam * 500 * altitudeRamp)
  let finalDensity = finalShaped * densityParam * 500.0 * altitudeRamp;

  // Return finalShaped for debug visualization (0..1 range).
  // Switch to finalDensity when ray marching is implemented.
  return finalShaped;
}

// ============================================================
// Ray-Box intersection (AABB slab method)
// ============================================================

const BOX_MIN = vec3f(-1.0, 0.0, -1.0);
const BOX_MAX = vec3f( 1.0, 1.0,  1.0);

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

// ============================================================
// Fragment — visualize cloud density as grayscale on the box
// ============================================================

const BG_COLOR = vec3f(0.075, 0.145, 0.25);

@fragment
fn fs(@location(0) uv : vec2f) -> @location(0) vec4f {
  let ndc_near = vec4f(uv, 0.0, 1.0);
  let ndc_far  = vec4f(uv, 1.0, 1.0);

  let world_near = camera.invViewProj * ndc_near;
  let world_far  = camera.invViewProj * ndc_far;

  let near = world_near.xyz / world_near.w;
  let far  = world_far.xyz  / world_far.w;

  let ro = camera.position;
  let rd = normalize(far - near);

  let hit = intersectBox(ro, rd);

  if (!hit.hit) {
    return vec4f(BG_COLOR, 1.0);
  }

  let tEntry = max(hit.tNear, 0.0);
  let entryPos = ro + rd * tEntry;

  // Sample cloud shape at entry point (already 0..1 from finalShaped)
  let d = cloudDensity(entryPos);
  return vec4f(vec3f(d), 1.0);
}
