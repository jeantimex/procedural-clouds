// ============================================================
// Uniforms
// ============================================================

struct Camera {
  invViewProj : mat4x4f,
  position    : vec3f,
  _pad        : f32,
};

struct Params {
  time : f32,
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
// Fragment — visualize noise on the bounding box
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

  // Sample position for noise (use entry point into box)
  let p = entryPos;
  let t = params.time;

  // Left half: Perlin fBM noise (scale=2, detail=0, like Blender Stage 1)
  // Right half: Voronoi F1 (scale=5, detail=1, like Blender Stage 2)
  if (uv.x < 0.0) {
    let n = noise_fbm_4d(vec4f(p * 2.0, t), 0.0, 0.0, 0.0);
    return vec4f(vec3f(n), 1.0);
  } else {
    let v = fractal_voronoi_f1_4d(vec4f(p, t), 5.0, 1.0, 0.5, 3.0, 1.0, true);
    return vec4f(vec3f(v), 1.0);
  }
}
