// ============================================================
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
