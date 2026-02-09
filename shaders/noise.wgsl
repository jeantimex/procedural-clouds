// ============================================================
// Blender-Compatible 4D Noise & Voronoi (WGSL Port)
// Optimized for Performance
// ============================================================

// ------------------------------------------------------------
// HASH FUNCTIONS (Jenkins Lookup3 - for Noise Texture)
// ------------------------------------------------------------

fn hash_uint4(kx: u32, ky: u32, kz: u32, kw: u32) -> u32 {
    var a = 0xdeadbeefu + (4u << 2u) + 13u;
    var b = a;
    var c = a;
    a += kx;
    b += ky;
    
    // mix_hash
    a -= c; a ^= ((c << 4u) | (c >> 28u)); c += b;
    b -= a; b ^= ((a << 6u) | (a >> 26u)); a += c;
    c -= b; c ^= ((b << 8u) | (b >> 24u)); b += a;
    a -= c; a ^= ((c << 16u) | (c >> 16u)); c += b;
    b -= a; b ^= ((a << 19u) | (a >> 13u)); a += c;
    c -= b; c ^= ((b << 4u) | (b >> 28u)); b += a;

    a += kz;
    b += kw;

    // final_hash
    c ^= b; c -= ((b << 14u) | (b >> 18u));
    a ^= c; a -= ((c << 11u) | (c >> 21u));
    b ^= a; b -= ((a << 25u) | (a >> 7u));
    c ^= b; c -= ((b << 16u) | (b >> 16u));
    a ^= c; a -= ((c << 4u) | (c >> 28u));
    b ^= a; b -= ((a << 14u) | (a >> 18u));
    c ^= b; c -= ((b << 24u) | (b >> 8u));

    return c;
}

// ------------------------------------------------------------
// HASH FUNCTIONS (PCG4D - for Voronoi)
// ------------------------------------------------------------

fn hash_pcg4d(v_in: vec4i) -> vec4i {
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

fn hash_vec4_to_vec4_voronoi(k: vec4i) -> vec4f {
    let h = hash_pcg4d(k);
    return vec4f(h & vec4i(0x7fffffff)) * (1.0 / f32(0x7fffffff));
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
    var u = x; if (h >= 24u) { u = y; }
    var v = y; if (h >= 16u) { v = z; }
    var s = z; if (h >= 8u) { s = w; }
    
    var res = u; if ((h & 1u) != 0u) { res = -u; }
    var res_v = v; if ((h & 2u) != 0u) { res_v = -v; }
    var res_s = s; if ((h & 4u) != 0u) { res_s = -s; }
    
    return res + res_v + res_s;
}

// ------------------------------------------------------------
// 4D PERLIN NOISE
// ------------------------------------------------------------

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

    let v0 = noiseg_4d(hash_uint4(u32(X),   u32(Y),   u32(Z),   u32(W)),   fx,       fy,       fz,       fw);
    let v1 = noiseg_4d(hash_uint4(u32(X+1), u32(Y),   u32(Z),   u32(W)),   fx-1.0,   fy,       fz,       fw);
    let v2 = noiseg_4d(hash_uint4(u32(X),   u32(Y+1), u32(Z),   u32(W)),   fx,       fy-1.0,   fz,       fw);
    let v3 = noiseg_4d(hash_uint4(u32(X+1), u32(Y+1), u32(Z),   u32(W)),   fx-1.0,   fy-1.0,   fz,       fw);
    let v4 = noiseg_4d(hash_uint4(u32(X),   u32(Y),   u32(Z+1), u32(W)),   fx,       fy,       fz-1.0,   fw);
    let v5 = noiseg_4d(hash_uint4(u32(X+1), u32(Y),   u32(Z+1), u32(W)),   fx-1.0,   fy,       fz-1.0,   fw);
    let v6 = noiseg_4d(hash_uint4(u32(X),   u32(Y+1), u32(Z+1), u32(W)),   fx,       fy-1.0,   fz-1.0,   fw);
    let v7 = noiseg_4d(hash_uint4(u32(X+1), u32(Y+1), u32(Z+1), u32(W)),   fx-1.0,   fy-1.0,   fz-1.0,   fw);
    let v8 = noiseg_4d(hash_uint4(u32(X),   u32(Y),   u32(Z),   u32(W+1)), fx,       fy,       fz,       fw-1.0);
    let v9 = noiseg_4d(hash_uint4(u32(X+1), u32(Y),   u32(Z),   u32(W+1)), fx-1.0,   fy,       fz,       fw-1.0);
    let v10 = noiseg_4d(hash_uint4(u32(X),   u32(Y+1), u32(Z),   u32(W+1)), fx,       fy-1.0,   fz,       fw-1.0);
    let v11 = noiseg_4d(hash_uint4(u32(X+1), u32(Y+1), u32(Z),   u32(W+1)), fx-1.0,   fy-1.0,   fz,       fw-1.0);
    let v12 = noiseg_4d(hash_uint4(u32(X),   u32(Y),   u32(Z+1), u32(W+1)), fx,       fy,       fz-1.0,   fw-1.0);
    let v13 = noiseg_4d(hash_uint4(u32(X+1), u32(Y),   u32(Z+1), u32(W+1)), fx-1.0,   fy,       fz-1.0,   fw-1.0);
    let v14 = noiseg_4d(hash_uint4(u32(X),   u32(Y+1), u32(Z+1), u32(W+1)), fx,       fy-1.0,   fz-1.0,   fw-1.0);
    let v15 = noiseg_4d(hash_uint4(u32(X+1), u32(Y+1), u32(Z+1), u32(W+1)), fx-1.0,   fy-1.0,   fz-1.0,   fw-1.0);

    return quad_mix(v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15, u, v, t, s);
}

// ------------------------------------------------------------
// 4D VORONOI F1 (Standard 3x3x3x3 = 81 neighbors)
// ------------------------------------------------------------

fn voronoi_f1_4d(coord: vec4f, randomness: f32) -> f32 {
    let cellPosition_f = floor(coord);
    let localPosition = coord - cellPosition_f;
    let cellPosition = vec4i(cellPosition_f);

    var minDistanceSq = 1e10;

    for (var u = -1; u <= 1; u++) {
        for (var k = -1; k <= 1; k++) {
            for (var j = -1; j <= 1; j++) {
                for (var i = -1; i <= 1; i++) {
                    let cellOffset = vec4i(i, j, k, u);
                    let p = vec4f(cellOffset) + hash_vec4_to_vec4_voronoi(cellPosition + cellOffset) * randomness;
                    let diff = p - localPosition;
                    let d2 = dot(diff, diff);
                    if (d2 < minDistanceSq) {
                        minDistanceSq = d2;
                    }
                }
            }
        }
    }
    return sqrt(minDistanceSq);
}

// ------------------------------------------------------------
// 4D VORONOI F1 FAST (2x2x2x2 = 16 neighbors)
// ------------------------------------------------------------

fn voronoi_f1_4d_fast(coord: vec4f, randomness: f32) -> f32 {
    let cellPosition_f = floor(coord);
    let localPosition = coord - cellPosition_f;
    let cellPosition = vec4i(cellPosition_f);
    
    // Determine which 16 cells to check based on local position
    let offset = vec4i(step(vec4f(0.5), localPosition));
    let base = cellPosition + offset - vec4i(1);

    var minDistanceSq = 1e10;

    for (var u = 0; u <= 1; u++) {
        for (var k = 0; k <= 1; k++) {
            for (var j = 0; j <= 1; j++) {
                for (var i = 0; i <= 1; i++) {
                    let cellOffset = vec4i(i, j, k, u);
                    let cell = base + cellOffset;
                    let p = vec4f(cell - cellPosition) + hash_vec4_to_vec4_voronoi(cell) * randomness;
                    let diff = p - localPosition;
                    let d2 = dot(diff, diff);
                    if (d2 < minDistanceSq) {
                        minDistanceSq = d2;
                    }
                }
            }
        }
    }
    return sqrt(minDistanceSq);
}

fn fractal_voronoi_4d_fast(coord: vec4f, detail: f32, roughness: f32, lacunarity: f32) -> f32 {
    var amplitude = 1.0;
    var max_amplitude = 0.0;
    var scale = 1.0;
    var total_distance = 0.0;

    let d = clamp(detail, 0.0, 5.0); // Limit octaves for performance

    for (var i = 0; i < 5; i++) {
        if (f32(i) > d) { break; }
        
        // Use fast version for detail octaves
        let dist = voronoi_f1_4d_fast(coord * scale, 1.0);
        
        total_distance += dist * amplitude;
        max_amplitude += amplitude;
        scale *= lacunarity;
        amplitude *= roughness;
    }

    return total_distance / max_amplitude;
}