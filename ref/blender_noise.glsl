// Blender GPU Shader Source Reference — Noise Texture
// From: source/blender/gpu/shaders/material/gpu_shader_material_noise.glsl
//       source/blender/gpu/shaders/common/gpu_shader_common_hash.glsl
// License: GPL v2+

// ============================================================
// HASH FUNCTIONS (Jenkins Lookup3)
// ============================================================

#define rot(x, k) (((x) << (k)) | ((x) >> (32 - (k))))

#define mix_hash(a, b, c) \
  { \
    a -= c; a ^= rot(c, 4);  c += b; \
    b -= a; b ^= rot(a, 6);  a += c; \
    c -= b; c ^= rot(b, 8);  b += a; \
    a -= c; a ^= rot(c, 16); c += b; \
    b -= a; b ^= rot(a, 19); a += c; \
    c -= b; c ^= rot(b, 4);  b += a; \
  }

#define final_hash(a, b, c) \
  { \
    c ^= b; c -= rot(b, 14); \
    a ^= c; a -= rot(c, 11); \
    b ^= a; b -= rot(a, 25); \
    c ^= b; c -= rot(b, 16); \
    a ^= c; a -= rot(c, 4);  \
    b ^= a; b -= rot(a, 14); \
    c ^= b; c -= rot(b, 24); \
  }

uint hash_uint(uint kx)
{
  uint a, b, c;
  a = b = c = 0xdeadbeefu + (1u << 2u) + 13u;
  a += kx;
  final_hash(a, b, c);
  return c;
}

uint hash_uint2(uint kx, uint ky)
{
  uint a, b, c;
  a = b = c = 0xdeadbeefu + (2u << 2u) + 13u;
  a += kx;
  b += ky;
  final_hash(a, b, c);
  return c;
}

uint hash_uint3(uint kx, uint ky, uint kz)
{
  uint a, b, c;
  a = b = c = 0xdeadbeefu + (3u << 2u) + 13u;
  a += kx;
  b += ky;
  c += kz;
  final_hash(a, b, c);
  return c;
}

uint hash_uint4(uint kx, uint ky, uint kz, uint kw)
{
  uint a, b, c;
  a = b = c = 0xdeadbeefu + (4u << 2u) + 13u;
  a += kx;
  b += ky;
  mix_hash(a, b, c);
  a += kz;
  b += kw;
  final_hash(a, b, c);
  return c;
}

float hash_uint_to_float(uint kx)
{
  return float(hash_uint(kx)) / float(0xFFFFFFFFu);
}

float hash_uint2_to_float(uint kx, uint ky)
{
  return float(hash_uint2(kx, ky)) / float(0xFFFFFFFFu);
}

float hash_uint3_to_float(uint kx, uint ky, uint kz)
{
  return float(hash_uint3(kx, ky, kz)) / float(0xFFFFFFFFu);
}

float hash_uint4_to_float(uint kx, uint ky, uint kz, uint kw)
{
  return float(hash_uint4(kx, ky, kz, kw)) / float(0xFFFFFFFFu);
}

float hash_float_to_float(float k)
{
  return hash_uint_to_float(floatBitsToUint(k));
}

float hash_vec2_to_float(vec2 k)
{
  return hash_uint2_to_float(floatBitsToUint(k.x), floatBitsToUint(k.y));
}

float hash_vec3_to_float(vec3 k)
{
  return hash_uint3_to_float(floatBitsToUint(k.x), floatBitsToUint(k.y), floatBitsToUint(k.z));
}

float hash_vec4_to_float(vec4 k)
{
  return hash_uint4_to_float(
      floatBitsToUint(k.x), floatBitsToUint(k.y), floatBitsToUint(k.z), floatBitsToUint(k.w));
}

vec4 hash_vec4_to_vec4(vec4 k)
{
  return vec4(hash_vec4_to_float(k),
              hash_vec4_to_float(vec4(k.w, k.x, k.y, k.z)),
              hash_vec4_to_float(vec4(k.z, k.w, k.x, k.y)),
              hash_vec4_to_float(vec4(k.y, k.z, k.w, k.x)));
}

// ============================================================
// PERLIN NOISE HELPERS
// ============================================================

float floorfrac(float x, out int i)
{
  float x_floor = floor(x);
  i = int(x_floor);
  return x - x_floor;
}

float fade(float t)
{
  return t * t * t * (t * (t * 6.0 - 15.0) + 10.0);
}

// Quadrilinear interpolation (4D)
float quad_mix(float v0,  float v1,  float v2,  float v3,
               float v4,  float v5,  float v6,  float v7,
               float v8,  float v9,  float v10, float v11,
               float v12, float v13, float v14, float v15,
               float x, float y, float z, float w)
{
  return mix(tri_mix(v0, v1, v2, v3, v4, v5, v6, v7, x, y, z),
             tri_mix(v8, v9, v10, v11, v12, v13, v14, v15, x, y, z),
             w);
}

float tri_mix(float v0, float v1, float v2, float v3,
              float v4, float v5, float v6, float v7,
              float x, float y, float z)
{
  float x1 = 1.0 - x;
  float y1 = 1.0 - y;
  float z1 = 1.0 - z;
  return z1 * (y1 * (v0 * x1 + v1 * x) + y * (v2 * x1 + v3 * x)) +
         z * (y1 * (v4 * x1 + v5 * x) + y * (v6 * x1 + v7 * x));
}

// 4D gradient function (32 gradient directions)
float noiseg_4d(uint hash, float x, float y, float z, float w)
{
  uint h = hash & 31u;
  float u = h < 24u ? x : y;
  float v = h < 16u ? y : z;
  float s = h < 8u ? z : w;
  return ((h & 1u) != 0u ? -u : u) + ((h & 2u) != 0u ? -v : v) + ((h & 4u) != 0u ? -s : s);
}

// ============================================================
// 4D PERLIN NOISE
// ============================================================

float perlin_noise(vec4 position)
{
  int X, Y, Z, W;
  float fx = floorfrac(position.x, X);
  float fy = floorfrac(position.y, Y);
  float fz = floorfrac(position.z, Z);
  float fw = floorfrac(position.w, W);

  float u = fade(fx);
  float v = fade(fy);
  float t = fade(fz);
  float s = fade(fw);

  float r = quad_mix(
      noiseg_4d(hash_uint4(uint(X),   uint(Y),   uint(Z),   uint(W)),   fx,       fy,       fz,       fw),
      noiseg_4d(hash_uint4(uint(X+1), uint(Y),   uint(Z),   uint(W)),   fx-1.0,   fy,       fz,       fw),
      noiseg_4d(hash_uint4(uint(X),   uint(Y+1), uint(Z),   uint(W)),   fx,       fy-1.0,   fz,       fw),
      noiseg_4d(hash_uint4(uint(X+1), uint(Y+1), uint(Z),   uint(W)),   fx-1.0,   fy-1.0,   fz,       fw),
      noiseg_4d(hash_uint4(uint(X),   uint(Y),   uint(Z+1), uint(W)),   fx,       fy,       fz-1.0,   fw),
      noiseg_4d(hash_uint4(uint(X+1), uint(Y),   uint(Z+1), uint(W)),   fx-1.0,   fy,       fz-1.0,   fw),
      noiseg_4d(hash_uint4(uint(X),   uint(Y+1), uint(Z+1), uint(W)),   fx,       fy-1.0,   fz-1.0,   fw),
      noiseg_4d(hash_uint4(uint(X+1), uint(Y+1), uint(Z+1), uint(W)),   fx-1.0,   fy-1.0,   fz-1.0,   fw),
      noiseg_4d(hash_uint4(uint(X),   uint(Y),   uint(Z),   uint(W+1)), fx,       fy,       fz,       fw-1.0),
      noiseg_4d(hash_uint4(uint(X+1), uint(Y),   uint(Z),   uint(W+1)), fx-1.0,   fy,       fz,       fw-1.0),
      noiseg_4d(hash_uint4(uint(X),   uint(Y+1), uint(Z),   uint(W+1)), fx,       fy-1.0,   fz,       fw-1.0),
      noiseg_4d(hash_uint4(uint(X+1), uint(Y+1), uint(Z),   uint(W+1)), fx-1.0,   fy-1.0,   fz,       fw-1.0),
      noiseg_4d(hash_uint4(uint(X),   uint(Y),   uint(Z+1), uint(W+1)), fx,       fy,       fz-1.0,   fw-1.0),
      noiseg_4d(hash_uint4(uint(X+1), uint(Y),   uint(Z+1), uint(W+1)), fx-1.0,   fy,       fz-1.0,   fw-1.0),
      noiseg_4d(hash_uint4(uint(X),   uint(Y+1), uint(Z+1), uint(W+1)), fx,       fy-1.0,   fz-1.0,   fw-1.0),
      noiseg_4d(hash_uint4(uint(X+1), uint(Y+1), uint(Z+1), uint(W+1)), fx-1.0,   fy-1.0,   fz-1.0,   fw-1.0),
      u, v, t, s);
  return r;
}

// ============================================================
// FRACTAL BROWNIAN MOTION (fBM)
// ============================================================

float noise_fbm(vec4 p, float detail, float roughness, float lacunarity, bool normalize)
{
  float fscale = 1.0;
  float amp = 1.0;
  float maxamp = 0.0;
  float sum = 0.0;

  for (int i = 0; i <= int(detail); i++) {
    float t = perlin_noise(fscale * p);
    sum += t * amp;
    maxamp += amp;
    amp *= roughness;
    fscale *= lacunarity;
  }

  float rmd = detail - floor(detail);
  if (rmd != 0.0) {
    float t = perlin_noise(fscale * p);
    float sum2 = sum + t * amp;
    return normalize ? mix(0.5 + 0.5 * (sum / maxamp), 0.5 + 0.5 * (sum2 / (maxamp + amp)), rmd) :
                       mix(sum, sum2, rmd);
  }

  return normalize ? 0.5 + 0.5 * (sum / maxamp) : sum;
}

// ============================================================
// NOISE TEXTURE NODE ENTRY POINT (4D)
// ============================================================

void node_noise_texture_4d(vec3 co,
                           float w,
                           float scale,
                           float detail,
                           float roughness,
                           float lacunarity,
                           float distortion,
                           float normalize,
                           out float value,
                           out vec4 color)
{
  vec4 p = vec4(co, w) * scale;

  if (distortion != 0.0) {
    p += vec4(perlin_noise(p + random_vec4_offset(0.0)) * distortion,
              perlin_noise(p + random_vec4_offset(1.0)) * distortion,
              perlin_noise(p + random_vec4_offset(2.0)) * distortion,
              perlin_noise(p + random_vec4_offset(3.0)) * distortion);
  }

  value = noise_fbm(p, detail, roughness, lacunarity, normalize != 0.0);
  color = vec4(value,
               noise_fbm(p + random_vec4_offset(4.0), detail, roughness, lacunarity, normalize != 0.0),
               noise_fbm(p + random_vec4_offset(5.0), detail, roughness, lacunarity, normalize != 0.0),
               1.0);
}
