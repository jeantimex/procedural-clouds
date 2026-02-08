// Blender GPU Shader Source Reference — 4D Voronoi F1
// From: source/blender/gpu/shaders/material/gpu_shader_material_voronoi.glsl
//       source/blender/gpu/shaders/material/gpu_shader_material_fractal_voronoi.glsl
//       source/blender/gpu/shaders/material/gpu_shader_material_tex_voronoi.glsl
//       source/blender/gpu/shaders/common/gpu_shader_common_hash.glsl
// License: GPL v2+

// ============================================================
// HASH FUNCTIONS (PCG4D)
// ============================================================

// PCG 4D hash function
// From "Hash Functions for GPU Rendering" JCGT 2020
// https://jcgt.org/published/0009/03/02/

int4 hash_pcg4d_i(int4 v)
{
  v = v * 1664525 + 1013904223;
  v.x += v.y * v.w;
  v.y += v.z * v.x;
  v.z += v.x * v.y;
  v.w += v.y * v.z;
  v = v ^ (v >> 16);
  v.x += v.y * v.w;
  v.y += v.z * v.x;
  v.z += v.x * v.y;
  v.w += v.y * v.z;
  return v;
}

float4 hash_int4_to_vec4(int4 k)
{
  int4 h = hash_pcg4d_i(k);
  return float4(h & 0x7fffffff) * (1.0 / float(0x7fffffff));
}

float3 hash_int4_to_vec3(int4 k)
{
  return hash_int4_to_vec4(k).xyz;
}

// ============================================================
// DATA STRUCTURES
// ============================================================

struct VoronoiParams {
  float scale;
  float detail;
  float roughness;
  float lacunarity;
  float smoothness;
  float exponent;
  float randomness;
  float max_distance;
  bool normalize;
  int feature;
  int metric;
};

struct VoronoiOutput {
  float Distance;
  float3 Color;
  float4 Position;
};

// ============================================================
// DISTANCE FUNCTIONS
// ============================================================

#define SHD_VORONOI_EUCLIDEAN 0
#define SHD_VORONOI_MANHATTAN 1
#define SHD_VORONOI_CHEBYCHEV 2
#define SHD_VORONOI_MINKOWSKI 3

#define SHD_VORONOI_F1 0
#define SHD_VORONOI_F2 1
#define SHD_VORONOI_SMOOTH_F1 2

float voronoi_distance(float4 a, float4 b, VoronoiParams params)
{
  if (params.metric == SHD_VORONOI_EUCLIDEAN) {
    return distance(a, b);
  }
  else if (params.metric == SHD_VORONOI_MANHATTAN) {
    return abs(a.x - b.x) + abs(a.y - b.y) + abs(a.z - b.z) + abs(a.w - b.w);
  }
  else if (params.metric == SHD_VORONOI_CHEBYCHEV) {
    return max(abs(a.x - b.x), max(abs(a.y - b.y), max(abs(a.z - b.z), abs(a.w - b.w))));
  }
  else if (params.metric == SHD_VORONOI_MINKOWSKI) {
    return pow(pow(abs(a.x - b.x), params.exponent) + pow(abs(a.y - b.y), params.exponent) +
                   pow(abs(a.z - b.z), params.exponent) + pow(abs(a.w - b.w), params.exponent),
               1.0 / params.exponent);
  }
  else {
    return 0.0;
  }
}

// ============================================================
// CORE: voronoi_f1 (4D, single octave)
// ============================================================
// Scans 3^4 = 81 neighboring cells to find closest feature point.

VoronoiOutput voronoi_f1(VoronoiParams params, float4 coord)
{
  float4 cellPosition_f = floor(coord);
  float4 localPosition = coord - cellPosition_f;
  int4 cellPosition = int4(cellPosition_f);

  float minDistance = FLT_MAX;
  int4 targetOffset = int4(0);
  float4 targetPosition = float4(0.0);

  for (int u = -1; u <= 1; u++) {
    for (int k = -1; k <= 1; k++) {
      for (int j = -1; j <= 1; j++) {
        for (int i = -1; i <= 1; i++) {
          int4 cellOffset = int4(i, j, k, u);
          float4 pointPosition = float4(cellOffset) +
                                 hash_int4_to_vec4(cellPosition + cellOffset) * params.randomness;
          float distanceToPoint = voronoi_distance(pointPosition, localPosition, params);
          if (distanceToPoint < minDistance) {
            targetOffset = cellOffset;
            minDistance = distanceToPoint;
            targetPosition = pointPosition;
          }
        }
      }
    }
  }

  VoronoiOutput octave;
  octave.Distance = minDistance;
  octave.Color = hash_int4_to_vec3(cellPosition + targetOffset);
  octave.Position = targetPosition + cellPosition_f;
  return octave;
}

// ============================================================
// FRACTAL WRAPPER: fractal_voronoi_x_fx (4D)
// Handles detail (octaves), roughness (persistence), lacunarity
// ============================================================

VoronoiOutput fractal_voronoi_x_fx(VoronoiParams params, float4 coord)
{
  float amplitude = 1.0;
  float max_amplitude = 0.0;
  float scale = 1.0;

  VoronoiOutput Output;
  Output.Distance = 0.0;
  Output.Color = float3(0.0, 0.0, 0.0);
  Output.Position = float4(0.0, 0.0, 0.0, 0.0);
  bool zero_input = params.detail == 0.0 || params.roughness == 0.0;

  for (int i = 0; i <= ceil(params.detail); ++i) {
    VoronoiOutput octave;
    if (params.feature == SHD_VORONOI_F2) {
      octave = voronoi_f2(params, coord * scale);
    }
    else if (params.feature == SHD_VORONOI_SMOOTH_F1 && params.smoothness != 0.0) {
      octave = voronoi_smooth_f1(params, coord * scale);
    }
    else {
      octave = voronoi_f1(params, coord * scale);  // <-- F1 path
    }

    if (zero_input) {
      max_amplitude = 1.0;
      Output = octave;
      break;
    }
    else if (i <= params.detail) {
      // Full-weight octave
      max_amplitude += amplitude;
      Output.Distance += octave.Distance * amplitude;
      Output.Color += octave.Color * amplitude;
      Output.Position = mix(Output.Position, octave.Position / scale, amplitude);
      scale *= params.lacunarity;
      amplitude *= params.roughness;
    }
    else {
      // Fractional octave (smooth blend for non-integer detail)
      float remainder = params.detail - floor(params.detail);
      if (remainder != 0.0) {
        max_amplitude = mix(max_amplitude, max_amplitude + amplitude, remainder);
        Output.Distance = mix(
            Output.Distance, Output.Distance + octave.Distance * amplitude, remainder);
        Output.Color = mix(Output.Color, Output.Color + octave.Color * amplitude, remainder);
        Output.Position = mix(
            Output.Position, mix(Output.Position, octave.Position / scale, amplitude), remainder);
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

// ============================================================
// NODE ENTRY POINT: node_tex_voronoi_f1_4d
// ============================================================

#define INITIALIZE_VORONOIPARAMS(FEATURE) \
  params.feature = FEATURE; \
  params.metric = int(metric); \
  params.scale = scale; \
  params.detail = clamp(detail, 0.0, 15.0); \
  params.roughness = clamp(roughness, 0.0, 1.0); \
  params.lacunarity = lacunarity; \
  params.smoothness = clamp(smoothness / 2.0, 0.0, 0.5); \
  params.exponent = exponent; \
  params.randomness = clamp(randomness, 0.0, 1.0); \
  params.max_distance = 0.0; \
  params.normalize = bool(normalize);

void node_tex_voronoi_f1_4d(float3 coord,
                            float w,
                            float scale,
                            float detail,
                            float roughness,
                            float lacunarity,
                            float smoothness,
                            float exponent,
                            float randomness,
                            float metric,
                            float normalize,
                            out float outDistance,
                            out float4 outColor,
                            out float3 outPosition,
                            out float outW,
                            out float outRadius)
{
  VoronoiParams params;

  INITIALIZE_VORONOIPARAMS(SHD_VORONOI_F1)

  w *= scale;
  coord *= scale;

  params.max_distance = voronoi_distance(
      float4(0.0), float4(0.5 + 0.5 * params.randomness), params);
  VoronoiOutput Output = fractal_voronoi_x_fx(params, float4(coord, w));
  outDistance = Output.Distance;
  outColor = float4(Output.Color, 1.0);
  outPosition = Output.Position.xyz;
  outW = Output.Position.w;
}
