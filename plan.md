# Procedural Clouds — Blender to WebGPU Port

## Blender Shader Analysis

### Architecture Overview

The shader is a **volumetric** material using a **Principled Volume** node (white color) connected to the Volume output (not Surface). All cloud shaping happens inside a single node group called **"Procedural Clouds Shader"** that outputs a scalar **Density** value.

### Group Interface (Inputs)

| Parameter            | Default | Purpose                                    |
| -------------------- | ------- | ------------------------------------------ |
| Density              | 1.0     | Global density multiplier                  |
| Low Altitude Density | 0.2     | Density floor at low altitudes             |
| Altitude             | 0.5     | Height at which cloud layer sits           |
| Factor               | 1.0     | Controls density range of Voronoi layers   |
| Scale                | 1.0     | Spatial scale (divides object coordinates) |
| Time                 | 0.0     | Animates the 4D noise W-dimension          |
| Detail               | 1.0     | Controls Voronoi detail/octaves            |

---

## The 5 Stages of Cloud Generation

### Stage 1 — Altitude Mask (base vertical profile)

- Takes the **Z** component from **Object** coordinates.
- Maps Z through a range based on `Altitude` (via `Map Range.010`, clamped).
- `Low Altitude Density` sets a non-zero floor via `MULTIPLY_ADD` into the Map Range's ToMin, so the bottom of the cloud volume isn't fully transparent.
- Result is multiplied (clamped) with a **4D Noise Texture** (`scale=2, detail=0, roughness=0`) — a very smooth, low-frequency noise animated by `Time` via the W input.
- **Effect**: A height-dependent soft mask with gentle animated variation — clouds are denser at certain altitudes and thin out above/below.

### Stage 2 — Large Voronoi Cells (macro cloud shapes)

- **4D Voronoi** (`F1 Euclidean, scale=5, lacunarity=3, roughness=0.5`).
- Object coords divided by `Scale`, animated via `Time` on W.
- `Detail` input drives the Voronoi detail directly.
- Distance output is remapped: `ToMin = Factor × 0.5`, `ToMax = Factor` — so there's always a base density within cells.
- Result is **multiplied** with the altitude mask, then **added** (clamped) to accumulated density.
- **Effect**: Defines the large-scale cloud cell structure — puffy blob boundaries.

### Stage 3 — Medium Voronoi Detail (internal cloud texture)

- **4D Voronoi** (`F1 Euclidean, scale=2, lacunarity=2.5, roughness=0.75`).
- Object coords divided by `Scale`, animated via `Time` on W.
- `Detail × 5` gives this layer much more detail than the macro layer.
- Same Factor-based remapping pattern (`ToMin = Factor × 0.5`, `ToMax = Factor`).
- **Added** (clamped) to the running density accumulation.
- **Effect**: Adds internal billowy texture and variation within the large cloud cells.

### Stage 4 — Upper Altitude Cutoff

- Extracts Z again from Object coords.
- Maps Z from `[Altitude × Scale, 1.0]` to `[0, 1]` (clamped).
- **Subtracted** (clamped) from the accumulated density.
- Then further modified by Factor via `MULTIPLY_ADD` and another `SUBTRACT` (clamped).
- **Effect**: Erodes cloud density at the top of the volume, giving clouds a flat-top / anvil ceiling.

### Stage 5 — Final Density & Max-Altitude Falloff

- Extracts Z one more time, maps it via `Altitude` as FromMax.
- Multiplied by the `Density` parameter.
- Then multiplied again with the altitude-mapped value.
- Finally multiplied with the output from Stage 4.
- **Effect**: Applies the user-controlled `Density` multiplier and creates a smooth density rolloff at the maximum cloud altitude.

### World Shader

Simple **Background** node with a dark desaturated blue (`RGB ≈ 0.075, 0.145, 0.25`) — a twilight/dusk sky.

---

## Key Techniques in the Blender Shader

1. **Layered 4D noise** — Two Voronoi textures at different scales (5 and 2) create macro shape + internal detail. The 4th dimension (W) is driven by `Time` for smooth animation.
2. **Altitude shaping** — Multiple Z-coordinate lookups with Map Range nodes sculpt the vertical cloud profile (fade-in at bottom, dense middle, hard cutoff at top).
3. **Additive density accumulation** — Each noise layer is added (with clamping) to build up density, giving an organic feel.
4. **Factor-controlled range** — The `Factor` input scales both the min and max of each Voronoi remapping, controlling how "filled" the cells are.
5. **Low-frequency altitude modulation** — A very simple noise texture (no detail/roughness) gently varies the altitude mask over time.

---

## WebGPU Porting Plan

### Rendering Approach

The Blender shader is a volumetric material evaluated per-voxel by Blender's internal renderer. In WebGPU we need to implement this as a **ray-marching** fragment shader:

1. Render a full-screen quad (or the bounding box of the cloud volume).
2. For each pixel, march a ray from the camera through the cloud volume.
3. At each step along the ray, evaluate the density function (the 5 stages above).
4. Accumulate color and opacity using Beer-Lambert absorption.

### WGSL Shader Functions Required

| Function                  | Purpose                                                    |
| ------------------------- | ---------------------------------------------------------- |
| `hash44(vec4f) -> vec4f`  | Hash function for 4D Voronoi cell randomization            |
| `noise4d(vec4f) -> f32`   | 4D value/gradient noise (for Stage 1's Noise Texture)      |
| `voronoi4d_f1(...)-> f32` | 4D Voronoi F1 Euclidean distance with detail and lacunarity |
| `mapRange(...)  -> f32`   | Clamped linear remap (equivalent to Blender's Map Range)   |
| `cloudDensity(pos, time, params) -> f32` | The full 5-stage density function    |
| `rayMarch(origin, dir, params) -> vec4f` | Ray marcher with Beer-Lambert        |

### Uniform Buffer Layout

```
struct CloudParams {
  density: f32,
  lowAltitudeDensity: f32,
  altitude: f32,
  factor: f32,
  scale: f32,
  time: f32,
  detail: f32,
};

struct Camera {
  invViewProj: mat4x4f,
  position: vec3f,
};
```

### Implementation Steps

1. **Set up render pipeline** — Full-screen quad with vertex + fragment shader, uniform buffers for camera and cloud params.
2. **Implement 4D noise in WGSL** — Port or write a 4D simplex/value noise function.
3. **Implement 4D Voronoi F1 in WGSL** — With configurable scale, detail (octaves), roughness, lacunarity. This is the most complex piece.
4. **Implement the 5-stage density function** — Translate the Blender node graph into a single WGSL function.
5. **Implement ray marching** — Fixed-step ray march through a bounding box with Beer-Lambert light absorption.
6. **Add camera controls** — Orbit/pan/zoom to navigate around the cloud volume.
7. **Animate** — Drive the `time` uniform from `requestAnimationFrame` for cloud animation.
8. **Add UI controls** — Sliders for Density, Altitude, Factor, Scale, Detail, etc.

### Challenges & Considerations

- **Performance**: 4D Voronoi with detail (fractal octaves) is expensive. May need to limit step count or use adaptive step sizing.
- **4D Voronoi accuracy**: Blender's Voronoi implementation searches neighboring cells with specific jitter; matching it exactly is important for visual parity.
- **Lighting**: Blender's Principled Volume includes full volumetric scattering. A simplified version (single forward-scatter + Beer-Lambert) should suffice for a real-time port.
- **Bounding volume**: Need to define the cloud volume extents. In Blender this is the mesh the material is assigned to; in WebGPU we'll use an axis-aligned bounding box.
- **Background**: Match the dark blue world color (`RGB ≈ 0.075, 0.145, 0.25`).

### File Structure (Planned)

```
procedural-clouds/
├── index.html
├── style.css
├── main.js              # WebGPU init, render loop, camera, UI
├── shaders/
│   ├── cloud.wgsl       # Vertex + fragment shader (ray march + density)
│   └── noise.wgsl       # 4D noise and Voronoi functions
├── plan.md              # This file
└── package.json
```
