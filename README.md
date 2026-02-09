# Procedural Clouds (WebGPU)

Real-time procedural volumetric clouds rendered with WebGPU. The project combines a compute pass that populates a 3D density cache with a full-screen ray-march render pass, exposing a set of artistic controls via `lil-gui`.

## Features
- WebGPU ray-marched volumetric clouds
- Compute shader density cache for faster rendering
- Adjustable cloud appearance and lighting controls
- Interactive orbit camera with inertia
- Performance panel via `stats.js`

## Requirements
- A WebGPU-capable browser. If WebGPU is unavailable, the page renders a simple message instead of the scene.
- Node.js + npm for local development

## Quick Start
```bash
npm install
npm run dev
```
Then open the URL printed by Vite (typically `http://localhost:5173`).

## Controls
The UI panel exposes these parameters:
- `Density`: Overall cloud density multiplier
- `Coverage`: Macro coverage factor
- `Scale`: Noise scale (now up to 15)
- `Altitude`: Base altitude of the cloud layer
- `Detail`: Higher-frequency noise contribution
- `Wind Speed`: Time scale for cloud motion
- `Skip Light March`: Toggles light marching in the shader
- `Ray Steps`: Number of ray-march steps per pixel
- `Light Steps`: Number of light-march steps
- `Shadow Dark`: Shadow intensity
- `Sun Intensity`: Direct light intensity
- `Cloud Height`: Vertical thickness of the cloud layer
- `Cache Res`: Density cache resolution (3D texture size)
- `Cache Update`: Update frequency of the cache (every N frames)
- `Cache Smooth`: Blend smoothing between cache updates

Camera controls:
- Drag to orbit
- Scroll to zoom

## How It Works
1. **Compute pass** (`shaders/noise.wgsl` + `shaders/cloud.wgsl`):
   - Writes a 3D density texture (the cache) at a configurable resolution.
2. **Render pass** (`shaders/cloud.wgsl`):
   - Ray-marches through the cached density to shade clouds with lighting.
3. **Parameter packing** (`main.js`):
   - UI values are packed into a uniform buffer and consumed by both passes.

## WebGPU Implementation Details
The renderer uses two pipelines and a small set of bind groups to keep per-frame work minimal.

### Pipelines
- **Compute pipeline** (`cs` entry point): populates a 3D density cache each update.
- **Render pipeline** (`vs`/`fs` entry points): draws a single full-screen triangle and performs ray marching in the fragment shader.

### Resources and Bind Groups
- **Uniform buffers**:
  - `cameraBuffer` (80 bytes): inverse view-projection matrix (64 bytes) + camera position (vec3 + pad).
  - `paramsBuffer` (96 bytes): packed params used by both compute and render stages.
- **Bind groups**:
  - Group 0: `cameraBuffer`, `paramsBuffer`.
  - Group 1: sampler + two 3D texture views (for sampling the density cache).
  - Group 2 (compute only): storage texture view for writing the current cache.

### Density Cache (3D Texture)
- Format: `rgba16float`, size `cacheResolution³`.
- Usage: `STORAGE_BINDING | TEXTURE_BINDING`.
- Two textures are used for **ping-pong**:
  - One texture is written by the compute pass.
  - The other is sampled by the render pass.
  - The roles swap based on `cacheUpdateRate`, with optional temporal smoothing via `cacheSmooth`.

### Sampling Strategy
- The density cache is sampled with a linear sampler for smooth transitions.
- The render pass uses the two cached volumes and blends between them to avoid popping.

### Camera and Matrices
- The camera orbits a target point using spherical coordinates (`theta`, `phi`, `dist`).
- Each frame builds `view`, `projection`, and `inverse view-projection` matrices on the CPU.
- The inverse view-projection is used in the fragment shader to reconstruct world-space ray directions.
- Camera motion uses simple exponential smoothing (inertia).

### Render Loop Overview
1. Update camera smoothing and time-dependent parameters.
2. Update uniform buffers via `queue.writeBuffer`.
3. If needed, run the compute pass to refresh the density cache.
4. Run the render pass to ray-march clouds into the swap chain texture.

## Performance Notes
- Increasing `Ray Steps`, `Light Steps`, and `Cache Res` can be expensive.
- `Cache Update` and `Cache Smooth` let you trade temporal stability for speed.

## License
See `LICENSE`.
