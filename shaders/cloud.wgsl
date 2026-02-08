// ============================================================
// Uniforms
// ============================================================

struct Camera {
  invViewProj : mat4x4f,
  position    : vec3f,
  _pad        : f32,
};

@group(0) @binding(0) var<uniform> camera : Camera;

// ============================================================
// Vertex — full-screen triangle (3 vertices, no buffers)
// ============================================================

struct VSOut {
  @builtin(position) pos : vec4f,
  @location(0)       uv  : vec2f,
};

@vertex
fn vs(@builtin(vertex_index) vi : u32) -> VSOut {
  // Generate a triangle that covers the full screen:
  //   vi=0 → (-1, -1)   vi=1 → ( 3, -1)   vi=2 → (-1,  3)
  let x = f32(i32(vi & 1u) * 4 - 1);
  let y = f32(i32(vi >> 1u) * 4 - 1);
  var out : VSOut;
  out.pos = vec4f(x, y, 0.0, 1.0);
  out.uv  = vec2f(x, y);
  return out;
}

// ============================================================
// Ray-Box intersection (AABB slab method)
// ============================================================

// Box spans [-1, 1] in XZ and [0, 1] in Y (cloud slab)
const BOX_MIN = vec3f(-1.0, 0.0, -1.0);
const BOX_MAX = vec3f( 1.0, 1.0,  1.0);

struct HitInfo {
  hit  : bool,
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
// Fragment — ray march stub, visualize box hit
// ============================================================

const BG_COLOR = vec3f(0.075, 0.145, 0.25);

@fragment
fn fs(@location(0) uv : vec2f) -> @location(0) vec4f {
  // Reconstruct ray from clip-space UV through inverse view-proj
  let ndc_near = vec4f(uv, 0.0, 1.0);
  let ndc_far  = vec4f(uv, 1.0, 1.0);

  let world_near = camera.invViewProj * ndc_near;
  let world_far  = camera.invViewProj * ndc_far;

  let near = world_near.xyz / world_near.w;
  let far  = world_far.xyz  / world_far.w;

  let ro = camera.position;
  let rd = normalize(far - near);

  // Intersect the cloud bounding box
  let hit = intersectBox(ro, rd);

  if (!hit.hit) {
    return vec4f(BG_COLOR, 1.0);
  }

  // Entry point into the box (clamp tNear to 0 if camera is inside)
  let tEntry = max(hit.tNear, 0.0);
  let entryPos = ro + rd * tEntry;

  // Normalize position within box to [0,1] for debug coloring
  let boxPos = (entryPos - BOX_MIN) / (BOX_MAX - BOX_MIN);

  return vec4f(boxPos, 1.0);
}
