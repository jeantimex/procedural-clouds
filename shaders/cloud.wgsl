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
// Optimized Noise & Hash
// ============================================================

fn hash33(p: vec3f) -> vec3f {
  var p3 = fract(p * vec3f(0.1031, 0.1030, 0.0973));
  p3 += dot(p3, p3.yxz + 33.33);
  return fract((p3.xxy + p3.yxx) * p3.zyx);
}

fn fade(t : f32) -> f32 {
  return t * t * t * (t * (t * 6.0 - 15.0) + 10.0);
}

fn noise_3d(p: vec3f) -> f32 {
    let i = floor(p);
    let f = fract(p);
    let u = fade(f.x);
    let v = fade(f.y);
    let w = fade(f.z);

    let a = dot(hash33(i + vec3f(0,0,0)) - 0.5, f - vec3f(0,0,0));
    let b = dot(hash33(i + vec3f(1,0,0)) - 0.5, f - vec3f(1,0,0));
    let c = dot(hash33(i + vec3f(0,1,0)) - 0.5, f - vec3f(0,1,0));
    let d = dot(hash33(i + vec3f(1,1,0)) - 0.5, f - vec3f(1,1,0));
    let e = dot(hash33(i + vec3f(0,0,1)) - 0.5, f - vec3f(0,0,1));
    let f_val = dot(hash33(i + vec3f(1,0,1)) - 0.5, f - vec3f(1,0,1));
    let g = dot(hash33(i + vec3f(0,1,1)) - 0.5, f - vec3f(0,1,1));
    let h = dot(hash33(i + vec3f(1,1,1)) - 0.5, f - vec3f(1,1,1));

    return mix(mix(mix(a, b, u), mix(c, d, u), v),
               mix(mix(e, f_val, u), mix(g, h, u), v), w);
}

// ------------------------------------------------------------
// Voronoi
// ------------------------------------------------------------

struct VoronoiResult {
  distance : f32,
  color    : vec3f,
};

fn voronoi_f1_3d(coord : vec3f, randomness : f32) -> VoronoiResult {
  let i = floor(coord);
  let f = fract(coord);
  var minDistSq = 100.0;
  var targetOffset = vec3i(0);

  for (var k = -1; k <= 1; k++) {
    for (var j = -1; j <= 1; j++) {
      for (var i_off = -1; i_off <= 1; i_off++) {
        let offset = vec3i(i_off, j, k);
        let h = hash33(i + vec3f(offset));
        let p = vec3f(offset) + h * randomness;
        let d = dot(p - f, p - f);
        if (d < minDistSq) {
          minDistSq = d;
          targetOffset = offset;
        }
      }
    }
  }
  var res : VoronoiResult;
  res.distance = sqrt(minDistSq);
  res.color = hash33(i + vec3f(targetOffset));
  return res;
}

fn fractal_voronoi_3d(coord : vec3f, detail : f32, roughness : f32, lacunarity : f32) -> f32 {
    var amp = 1.0;
    var max_amp = 0.0;
    var freq = 1.0;
    var total = 0.0;
    let d = clamp(detail, 0.0, 5.0);
    for (var i = 0; i < 5; i++) {
        if (f32(i) > d) { break; }
        let res = voronoi_f1_3d(coord * freq, 1.0);
        total += res.distance * amp;
        max_amp += amp;
        amp *= roughness;
        freq *= lacunarity;
    }
    return total / max_amp;
}

fn mapRange(value : f32, fromMin : f32, fromMax : f32, toMin : f32, toMax : f32) -> f32 {
  let t = (value - fromMin) / (fromMax - fromMin);
  return clamp(mix(toMin, toMax, t), min(toMin, toMax), max(toMin, toMax));
}

// ------------------------------------------------------------
// Cloud Density
// ------------------------------------------------------------

fn cloudDensity(pos : vec3f, is_cheap : bool) -> f32 {
  let altitude     = params.altitude;
  let factor       = params.factor;
  let scale        = params.scale;
  let time         = params.time;
  let detail       = params.detail;
  let lowAltDens   = params.lowAltDensity;
  let densityParam = params.density;

  // Flip the vertical coordinate
  let Z = 1.0 - pos.y;

  // Stage 1: Altitude profile
  let altDiv5 = altitude / 5.0;
  let altToMin = 1.0 - lowAltDens;
  let altitudeProfile = mapRange(Z, 0.0, altDiv5, altToMin, 1.0);

  // Noise modulation
  let noiseFac = noise_3d(pos / scale * 1.5 + vec3f(time * 0.1)) * 0.5 + 0.5;
  let altitudeMask = clamp(altitudeProfile * noiseFac, 0.0, 1.0);

  // Stage 2: Macro Voronoi
  let v1dist = voronoi_f1_3d(pos / scale + vec3f(0.0, 0.0, time * 0.05), 1.0).distance;
  let v1mapped = mapRange(v1dist, 0.0, 0.8, factor * (-0.4), factor);
  let v1scaled = clamp(v1mapped * 0.5, 0.0, 1.0);
  let accumulated1 = clamp(altitudeMask + v1scaled, 0.0, 1.0);

  if (is_cheap) {
      return accumulated1 * densityParam * 500.0 * mapRange(Z, 0.0, altitude, 0.0, 1.0);
  }

  // Stage 3: Detail Voronoi
  let v2dist = fractal_voronoi_3d(pos / scale + vec3f(0.0, 0.0, time * 0.15), detail, 0.7, 2.5);
  let v2mapped = mapRange(v2dist, 0.0, 0.8, factor * (-0.25), factor);
  let accumulated2 = clamp(accumulated1 + v2mapped, 0.0, 1.0);

  // Stage 4: Upper cutoff (Modulated by noise for uneven peaks)
  // Higher noiseFac areas (thick clouds) will break through the ceiling more
  let cutoffBase = mapRange(Z, altitude * 0.4, altitude, 0.0, 1.0);
  let unevenCutoff = cutoffBase * (1.1 - 0.4 * noiseFac); 
  
  let shaped = clamp(accumulated2 - unevenCutoff, 0.0, 1.0);
  let finalShaped = clamp(shaped - (1.0 - factor), 0.0, 1.0);

  // Stage 5: Final ramp (Bottom fade)
  let altitudeRamp = mapRange(Z, 0.0, 0.1, 0.0, 1.0);
  return finalShaped * densityParam * 500.0 * altitudeRamp;
}

// ------------------------------------------------------------
// Ray Marching
// ------------------------------------------------------------

const BOX_MIN = vec3f(-3.0, 0.0, -3.0);
const BOX_MAX = vec3f( 3.0, 1.0,  3.0);

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

const SUN_DIR   = vec3f(0.189, 0.943, 0.283); 

const SUN_COLOR = vec3f(1.0, 0.98, 0.94);

const AMBIENT   = vec3f(0.15, 0.2, 0.35); // Darker, cooler ambient for more contrast

const BG_COLOR  = vec3f(0.075, 0.145, 0.25);

const NUM_STEPS = 64; // Increased for better detail



// Henyey-Greenstein Phase Function for forward scattering

fn hgPhase(cosTheta: f32, g: f32) -> f32 {

    let g2 = g * g;

    return (1.0 - g2) / (4.0 * 3.14159 * pow(1.0 + g2 - 2.0 * g * cosTheta, 1.5));

}



// Improved multi-step light march for self-shadowing

fn lightMarch(pos : vec3f) -> f32 {

  var shadow = 0.0;

  let steps = 4;

  let stepSize = 0.12;

  for (var i = 1; i <= steps; i++) {

    let p = pos + SUN_DIR * (f32(i) * stepSize);

    // Use cheap density for performance in light march

    shadow += cloudDensity(p, true) * stepSize;

  }

  // Beer's Law for light absorption

  return exp(-shadow * 2.0); 

}



fn interleavedGradientNoise(uv: vec2f) -> f32 {

    let magic = vec3f(0.06711056, 0.00583715, 52.9829189);

    return fract(magic.z * fract(dot(uv, magic.xy)));

}



@fragment

fn fs(@builtin(position) fragCoord : vec4f, @location(0) uv : vec2f) -> @location(0) vec4f {

  let ndc_near = vec4f(uv, 0.0, 1.0);

  let ndc_far  = vec4f(uv, 1.0, 1.0);



  let world_near = camera.invViewProj * ndc_near;

  let world_far  = camera.invViewProj * ndc_far;



  let near = world_near.xyz / world_near.w;

  let far  = world_far.xyz  / world_far.w;



  let ro = camera.position;

  let rd = normalize(far - near);



  let hit = intersectBox(ro, rd);



    // Sky gradient with sun glow



    let sky = mix(BG_COLOR, vec3f(0.1, 0.2, 0.4), clamp(rd.y * 0.5 + 0.5, 0.0, 1.0));



    let sunTheta = dot(rd, SUN_DIR);



    let sunGlow = pow(max(sunTheta, 0.0), 64.0) * SUN_COLOR * 0.8;



    let finalSky = sky + sunGlow;



  



    var outColor = finalSky;



  



    if (hit.hit) {



      let tEntry = max(hit.tNear, 0.0);



      let tExit  = hit.tFar;



      let stepSize = (tExit - tEntry) / f32(NUM_STEPS);



  



      let dither = interleavedGradientNoise(fragCoord.xy);



      var pos = ro + rd * (tEntry + stepSize * dither);



  



      var transmittance = 1.0;



      var color = vec3f(0.0);



  



      // Scattering phase (Anisotropy)



      let phase = mix(1.0, hgPhase(sunTheta, 0.45), 0.6);



  



      for (var i = 0; i < NUM_STEPS; i++) {



        let d = cloudDensity(pos, false);



  



        if (d > 0.01) {



          let step_trans = exp(-d * stepSize);



          let sunTrans = lightMarch(pos);



          let powder = 1.0 - exp(-d * 2.0);



          let scattering = sunTrans * phase * powder;



          let litColor = SUN_COLOR * scattering * 2.5 + AMBIENT * (1.0 - sunTrans);



  



          color += transmittance * (1.0 - step_trans) * litColor;



          transmittance *= step_trans;



  



          if (transmittance < 0.01) { break; }



        }



        pos += rd * stepSize;



      }



      outColor = color + transmittance * finalSky;



    }



    



    // Apply Tone Mapping and Gamma Correction to everything consistently



    outColor = outColor / (outColor + vec3f(1.0));



    outColor = pow(outColor, vec3f(1.0 / 2.2));



  



    return vec4f(outColor, 1.0);



  }



  
