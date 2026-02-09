Yes — in the WebGPU shader we already cast self‑shadows inside the cloud volume.
  That’s the lightMarch() function: it samples density a few steps toward the sun
  direction and uses exp(-shadow * 1.5) to attenuate light. That’s a simple
  volumetric shadow approximation (single scattering with a short shadow ray), not
  full multiple scattering.

  In Blender:

  - The Principled Volume uses Cycles’ volumetric path tracing, so shadows are
    handled physically with true volume scattering and absorption.
  - It’s much more accurate (and more expensive): multiple scattering, proper
    integration along rays, and shadows depend on the entire volume and lighting
    setup.

  So:

  - WebGPU = fast approximate self‑shadow via a short secondary march.
  - Blender/Cycles = full volumetric lighting/shadowing.