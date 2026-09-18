# Local MediaPipe runtime

Unmodified runtime files from `@mediapipe/tasks-vision` **0.10.32**:

https://registry.npmjs.org/@mediapipe/tasks-vision/-/tasks-vision-0.10.32.tgz

Includes `vision_bundle.mjs` and both SIMD/non-SIMD WASM loaders and binaries.
The worker loads this module dynamically in a classic worker because the WASM
loader uses `importScripts`. All assets are served by Flask on the same origin.

MediaPipe is copyright Google LLC and contributors, distributed under Apache
2.0; see the included `LICENSE`, sourced from:
https://github.com/google-ai-edge/mediapipe/blob/master/LICENSE

The version-1 Google MediaPipe model sources and checksums for the runtime and
models are in `../../models/privacy/manifest.json`. Model files are distributed
under Apache 2.0 through Google's MediaPipe model collection. Keep the local
runtime, its WASM files, and the asset manifest together when updating versions.
