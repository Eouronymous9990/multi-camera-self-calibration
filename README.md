# Multi-Camera Self-Calibration and Metric 3D Reconstruction

![Multi-camera views reconstructed into a metric 3D world](assets/overview.png)

This project reconstructs a **metric 3D scene from multiple ordinary, static cameras** while automatically estimating the geometry that relates those cameras to each other. Given synchronized footage from two or more cameras pointed at the same physical space, the pipeline in [pipeline.ipynb](pipeline.ipynb) recovers each camera's pose relative to a shared reference frame, resolves the real-world (metric) scale of the reconstruction, and defines a consistent 3D world coordinate system anchored to the physical ground — all without a checkerboard or calibration wand present in the scene during the main calibration workflow.

The current experimental scene is a football/soccer player performing a dribbling skill with a ball, captured by two static cameras. That scene is the **observation source**, not the deliverable: the player's body and the ball are used purely as geometric targets that supply the correspondences and metric reference the calibration math needs. The actual output of this project is multi-camera calibration + 3D reconstruction infrastructure intended to sit underneath a separate, downstream sports-analysis layer.

## The Problem

Reconstructing accurate 3D geometry from multiple cameras requires knowing, for every camera:

- **Intrinsics** — focal length, principal point, and lens distortion.
- **Extrinsics** — each camera's rotation and translation relative to the others.
- **Relative geometry** — a single, jointly-consistent multi-camera configuration, not just independent pairwise estimates.
- **Metric scale** — cameras alone only ever recover geometry up to an unknown, arbitrary scale factor; something of known physical size has to anchor that scale to real units (meters).
- **A consistent world coordinate system** — an origin and axis convention (e.g., "up" defined by the physical ground) that downstream 3D analysis can reason about.

Traditionally this is solved with a printed checkerboard or a calibration wand waved through the scene. This project instead estimates the extrinsic geometry, the metric scale, and the world coordinate frame directly from what the cameras already see during normal filming: a person moving through the shared field of view, and a ball of known physical size.

## Input and Output

### Input

- **Camera intrinsics** for each physical camera (K matrix + lens distortion coefficients), supplied as pre-computed values — see [Stage 0](#stage-0--camera-intrinsics-pre-calibrated-input) below for exactly how these enter the pipeline.
- **Synchronized frames** from each camera (folders of images, one per camera), showing a single player and, for at least some frames, a visible soccer ball.
- **A known physical reference**: the ball's real radius (`BALL_RADIUS_CM`, default 11.0 cm), used as the metric anchor.

### Output

A single serialized calibration file (`stereo_calibration_hybrid.pkl`) containing, per camera:

- Intrinsics (`Ks`, `dists`)
- Extrinsics relative to camera 1 (`Rs`, `Ts`)
- Essential/fundamental matrices (`Es`, `Fs`)
- The recovered metric scale (`SCALE`, `SCALE_UNIT`)
- The world coordinate transform (`WORLD_TRANSFORM`: world rotation, origin, ground-plane and axis diagnostics)

This file is the calibration package a downstream 3D analysis tool (player tracking, kinematics, etc.) would load — the calibration/reconstruction stage does not itself perform that analysis.

## Pipeline Overview

![Multi-camera self-calibration pipeline diagram](assets/pipeline_overview.jpg)

> The diagram above illustrates the intended end-to-end pipeline and labels the first stage "mrcal." **In the current code, intrinsics are not computed by an intrinsic-calibration routine at all** — they are loaded as fixed, pre-computed `K`/distortion values hardcoded in the notebook (see Stage 0 below). Whether those particular numbers were originally produced with mrcal is not something `pipeline.ipynb` or `scripts/` can confirm — that step, if it happened, occurred outside this repository. Treat the diagram as the target architecture and the text below as what the code actually does today.

```text
Camera Intrinsics (pre-calibrated input)
        |
Synchronized Multi-Camera Frames + Correspondence-Mode Selection
        |
Cross-Camera Person Correspondences  (manual clicks / OSNet Re-ID + pose / both)
        |
Initial Extrinsics per camera  (Essential/Fundamental matrix, R + unit-direction T)
        |
Stage 1 — Unscaled Joint Bundle Adjustment
        |
Ball Detection (YOLO, all cameras/frames)
        |
Stage 2 — Per-Camera Metric Scale Recovery (ball diameter)
        |
Stage 3 — Scale-Aware Joint Bundle Adjustment
        |
World-Axis / Ground-Plane Construction  (ball-plane fit or manual clicks)
        |
Validation  (reprojection error, rectification, epipolar tests)
        |
Final Calibration Package (stereo_calibration_hybrid.pkl)
        |
Ready for a downstream analysis pipeline
```

## Pipeline Stages

### Stage 0 — Camera Intrinsics (pre-calibrated input)

`pipeline.ipynb` Cell 2 hardcodes two intrinsic profiles as NumPy arrays — `K_cam12`/`dist_cam12` and `K_cam3`/`dist_cam3` — each a 3x3 `K` matrix plus a 5-parameter OpenCV radial-tangential distortion vector (`k1, k2, p1, p2, k3`). A `PATTERN` string (e.g. `"01"`) assigns one of these two profiles to each camera slot, so physically identical camera bodies can reuse the same profile across a rig with more than two cameras. Nothing in this repository *computes* these values from checkerboard images — they are treated as already-known input, presumably calibrated upstream by some other tool.

### Stage 1 — Video Preparation and Correspondence-Mode Selection

Cell 1/3 configure the camera count (`NUM_CAMS`) and per-camera frame folders, and warn if different cameras have different frame counts (frames are assumed pre-synchronized by matching order/filename — there is no timestamp-based alignment). The Mode Selector cell then reads `PIPELINE_INDEX` to choose how 2D correspondences will be collected: `'1'` manual clicking only, `'2'` fully automatic Re-ID, or `'3'` both combined.

### Stage 2 — Cross-Camera Correspondences

Three interchangeable sources feed the same downstream structure, always matched against camera 1 (star topology: cam1 <-> cam2, cam1 <-> cam3, ...):

- **Manual** ([scripts/manual_match.py](scripts/manual_match.py)): an interactive Matplotlib point-and-click tool, one window per non-reference camera.
- **Automatic Re-ID + pose** ([scripts/reid_match.py](scripts/reid_match.py), [scripts/match.py](scripts/match.py)): YOLO11 (`models/yolo11x.pt`, COCO class 0) detects people in both views of a frame; each detected crop is embedded with **OSNet** (`osnet_ain_x1_0`, via `torchreid`) using horizontal-flip-averaged embeddings, combined with a torso-color histogram signal into a weighted similarity score, and matched with Hungarian assignment against a confidence threshold. For each matched pair, MediaPipe Pose extracts body keypoints in both crops; the intersection of visible keypoints (restricted to shoulders/hips/knees/ankles, landmark IDs `{11,12,23,24,25,26,27,28}`) becomes that frame's 2D correspondence points. An optional review UI lets a bad automatic match be discarded before it reaches calibration.
- **Merge** (Cell 4D): manual and automatic correspondences for the same camera pair are concatenated into one point set.

### Stage 3 — Initial Extrinsic Estimation

For each cam1 <-> camK pair independently (Cell 5): correspondence points are undistorted and normalized, an Essential matrix is estimated with RANSAC (falling back to a Fundamental matrix if the Essential estimate is weak), and `cv2.recoverPose` decomposes it into a rotation `R` and a translation **direction** `T` (unit norm — no scale yet). This is an initialization, not a final calibration: it is computed independently per pair, so different pairs' geometries are not yet mutually consistent, and the reconstructed structure still lives in an arbitrary, unscaled coordinate system.

### Stage 4 — Unscaled Joint Bundle Adjustment

All non-reference cameras' `(R, T-direction)` and every triangulated 3D scene point are refined **jointly** in one `scipy.optimize.least_squares` problem, minimizing reprojection error across every camera at once (not pair by pair), with each camera's translation still constrained to unit norm. The solver uses a sparse Jacobian for efficiency and an outer IRLS loop (3 iterations) that approximates a Cauchy robust loss, so noisy or outlier correspondences are progressively down-weighted; whichever IRLS iterate achieves the best true (unweighted) reprojection RMS is kept, not necessarily the last one. The output is a single, internally consistent multi-camera reconstruction — still at an arbitrary, non-metric scale.

### Stage 5 — Metric Scale Recovery from the Ball

A YOLO ball detector (COCO class 32) scans every frame of every camera. For each non-reference camera **separately**, cam1<->camK ball detections are gated by symmetric epipolar distance (rejecting frames where the two cameras likely detected different balls), triangulated, and converted to an estimated real-world diameter from pixel radius and depth; a MAD-based outlier filter discards inconsistent frames before averaging. The scale factor is `s_k = true_diameter / estimated_diameter`, applied to rescale camera k's translation into meters.

Scale is recovered **per camera, not as one shared factor** — a deliberate deviation from pooling everything into a single scalar, documented directly in the code: because correspondences are only ever pairwise-to-cam1 (no point is jointly seen by two non-reference cameras), Stage 4's bundle adjustment has no information tying one camera's baseline length to another's, so forcing a single shared scale on every camera was found to produce roughly 2x real-world measurement errors in testing.

### Stage 6 — Scale-Aware Joint Bundle Adjustment

A second joint least-squares refinement, now in full metric space (camera translations get all 3 degrees of freedom, no unit-norm constraint). It reuses Stage 4's exact point/observation bookkeeping and adds a per-frame ball-radius residual that keeps the reconstructed ball's physical size anchored to its known radius, preventing scale drift during optimization. An optional smoothness penalty over the ball's trajectory exists but is disabled by default. This is the final refinement of camera geometry and 3D structure — both geometrically consistent (reprojection-minimizing) and metrically meaningful (anchored to real units).

### Stage 7 — Ground Plane and World Coordinate System

A separate stage from camera calibration: ball centers detected near/on the ground are triangulated, and a plane is fit through them by SVD. That plane's normal becomes the world **+Y (up)** axis; the world **+X** axis is built by projecting the camera's own +X direction onto the ground plane, and **+Z = X x Y** completes a right-handed frame (labeled "forward" in the code — it is derived from camera orientation, not from any specific field/goal direction). Several automated quality checks guard this fit: a minimum point count, an area-ratio check that rejects a near-collinear ball track, a relative-RMS planarity threshold, and a leave-one-out normal-stability check. A configurable offset (`GROUND_Y_OFFSET_CM`, default 9 cm) shifts the world origin down along -Y to compensate for the ball's center sitting roughly one radius above the true floor. Optionally, the reference pair's `R`/`T` are jointly re-refined together with the ground-plane fit (a bundle adjustment with an added planarity residual). A manual alternative (`AXIS_MODE = "click"` with `AUTO_AXIS_FROM_BALL = False`) exists: a two-phase interactive tool for clicking Y-axis-defining point pairs and ground-surface point pairs.

## Validation

- **Reprojection error analysis** (Cell 6): per camera-pair mean/max reprojection error in pixels, with per-point bar plots.
- **Stereo rectification** (Cell 7): `cv2.stereoRectify` plus a visual check — rectified image pairs overlaid with horizontal lines that should pass through matching features.
- **Interactive epipolar-line tester** (Cell 11): click a point in one camera view and see its corresponding epipolar line drawn in the other, for spot-checking any camera pair.
- **World-frame verification** (Cell AX2): a 3D plot of the recovered world frame plus a numeric check that ground points land near `Y = 0` and the recovered Y axis is close to `[0, 1, 0]`.
- **Saved-calibration round-trip validator with 3D skeleton reconstruction** (Cell 12): a full implementation exists that reloads *only* the saved pickle from disk, re-detects people and MediaPipe pose keypoints, matches people across cameras purely by epipolar consistency, triangulates the matched skeleton, and renders it interactively (Plotly) alongside match-quality overlays — this is the project's intended "does the calibration produce a physically sensible player?" sanity check. **As shipped, this entire cell is commented out** in `pipeline.ipynb` and is not executed by the current pipeline run; it is present as implemented, disabled tooling.

## Final Calibration Output

Cell 8 saves `stereo_calibration_hybrid.pkl` into the configured output directory, after asserting both `USE_WORLD_AXIS` and `USE_SCALE` are enabled. It bundles per-camera intrinsics/extrinsics/essential-fundamental matrices, the metric `SCALE`/`SCALE_UNIT`, and the full `WORLD_TRANSFORM`, plus flat `K1/K2/R/T/...`-style aliases for the cam1<->cam2 pair for any older 2-camera-only consumer. This project's scope ends here: **camera calibration and metric 3D reconstruction infrastructure**, ready to be loaded by a separate downstream analysis pipeline. It does not itself perform player tracking, kinematics, or sports analysis.

## Results

The notebook's stored execution output reflects one recorded run: a **2-camera rig** (`NUM_CAMS=2`), `PIPELINE_INDEX='2'` (fully automatic Re-ID), `BALL_RADIUS_CM=11.0`. These are the only numbers currently baked into the repository — they describe this one experiment, not a general benchmark.

| Metric | Value |
|---|---|
| Correspondence frames (Auto Re-ID) | 35 / 39 accepted, 253 total point correspondences |
| Essential-matrix RANSAC inliers | 100 / 253 |
| Stage 4 (unscaled BA) reprojection RMS | 0.518 px overall (cam1 0.527 px, cam2 0.510 px) |
| Ball frames detected in >=2 cameras | 19 / 38 |
| Stage 5 scale factor (cam2) | `s = 9.559` (1 reconstruction unit = 10.46 cm), `||T2|| = 9.56` m |
| Stage 6 (scale-aware BA) reprojection RMS | 130.5 px -> 0.551 px (cam1), 231.1 px -> 0.533 px (cam2) before/after re-optimizing |
| Ground-plane fit quality (AX1) | relative RMS 0.0053, area ratio 0.486, normal stability 0.23 deg |
| World-frame axis orthogonality | ~1e-14 (X.Y, Y.Z, X.Z all effectively zero) |
| **Final reprojection error (Cell 6, post-refinement)** | **cam1: mean 0.452 px / max 1.718 px — cam2: mean 0.568 px / max 2.366 px** |
| Saved translation magnitude | `||T2|| = 9.162` m |

Notes on reading this table: the large 130.5 px / 231.1 px "before" numbers in Stage 6 are expected, not a bug — they reflect the reprojection error immediately after Stage 5 rescales the translation by a factor of ~9.56x, before Stage 6's optimizer has run to reconcile the rest of the geometry with that new scale. The stereo-rectification ROI for this particular camera pair came out very narrow (29 px wide for cam1), consistent with the ~70 degree relative rotation (`Ry`) recovered between these two specific camera placements — a wide-baseline/highly-converged rig, not a general system limitation.

## Demo / Visual Results

| | |
|---|---|
| ![Two camera views and the resulting 3D reconstruction](assets/overview.png) | Two synchronized camera views (top) and the reconstructed 3D scene — player skeleton and ball — in a shared world frame (bottom). |
| ![Pipeline architecture](assets/pipeline_overview.jpg) | The intended end-to-end pipeline architecture (see the caveat on the intrinsics stage above). |

## Technology Stack

Verified against actual imports in `pipeline.ipynb` and `scripts/`:

- **Python**, **NumPy**, **SciPy** (`optimize.least_squares`, `sparse`, `optimize.linear_sum_assignment`)
- **OpenCV** (`cv2`) — undistortion, Essential/Fundamental matrix estimation, pose recovery, triangulation, stereo rectification
- **PyTorch** + **torchvision**
- **torchreid** — OSNet (`osnet_ain_x1_0`) person re-identification embeddings
- **Ultralytics YOLO11** (`models/yolo11x.pt`) — person and sports-ball detection
- **MediaPipe** (`solutions.pose`) — body keypoint estimation
- **Matplotlib** (incl. `mpl_toolkits.mplot3d`, interactive widgets) — all visualization and click-based tools
- **Pillow** — image preprocessing for OSNet
- **Plotly** — only used by the (currently disabled) Cell 12 3D skeleton viewer

No dependency manifest (`requirements.txt` / `environment.yml`) currently exists in the repository — the list above is derived from source, not a lockfile.

## Project Structure

```text
My_3D_Workflow/
├── pipeline.ipynb        Main notebook: the full calibration pipeline, cell by cell
├── models/
│   └── yolo11x.pt         YOLO11-x weights (person + ball detection; auto-downloaded if missing)
├── scripts/
│   ├── detectors.py        Shared cached YOLO-detector loader
│   ├── geometry.py         Shared undistort / projection-matrix / triangulation / skew helpers
│   ├── manual_match.py      Interactive manual point-correspondence click tool
│   ├── match.py            OSNet person Re-ID core (detection, embeddings, matching)
│   └── reid_match.py        Orchestrates automatic Re-ID + pose matching across camera pairs
├── assets/                 Images used by this README
└── README.md
```

## Current Status

**Implemented and exercised end-to-end** (see Results): 2-camera calibration through all three bundle-adjustment stages, automatic Re-ID + pose correspondence collection, ball-based metric scale recovery, automatic ground-plane/world-axis construction, reprojection-error/rectification/epipolar validation, and a single serialized calibration output. The code is written generically for `NUM_CAMS >= 2` (star topology around a reference camera), but the only stored, executed run in the repository is a 2-camera one.

**Implemented but not currently active:** Cell 12's full round-trip validator — reload from disk, cross-camera person matching by epipolar consistency, 3D skeleton triangulation, and an interactive Plotly viewer — exists in the notebook entirely commented out.

**Known limitations / not yet done:**
- No committed dependency manifest.
- Input frame-folder paths (`BASE`, `MATCH_FRAMES_DIRS`, etc.) are hardcoded absolute local paths specific to one machine; not yet a configurable/portable setup.
- Frame synchronization across cameras is assumed by ordering/filename, not verified by timestamp.
- Correspondences are pairwise-to-reference only (no true N-way multi-camera point tracks), which is why per-camera (not pooled) scale recovery was necessary.

**Future work / potential extensibility:** validated runs with more than two cameras; re-enabling the Cell 12 skeleton-based validation as a standard pipeline step; packaging dependencies and parameterizing input paths. Using a different known-size object in place of the ball, or extending beyond the current football/dribbling scene, are plausible future directions but are explicitly out of scope for the system as it stands today.
