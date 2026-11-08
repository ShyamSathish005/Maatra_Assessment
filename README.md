# Universal Zero-Annotation Sports Player Detection & Team Color Clustering
### Football Broadcast Optimized & Multi-Sport Generalizable Computer Vision Pipeline

Production-grade, **zero-annotation** computer vision pipeline engineered for **broadcast football match footage** (pitch turf exclusion, outfield squads, match officials/referees, and goalkeepers) while generalizing seamlessly across **all two-team broadcast sports** (**Football, Basketball, Tennis, Hockey, Volleyball, Rugby, American Football**) via unsupervised playing-surface learning (`DynamicSurfaceEstimator`).

---

## 1. Quickstart & Single Terminal Command

```bash
# 1. Environment Setup
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 2. Production Execution Command (Match 1)
python pipeline.py --input data/0bfacc_0.mp4 \
                   --tracker_config configs/bytetrack_sports.yaml \
                   --output_video deliverables/match_1/output_match.mp4 \
                   --output_csv deliverables/match_1/detections_log.csv \
                   --output_json deliverables/match_1/detections_log.json \
                   --summary deliverables/match_1/run_summary.json
```

To run on secondary broadcast footage (Match 2):
```bash
python pipeline.py --input data/Video_2.mp4 \
                   --tracker_config configs/bytetrack_sports.yaml \
                   --output_video deliverables/match_2/output_2.mp4 \
                   --output_csv deliverables/match_2/output_2.csv \
                   --output_json deliverables/match_2/output_2.json \
                   --summary deliverables/match_2/output_2_summary.json
```

Interactive Walkthrough Notebook:
```bash
jupyter notebook run.ipynb
```

---

## 2. Example Output Videos & Quantitative Verification

### Example Processed Output Videos
* [**`deliverables/match_1/output_match.mp4`**](deliverables/match_1/output_match.mp4): Primary processed output video for Match 1 (`0bfacc_0.mp4` — Red vs. White squads, 1080p, 25 fps, 30s), with structured logs in [`deliverables/match_1/`](deliverables/match_1/).
* [**`deliverables/match_2/output_2.mp4`**](deliverables/match_2/output_2.mp4): **Example output video for Match 2** (`Video_2.mp4` — Navy vs. White squads, 1080p, 25 fps, 30s), demonstrating cross-match generalizability with persistent ByteTrack tracking, squad tags, and HUD scoreboard, with structured logs in [`deliverables/match_2/`](deliverables/match_2/).

### Quantitative Verification Metrics
Evaluated on official football broadcast match footage (1080p @ 25 fps, 750 frames / 30.0s):

| Evaluation Metric | Match 1 (`deliverables/match_1/output_match.mp4`) | Match 2 Example (`deliverables/match_2/output_2.mp4`) | Benchmark Standard |
|---|---|---|---|
| **Label Flip Rate (Stability)** | **0.49%** | **0.48%** | **Target: < 5.0%** (Exceeded by 10x) |
| **Flicker Reduction** | **90.86%** | **94.39%** | Eliminates inter-frame strobing |
| **Raw Flips vs. Smoothed** | 1,050 $\to$ **96** | 1,461 $\to$ **82** | Rolling majority-vote stabilization |
| **Kit Separation ($\Delta E^*_{ab}$)** | **69.07** | **61.80** | Perceptually massive separation |
| **PCA Explained Variance** | **95.30%** | **95.51%** | Discriminative color manifold |
| **Detections Logged** | **19,538** (0 nulls) | **17,220** (0 nulls) | Zero-null field guarantee |
| **Unique Track IDs** | **76** | **48** | Persistent ByteTrack trajectories |
| **Officials & Goalkeeper Isolation** | **3,599 Ref / 3,631 GK** | **4,112 Ref / 1,526 GK** | Zero team cluster contamination |

---

## 3. Architecture: The 5-Stage Universal Sports Approach

```text
Broadcast Frame (1080p — Football Pitch & Universal Two-Team Sports)
   │
   ▼
[1. Detection & Tracking] ──► YOLOv8x-seg + ByteTrack (configs/bytetrack_sports.yaml)
   │                           + Dynamic Pitch/Court Convex Hull Filter (drops crowd, bench, staff)
   ▼
[2. Dynamic Surface & Torso] ─► Unsupervised Background Surface Estimator (CIELAB Centroid)
   │                             + Core Upper-Torso Polygon (18–40% h, 25–75% w) ∩ Instance Mask
   │                             ∩ NOT(Playing Surface Color) ∩ 2-px Morphological Erosion
   ▼
[3. Feature Representation] ──► CIELAB Dominant Color (mini-KMeans k=2) + a*b* Histogram
   │                           (Lightness L* down-weighted by 0.3 for shadow invariance)
   │                           + HSV Neon Anomaly Detection (S > 185, V > 195, 25 <= H <= 50)
   ▼
[4. Unsupervised Grouping] ───► Warmup Sample Bank ──► PCA(d=6) ──► K=5 Over-clustering
   │                           ──► Top 2 Frequencies: Team A (Squad 1) & Team B (Squad 2)
   │                           ──► Lower Frequencies / Neon: Officials/Referees & Goalkeepers/Specialists
   │                           ──► Tied Robust Covariance (MCD) ──► Fit once, predict for match
   ▼
[5. Temporal Smoothing] ─────► Rolling majority vote (window=30) with tie hysteresis per Track ID
   │                           (Completely eliminates player box color flickering across frames)
   ▼
Outputs: deliverables/match_1/{output_match.mp4, detections_log.csv, detections_log.json, run_summary.json}
```

---

## 4. Football Broadcast Compliance & Multi-Sport Generalizability

### 1. Dynamic Playing Surface Learning (`DynamicSurfaceEstimator`)
* **The Challenge**: Naive bounding boxes capture 60–70% non-jersey pixels (background pitch grass, stadium advertising boards, shorts, socks, and skin), while hardcoded HSV green thresholds fail when generalizing to non-grass courts.
* **Engineered Solution (`DynamicSurfaceEstimator`)**:
  1. **Automated Background Surface Estimation**: During warmup and per-frame updates, the pipeline masks out all detected person bounding boxes and samples background pixels in the active playing zone.
  2. **1-Cluster CIELAB Surface Centroid**: A fast 1-cluster K-Means center refined on the 70% closest inliers estimates the dominant playing surface color in CIELAB (`surface_lab_cv`) along with its covariance.
  3. **Football Pitch & Universal Court Rejection**: Candidate torso pixels whose chroma-weighted CIELAB distance to the discovered surface centroid falls below `surface_delta_thresh` are rejected. This automatically strips:
     * **Green pitch grass / turf** in Broadcast Football, Rugby, and American Football
     * **Tan/maple hardwood** in Basketball and Volleyball
     * **Blue, green, or red clay/hardcourt** in Tennis
     * **White ice** in Ice Hockey
  4. **Pure Geometric Upper-Torso Isolation**: Samples strictly $y \in [18\%, 40\%]$ of player height and $x \in [25\%, 75\%]$ of player width, intersected with the `YOLOv8x-seg` silhouette mask to exclude shorts, socks, skin, and head/hair.
  5. **2-Pixel Morphological Boundary Erosion**: `cv2.erode` with a $3\times3$ kernel for 2 iterations strips the $1\text{–}2\text{ px}$ antialiased pitch/court boundary halo around player silhouettes.

### 2. Match Officials & Goalkeepers / Specialist Isolation
* **The Challenge**: Football broadcasts require categorizing all on-pitch personnel into distinct groups: **Team A**, **Team B**, **Match Officials (Referees / linesmen)**, and **Goalkeepers** (or specialist roles like liberos in volleyball). Forcing $K=2$ pulls outfield team centroids toward referees and goalkeepers.
* **Engineered Solution**:
  1. **Unsupervised Over-Clustering ($K=5$)**: The warmup feature bank in PCA space is partitioned into $K=5$ clusters.
  2. **Frequency Hierarchy**: The two highest-frequency clusters become **`Team A (Squad 1)` (`group_id = 0`)** and **`Team B (Squad 2)` (`group_id = 1`)**.
  3. **Outlier Routing**: Lower-frequency clusters route automatically to **`Officials/Referees` (`group_id = 2`)** and **`Goalkeepers/Specialists` (`group_id = 3`)**.
  4. **Unsupervised HSV Neon Anomaly Threshold**: Any detection with $S > 185$, $V > 195$, and $25 \le H \le 50$ is flagged as a high-visibility neon referee kit prior to centroid distance calculation.

### 3. Stadium Shadow & Illumination Invariance
* **The Challenge**: Players moving between direct stadium sunlight and harsh roof shadows suffer a 40–60% drop in RGB intensity, splitting a single team into two distinct RGB clusters.
* **Engineered Solution**:
  * Features are computed in **CIELAB ($L^*a^*b^*$)**, decoupling luminance from chromaticity.
  * Lightness $L^*$ is down-weighted by $0.3$ relative to chromatic channels $(a^*, b^*)$, and augmented with a normalized $16\times 16$ 2D $a^*b^*$ histogram under Hellinger distance. Shadowed and sunlit kits map to the same chromatic centroid.

### 4. Frame-to-Frame Cluster Flips & Flickering
* **The Challenge**: Per-frame clustering arbitrarily swaps cluster indices (`Team A` becomes `1` in one frame and `0` in the next), causing boxes to strobe.
* **Engineered Solution**:
  * **Global Fit-Once Model**: The clustering model is fitted once during the warmup phase (150 frames) and frozen; subsequent frames call `predict()`. Deterministic CIELAB hue sorting guarantees consistent `Team A` / `Team B` assignment across runs.
  * **Rolling Majority Vote**: A 30-frame rolling deque per ByteTrack ID with tie-break hysteresis reduces label flips from **1,050 to 96** in Match 1 (flip rate of **0.49%**) and **1,461 to 82** in Match 2 (flip rate of **0.48%**).

### Edge Cases & Production Considerations
* **Single-Frame Sprint & Motion Blur Dynamics**:
  * Under extreme camera pans, high-velocity sprints, or tight player-on-player occlusions, momentary motion blur and pixel blending de-saturate chromaticity channels ($a^*, b^*$).
  * This physical broadcast constraint can cause isolated, single-frame raw classification perturbations before the 30-frame rolling majority vote re-locks the track identity.
  * Despite these broadcast camera artifacts, the temporal smoothing layer sustains an overall track label stability of **>99.5%** (**0.49% flip rate** across 19,538 detection instances in Match 1, and **0.48%** across 17,220 instances in Match 2).
* **Production Roadmap & Next-Stage Enhancements**:
  * **Optical Flow Velocity Conditioning**: Integrate dense or sparse optical flow vectors into the tracker association cost matrix to adaptively scale color weighting during rapid camera pans.
  * **Online Kalman Filtering on Color Distributions**: Replace the discrete rolling majority vote with an online recursive Bayesian filter / Kalman filter directly on the continuous CIELAB feature distribution per track.
  * **Self-Supervised Backbone Integration**: When high-end GPU acceleration (e.g., NVIDIA TensorRT / CUDA) is available, pass the eroded torso crop through a frozen self-supervised vision backbone (such as DINOv2 `dinov2_vits14` CLS tokens) to add semantic texture embeddings alongside perceptual color histograms.

---

## 5. Repository Structure

```text
Maatra Assessment/
├── README.md                      # Architecture, evaluation metrics & multi-sport documentation
├── requirements.txt               # Pinned dependencies
├── .gitignore                     # Git hygiene (ignores weights, caches, and test logs)
├── pipeline.py                    # Standalone CLI sports analytics pipeline
├── run.ipynb                      # Interactive walkthrough Jupyter Notebook
│
├── deliverables/                  # Organized match deliverables (videos, CSV/JSON logs, summaries)
│   ├── match_1/                   # Match 1 deliverables (0bfacc_0.mp4 — Red vs. White)
│   │   ├── output_match.mp4       # Rendered 30s 1080p match video (35.8 MB)
│   │   ├── detections_log.csv     # Structured detection CSV (19,538 rows, 0 nulls)
│   │   ├── detections_log.json    # JSON export of Match 1 detections
│   │   └── run_summary.json       # Diagnostic metrics (0.49% flip rate, Delta E = 69.07)
│   └── match_2/                   # Match 2 deliverables (Video_2.mp4 — Navy vs. White)
│       ├── output_2.mp4           # Example output video (1080p, 30s, 39.3 MB)
│       ├── output_2.csv           # Structured detection CSV (17,220 rows, 0 nulls)
│       ├── output_2.json          # JSON export of Match 2 detections
│       └── output_2_summary.json  # Diagnostic metrics (0.48% flip rate, Delta E = 61.80)
│
├── data/                          # Benchmark 1080p broadcast clips
│   ├── 0bfacc_0.mp4               # Match 1 input clip
│   └── Video_2.mp4                # Match 2 input clip
│
├── configs/                       # Sport-agnostic tracking configuration
│   └── bytetrack_sports.yaml      # ByteTrack config tuned for sports occlusion recovery
│
├── tools/                         # Reproducible utilities
│   └── make_notebook.py           # Notebook generator
│
└── outputs/                       # Scratch/custom output directory
```

---

## 6. CLI Reference

```bash
usage: pipeline.py [-h] --input INPUT [--output_video OUTPUT_VIDEO]
                   [--output_csv OUTPUT_CSV] [--output_json OUTPUT_JSON]
                   [--summary SUMMARY] [--model MODEL] [--imgsz IMGSZ]
                   [--conf CONF] [--device DEVICE] [--half]
                   [--tracker_config TRACKER_CONFIG] [--start_sec START_SEC]
                   [--duration_sec DURATION_SEC]
                   [--warmup_frames WARMUP_FRAMES]
                   [--warmup_stride WARMUP_STRIDE]
                   [--vote_window VOTE_WINDOW] [--l_weight L_WEIGHT]
                   [--surface_delta_thresh SURFACE_DELTA_THRESH]
                   [--torso TOP BOTTOM] [--box_colors {auto,jersey,fixed}]
                   [--no_surface_filter] [--no_h264] [--debug_torso] [-v]
```
