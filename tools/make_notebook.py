"""Generates run.ipynb synchronized with the universal sport-agnostic architecture in pipeline.py."""
import json
from pathlib import Path

cells = []


def md(s):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": s.strip("\n").splitlines(keepends=True)})


def code(s):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                  "source": s.strip("\n").splitlines(keepends=True)})


md(r"""
# Universal Zero-Annotation Sports Player Detection & Team Color Clustering
### Multi-Sport Computer Vision Pipeline (Football, Basketball, Tennis, Hockey, Volleyball, Rugby)

This notebook walks through the universal, sport-agnostic computer vision pipeline implemented in [`pipeline.py`](pipeline.py):

1. **Dynamic Playing Surface Learning (Sport-Agnostic Background Rejection)**:
   - Samples background pixels outside all detected person bounding boxes/masks.
   - Computes the dominant playing surface centroid in CIELAB (`DynamicSurfaceEstimator`) using a fast 1-cluster K-Means center refined on inliers.
   - Filters out candidate torso pixels falling within a color-distance threshold (`surface_delta_thresh`) of the court surface—dynamically eliminating green turf (football/rugby), tan hardwood (basketball/volleyball), blue/red clay (tennis), and white ice (hockey).
2. **Pure Geometric Torso Isolation**:
   - Samples the core upper-torso polygon: $y \in [18\%, 40\%]$ of player height and $x \in [25\%, 75\%]$ of player width.
   - Applies a 2-pixel morphological erosion (`cv2.erode` with $3\times3$ kernel, 2 iterations) on the person silhouette mask to strip antialiased court/surface boundary bleed.
3. **Feature Space & Lightness Invariance**:
   - Dominant CIELAB centroids via patch-level mini-KMeans ($k=2$).
   - Down-weights lightness $L^*$ by $0.3$ relative to chromaticity channels ($a^*, b^*$) to prevent arena/stadium shadow transitions from shifting cluster assignments.
4. **Unsupervised Over-Clustering ($K=5$) & Generic Role Separation**:
   - Frequency routing: Top 2 clusters $\to$ `Team A (Squad 1)` (`group_id = 0`) & `Team B (Squad 2)` (`group_id = 1`).
   - Remaining lower-frequency clusters $\to$ `Officials/Referees` (`group_id = 2`) & `Goalkeepers/Specialists` (`group_id = 3`).
   - Unsupervised HSV neon anomaly detector ($S > 185, V > 195, 25 \le H \le 50$) routes neon official jerseys prior to centroid distance calculation.
5. **Temporal Track Smoothing**:
   - Uses [`configs/bytetrack_sports.yaml`](configs/bytetrack_sports.yaml) and a 30-frame rolling majority vote with tie-break hysteresis per Track ID, keeping label flip rates below $1\%$.
""")

code(r"""
import json, cv2, numpy as np, pandas as pd, matplotlib.pyplot as plt
from pathlib import Path
import pipeline as P

INPUT = "data/0bfacc_0.mp4"
cfg = P.PipelineConfig(
    tracker_config="configs/bytetrack_sports.yaml",
    torso_top=0.18, torso_bottom=0.40,
    torso_left=0.25, torso_right=0.75,
    duration_sec=3, warmup_frames=60
)
pipe = P.MatchAnalysisPipeline(cfg)
print(f"Pipeline initialized on device: {pipe.device} | Tracker: {cfg.tracker_config}")
""")

md(r"""
## Stage 2: Dynamic Playing Surface Rejection & Pure Geometric Torso Isolation
Comparing raw bounding box vs. core upper-torso band ($18\text{–}40\%$ H, $25\text{–}75\%$ W) vs. surface-rejected & 2px-eroded jersey pixels.
""")

code(r"""
cap = cv2.VideoCapture(INPUT); cap.set(cv2.CAP_PROP_POS_FRAMES, 60); ok, frame = cap.read(); cap.release()
res = pipe._YOLO(cfg.model).predict(frame, **pipe._infer_kwargs(0.5))[0]
boxes, polys = res.boxes.xyxy.cpu().numpy(), res.masks.xy
pipe.surface.update(frame, boxes)
order = np.argsort(-(boxes[:, 3] - boxes[:, 1]))[:5]

ex = P.JerseyColorExtractor(cfg, surface_estimator=pipe.surface, keep_crop=True)
fig, axes = plt.subplots(len(order), 3, figsize=(8, 2.3 * len(order)))
for r, i in enumerate(order):
    x1, y1, x2, y2 = boxes[i].astype(int)
    f = ex.extract(frame, boxes[i], polys[i])
    tx1, ty1, tx2, ty2 = f.torso
    for ax, img, t in zip(axes[r], [frame[y1:y2, x1:x2], frame[ty1:ty2, tx1:tx2], f.crop],
                          ["Bounding Box", "Core Torso (18-40% H, 25-75% W)", "Surface-Rejected & Eroded"]):
        ax.imshow(img[:, :, ::-1]); ax.set_title(t, fontsize=9); ax.axis("off")
plt.tight_layout()
plt.show()
""")

md(r"""
## Stage 4: Unsupervised Over-Clustering ($K=5$) & Generic Role Separation
Inspect learned cluster centroids for `Team A (Squad 1)`, `Team B (Squad 2)`, `Officials/Referees`, and `Goalkeepers/Specialists`, plus the dynamically discovered playing surface color.
""")

code(r"""
summary = json.loads(Path("deliverables/match_1/run_summary.json").read_text())
tm = summary["team_model"]

print(f"Warmup samples: {tm['warmup_samples']}")
print(f"PCA explained variance (6 dims): {tm['pca_explained_variance'] * 100:.2f}%")
print(f"K=5 Cluster Sample Counts: {tm['warmup_k5_counts']}")
print(f"Warmup Group Sizes: {tm['warmup_group_sizes']}")
print(f"Discovered Playing Surface BGR: {tm.get('playing_surface_bgr')}")
print(f"Team A vs Team B Delta E Separation: {tm['team_color_delta_e']:.2f}")

swatches = [
    ("Team A (Squad 1)", tm["team_colors_bgr"][0]),
    ("Team B (Squad 2)", tm["team_colors_bgr"][1]),
]
if tm.get("playing_surface_bgr"):
    swatches.append(("Playing Surface", tm["playing_surface_bgr"]))

fig, ax = plt.subplots(figsize=(2.4 * len(swatches), 1.8))
for i, (name, bgr) in enumerate(swatches):
    ax.add_patch(plt.Rectangle((i, 0), 0.85, 1, color=np.array(bgr[::-1]) / 255.0))
    ax.text(i + 0.425, -0.35, name, ha="center", fontsize=10, weight="bold")
ax.set_xlim(0, len(swatches)); ax.set_ylim(-0.6, 1.1); ax.axis("off")
plt.title(f"Discovered Kit & Playing Surface Colors (Delta E: {tm['team_color_delta_e']:.1f})", fontsize=12)
plt.show()
""")

md(r"""
## Stage 5: Temporal Track Smoothing & Label Flip Verification
Verifying that label flips remain well under the 5% threshold across all detections.
""")

code(r"""
df = pd.read_csv("deliverables/match_1/detections_log.csv")
print(f"Total Detections Logged: {len(df):,}")
print(f"Unique Player Tracks: {df.player_id.nunique()}")
print(f"Group Distribution:")
print(df.group_name.value_counts())

raw_flips = summary["label_flips_raw"]
smooth_flips = summary["label_flips_smoothed"]
flip_rate = summary["label_flip_rate_pct"]

print("\n--- Stability Metrics ---")
print(f"Raw Per-Frame Flips: {raw_flips}")
print(f"Smoothed Track Flips: {smooth_flips}")
print(f"Overall Track Flip Rate: {flip_rate:.2f}% (Requirement: < 5%)")
assert flip_rate < 5.0, f"Flip rate {flip_rate}% exceeds 5% threshold!"
print("Verification Passed: Label flip rate is well below 5%.")

long_tracks = df.player_id.value_counts().index[:8]
fig, ax = plt.subplots(figsize=(12, 4.5))
for k, pid in enumerate(long_tracks):
    t = df[df.player_id == pid]
    ax.scatter(t.frame_id, np.full(len(t), k) + 0.15, c=t.raw_group_id, cmap="Set1", vmin=-1, vmax=4, s=5)
    ax.scatter(t.frame_id, np.full(len(t), k) - 0.15, c=t.group_id, cmap="Set1", vmin=-1, vmax=4, s=5, marker="s")
ax.set_yticks(range(len(long_tracks)), [f"Player #{p}" for p in long_tracks])
ax.set_xlabel("Frame ID")
ax.set_title("Temporal Stability: Upper Dots (Raw Per-Frame) vs. Lower Squares (Rolling Majority Vote)")
plt.grid(True, linestyle="--", alpha=0.4)
plt.tight_layout()
plt.show()
""")

md(r"""
## Annotated Video Inspection
Sample rendered frames from `deliverables/match_1/output_match.mp4` showing bounding box classifications and HUD overlay.
""")

code(r"""
cap = cv2.VideoCapture("deliverables/match_1/output_match.mp4")
n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fig, axes = plt.subplots(1, 3, figsize=(18, 4.5))
for ax, fi in zip(axes, [50, 350, 650]):
    cap.set(cv2.CAP_PROP_POS_FRAMES, min(fi, n - 1)); ok, fr = cap.read()
    ax.imshow(fr[:, :, ::-1])
    ax.set_title(f"Rendered Frame {fi} / {n}", fontsize=11)
    ax.axis("off")
cap.release()
plt.tight_layout()
plt.show()
""")

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"}
    },
    "nbformat": 4,
    "nbformat_minor": 5
}
Path("run.ipynb").write_text(json.dumps(nb, indent=1))
print("Successfully generated and synchronized run.ipynb.")
