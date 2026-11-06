#!/usr/bin/env python3
"""
Unsupervised football team identification from broadcast video.

5-Stage Sports Analytics Pipeline:
1. Detection, Segmentation & Tracking:
   - YOLOv8x-seg (COCO class 0: person) + ByteTrack
   - Convex pitch-hull filtering to remove crowd, bench, and ball boys
2. Pitch & Torso Isolation:
   - Upper-torso geometric crop: 15% to 42% height, 22% to 78% width
   - Dynamic HSV pitch-green exclusion mask (H in [35, 85], S > 40, V > 40)
   - 2-pixel morphological erosion (cv2.erode with 3x3 kernel, 2 iterations)
     prior to color sampling to strip antialiased grass boundary bleed
3. Feature Representation & Lightness Invariance:
   - Mini-KMeans (k=2) on masked torso pixels to extract dominant CIELAB cluster
   - Down-weight lightness L* by 0.3 relative to chromatic channels (a*, b*)
   - Normalized 16x16 a*b* 2D histogram under Hellinger distance
   - Unsupervised neon anomaly filter in HSV (S > 185, V > 195, 25 <= H <= 50)
4. Neon & Outlier Separation (Referee / GK Isolation):
   - Warmup sample bank fitted with unsupervised K=5 over-clustering (GMM/K-Means)
   - Two highest-frequency clusters assigned to Team A (group_id = 0) and Team B (group_id = 1)
   - Remaining lower-frequency clusters routed to Officials / Goalkeepers (group_id = 2 and group_id = 3)
   - Neon detections tagged as official/outlier anomaly before centroid distance calculation
5. Temporal Track Smoothing:
   - Rolling majority vote (default 30 frames) with tie-break hysteresis per Track ID
   - Eliminates player bounding box color flickering across frames

Outputs: Annotated Video + Structured CSV/JSON Logs + Run Summary Diagnostics
"""
from __future__ import annotations

import argparse
import json
import logging
import shutil
import subprocess
import time
from collections import Counter, defaultdict, deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import pandas as pd
from scipy.stats import chi2
from sklearn.cluster import KMeans
from sklearn.covariance import MinCovDet
from sklearn.decomposition import PCA

LOG = logging.getLogger("team_pipeline")

GROUP_NAMES: Dict[int, str] = {
    0: "Team A",
    1: "Team B",
    2: "Officials",
    3: "Goalkeepers",
    -1: "Unassigned",
}

TEAM_A_GROUP = 0
TEAM_B_GROUP = 1
OFFICIALS_GROUP = 2
GOALKEEPERS_GROUP = 3
UNASSIGNED = -1

# High-visibility BGR display palette
FIXED_PALETTE: Dict[int, Tuple[int, int, int]] = {
    0: (40, 60, 240),     # Team A: Bright Red/Coral
    1: (240, 210, 40),    # Team B: Cyan/Sky Blue
    2: (0, 240, 255),     # Officials: Neon Yellow / Gold
    3: (255, 0, 255),     # Goalkeepers: Magenta / Fuchsia
    -1: (170, 170, 170),  # Unassigned: Neutral Gray
}


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
@dataclass
class PipelineConfig:
    # Stage 1 - detection / tracking
    model: str = "yolov8x-seg.pt"
    imgsz: int = 1280
    conf: float = 0.1            # low conf to feed ByteTrack 2nd association stage
    iou: float = 0.6
    device: str = "auto"
    half: bool = False
    tracker: str = "trackers/bytetrack_football.yaml"
    pitch_filter: bool = True
    pitch_min_green_frac: float = 0.15

    # Stage 2 - strict torso mask & grass erosion
    torso_top: float = 0.15       # 15% height
    torso_bottom: float = 0.42    # 42% height
    torso_left: float = 0.22      # 22% width
    torso_right: float = 0.78     # 78% width
    green_h_low: int = 35
    green_h_high: int = 85
    green_s_min: int = 40
    green_v_min: int = 40
    erode_kernel_size: int = 3
    erode_iterations: int = 2     # 2-pixel morphological erosion
    min_box_h: int = 20
    min_valid_pixels: int = 20

    # Stage 3 - feature space & lightness invariance
    features: str = "lab"        # "lab" | "lab+dinov2"
    hist_bins: int = 16
    l_weight: float = 0.3        # down-weight lightness L* by 0.3 for shadow invariance
    hist_weight: float = 40.0
    dino_weight: float = 40.0

    # Neon referee / outlier threshold check in HSV space
    neon_h_min: int = 25
    neon_h_max: int = 50
    neon_s_min: int = 185
    neon_v_min: int = 195
    neon_pixel_ratio_thresh: float = 0.12  # >= 12% neon pixels -> official anomaly

    # Stage 4 - unsupervised over-clustering
    n_teams: int = 2
    warmup_k: int = 5            # K = 5 over-clustering
    warmup_frames: int = 150
    warmup_stride: int = 3
    warmup_conf: float = 0.5
    warmup_min_box_frac: float = 0.035
    warmup_rel_box_h: float = 0.7
    warmup_min_purity: float = 0.5
    warmup_max_samples: int = 4000
    pca_components: int = 6
    outlier_quantile: float = 0.995

    # Stage 5 - temporal smoothing
    vote_window: int = 30

    # Rendering & output
    box_colors: str = "auto"     # "auto" | "jersey" | "fixed"
    debug_torso: bool = False
    h264: bool = True
    start_sec: float = 0.0
    duration_sec: float = 0.0    # 0 = entire clip


# --------------------------------------------------------------------------- #
# Helpers & Device Selection
# --------------------------------------------------------------------------- #
def resolve_device(device: str) -> str:
    """Robust dynamic device selection: CUDA -> MPS -> CPU fallback."""
    if device != "auto":
        return device
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda:0"
        if torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def find_ffmpeg() -> Optional[str]:
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def cv_lab_to_true(lab: np.ndarray) -> np.ndarray:
    """OpenCV 8-bit Lab (L*255/100, a+128, b+128) -> true CIELAB units."""
    lab = np.asarray(lab, dtype=np.float32)
    out = lab.copy()
    out[..., 0] = lab[..., 0] * 100.0 / 255.0
    out[..., 1:] = lab[..., 1:] - 128.0
    return out


def cv_lab_to_bgr(lab_cv: np.ndarray) -> Tuple[int, int, int]:
    px = np.clip(np.asarray(lab_cv, dtype=np.float32), 0, 255).astype(np.uint8).reshape(1, 1, 3)
    b, g, r = cv2.cvtColor(px, cv2.COLOR_LAB2BGR)[0, 0]
    return int(b), int(g), int(r)


def delta_e(lab_cv_1: np.ndarray, lab_cv_2: np.ndarray) -> float:
    return float(np.linalg.norm(cv_lab_to_true(lab_cv_1) - cv_lab_to_true(lab_cv_2)))


# --------------------------------------------------------------------------- #
# Stage 1b - Pitch Hull Filter
# --------------------------------------------------------------------------- #
class PitchFilter:
    def __init__(self, cfg: PipelineConfig, scale: float = 0.25):
        self.cfg = cfg
        self.scale = scale
        self.mask: Optional[np.ndarray] = None

    def update(self, frame: np.ndarray) -> None:
        cfg = self.cfg
        small = cv2.resize(frame, None, fx=self.scale, fy=self.scale, interpolation=cv2.INTER_AREA)
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        green = cv2.inRange(
            hsv,
            (cfg.green_h_low, cfg.green_s_min, cfg.green_v_min),
            (cfg.green_h_high, 255, 255),
        )
        if green.mean() / 255.0 < cfg.pitch_min_green_frac:
            self.mask = None
            return
        k = max(5, int(0.02 * small.shape[1]) | 1)
        closed = cv2.morphologyEx(green, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        min_area = 0.01 * small.shape[0] * small.shape[1]
        big = [c for c in contours if cv2.contourArea(c) > min_area]
        if not big:
            self.mask = None
            return
        hull = cv2.convexHull(np.vstack(big))
        mask = np.zeros(small.shape[:2], np.uint8)
        cv2.fillConvexPoly(mask, hull, 255)
        m = max(3, int(0.015 * small.shape[1]) | 1)
        self.mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (m, m)))

    def on_pitch(self, bbox: Sequence[float]) -> bool:
        if self.mask is None:
            return True
        x1, y1, x2, y2 = bbox
        fx = int(np.clip((x1 + x2) / 2 * self.scale, 0, self.mask.shape[1] - 1))
        fy = int(np.clip(y2 * self.scale, 0, self.mask.shape[0] - 1))
        return bool(self.mask[fy, fx])


# --------------------------------------------------------------------------- #
# Stages 2 & 3 - Strict Torso Mask, Green Erosion & CIELAB Features
# --------------------------------------------------------------------------- #
@dataclass
class JerseyFeature:
    lab_cv: np.ndarray        # dominant colour, OpenCV 8-bit Lab
    hist: np.ndarray          # normalised a*b* histogram (bins*bins,)
    n_valid: int              # valid pixels after torso mask, green filter & 2px erosion
    purity: float             # share of dominant colour cluster
    box_h: int
    is_neon: bool             # unsupervised neon anomaly flag
    crop: Optional[np.ndarray] = None
    torso: Tuple[int, int, int, int] = (0, 0, 0, 0)


class JerseyColorExtractor:
    def __init__(self, cfg: PipelineConfig, keep_crop: bool = False):
        self.cfg = cfg
        self.keep_crop = keep_crop
        self.lower_green = np.array([cfg.green_h_low, cfg.green_s_min, cfg.green_v_min], np.uint8)
        self.upper_green = np.array([cfg.green_h_high, 255, 255], np.uint8)
        self.km_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.5)
        self.erode_kernel = np.ones((cfg.erode_kernel_size, cfg.erode_kernel_size), np.uint8)

    def torso_box(self, bbox: Sequence[float], shape: Tuple[int, ...]) -> Optional[Tuple[int, int, int, int]]:
        H, W = shape[:2]
        x1, y1, x2, y2 = bbox
        # Clamped bounding box coordinates to prevent arbitrary resolution IndexError
        x1, y1 = max(0, min(W, int(x1))), max(0, min(H, int(y1)))
        x2, y2 = max(0, min(W, int(x2))), max(0, min(H, int(y2)))
        h, w = y2 - y1, x2 - x1
        if h < self.cfg.min_box_h or w < 8:
            return None
        c = self.cfg
        ty1, ty2 = y1 + int(h * c.torso_top), y1 + int(h * c.torso_bottom)
        tx1, tx2 = x1 + int(w * c.torso_left), x1 + int(w * c.torso_right)
        ty1, ty2 = max(0, min(H, ty1)), max(0, min(H, ty2))
        tx1, tx2 = max(0, min(W, tx1)), max(0, min(W, tx2))
        if ty2 - ty1 < 3 or tx2 - tx1 < 3:
            return None
        return tx1, ty1, tx2, ty2

    def check_neon(self, hsv_crop: np.ndarray, mask: np.ndarray) -> bool:
        """
        Unsupervised threshold check in HSV space:
        Any detection with saturation S > 185, value V > 195, and 25 <= H <= 50
        is flagged as an official/neon outlier anomaly.
        """
        if mask is None or cv2.countNonZero(mask) == 0:
            return False
        c = self.cfg
        px_hsv = hsv_crop[mask > 0]
        if len(px_hsv) < 5:
            return False
        neon_px = (
            (px_hsv[:, 0] >= c.neon_h_min) &
            (px_hsv[:, 0] <= c.neon_h_max) &
            (px_hsv[:, 1] > c.neon_s_min) &
            (px_hsv[:, 2] > c.neon_v_min)
        )
        return bool(np.mean(neon_px) >= c.neon_pixel_ratio_thresh)

    def extract(
        self, frame: np.ndarray, bbox: Sequence[float], polygon: Optional[np.ndarray] = None
    ) -> Optional[JerseyFeature]:
        tb = self.torso_box(bbox, frame.shape)
        if tb is None:
            return None
        tx1, ty1, tx2, ty2 = tb
        crop = frame[ty1:ty2, tx1:tx2]

        # 1. Pitch-green HSV exclusion mask
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        valid = cv2.bitwise_not(cv2.inRange(hsv, self.lower_green, self.upper_green))

        # 2. Instance-segmentation silhouette intersection
        if polygon is not None and len(polygon) >= 3:
            seg = np.zeros(crop.shape[:2], np.uint8)
            poly = np.round(polygon - np.array([tx1, ty1], np.float32)).astype(np.int32)
            cv2.fillPoly(seg, [poly], 255)
            if cv2.countNonZero(seg) >= self.cfg.min_valid_pixels:
                valid = cv2.bitwise_and(valid, seg)

        # 3. 2-pixel morphological erosion prior to color sampling
        eroded = cv2.erode(valid, self.erode_kernel, iterations=self.cfg.erode_iterations)
        if cv2.countNonZero(eroded) >= self.cfg.min_valid_pixels:
            valid = eroded

        # Check neon referee anomaly on the valid torso mask
        is_neon = self.check_neon(hsv, valid)

        lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
        px = lab[valid > 0].astype(np.float32)
        if len(px) < self.cfg.min_valid_pixels:
            return None

        # 4. Mini-KMeans (k=2) on masked torso pixels to extract dominant cluster centroid
        if len(px) >= 40:
            _, labels, centers = cv2.kmeans(px, 2, None, self.km_criteria, 3, cv2.KMEANS_PP_CENTERS)
            counts = np.bincount(labels.ravel(), minlength=2)
            dom = centers[int(np.argmax(counts))]
            purity = float(counts.max() / counts.sum())
        else:
            dom = np.median(px, axis=0)
            purity = 1.0

        b = self.cfg.hist_bins
        hist, _, _ = np.histogram2d(px[:, 1], px[:, 2], bins=b, range=[[0, 256], [0, 256]])
        hist = (hist / max(hist.sum(), 1.0)).astype(np.float32).ravel()

        masked = None
        if self.keep_crop:
            masked = crop.copy()
            masked[valid == 0] = 0

        return JerseyFeature(dom, hist, int(len(px)), purity, int(bbox[3] - bbox[1]), is_neon, masked, tb)

    def to_vector(self, f: JerseyFeature) -> np.ndarray:
        # Down-weight lightness L* by 0.3 relative to chromatic channels (a*, b*)
        lab = cv_lab_to_true(f.lab_cv)
        lab[0] *= self.cfg.l_weight
        return np.concatenate([lab, np.sqrt(f.hist) * self.cfg.hist_weight]).astype(np.float32)


# --------------------------------------------------------------------------- #
# Stage 4 - Unsupervised Over-clustering Engine (K=5)
# --------------------------------------------------------------------------- #
class TeamClusterer:
    """
    Unsupervised Over-Clustering Engine (K = 5):
    - Fits KMeans / GMM with K = 5 over-clustering on warm-up features in PCA space.
    - Designates the two highest-volume clusters as Team A (group_id = 0) and Team B (group_id = 1).
    - Routes remaining lower-frequency clusters to Officials (group_id = 2) and Goalkeepers (group_id = 3).
    - Hard neon threshold overrides route neon anomalies directly to Officials before centroid distance calculation.
    """

    def __init__(self, cfg: PipelineConfig, n_iter: int = 3):
        self.cfg = cfg
        self.n_iter = n_iter
        self.pca: Optional[PCA] = None
        self.locs: List[np.ndarray] = []
        self.precs: List[np.ndarray] = []
        self.logdets: List[float] = []
        self.comp_group: List[int] = []
        self.threshold = np.inf
        self.team_lab_cv: List[np.ndarray] = []
        self.outlier_lab_cv: Dict[int, List[np.ndarray]] = {OFFICIALS_GROUP: [], GOALKEEPERS_GROUP: []}
        self.stats: Dict = {}

    @staticmethod
    def _robust_cov(pts: np.ndarray, df: int) -> Tuple[np.ndarray, np.ndarray]:
        if len(pts) >= max(5 * df, 20):
            mcd = MinCovDet(support_fraction=0.8, random_state=0).fit(pts)
            loc, cov = mcd.location_, mcd.covariance_
        elif len(pts) > df:
            loc, cov = pts.mean(0), np.cov(pts.T)
        else:
            loc, cov = pts.mean(0), np.eye(df)
        cov = np.atleast_2d(cov)
        cov = cov + np.eye(df) * (1e-3 * max(np.trace(cov) / df, 1e-6))
        return loc, cov

    def _set_components(self, comps: List[Tuple[np.ndarray, np.ndarray, int]]) -> None:
        self.locs, self.precs, self.logdets, self.comp_group = [], [], [], []
        for loc, cov, grp in comps:
            self.locs.append(loc)
            self.precs.append(np.linalg.inv(cov))
            self.logdets.append(float(np.linalg.slogdet(cov)[1]))
            self.comp_group.append(grp)

    def _d2(self, Z: np.ndarray) -> np.ndarray:
        out = []
        for loc, prec in zip(self.locs, self.precs):
            d = Z - loc
            out.append(np.einsum("ij,jk,ik->i", d, prec, d))
        return np.stack(out, axis=1)

    def _loglik(self, d2: np.ndarray) -> np.ndarray:
        return -0.5 * (d2 + np.array(self.logdets)[None, :])

    def fit(self, X: np.ndarray, labs_cv: np.ndarray, neon_flags: np.ndarray) -> "TeamClusterer":
        cfg = self.cfg
        n = len(X)
        if n < 20:
            raise RuntimeError(f"Insufficient warmup jersey samples ({n}) to fit clustering model.")

        rng = np.random.default_rng(0)
        if n > cfg.warmup_max_samples:
            idx = rng.choice(n, cfg.warmup_max_samples, replace=False)
            X, labs_cv, neon_flags, n = X[idx], labs_cv[idx], neon_flags[idx], cfg.warmup_max_samples

        self.pca = PCA(n_components=min(cfg.pca_components, X.shape[1], n - 1), random_state=0).fit(X)
        Z = self.pca.transform(X)
        df = Z.shape[1]

        # 1. Unsupervised Over-clustering with K = 5
        K = cfg.warmup_k
        km = KMeans(n_clusters=K, n_init=10, random_state=0).fit(Z)
        labels = km.labels_
        counts = np.bincount(labels, minlength=K)

        # 2. Designate two highest-volume clusters as Team A and Team B
        freq_sorted = np.argsort(-counts)
        team_a_cluster = freq_sorted[0]
        team_b_cluster = freq_sorted[1]
        outlier_clusters = freq_sorted[2:]

        # Map K=5 clusters to groups:
        # Cluster 0 -> Team A, Cluster 1 -> Team B
        # Remaining lower-frequency clusters -> Officials (2) / Goalkeepers (3)
        cluster_to_group = {team_a_cluster: TEAM_A_GROUP, team_b_cluster: TEAM_B_GROUP}
        for rank, c_idx in enumerate(outlier_clusters):
            # Split remaining clusters between Officials (2) and Goalkeepers (3)
            # If cluster has neon detections, prefer Officials
            is_cluster_neon = bool(np.mean(neon_flags[labels == c_idx]) >= 0.05) if (labels == c_idx).any() else False
            if is_cluster_neon or rank == 0:
                cluster_to_group[c_idx] = OFFICIALS_GROUP
            else:
                cluster_to_group[c_idx] = GOALKEEPERS_GROUP

        # 3. Fit robust tied-covariance components for each group
        initial_groups = np.array([cluster_to_group[l] for l in labels])
        comp_members = [labels == c for c in [team_a_cluster, team_b_cluster] + list(outlier_clusters)]
        comp_grps = [TEAM_A_GROUP, TEAM_B_GROUP] + [cluster_to_group[c] for c in outlier_clusters]

        for it in range(self.n_iter + 1):
            comps = []
            team_locs, team_covs, team_n = [], [], []
            for m in comp_members[:2]:
                loc, cov = self._robust_cov(Z[m], df)
                team_locs.append(loc)
                team_covs.append(cov)
                team_n.append(max(int(m.sum()), 1))

            pooled = np.average(team_covs, axis=0, weights=team_n)
            for ci, loc in enumerate(team_locs):
                comps.append((loc, pooled, comp_grps[ci]))

            for ci_offset, m in enumerate(comp_members[2:]):
                if m.sum() == 0:
                    continue
                loc = np.median(Z[m], axis=0)
                comps.append((loc, pooled, comp_grps[2 + ci_offset]))

            self._set_components(comps)
            if it == self.n_iter:
                break

            assign = self._loglik(self._d2(Z)).argmax(1)
            comp_members = [assign == i for i in range(len(comps))]

        d2 = self._d2(Z)
        assign = self._loglik(d2).argmax(1)
        grp = np.array([self.comp_group[a] for a in assign])

        # Mahalanobis threshold check: outliers far from team centroids
        d2_own = d2[np.arange(n), assign]
        team_d2 = d2_own[grp < OFFICIALS_GROUP]
        self.threshold = float(max(
            chi2.ppf(cfg.outlier_quantile, df),
            np.quantile(team_d2, cfg.outlier_quantile) if len(team_d2) else 0.0
        ))
        grp = np.where((grp < OFFICIALS_GROUP) & (d2_own > self.threshold), OFFICIALS_GROUP, grp)

        # Deterministic Team A / Team B ordering via CIELAB hue
        meds_cv = [np.median(labs_cv[grp == k], axis=0) if (grp == k).any() else np.array([128, 128, 128])
                   for k in (TEAM_A_GROUP, TEAM_B_GROUP)]
        hue = [np.arctan2(m[2] - 128.0, m[1] - 128.0) for m in meds_cv]
        order = [int(i) for i in np.argsort(hue)]
        remap = {old: new for new, old in enumerate(order)}
        self.comp_group = [remap.get(g, g) if g < OFFICIALS_GROUP else g for g in self.comp_group]
        grp = np.array([remap.get(int(g), int(g)) for g in grp])
        self.team_lab_cv = [meds_cv[i] for i in order]

        self.stats = {
            "warmup_samples": int(n),
            "pca_dims": int(df),
            "pca_explained_variance": round(float(self.pca.explained_variance_ratio_.sum()), 4),
            "overcluster_k": int(K),
            "warmup_k5_counts": [int(counts[c]) for c in freq_sorted],
            "mahalanobis_threshold": round(self.threshold, 2),
            "warmup_group_sizes": {GROUP_NAMES[k]: int((grp == k).sum()) for k in (0, 1, 2, 3)},
            "team_colors_bgr": [cv_lab_to_bgr(m) for m in self.team_lab_cv],
            "team_color_delta_e": round(delta_e(self.team_lab_cv[0], self.team_lab_cv[1]), 2),
        }
        return self

    def predict(self, X: np.ndarray, neon_flags: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        if len(X) == 0:
            return np.zeros(0, int), np.zeros(0)
        Z = self.pca.transform(X)
        d2 = self._d2(Z)
        ll = self._loglik(d2)
        best = ll.argmax(1)
        groups = np.array([self.comp_group[b] for b in best])
        d2_best = d2[np.arange(len(Z)), best]

        # Mahalanobis distance outlier check
        groups[(groups < OFFICIALS_GROUP) & (d2_best > self.threshold)] = OFFICIALS_GROUP

        # Unsupervised neon threshold override:
        # Any detection tagged with neon HSV anomaly is routed directly to Officials (group_id = 2)
        if neon_flags is not None:
            groups[neon_flags] = OFFICIALS_GROUP

        p = np.exp(ll - ll.max(1, keepdims=True))
        conf = p.max(1) / p.sum(1)
        if neon_flags is not None:
            conf[neon_flags] = 1.0

        return groups.astype(int), conf


# --------------------------------------------------------------------------- #
# Stage 5 - Temporal Smoothing
# --------------------------------------------------------------------------- #
class TrackSmoother:
    def __init__(self, window: int = 30):
        self.votes: Dict[int, deque] = defaultdict(lambda: deque(maxlen=window))
        self.current: Dict[int, int] = {}

    def update(self, pid: int, group: Optional[int]) -> int:
        if group is not None:
            self.votes[pid].append(int(group))
        v = self.votes[pid]
        if not v:
            return UNASSIGNED
        counts = Counter(v)
        top = max(counts.values())
        cands = [g for g, c in counts.items() if c == top]
        if len(cands) == 1:
            new = cands[0]
        else:
            prev = self.current.get(pid)
            new = prev if prev in cands else v[-1]
        self.current[pid] = new
        return new


# --------------------------------------------------------------------------- #
# Orchestrator & CLI Runner
# --------------------------------------------------------------------------- #
class MatchAnalysisPipeline:
    def __init__(self, cfg: Optional[PipelineConfig] = None):
        from ultralytics import YOLO

        self.cfg = cfg or PipelineConfig()
        self.device = resolve_device(self.cfg.device)
        self._YOLO = YOLO
        self.model = YOLO(self.cfg.model)
        self.extractor = JerseyColorExtractor(self.cfg, keep_crop=False)
        self.pitch = PitchFilter(self.cfg)
        self.clusterer = TeamClusterer(self.cfg)
        self.smoother = TrackSmoother(self.cfg.vote_window)
        self.palette: Dict[int, Tuple[int, int, int]] = dict(FIXED_PALETTE)
        self.is_fitted = False
        cv2.setRNGSeed(42)
        LOG.info("Initialized pipeline. Device: %s, Model: %s, Imgsz: %d",
                 self.device, self.cfg.model, self.cfg.imgsz)

    def _infer_kwargs(self, conf: float) -> dict:
        c = self.cfg
        kw = dict(classes=[0], conf=conf, iou=c.iou, imgsz=c.imgsz, device=self.device,
                  retina_masks=False, verbose=False)
        if c.half:
            kw["half"] = True
        return kw

    def _vectors(self, feats: List[JerseyFeature]) -> np.ndarray:
        if not feats:
            return np.zeros((0, 1), np.float32)
        return np.stack([self.extractor.to_vector(f) for f in feats])

    def _open(self, path: str) -> Tuple[cv2.VideoCapture, float, int, int, int, int]:
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {path}")
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        start = int(round(self.cfg.start_sec * fps))
        if start > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        n = total - start
        if self.cfg.duration_sec > 0:
            n = min(n, int(round(self.cfg.duration_sec * fps)))
        return cap, fps, W, H, start, max(n, 0)

    def warmup_and_fit(self, video_path: str) -> Dict:
        c = self.cfg
        cap, fps, W, H, _, n_frames = self._open(video_path)
        det_model = self._YOLO(c.model)
        min_h = c.warmup_min_box_frac * H
        feats: List[JerseyFeature] = []
        limit = min(n_frames, c.warmup_frames)
        hard_limit = min(n_frames, c.warmup_frames * 4)
        idx = 0
        t0 = time.time()

        while idx < hard_limit:
            ok, frame = cap.read()
            if not ok:
                break
            if idx >= limit and len(feats) >= 200:
                break
            if idx % c.warmup_stride == 0:
                res = det_model.predict(frame, **self._infer_kwargs(c.warmup_conf))[0]
                if res.boxes is not None and len(res.boxes):
                    if c.pitch_filter:
                        self.pitch.update(frame)
                    boxes = res.boxes.xyxy.cpu().numpy()
                    polys = res.masks.xy if res.masks is not None else [None] * len(boxes)
                    med_h = float(np.median(boxes[:, 3] - boxes[:, 1]))
                    thr_h = max(min_h, c.warmup_rel_box_h * med_h)
                    for bbox, poly in zip(boxes, polys):
                        if bbox[3] - bbox[1] < thr_h:
                            continue
                        if c.pitch_filter and not self.pitch.on_pitch(bbox):
                            continue
                        f = self.extractor.extract(frame, bbox, poly)
                        if f is None or f.purity < c.warmup_min_purity or f.n_valid < c.min_valid_pixels:
                            continue
                        feats.append(f)
            idx += 1
        cap.release()
        del det_model
        LOG.info("Warmup complete: %d frames, %d samples (%.1fs)", idx, len(feats), time.time() - t0)

        X = self._vectors(feats)
        labs = np.stack([f.lab_cv for f in feats]) if feats else np.zeros((0, 3))
        neons = np.array([f.is_neon for f in feats], dtype=bool) if feats else np.zeros(0, bool)
        self.clusterer.fit(X, labs, neons)
        self.is_fitted = True
        self._build_palette()
        st = dict(self.clusterer.stats, warmup_frames_scanned=idx)
        LOG.info("Team model stats: %s", json.dumps(st))
        return st

    def _build_palette(self) -> None:
        self.palette = dict(FIXED_PALETTE)
        if self.cfg.box_colors == "jersey":
            for k, lab in enumerate(self.clusterer.team_lab_cv):
                l_adj = lab.copy()
                l_adj[0] = max(l_adj[0], 90)
                self.palette[k] = cv_lab_to_bgr(l_adj)

    def _draw(self, frame: np.ndarray, bbox: Sequence[float], pid: int, group: int,
              feat: Optional[JerseyFeature]) -> None:
        H, W = frame.shape[:2]
        x1, y1, x2, y2 = map(int, bbox)
        x1, y1 = max(0, min(W, x1)), max(0, min(H, y1))
        x2, y2 = max(0, min(W, x2)), max(0, min(H, y2))

        color = self.palette.get(group, FIXED_PALETTE[-1])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        if self.cfg.debug_torso and feat is not None:
            tx1, ty1, tx2, ty2 = feat.torso
            cv2.rectangle(frame, (tx1, ty1), (tx2, ty2), (255, 255, 255), 1)

        label = f"#{pid} {GROUP_NAMES.get(group, 'Unknown')}"
        font, scale, th = cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1
        (tw, tht), base = cv2.getTextSize(label, font, scale, th)
        ly = y1 - 4 if y1 - tht - 8 > 0 else y2 + tht + 6
        cv2.rectangle(frame, (x1 - 1, ly - tht - 4), (x1 + tw + 4, ly + base - 1), color, -1)
        lum = 0.114 * color[0] + 0.587 * color[1] + 0.299 * color[2]
        txt = (0, 0, 0) if lum > 140 else (255, 255, 255)
        cv2.putText(frame, label, (x1 + 2, ly - 2), font, scale, txt, th, cv2.LINE_AA)

    def _hud(self, frame: np.ndarray, frame_id: int, counts: Counter) -> None:
        overlay = frame.copy()
        cv2.rectangle(overlay, (12, 12), (360, 140), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)
        cv2.putText(frame, f"Frame {frame_id} | Unsupervised Tracking", (22, 34),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        for i, g in enumerate([0, 1, 2, 3]):
            y = 56 + i * 22
            cv2.rectangle(frame, (22, y - 12), (40, y + 2), self.palette[g], -1)
            cv2.putText(frame, f"{GROUP_NAMES[g]}: {counts.get(g, 0)}", (50, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)

    def run(self, input_video: str, output_video: str = "output_match.mp4",
            output_csv: Optional[str] = "detections_log.csv", output_json: Optional[str] = None,
            summary_path: Optional[str] = None) -> Dict:
        c = self.cfg
        warm = self.warmup_and_fit(input_video) if not self.is_fitted else self.clusterer.stats

        cap, fps, W, H, start, n_frames = self._open(input_video)
        Path(output_video).parent.mkdir(parents=True, exist_ok=True)
        ffmpeg = find_ffmpeg() if c.h264 else None
        raw_path = str(Path(output_video).with_suffix(".raw.mp4")) if ffmpeg else output_video
        out = cv2.VideoWriter(raw_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))

        records: List[dict] = []
        raw_hist: Dict[int, List[int]] = defaultdict(list)
        smooth_hist: Dict[int, List[int]] = defaultdict(list)
        try:
            from tqdm import tqdm
            pbar = tqdm(total=n_frames, desc="Processing Match", unit="f")
        except Exception:
            pbar = None

        t0 = time.time()
        frame_id = 0
        while frame_id < n_frames:
            ok, frame = cap.read()
            if not ok:
                break
            res = self.model.track(frame, persist=True, tracker=c.tracker, **self._infer_kwargs(c.conf))[0]
            if c.pitch_filter:
                self.pitch.update(frame)

            dets = []
            if res.boxes is not None and res.boxes.id is not None and len(res.boxes):
                boxes = res.boxes.xyxy.cpu().numpy()
                ids = res.boxes.id.cpu().numpy().astype(int)
                confs = res.boxes.conf.cpu().numpy()
                polys = res.masks.xy if res.masks is not None else [None] * len(boxes)
                feats, fidx = [], []
                for bbox, pid, cf, poly in zip(boxes, ids, confs, polys):
                    if c.pitch_filter and not self.pitch.on_pitch(bbox):
                        continue
                    f = self.extractor.extract(frame, bbox, poly)
                    # Clamped bounding boxes
                    bx1, by1, bx2, by2 = bbox
                    clamped_box = [
                        max(0.0, min(float(W), float(bx1))),
                        max(0.0, min(float(H), float(by1))),
                        max(0.0, min(float(W), float(bx2))),
                        max(0.0, min(float(H), float(by2))),
                    ]
                    dets.append([clamped_box, int(pid), float(cf), f, None, None])
                    if f is not None:
                        feats.append(f)
                        fidx.append(len(dets) - 1)

                if feats:
                    neons = np.array([f.is_neon for f in feats], dtype=bool)
                    groups, gconf = self.clusterer.predict(self._vectors(feats), neons)
                    for j, g, gc in zip(fidx, groups, gconf):
                        dets[j][4], dets[j][5] = int(g), float(gc)

            counts: Counter = Counter()
            for bbox, pid, cf, f, raw_g, raw_conf in dets:
                grp = self.smoother.update(pid, raw_g)
                if raw_g is not None:
                    raw_hist[pid].append(raw_g)
                smooth_hist[pid].append(grp)
                counts[grp] += 1
                self._draw(frame, bbox, pid, grp, f)
                x1, y1, x2, y2 = (round(float(v), 1) for v in bbox)

                # Ensure fully populated records with no null values
                records.append({
                    "frame_id": int(frame_id),
                    "player_id": int(pid),
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "group_id": int(grp),
                    "group_name": GROUP_NAMES.get(grp, "Unknown"),
                    "raw_group_id": int(-1 if raw_g is None else raw_g),
                    "group_conf": round(float(0.0 if raw_conf is None else raw_conf), 3),
                    "det_conf": round(float(cf), 3),
                    "timestamp_s": round((start + frame_id) / fps, 3),
                })

            self._hud(frame, frame_id, counts)
            out.write(frame)
            frame_id += 1
            if pbar:
                pbar.update(1)

        if pbar:
            pbar.close()
        cap.release()
        out.release()
        elapsed = time.time() - t0

        if raw_path != output_video:
            cmd = [ffmpeg, "-y", "-loglevel", "error", "-i", raw_path, "-c:v", "libx264",
                   "-pix_fmt", "yuv420p", "-crf", "20", "-preset", "medium", output_video]
            if subprocess.run(cmd).returncode == 0:
                Path(raw_path).unlink(missing_ok=True)
            else:
                shutil.move(raw_path, output_video)

        cols = ["frame_id", "player_id", "x1", "y1", "x2", "y2", "group_id",
                "group_name", "raw_group_id", "group_conf", "det_conf", "timestamp_s"]
        df = pd.DataFrame.from_records(records) if records else pd.DataFrame(columns=cols)
        if output_csv:
            df.to_csv(output_csv, index=False)
        if output_json:
            df.to_json(output_json, orient="records", indent=1)

        def flips(h: Dict[int, List[int]]) -> int:
            return int(sum(sum(a != b for a, b in zip(s, s[1:])) for s in h.values()))

        raw_flips = flips(raw_hist)
        smooth_flips = flips(smooth_hist)
        total_transitions = max(1, sum(max(0, len(s) - 1) for s in smooth_hist.values()))
        flip_rate = round(float(smooth_flips / total_transitions) * 100.0, 2)

        summary = {
            "input": str(input_video),
            "frames_processed": frame_id,
            "fps_video": fps,
            "resolution": [W, H],
            "processing_fps": round(frame_id / max(elapsed, 1e-6), 2),
            "unique_tracks": int(df["player_id"].nunique()) if len(df) else 0,
            "detections": int(len(df)),
            "group_counts": {GROUP_NAMES[int(k)]: int(v) for k, v in df["group_id"].value_counts().items()} if len(df) else {},
            "label_flips_raw": raw_flips,
            "label_flips_smoothed": smooth_flips,
            "label_flip_rate_pct": flip_rate,
            "team_model": warm,
            "config": asdict(c),
        }
        if summary_path:
            Path(summary_path).write_text(json.dumps(summary, indent=2, default=str))

        LOG.info("Run finished: %d frames in %.1fs | raw flips=%d, smoothed flips=%d (flip rate=%.2f%%)",
                 frame_id, elapsed, raw_flips, smooth_flips, flip_rate)
        return summary


# --------------------------------------------------------------------------- #
# CLI Parser
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    d = PipelineConfig()
    p = argparse.ArgumentParser(description="Unsupervised team identification for broadcast footage.",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--input", required=True, help="Input video path")
    p.add_argument("--output_video", default="output_match.mp4")
    p.add_argument("--output_csv", default="detections_log.csv")
    p.add_argument("--output_json", default="detections_log.json")
    p.add_argument("--summary", default="run_summary.json")
    p.add_argument("--model", default=d.model)
    p.add_argument("--imgsz", type=int, default=d.imgsz)
    p.add_argument("--conf", type=float, default=d.conf)
    p.add_argument("--device", default=d.device, help="auto | cpu | mps | cuda:0")
    p.add_argument("--half", action="store_true")
    p.add_argument("--tracker", default=d.tracker)
    p.add_argument("--start_sec", type=float, default=d.start_sec)
    p.add_argument("--duration_sec", type=float, default=d.duration_sec)
    p.add_argument("--warmup_frames", type=int, default=d.warmup_frames)
    p.add_argument("--warmup_stride", type=int, default=d.warmup_stride)
    p.add_argument("--vote_window", type=int, default=d.vote_window)
    p.add_argument("--l_weight", type=float, default=d.l_weight, help="Down-weight lightness L*")
    p.add_argument("--torso", type=float, nargs=2, default=[d.torso_top, d.torso_bottom],
                   metavar=("TOP", "BOTTOM"))
    p.add_argument("--box_colors", choices=["auto", "jersey", "fixed"], default=d.box_colors)
    p.add_argument("--no_pitch_filter", action="store_true")
    p.add_argument("--no_h264", action="store_true")
    p.add_argument("--debug_torso", action="store_true")
    p.add_argument("-v", "--verbose", action="store_true")
    return p


def main(argv: Optional[Sequence[str]] = None) -> Dict:
    a = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if a.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    tracker = a.tracker
    if not Path(tracker).exists():
        local = Path(__file__).resolve().parent / tracker
        tracker = str(local) if local.exists() else "bytetrack.yaml"

    cfg = PipelineConfig(
        model=a.model, imgsz=a.imgsz, conf=a.conf, device=a.device, half=a.half, tracker=tracker,
        start_sec=a.start_sec, duration_sec=a.duration_sec, warmup_frames=a.warmup_frames,
        warmup_stride=a.warmup_stride, vote_window=a.vote_window, l_weight=a.l_weight,
        torso_top=a.torso[0], torso_bottom=a.torso[1], box_colors=a.box_colors,
        pitch_filter=not a.no_pitch_filter, h264=not a.no_h264, debug_torso=a.debug_torso,
    )
    pipe = MatchAnalysisPipeline(cfg)
    summary = pipe.run(a.input, a.output_video, a.output_csv, a.output_json, a.summary)
    print(json.dumps({
        "frames_processed": summary["frames_processed"],
        "unique_tracks": summary["unique_tracks"],
        "group_counts": summary["group_counts"],
        "label_flips_raw": summary["label_flips_raw"],
        "label_flips_smoothed": summary["label_flips_smoothed"],
        "label_flip_rate_pct": summary["label_flip_rate_pct"],
        "team_colors_bgr": summary["team_model"]["team_colors_bgr"],
        "team_color_delta_e": summary["team_model"]["team_color_delta_e"],
    }, indent=2))
    return summary


if __name__ == "__main__":
    main()
