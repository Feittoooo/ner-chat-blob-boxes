"""Find, for each blob, the rotated rectangle with the highest pixel IoU.

1. Starting rectangles: OpenCV's minimum-area rectangle (raw and grown by one pixel) and two built from
   the blob's principal axes.
2. From each one, Nelder-Mead maximizes `soft_iou` over (cx, cy, width, height, angle).
3. A pattern search then polishes every start and every result on the exact `discrete_iou`.
4. Restarts: the winner is nudged at random and polished again, keeping any improvement.

Every rectangle tried along the way is a candidate, and the one with the highest exact IoU wins (ties go to the
earliest), so the answer is never worse than OpenCV's rectangle (up to the rounding below). The restarts use a fixed seed per blob, so
the output is the same on every run and does not depend on the other blobs.

The search tends to end with an edge exactly on a row of pixel centres, where rounding the output to
DECIMALS can drop that row. So the reported box is the rounded one, and its IoU is the IoU of the CSV values.
"""

import math
from dataclasses import dataclass

import cv2
import numpy as np
from scipy.optimize import minimize

from blob_boxes.blobs import Blob, find_blobs
from blob_boxes.box import DECIMALS, RotatedBox
from blob_boxes.iou import discrete_iou, soft_iou

# Nelder-Mead's first moves, per parameter (cx, cy, width, height in pixels; angle in degrees).
_SIMPLEX_STEPS = (1.0, 1.0, 1.0, 1.0, 5.0)
# Pattern search: base step per parameter, then shrunk by each factor in turn.
_POLISH_STEPS = (1.0, 1.0, 1.0, 1.0, 2.0)
_POLISH_SCALES = (1.0, 0.5, 0.25, 0.1, 0.05)
# Restarts: how many, and the spread of the random nudge per parameter. On the 2048x2048 sample, 10/20/30/40
# restarts give a mean IoU of 0.947/0.948/0.950/0.951 for 8/14/20/27 extra seconds (see DECISIONS.md).
_RESTARTS = 30
_RESTART_SPREAD = (1.0, 1.0, 1.5, 1.5, 8.0)
_SEED = 0
# Growth tried before rounding, to keep edge pixels covered: well above the rounding error (half of
# 10**-DECIMALS), well below a pixel.
_ROUNDING_MARGIN = 10 * 10**-DECIMALS
# The spread-based start of a 1-pixel blob would be 0 x 0; one pixel across is its honest size.
_MIN_SIDE = 1.0


@dataclass(frozen=True)
class BlobBox:
    blob_id: int
    box: RotatedBox  # normalized (width >= height, angle in [-90, 90)) and rounded to DECIMALS
    iou: float
    area: int  # blob pixels


def fit_boxes(mask: np.ndarray, connectivity: int = 8) -> list[BlobBox]:
    """The best rotated rectangle for every blob of a 2-D mask (non-zero = blob), in blob id order."""
    mask = np.asarray(mask)
    blobs = find_blobs(mask, connectivity)
    return [fit_blob(blob, mask.shape) for blob in blobs]


def fit_blob(blob: Blob, image_shape: tuple[int, int]) -> BlobBox:
    starts = initial_boxes(blob)
    candidates = list(starts)
    best = _best(candidates, blob, image_shape)

    if discrete_iou(best, blob, image_shape) < 1.0:  # a perfect start (e.g. a 1-pixel blob) needs no search
        candidates += [_optimize_soft(start, blob, image_shape) for start in starts]
        # Small blobs have many local optima, so every candidate is polished, not only the best one.
        candidates = [_polish(c, blob, image_shape) for c in candidates]
        best = _restart(_best(candidates, blob, image_shape), blob, image_shape)

    best = best.normalized()
    grown = RotatedBox(best.cx, best.cy, best.width + _ROUNDING_MARGIN, best.height + _ROUNDING_MARGIN, best.angle)
    box = _best([best.rounded(), grown.rounded()], blob, image_shape)
    return BlobBox(blob.id, box, discrete_iou(box, blob, image_shape), blob.area)


def initial_boxes(blob: Blob) -> list[RotatedBox]:
    """Starting rectangles, in tie-break order: OpenCV's first."""
    rows, cols = np.nonzero(blob.pixels)
    points = np.column_stack([cols + blob.x0, rows + blob.y0]).astype(np.float32)  # pixel centres (x, y)

    # minAreaRect hugs the pixel centres, so its sides are one pixel short of the pixels' own squares: a
    # 1-pixel-wide line comes back 0 wide. With 1 added to each side it covers the same centres with honest
    # sizes; the raw rectangle is kept too, because on a staircase line its thinner shape can score higher.
    (cx, cy), (w, h), angle = cv2.minAreaRect(points)
    if w < h:  # OpenCV 5.0 returns angle in [-90, 0) and swaps sides instead (verified, see DECISIONS.md)
        w, h, angle = h, w, angle + 90
    cx, cy, w, h, angle = float(cx), float(cy), float(w), float(h), float(angle)
    starts = [RotatedBox(cx, cy, w + 1, h + 1, angle), RotatedBox(cx, cy, w, h, angle)]

    # Principal axes: the direction in which the blob is longest, from the covariance of its pixel centres.
    centre = points.mean(axis=0, dtype=np.float64)
    offsets = points - centre
    cov = offsets.T @ offsets / len(points)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)  # ascending: the last one is the long axis
    major = eigenvectors[:, 1]
    angle = math.degrees(math.atan2(major[1], major[0]))
    probe = RotatedBox(float(centre[0]), float(centre[1]), 0.0, 0.0, angle)
    u, v = probe.to_frame(points[:, 0].astype(np.float64), points[:, 1].astype(np.float64))

    # (a) The tightest rectangle along those axes, like minAreaRect but on a different orientation.
    mid_u, mid_v = (u.max() + u.min()) / 2, (v.max() + v.min()) / 2
    t = math.radians(angle)
    starts.append(
        RotatedBox(
            float(centre[0] + mid_u * math.cos(t) - mid_v * math.sin(t)),
            float(centre[1] + mid_u * math.sin(t) + mid_v * math.cos(t)),
            float(u.max() - u.min()) + 1,
            float(v.max() - v.min()) + 1,
            angle,
        )
    )
    # (b) The rectangle with the same spread: a filled w x h rectangle has variance w^2 / 12 along its width.
    # It cuts through ragged ends (ellipses, noisy outlines) instead of stretching to reach them.
    starts.append(
        RotatedBox(
            float(centre[0]),
            float(centre[1]),
            max(_MIN_SIDE, math.sqrt(12 * max(eigenvalues[1], 0.0))),
            max(_MIN_SIDE, math.sqrt(12 * max(eigenvalues[0], 0.0))),
            angle,
        )
    )
    return starts


def _box(params: np.ndarray) -> RotatedBox:
    cx, cy, w, h, angle = (float(p) for p in params)
    return RotatedBox(cx, cy, abs(w), abs(h), angle)


def _params(box: RotatedBox) -> np.ndarray:
    return np.array([box.cx, box.cy, box.width, box.height, box.angle])


def _best(candidates: list[RotatedBox], blob: Blob, image_shape: tuple[int, int]) -> RotatedBox:
    scores = [discrete_iou(c, blob, image_shape) for c in candidates]
    return candidates[int(np.argmax(scores))]  # argmax returns the first of equal scores


def _optimize_soft(start: RotatedBox, blob: Blob, image_shape: tuple[int, int]) -> RotatedBox:
    x0 = _params(start)
    simplex = np.vstack([x0] + [x0 + step * np.eye(5)[i] for i, step in enumerate(_SIMPLEX_STEPS)])
    result = minimize(
        lambda p: -soft_iou(_box(p), blob, image_shape),
        x0,
        method="Nelder-Mead",
        options={"initial_simplex": simplex, "xatol": 1e-3, "fatol": 1e-6, "maxiter": 2000},
    )
    return _box(result.x)


def _polish(box: RotatedBox, blob: Blob, image_shape: tuple[int, int]) -> RotatedBox:
    """Coordinate search on the exact IoU: try +/- a step on each parameter, keep strict improvements only."""
    params = _params(box)
    best = discrete_iou(_box(params), blob, image_shape)
    for scale in _POLISH_SCALES:
        improved = True
        while improved:
            improved = False
            for i, base in enumerate(_POLISH_STEPS):
                for sign in (1, -1):
                    trial = params.copy()
                    trial[i] += sign * base * scale
                    score = discrete_iou(_box(trial), blob, image_shape)
                    if score > best:
                        params, best, improved = trial, score, True
    return _box(params)


def _restart(box: RotatedBox, blob: Blob, image_shape: tuple[int, int]) -> RotatedBox:
    """Escape local optima: polish random nudges of the best rectangle, keep strict improvements only."""
    rng = np.random.default_rng(_SEED)
    best = discrete_iou(box, blob, image_shape)
    for _ in range(_RESTARTS):
        if best == 1.0:
            break
        trial = _polish(_box(_params(box) + rng.normal(0.0, _RESTART_SPREAD)), blob, image_shape)
        score = discrete_iou(trial, blob, image_shape)
        if score > best:
            box, best = trial, score
    return box
