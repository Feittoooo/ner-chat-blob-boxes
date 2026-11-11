"""Split a binary mask into blobs (connected components), numbered in reading order."""

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True, eq=False)
class Blob:
    """One connected component, kept as a crop of its bounding box.

    `pixels` is True only where this blob is: another blob reaching into the same box is left out.
    Pixel (r, c) of the crop is pixel (y0 + r, x0 + c) of the full mask.
    """

    id: int  # 1, 2, ... in the order the blobs are met reading the mask row by row
    x0: int  # column of the crop's left edge in the full mask
    y0: int  # row of the crop's top edge in the full mask
    pixels: np.ndarray

    @property
    def area(self) -> int:
        return int(self.pixels.sum())


def find_blobs(mask: np.ndarray, connectivity: int = 8) -> list[Blob]:
    """Connected components of `mask` (any 2-D array; non-zero = blob), ordered by id."""
    mask = np.asarray(mask)
    if mask.ndim != 2:
        raise ValueError(f"mask must be 2-D, got shape {mask.shape}")
    if connectivity not in (4, 8):
        raise ValueError(f"connectivity must be 4 or 8, got {connectivity}")

    # SAUF numbers components in reading order. OpenCV's default algorithm does not guarantee that order, and on
    # the 2048x2048 sample processed in parallel it does break it, so blob ids would not be stable.
    n, labels, stats, _ = cv2.connectedComponentsWithStatsWithAlgorithm(
        (mask != 0).astype(np.uint8), connectivity, cv2.CV_32S, cv2.CCL_SAUF
    )

    blobs = []
    for label in range(1, n):  # label 0 is the background
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        w = int(stats[label, cv2.CC_STAT_WIDTH])
        h = int(stats[label, cv2.CC_STAT_HEIGHT])
        blobs.append(Blob(id=label, x0=x, y0=y, pixels=labels[y : y + h, x : x + w] == label))
    return blobs
