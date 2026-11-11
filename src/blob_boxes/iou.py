"""How well a rotated rectangle matches a blob: intersection over union (IoU) of their pixels.

The rectangle is drawn on the image grid, so only the pixels it covers inside the image count: a part that
sticks out of the image has no pixels. Other blobs inside the rectangle count as background.
"""

import numpy as np

from blob_boxes.blobs import Blob
from blob_boxes.box import RotatedBox

Window = tuple[int, int, int, int]  # x_min, y_min, x_max, y_max: an inclusive pixel range of the image


def window_around(box: RotatedBox, blob: Blob, image_shape: tuple[int, int]) -> Window:
    """Every pixel the rectangle or the blob reaches, clipped to the image."""
    rows, cols = image_shape
    h, w = blob.pixels.shape
    bx_min, by_min, bx_max, by_max = box.pixel_bounds()
    return (
        max(0, min(bx_min, blob.x0)),
        max(0, min(by_min, blob.y0)),
        min(cols - 1, max(bx_max, blob.x0 + w - 1)),
        min(rows - 1, max(by_max, blob.y0 + h - 1)),
    )


def blob_in_window(blob: Blob, window: Window) -> np.ndarray:
    """The blob's pixels placed on the window's grid (the window must contain the blob's bounding box)."""
    x_min, y_min, x_max, y_max = window
    grid = np.zeros((y_max - y_min + 1, x_max - x_min + 1), dtype=bool)
    h, w = blob.pixels.shape
    top, left = blob.y0 - y_min, blob.x0 - x_min
    grid[top : top + h, left : left + w] = blob.pixels
    return grid


def pixel_centres(window: Window) -> tuple[np.ndarray, np.ndarray]:
    x_min, y_min, x_max, y_max = window
    ys, xs = np.mgrid[y_min : y_max + 1, x_min : x_max + 1]
    return xs, ys


def discrete_iou(box: RotatedBox, blob: Blob, image_shape: tuple[int, int]) -> float:
    """The exact pixel IoU: this is the quantity blob-boxes maximizes and reports."""
    window = window_around(box, blob, image_shape)
    covered = box.covers(*pixel_centres(window))
    intersection = int((covered & blob_in_window(blob, window)).sum())
    union = int(covered.sum()) + blob.area - intersection
    return intersection / union if union else 0.0


def soft_iou(box: RotatedBox, blob: Blob, image_shape: tuple[int, int]) -> float:
    """A continuous stand-in for `discrete_iou`, used only to guide the search.

    Each pixel counts by the share of its 1x1 square inside the rectangle, measured along the rectangle's two
    axes (exact when the rectangle is axis-aligned, a close estimate when rotated). A pixel whose centre is on
    an edge counts half, the same threshold `discrete_iou` uses, and a perfect fit scores 1. Unlike the exact
    IoU, it changes whenever an edge moves across a pixel, so the search always knows which way is better.
    """
    # Coverage reaches half a pixel past each edge, so look one pixel further than the rectangle itself.
    grown = RotatedBox(box.cx, box.cy, abs(box.width) + 1, abs(box.height) + 1, box.angle)
    window = window_around(grown, blob, image_shape)
    u, v = box.to_frame(*pixel_centres(window))
    coverage = np.clip(abs(box.width) / 2 + 0.5 - np.abs(u), 0, 1) * np.clip(
        abs(box.height) / 2 + 0.5 - np.abs(v), 0, 1
    )
    pixels = blob_in_window(blob, window)
    intersection = float(coverage[pixels].sum())
    union = float(coverage.sum()) + blob.area - intersection
    return intersection / union if union else 0.0
