"""Generate the blob-boxes sample mask: separate shapes with known geometry, covering the hard cases.

Both files are committed; rerun only if SHAPES changes:

    uv run python scripts/make_masks.py
"""

from pathlib import Path

import cv2
import numpy as np

from blob_boxes.box import RotatedBox

OUT_DIR = Path(__file__).resolve().parent.parent / "samples" / "masks"
SIZE = 256

# Pixel (row r, column c) has its centre at x = c, y = r; y grows downwards. Angles are those of the long side,
# from +x towards +y (clockwise on screen). Every shape sits far enough from the others not to touch them.
RECTS = [
    # (centre x, centre y, long side, short side, angle in degrees)
    (60, 50, 60, 20, 30),     # plain rotated rectangle
    (185, 55, 44, 12, -60),   # rotated the other way
    (200, 205, 24, 24, 45),   # square at 45°: the angle is ambiguous every 90°
    (6, 215, 34, 14, 20),     # touches the left border, so part of it is cut off
]
ELLIPSE = (70, 150, 25, 10, 15)          # centre x, centre y, semi-axes a and b, angle
LINE = ((150, 120), (205, 165))          # 1-pixel-wide diagonal line, (x, y) end points
SINGLE_PIXEL = (240, 12)                 # (x, y)
IRREGULAR = [(120, 215, 40, 10, 0), (140, 215, 8)]  # a bar plus a disc at its right end


def centres() -> tuple[np.ndarray, np.ndarray]:
    ys, xs = np.mgrid[0:SIZE, 0:SIZE]
    return xs, ys


def rotated_rect(cx: float, cy: float, long_side: float, short_side: float, angle: float) -> np.ndarray:
    """Pixels whose centre lies inside the rotated rectangle.

    Uses blob-boxes' own coverage rule, so a rectangle drawn here has IoU exactly 1 with the same box: with a
    separate rule, centres lying on an edge fell on either side depending on floating-point rounding.
    """
    return RotatedBox(cx, cy, long_side, short_side, angle).covers(*centres())


def rotated_ellipse(cx: float, cy: float, a: float, b: float, angle: float) -> np.ndarray:
    xs, ys = centres()
    t = np.deg2rad(angle)
    u = (xs - cx) * np.cos(t) + (ys - cy) * np.sin(t)
    v = -(xs - cx) * np.sin(t) + (ys - cy) * np.cos(t)
    return (u / a) ** 2 + (v / b) ** 2 <= 1


def disc(cx: float, cy: float, r: float) -> np.ndarray:
    xs, ys = centres()
    return (xs - cx) ** 2 + (ys - cy) ** 2 <= r**2


def build_mask() -> np.ndarray:
    mask = np.zeros((SIZE, SIZE), dtype=bool)
    for rect in RECTS:
        mask |= rotated_rect(*rect)
    mask |= rotated_ellipse(*ELLIPSE)
    line = np.zeros((SIZE, SIZE), dtype=np.uint8)
    cv2.line(line, LINE[0], LINE[1], color=1, thickness=1, lineType=cv2.LINE_8)
    mask |= line.astype(bool)
    x, y = SINGLE_PIXEL
    mask[y, x] = True
    mask |= rotated_rect(*IRREGULAR[0]) | disc(*IRREGULAR[1])
    return mask


def main() -> None:
    mask = build_mask()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.save(OUT_DIR / "example_mask.npy", mask.astype(np.uint8))           # values 0/1
    cv2.imwrite(str(OUT_DIR / "example_mask.png"), mask.astype(np.uint8) * 255)  # black/white, to look at
    print(f"wrote {OUT_DIR / 'example_mask.npy'} and {OUT_DIR / 'example_mask.png'}")


if __name__ == "__main__":
    main()
