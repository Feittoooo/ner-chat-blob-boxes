"""Files in and out: read a mask into the boolean array the algorithm works on, write the boxes as CSV."""

import csv
from pathlib import Path

import cv2
import numpy as np

from blob_boxes.box import DECIMALS, RotatedBox

# Image pixels brighter than mid-grey are blob pixels. A plain "non-zero" rule would turn the near-black noise
# of lossy formats (JPEG) or anti-aliased edges into extra blobs.
IMAGE_THRESHOLD = 127


class MaskError(Exception):
    """The mask file is missing, unreadable or not 2-D; the message is meant for the CLI user."""


def load_mask(path: str | Path) -> np.ndarray:
    """Load a `.npy` array (non-zero = blob) or an image (brighter than mid-grey = blob) as a 2-D bool array."""
    path = Path(path)
    if not path.is_file():
        raise MaskError(f"{path} does not exist")

    if path.suffix.lower() == ".npy":
        try:
            mask = np.load(path, allow_pickle=False) != 0
        except ValueError as exc:
            raise MaskError(f"{path} is not a plain NumPy array: {exc}") from exc
    else:
        # cv2.imread can't open non-ASCII paths on Windows, and it returns None instead of raising.
        image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise MaskError(f"{path} is not an image OpenCV can read")
        mask = image > IMAGE_THRESHOLD

    if mask.ndim != 2:
        raise MaskError(f"{path} must be a 2-D mask, got shape {mask.shape}")
    return mask


CSV_COLUMNS = ("blob_id", "center_x", "center_y", "width", "height", "angle")


def box_row(blob_id: int, box: RotatedBox) -> list[str]:
    """One CSV row with DECIMALS decimals; `fit_boxes` already rounded the box, so nothing changes here."""
    box = box.rounded()
    values = [box.cx, box.cy, box.width, box.height, box.angle]
    return [str(blob_id)] + [f"{v:.{DECIMALS}f}" for v in values]


def write_csv(rows: list[tuple[int, RotatedBox]], path: str | Path) -> None:
    """Write (blob_id, box) pairs with the CSV_COLUMNS header. Unix line endings on every OS."""
    with Path(path).open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(CSV_COLUMNS)
        writer.writerows(box_row(blob_id, box) for blob_id, box in rows)
