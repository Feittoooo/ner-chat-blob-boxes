"""Read a mask file into the boolean array the algorithm works on."""

from pathlib import Path

import cv2
import numpy as np

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
