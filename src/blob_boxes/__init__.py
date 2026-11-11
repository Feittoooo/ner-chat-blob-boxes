"""blob-boxes: for each blob in a binary mask, the rotated rectangle with the highest IoU."""

from blob_boxes.box import RotatedBox
from blob_boxes.fit import BlobBox, fit_boxes

__all__ = ["BlobBox", "RotatedBox", "fit_boxes"]
