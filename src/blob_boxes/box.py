"""The rotated rectangle and the pixels it covers.

Coordinates: pixel (row r, column c) has its centre at x = c, y = r, and y grows downwards. A rectangle covers
a pixel when the pixel's centre lies inside it (edges included). `angle` is the direction of the `width` side,
in degrees from +x towards +y, which is clockwise on screen.
"""

import math
from dataclasses import dataclass

import numpy as np

# Pixel centres exactly on an edge must stay covered whatever the floating-point rounding of the rotation.
_EDGE_TOLERANCE = 1e-9


@dataclass(frozen=True)
class RotatedBox:
    cx: float
    cy: float
    width: float
    height: float
    angle: float

    def normalized(self) -> "RotatedBox":
        """The same rectangle written the canonical way: width >= height and angle in [-90, 90).

        A square has the same pixels every 90 degrees, so its angle is brought into [-45, 45).
        """
        width, height, angle = abs(self.width), abs(self.height), self.angle
        if height > width:
            width, height, angle = height, width, angle + 90
        angle = (angle + 90) % 180 - 90
        if math.isclose(width, height):
            angle = (angle + 45) % 90 - 45
        return RotatedBox(self.cx, self.cy, width, height, angle)

    def to_frame(self, xs: np.ndarray, ys: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Coordinates of points in the rectangle's own frame: u along the width, v along the height."""
        t = math.radians(self.angle)
        dx, dy = xs - self.cx, ys - self.cy
        return dx * math.cos(t) + dy * math.sin(t), -dx * math.sin(t) + dy * math.cos(t)

    def covers(self, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
        """True for each point (pixel centre) inside the rectangle."""
        u, v = self.to_frame(xs, ys)
        return (np.abs(u) <= abs(self.width) / 2 + _EDGE_TOLERANCE) & (
            np.abs(v) <= abs(self.height) / 2 + _EDGE_TOLERANCE
        )

    def pixel_bounds(self) -> tuple[int, int, int, int]:
        """Smallest pixel range (x_min, y_min, x_max, y_max), inclusive, that holds every covered pixel."""
        u = abs(self.width) / 2
        v = abs(self.height) / 2
        t = math.radians(self.angle)
        half_x = abs(u * math.cos(t)) + abs(v * math.sin(t))
        half_y = abs(u * math.sin(t)) + abs(v * math.cos(t))
        return (
            math.floor(self.cx - half_x),
            math.floor(self.cy - half_y),
            math.ceil(self.cx + half_x),
            math.ceil(self.cy + half_y),
        )
