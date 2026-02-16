"""Depth reprojection: camera-perspective depth → projector-perspective depth.

Uses the Gray code correspondence maps (map_x, map_y) to warp camera-space
depth into projector pixel space via cv2.remap — a dense gather operation
with bilinear interpolation.
"""

from __future__ import annotations

import cv2
import numpy as np


class DepthReprojector:
    """Reprojects camera-view depth to projector-view depth.

    Uses pre-computed Gray code correspondence maps for a dense, clean warp.
    The maps are at projector resolution and map each projector pixel to the
    camera pixel that the projector illuminates — so cv2.remap gives us a
    projector-perspective view of any camera-space image.
    """

    def __init__(
        self,
        map_x: np.ndarray,
        map_y: np.ndarray,
        proj_w: int,
        proj_h: int,
    ):
        self.proj_w = proj_w
        self.proj_h = proj_h

        # Store maps as float32 for cv2.remap
        self._map_x = map_x.astype(np.float32)
        self._map_y = map_y.astype(np.float32)

    def reproject(self, depth_cam: np.ndarray) -> np.ndarray:
        """Convert camera-perspective depth to projector-perspective depth.

        Args:
            depth_cam: (cam_h, cam_w) float32 [0,1] from Depth Anything V2

        Returns:
            (proj_h, proj_w) float32 [0,1] depth from projector's viewpoint
        """
        # Warp camera-space depth to projector space using correspondence maps
        proj_depth = cv2.remap(
            depth_cam,
            self._map_x,
            self._map_y,
            cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )

        return proj_depth
