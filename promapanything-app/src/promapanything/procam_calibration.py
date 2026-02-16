"""ProCam calibration engine — all projection-based, no physical checkerboard.

Pipeline:
  1. Project checkerboard at multiple positions across the projector
     → detect in camera → calibrate camera intrinsics (Zhang's method)
  2. Use camera calibration rvecs/tvecs to get 3D corner positions
     → calibrate projector intrinsics
  3. Gray code structured light → dense camera↔projector correspondence
  4. Stereo calibrate → R, T (camera→projector extrinsics)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import cv2
import numpy as np

from .calibration import CalibrationRunner
from .state import ProCamCalibration

logger = logging.getLogger(__name__)


def _generate_checkerboard_image(
    proj_w: int, proj_h: int,
    board_cols: int, board_rows: int,
    square_px: int,
    offset_x: int, offset_y: int,
    white_level: int = 255,
    black_level: int = 0,
) -> np.ndarray:
    """Generate a checkerboard image at projector resolution.

    Returns (proj_h, proj_w) uint8 with the checkerboard at the given offset.
    Board has (board_cols+1) x (board_rows+1) squares, surrounded by a white
    border so OpenCV can detect the board boundary against a dark background.
    """
    img = np.zeros((proj_h, proj_w), dtype=np.uint8)
    n_sq_x = board_cols + 1
    n_sq_y = board_rows + 1

    # Draw white border (1 square-width padding) around the checkerboard
    border = square_px
    bx0 = max(0, offset_x - border)
    by0 = max(0, offset_y - border)
    bx1 = min(proj_w, offset_x + n_sq_x * square_px + border)
    by1 = min(proj_h, offset_y + n_sq_y * square_px + border)
    img[by0:by1, bx0:bx1] = white_level

    # Draw black squares on top of the white border
    for sy in range(n_sq_y):
        for sx in range(n_sq_x):
            if (sx + sy) % 2 == 1:  # black squares
                x0 = offset_x + sx * square_px
                y0 = offset_y + sy * square_px
                x1 = x0 + square_px
                y1 = y0 + square_px
                x0c = max(0, min(x0, proj_w))
                y0c = max(0, min(y0, proj_h))
                x1c = max(0, min(x1, proj_w))
                y1c = max(0, min(y1, proj_h))
                if x1c > x0c and y1c > y0c:
                    img[y0c:y1c, x0c:x1c] = black_level
    return img


def generate_board_positions(
    proj_w: int, proj_h: int,
    board_cols: int, board_rows: int,
    square_px: int,
    n_positions: int = 7,
) -> list[tuple[int, int]]:
    """Generate checkerboard offset positions spread across the projector.

    Returns list of (offset_x, offset_y) in projector pixels.
    """
    board_w = (board_cols + 1) * square_px
    board_h = (board_rows + 1) * square_px
    border = square_px  # white border around board
    margin = border + square_px // 2  # border + extra breathing room

    max_x = proj_w - board_w - margin
    max_y = proj_h - board_h - margin
    min_x = margin
    min_y = margin

    if max_x < min_x:
        max_x = min_x
    if max_y < min_y:
        max_y = min_y

    cx = (min_x + max_x) // 2
    cy = (min_y + max_y) // 2

    # Spread positions: center, corners, and midpoints
    positions = [
        (cx, cy),                        # center
        (min_x, min_y),                  # top-left
        (max_x, min_y),                  # top-right
        (min_x, max_y),                  # bottom-left
        (max_x, max_y),                  # bottom-right
        (cx, min_y),                     # top-center
        (cx, max_y),                     # bottom-center
        (min_x, cy),                     # left-center
        (max_x, cy),                     # right-center
    ]
    return positions[:n_positions]


def get_projected_corner_positions(
    board_cols: int, board_rows: int,
    square_px: int,
    offset_x: int, offset_y: int,
) -> np.ndarray:
    """Get the projector-pixel coordinates of the inner corners.

    Returns (board_cols * board_rows, 1, 2) float32.
    """
    corners = np.zeros((board_rows * board_cols, 1, 2), dtype=np.float32)
    for r in range(board_rows):
        for c in range(board_cols):
            # Inner corners start at (1,1) square offset
            px = offset_x + (c + 1) * square_px
            py = offset_y + (r + 1) * square_px
            corners[r * board_cols + c, 0] = (px, py)
    return corners


class ProCamCalibrator:
    """Multi-step ProCam calibration engine using projected checkerboards."""

    def __init__(
        self,
        proj_w: int = 1920,
        proj_h: int = 1080,
        board_cols: int = 5,
        board_rows: int = 3,
        square_px: int = 80,
        white_level: int = 255,
        black_level: int = 0,
    ):
        self.proj_w = proj_w
        self.proj_h = proj_h
        self.board_cols = max(board_cols, 3)
        self.board_rows = max(board_rows, 3)
        self.square_px = square_px
        self.white_level = white_level
        self.black_level = black_level
        self.board_size = (board_cols, board_rows)

        # Object points template (arbitrary units, using mm-like spacing)
        # Physical size doesn't matter for intrinsics recovery
        self._obj_pts_template = np.zeros(
            (board_cols * board_rows, 3), dtype=np.float32
        )
        for r in range(board_rows):
            for c in range(board_cols):
                self._obj_pts_template[r * board_cols + c] = (
                    c * square_px, r * square_px, 0.0
                )

        # Board positions for sequential projection
        self._positions = generate_board_positions(
            proj_w, proj_h, board_cols, board_rows, square_px, n_positions=7
        )
        self._current_position_idx = 0

        # Collected calibration data
        self._obj_points: list[np.ndarray] = []
        self._cam_img_points: list[np.ndarray] = []
        self._proj_img_points: list[np.ndarray] = []
        self._cam_image_size: tuple[int, int] | None = None

    # ── Board pattern generation ────────────────────────────────────────

    @property
    def total_positions(self) -> int:
        return len(self._positions)

    @property
    def current_position_idx(self) -> int:
        return self._current_position_idx

    @property
    def capture_count(self) -> int:
        return len(self._obj_points)

    def get_current_board_image(self) -> np.ndarray | None:
        """Get the checkerboard image for the current position.

        Returns (proj_h, proj_w) uint8, or None if all positions done.
        """
        if self._current_position_idx >= len(self._positions):
            return None
        ox, oy = self._positions[self._current_position_idx]
        return _generate_checkerboard_image(
            self.proj_w, self.proj_h,
            self.board_cols, self.board_rows,
            self.square_px, ox, oy,
            self.white_level, self.black_level,
        )

    def get_current_proj_corners(self) -> np.ndarray | None:
        """Get projector-pixel corner positions for the current board position."""
        if self._current_position_idx >= len(self._positions):
            return None
        ox, oy = self._positions[self._current_position_idx]
        return get_projected_corner_positions(
            self.board_cols, self.board_rows, self.square_px, ox, oy,
        )

    # ── Capture + detection ─────────────────────────────────────────────

    def capture_board(self, frame_bgr: np.ndarray) -> tuple[bool, np.ndarray | None]:
        """Capture the current projected checkerboard from the camera.

        Detects corners in the camera image. If successful, stores the data
        and advances to the next position.

        Returns (success, detected_corners_for_overlay).
        """
        if self._current_position_idx >= len(self._positions):
            return False, None

        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        self._cam_image_size = (gray.shape[1], gray.shape[0])

        # Try multiple flag combinations for robustness
        flag_sets = [
            cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE,
            cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_FILTER_QUADS,
            cv2.CALIB_CB_ADAPTIVE_THRESH,
        ]
        found = False
        corners = None
        for flags in flag_sets:
            found, corners = cv2.findChessboardCorners(gray, self.board_size, flags)
            if found and corners is not None:
                break
        if not found or corners is None:
            return False, None

        # Sub-pixel refinement
        criteria = (
            cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001
        )
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

        # Store this capture
        proj_corners = self.get_current_proj_corners()
        self._obj_points.append(self._obj_pts_template.copy())
        self._cam_img_points.append(corners)
        self._proj_img_points.append(proj_corners)

        logger.info(
            "Board capture %d at position %d: %d corners detected",
            len(self._obj_points),
            self._current_position_idx,
            len(corners),
        )

        self._current_position_idx += 1
        return True, corners

    def detect_current_board(self, frame_bgr: np.ndarray) -> tuple[bool, np.ndarray | None]:
        """Detect corners in the current frame for live preview overlay.

        Does NOT store data or advance position.
        """
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        flags = (
            cv2.CALIB_CB_ADAPTIVE_THRESH
            | cv2.CALIB_CB_NORMALIZE_IMAGE
            | cv2.CALIB_CB_FAST_CHECK
        )
        found, corners = cv2.findChessboardCorners(gray, self.board_size, flags)
        if found and corners is not None:
            criteria = (
                cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001
            )
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        return found, corners

    # ── Camera intrinsics ───────────────────────────────────────────────

    def compute_camera_intrinsics(self) -> tuple[np.ndarray, np.ndarray, float]:
        """Calibrate camera intrinsics from projected checkerboard captures.

        Returns (K_cam, dist_cam, mean_reprojection_error).
        """
        if len(self._obj_points) < 3:
            raise ValueError(
                f"Need at least 3 captures, have {len(self._obj_points)}"
            )

        ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
            self._obj_points,
            self._cam_img_points,
            self._cam_image_size,
            None,
            None,
        )
        self._cam_rvecs = rvecs
        self._cam_tvecs = tvecs
        logger.info(
            "Camera intrinsics: fx=%.1f fy=%.1f cx=%.1f cy=%.1f error=%.3f px",
            K[0, 0], K[1, 1], K[0, 2], K[1, 2], ret,
        )
        return K, dist, ret

    # ── Projector intrinsics ────────────────────────────────────────────

    def compute_projector_intrinsics(
        self, K_cam: np.ndarray, dist_cam: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """Calibrate projector intrinsics using 3D points from camera calibration.

        Uses the camera's rvecs/tvecs to compute 3D positions of board corners,
        then calibrates the projector using those 3D points and the known
        projector pixel positions.

        Returns (K_proj, dist_proj, mean_reprojection_error).
        """
        if not hasattr(self, "_cam_rvecs"):
            raise ValueError("Must compute camera intrinsics first")

        # Build 3D object points in camera coordinate system for each view
        obj_3d_list = []
        for i in range(len(self._obj_points)):
            R_cam, _ = cv2.Rodrigues(self._cam_rvecs[i])
            t_cam = self._cam_tvecs[i].reshape(3)
            # Transform template points to camera coordinates
            pts_3d = (R_cam @ self._obj_pts_template.T).T + t_cam
            obj_3d_list.append(pts_3d.astype(np.float32))

        proj_size = (self.proj_w, self.proj_h)

        # Non-planar 3D points require an initial intrinsic guess.
        # Use a reasonable estimate: focal length ~ image diagonal,
        # principal point at center.
        fx_init = float(max(self.proj_w, self.proj_h))
        K_init = np.array([
            [fx_init, 0, self.proj_w / 2.0],
            [0, fx_init, self.proj_h / 2.0],
            [0, 0, 1],
        ], dtype=np.float64)

        ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
            obj_3d_list,
            self._proj_img_points,
            proj_size,
            K_init,
            None,
            flags=cv2.CALIB_USE_INTRINSIC_GUESS,
        )
        logger.info(
            "Projector intrinsics: fx=%.1f fy=%.1f cx=%.1f cy=%.1f error=%.3f px",
            K[0, 0], K[1, 1], K[0, 2], K[1, 2], ret,
        )
        return K, dist, ret

    # ── Stereo extrinsics ───────────────────────────────────────────────

    def compute_stereo(
        self,
        K_cam: np.ndarray,
        dist_cam: np.ndarray,
        K_proj: np.ndarray,
        dist_proj: np.ndarray,
        fix_intrinsics: bool = True,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """Stereo calibration: solve for R, T from camera to projector.

        Args:
            fix_intrinsics: If True, keep K_cam and K_proj fixed. Set False
                when K_proj is only an estimate (e.g. projector intrinsics
                calibration failed) so stereo calibrate can refine it.

        Returns (R, T, reprojection_error).
        """
        if len(self._obj_points) < 3:
            raise ValueError("Need at least 3 captures for stereo calibration")

        flags = cv2.CALIB_FIX_INTRINSIC if fix_intrinsics else cv2.CALIB_USE_INTRINSIC_GUESS
        ret, K_cam_out, _, K_proj_out, _, R, T, E, F = cv2.stereoCalibrate(
            self._obj_points,
            self._cam_img_points,
            self._proj_img_points,
            K_cam.copy(),
            dist_cam.copy(),
            K_proj.copy(),
            dist_proj.copy(),
            self._cam_image_size,
            flags=flags,
        )
        # If intrinsics were refined, update them
        if not fix_intrinsics:
            self._refined_K_proj = K_proj_out
        logger.info(
            "Stereo calibration: error=%.3f px\n  R=%s\n  T=%s",
            ret, R.tolist(), T.flatten().tolist(),
        )
        return R, T, ret

    # ── RoomAlive-style dense calibration ─────────────────────────────

    @staticmethod
    def calibrate_from_dense_correspondences(
        decoded_col: np.ndarray,
        decoded_row: np.ndarray,
        decode_mask: np.ndarray,
        depth_frame: np.ndarray,
        K_cam: np.ndarray,
        dist_cam: np.ndarray,
        K_proj: np.ndarray,
        proj_w: int,
        proj_h: int,
        max_points: int = 50000,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """RoomAlive-style: use dense Gray code correspondences + depth to find R, T.

        This uses the depth map from the camera (Depth Anything V2) to unproject
        camera pixels to 3D, then pairs them with projector pixels from Gray code
        decoding. cv2.solvePnPRansac finds the projector pose robustly.

        Args:
            decoded_col: (cam_h, cam_w) int32 — projector column for each camera pixel
            decoded_row: (cam_h, cam_w) int32 — projector row for each camera pixel
            decode_mask: (cam_h, cam_w) bool — valid Gray code pixels
            depth_frame: (cam_h, cam_w) float32 [0,1] — monocular depth
            K_cam: 3x3 camera intrinsic matrix
            dist_cam: camera distortion coefficients
            K_proj: 3x3 projector intrinsic matrix (initial estimate)
            proj_w, proj_h: projector resolution
            max_points: subsample to this many points for speed

        Returns:
            (R, T, reprojection_error) — camera-to-projector transform
        """
        cam_h, cam_w = depth_frame.shape

        # Get valid pixels: both Gray code and depth must be valid
        valid = decode_mask & (depth_frame > 0.01)
        valid_ys, valid_xs = np.where(valid)

        if len(valid_xs) < 100:
            raise ValueError(f"Only {len(valid_xs)} valid correspondences — need at least 100")

        # Subsample if too many points
        if len(valid_xs) > max_points:
            indices = np.random.default_rng(42).choice(
                len(valid_xs), max_points, replace=False
            )
            valid_xs = valid_xs[indices]
            valid_ys = valid_ys[indices]

        N = len(valid_xs)
        logger.info("Dense calibration: using %d 3D-2D correspondences", N)

        # Unproject camera pixels to 3D using depth + camera intrinsics
        # First undistort the pixel coordinates
        pts_2d = np.column_stack([
            valid_xs.astype(np.float64),
            valid_ys.astype(np.float64),
        ]).reshape(-1, 1, 2)
        pts_norm = cv2.undistortPoints(pts_2d, K_cam, dist_cam)  # normalized coords
        pts_norm = pts_norm.reshape(-1, 2)

        # Get depth at each pixel
        depths = depth_frame[valid_ys, valid_xs].astype(np.float64)

        # Scale monocular depth to a plausible range.
        # The exact scale doesn't matter much — solvePnP recovers the pose
        # from the 3D-2D correspondences regardless.  We use a range that
        # keeps the point cloud well-conditioned for the solver.
        depths = 1.0 + depths * 9.0  # [0,1] → [1, 10] arbitrary units

        # 3D points in camera space: P = [x_norm * z, y_norm * z, z]
        pts_3d = np.zeros((N, 3), dtype=np.float64)
        pts_3d[:, 0] = pts_norm[:, 0] * depths
        pts_3d[:, 1] = pts_norm[:, 1] * depths
        pts_3d[:, 2] = depths

        # Corresponding projector pixels
        proj_xs = decoded_col[valid_ys, valid_xs].astype(np.float64)
        proj_ys = decoded_row[valid_ys, valid_xs].astype(np.float64)
        pts_proj = np.column_stack([proj_xs, proj_ys]).astype(np.float64)

        # Filter out-of-range projector pixels
        in_range = (
            (proj_xs >= 0) & (proj_xs < proj_w)
            & (proj_ys >= 0) & (proj_ys < proj_h)
        )
        pts_3d = pts_3d[in_range]
        pts_proj = pts_proj[in_range]

        if len(pts_3d) < 100:
            raise ValueError(f"Only {len(pts_3d)} in-range correspondences")

        # Solve PnP with RANSAC for robustness
        dist_proj = np.zeros(5, dtype=np.float64)  # assume minimal projector distortion
        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            pts_3d.reshape(-1, 1, 3),
            pts_proj.reshape(-1, 1, 2),
            K_proj,
            dist_proj,
            iterationsCount=500,
            reprojectionError=5.0,
            confidence=0.99,
            flags=cv2.SOLVEPNP_EPNP,
        )

        if not success or inliers is None:
            raise ValueError("solvePnPRansac failed — not enough inliers")

        n_inliers = len(inliers)
        logger.info("solvePnPRansac: %d/%d inliers (%.1f%%)",
                     n_inliers, len(pts_3d), 100.0 * n_inliers / len(pts_3d))

        # Refine with Levenberg-Marquardt on inlier set
        inlier_3d = pts_3d[inliers.ravel()]
        inlier_2d = pts_proj[inliers.ravel()]

        rvec_refined, tvec_refined = cv2.solvePnPRefineLM(
            inlier_3d.reshape(-1, 1, 3),
            inlier_2d.reshape(-1, 1, 2),
            K_proj,
            dist_proj,
            rvec,
            tvec,
        )

        R, _ = cv2.Rodrigues(rvec_refined)
        T = tvec_refined.reshape(3, 1)

        # Compute reprojection error on inliers
        projected, _ = cv2.projectPoints(
            inlier_3d, rvec_refined, tvec_refined, K_proj, dist_proj
        )
        projected = projected.reshape(-1, 2)
        errors = np.linalg.norm(projected - inlier_2d, axis=1)
        mean_error = float(np.mean(errors))
        logger.info("Dense calibration reprojection error: %.3f px (%d inliers)",
                     mean_error, n_inliers)

        return R, T, mean_error

    # ── Gray Codes (wraps existing calibration.py) ──────────────────────

    @staticmethod
    def create_gray_code_runner(
        proj_w: int, proj_h: int, settle_ms: float = 200.0,
    ) -> CalibrationRunner:
        return CalibrationRunner(proj_w, proj_h, settle_ms)

    @staticmethod
    def decode_gray_codes(
        runner: CalibrationRunner,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Decode captured Gray codes.

        Returns (map_x, map_y, correspondence_mask).
        """
        map_x, map_y = runner.decode()
        correspondence_mask = (map_x >= 0) & (map_y >= 0)
        return map_x, map_y, correspondence_mask


# ── Serialization ───────────────────────────────────────────────────────


def save_procam(
    path: str | Path,
    K_cam: np.ndarray,
    dist_cam: np.ndarray,
    K_proj: np.ndarray,
    dist_proj: np.ndarray,
    R: np.ndarray,
    T: np.ndarray,
    map_x: np.ndarray,
    map_y: np.ndarray,
    proj_w: int,
    proj_h: int,
) -> None:
    """Save full ProCam calibration to JSON."""
    _zeros5 = np.zeros((1, 5), dtype=np.float64)
    data = {
        "version": 2,
        "projector_width": proj_w,
        "projector_height": proj_h,
        "K_cam": K_cam.tolist(),
        "dist_cam": (dist_cam if dist_cam is not None else _zeros5).tolist(),
        "K_proj": K_proj.tolist(),
        "dist_proj": (dist_proj if dist_proj is not None else _zeros5).tolist(),
        "R": R.tolist(),
        "T": T.tolist(),
        "map_x": map_x.tolist(),
        "map_y": map_y.tolist(),
    }
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data))
    logger.info("ProCam calibration saved to %s", path)


def load_procam(path: str | Path) -> ProCamCalibration:
    """Load ProCam calibration from JSON."""
    from .calibration import sanitize_correspondence_maps

    data = json.loads(Path(path).read_text())

    cal = ProCamCalibration()
    map_x = np.array(data["map_x"], dtype=np.float32)
    map_y = np.array(data["map_y"], dtype=np.float32)
    cal.map_x, cal.map_y = sanitize_correspondence_maps(map_x, map_y)

    if data.get("version", 1) >= 2:
        cal.K_cam = np.array(data["K_cam"], dtype=np.float64)
        cal.dist_cam = np.array(data["dist_cam"], dtype=np.float64)
        cal.K_proj = np.array(data["K_proj"], dtype=np.float64)
        cal.dist_proj = np.array(data["dist_proj"], dtype=np.float64)
        cal.R = np.array(data["R"], dtype=np.float64)
        cal.T = np.array(data["T"], dtype=np.float64)

    return cal
