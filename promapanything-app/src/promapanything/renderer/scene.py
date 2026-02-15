"""3D scene renderer — point cloud, mesh, projector frustum wireframe."""

from __future__ import annotations

import math
from pathlib import Path

import moderngl
import numpy as np

from ..state import AppState, ViewMode

SHADER_DIR = Path(__file__).parent / "shaders"


def _read_shader(name: str) -> str:
    return (SHADER_DIR / name).read_text()


def _perspective(fov_deg: float, aspect: float, near: float, far: float) -> np.ndarray:
    """Build a perspective projection matrix (column-major for OpenGL)."""
    f = 1.0 / math.tan(math.radians(fov_deg) / 2.0)
    nf = near - far
    m = np.zeros((4, 4), dtype=np.float32)
    m[0, 0] = f / aspect
    m[1, 1] = f
    m[2, 2] = (far + near) / nf
    m[2, 3] = -1.0
    m[3, 2] = (2.0 * far * near) / nf
    return m


def _look_at(eye: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
    """Build a look-at view matrix (column-major)."""
    f = target - eye
    f = f / np.linalg.norm(f)
    u = up / np.linalg.norm(up)
    s = np.cross(f, u)
    s = s / (np.linalg.norm(s) + 1e-9)
    u = np.cross(s, f)

    m = np.eye(4, dtype=np.float32)
    m[0, :3] = s
    m[1, :3] = u
    m[2, :3] = -f
    m[3, 0] = -np.dot(s, eye)
    m[3, 1] = -np.dot(u, eye)
    m[3, 2] = np.dot(f, eye)
    return m


class SceneRenderer:
    """Renders the 3D scene (point cloud or mesh) with orbit camera."""

    def __init__(self, ctx: moderngl.Context):
        self.ctx = ctx

        # Point cloud program
        self._pc_prog = ctx.program(
            vertex_shader=_read_shader("pointcloud.vert"),
            fragment_shader=_read_shader("pointcloud.frag"),
        )
        self._pc_vao = ctx.vertex_array(self._pc_prog, [])

        # Mesh program
        self._mesh_prog = ctx.program(
            vertex_shader=_read_shader("mesh.vert"),
            fragment_shader=_read_shader("mesh.frag"),
        )
        # Mesh needs an index buffer — built lazily when resolution is known
        self._mesh_vao = None
        self._mesh_ibo = None
        self._mesh_res = (0, 0)
        self._mesh_num_indices = 0

        # Frustum wireframe
        self._frustum_prog = ctx.program(
            vertex_shader=_read_shader("frustum.vert"),
            fragment_shader=_read_shader("frustum.frag"),
        )
        self._frustum_vbo = ctx.buffer(reserve=12 * 8 * 4)  # 8 verts * vec3
        self._frustum_ibo = ctx.buffer(reserve=24 * 4)  # 12 edges * 2 indices
        self._frustum_vao = ctx.vertex_array(
            self._frustum_prog,
            [(self._frustum_vbo, "3f", "in_position")],
            index_buffer=self._frustum_ibo,
            index_element_size=4,
        )
        self._frustum_built = False

        # Depth texture resolution for vertex count
        self._depth_w = 0
        self._depth_h = 0

    def _build_mesh_index_buffer(self, w: int, h: int) -> None:
        """Build triangle index buffer for a w x h grid.

        Every quad (px, py)→(px+1, py)→(px, py+1)→(px+1, py+1) becomes 2 triangles.
        Skip boundary pixels at the edge.
        """
        if w == self._mesh_res[0] and h == self._mesh_res[1]:
            return

        # Downsample if too many triangles (>2M triangles)
        step = 1
        while (w // step) * (h // step) * 2 > 2_000_000:
            step += 1

        sw, sh = w // step, h // step
        indices = []
        for py in range(sh - 1):
            for px in range(sw - 1):
                # Vertex IDs in the full grid
                i00 = (py * step) * w + (px * step)
                i10 = (py * step) * w + ((px + 1) * step)
                i01 = ((py + 1) * step) * w + (px * step)
                i11 = ((py + 1) * step) * w + ((px + 1) * step)
                indices.extend([i00, i10, i01, i10, i11, i01])

        indices = np.array(indices, dtype=np.int32)
        self._mesh_num_indices = len(indices)

        if self._mesh_ibo is not None:
            self._mesh_ibo.release()
        if self._mesh_vao is not None:
            self._mesh_vao.release()

        self._mesh_ibo = self.ctx.buffer(indices.tobytes())
        self._mesh_vao = self.ctx.vertex_array(
            self._mesh_prog,
            [],
            index_buffer=self._mesh_ibo,
            index_element_size=4,
        )
        self._mesh_res = (w, h)

    def build_frustum(self, proj_w: int, proj_h: int, focal: float) -> None:
        """Build a projector frustum wireframe from intrinsics."""
        aspect = proj_w / max(proj_h, 1)
        fov_h = 2.0 * math.atan(proj_w / (2.0 * focal))
        fov_v = 2.0 * math.atan(proj_h / (2.0 * focal))

        near, far = 0.3, 3.0
        hn = near * math.tan(fov_v / 2.0)
        wn = near * math.tan(fov_h / 2.0)
        hf = far * math.tan(fov_v / 2.0)
        wf = far * math.tan(fov_h / 2.0)

        # 8 corners of the frustum (camera space: +Z = into screen)
        verts = np.array([
            # Near plane
            [-wn, -hn, -near],
            [ wn, -hn, -near],
            [ wn,  hn, -near],
            [-wn,  hn, -near],
            # Far plane
            [-wf, -hf, -far],
            [ wf, -hf, -far],
            [ wf,  hf, -far],
            [-wf,  hf, -far],
        ], dtype=np.float32)

        # 12 edges connecting near/far planes
        edges = np.array([
            0, 1,  1, 2,  2, 3,  3, 0,  # Near
            4, 5,  5, 6,  6, 7,  7, 4,  # Far
            0, 4,  1, 5,  2, 6,  3, 7,  # Connecting
        ], dtype=np.int32)

        self._frustum_vbo.orphan(verts.nbytes)
        self._frustum_vbo.write(verts.tobytes())
        self._frustum_ibo.orphan(edges.nbytes)
        self._frustum_ibo.write(edges.tobytes())
        self._frustum_built = True

    def build_frustum_from_intrinsics(
        self,
        K_proj: np.ndarray,
        R: np.ndarray,
        T: np.ndarray,
        proj_w: int,
        proj_h: int,
    ) -> None:
        """Build projector frustum wireframe using real calibrated intrinsics.

        K_proj: 3x3 projector intrinsic matrix
        R: 3x3 rotation (camera→projector)
        T: 3x1 translation (camera→projector)
        """
        fx = K_proj[0, 0]
        fy = K_proj[1, 1]
        cx = K_proj[0, 2]
        cy = K_proj[1, 2]

        # Compute FOV from intrinsics
        fov_h = 2.0 * math.atan(proj_w / (2.0 * fx))
        fov_v = 2.0 * math.atan(proj_h / (2.0 * fy))

        near, far = 0.3, 3.0
        hn = near * math.tan(fov_v / 2.0)
        wn = near * math.tan(fov_h / 2.0)
        hf = far * math.tan(fov_v / 2.0)
        wf = far * math.tan(fov_h / 2.0)

        # Frustum corners in projector local space
        local_verts = np.array([
            [-wn, -hn, near],
            [ wn, -hn, near],
            [ wn,  hn, near],
            [-wn,  hn, near],
            [-wf, -hf, far],
            [ wf, -hf, far],
            [ wf,  hf, far],
            [-wf,  hf, far],
        ], dtype=np.float64)

        # Transform from projector space to camera space
        # P_proj = R @ P_cam + T  →  P_cam = R^T @ (P_proj - T)
        R_inv = R.T
        T_vec = T.reshape(3)
        cam_verts = (R_inv @ (local_verts.T - T_vec[:, np.newaxis])).T

        # Convert to OpenGL convention (-Z forward)
        gl_verts = cam_verts.copy().astype(np.float32)
        gl_verts[:, 2] = -gl_verts[:, 2]

        edges = np.array([
            0, 1,  1, 2,  2, 3,  3, 0,
            4, 5,  5, 6,  6, 7,  7, 4,
            0, 4,  1, 5,  2, 6,  3, 7,
        ], dtype=np.int32)

        self._frustum_vbo.orphan(gl_verts.nbytes)
        self._frustum_vbo.write(gl_verts.tobytes())
        self._frustum_ibo.orphan(edges.nbytes)
        self._frustum_ibo.write(edges.tobytes())
        self._frustum_built = True

    def compute_mvp(self, state: AppState, aspect: float) -> np.ndarray:
        """Compute model-view-projection matrix from orbit camera state."""
        orbit = state.orbit
        yaw_rad = math.radians(orbit.yaw)
        pitch_rad = math.radians(orbit.pitch)

        # Camera position on sphere around target
        cos_p = math.cos(pitch_rad)
        eye = np.array([
            orbit.target_x + orbit.distance * cos_p * math.sin(yaw_rad),
            orbit.target_y + orbit.distance * math.sin(pitch_rad),
            orbit.target_z + orbit.distance * cos_p * math.cos(yaw_rad),
        ], dtype=np.float32)

        target = np.array([orbit.target_x, orbit.target_y, orbit.target_z], dtype=np.float32)
        up = np.array([0.0, 1.0, 0.0], dtype=np.float32)

        view = _look_at(eye, target, up)
        proj = _perspective(orbit.fov, aspect, 0.01, 100.0)
        return (proj @ view).astype(np.float32)

    def render(
        self,
        state: AppState,
        depth_tex: moderngl.Texture,
        color_tex: moderngl.Texture,
        viewport: tuple[int, int, int, int],
    ) -> None:
        """Render the 3D scene into the given viewport."""
        ctx = self.ctx
        x, y, w, h = viewport
        ctx.viewport = viewport

        dw, dh = depth_tex.size
        self._depth_w = dw
        self._depth_h = dh

        aspect = w / max(h, 1)
        mvp = self.compute_mvp(state, aspect)

        # Use real camera intrinsics if available, else heuristic
        if state.procam.K_cam is not None:
            K = state.procam.K_cam
            focal = (K[0, 0] + K[1, 1]) / 2.0
            cx = K[0, 2]
            cy = K[1, 2]
        else:
            focal = max(dw, dh) * 0.8
            cx = dw / 2.0
            cy = dh / 2.0

        ctx.enable(moderngl.DEPTH_TEST)

        if state.view_mode == ViewMode.POINT_CLOUD:
            self._render_point_cloud(depth_tex, color_tex, mvp, focal, cx, cy, dw, dh)
        else:
            self._render_mesh(depth_tex, color_tex, mvp, focal, cx, cy, dw, dh)

        # Projector frustum
        if self._frustum_built:
            self._render_frustum(mvp)

        ctx.disable(moderngl.DEPTH_TEST)

    def _set_common_uniforms(
        self, prog, depth_tex, color_tex, mvp, focal, cx, cy, dw, dh, state=None
    ):
        depth_tex.use(0)
        color_tex.use(1)
        prog["depth_tex"].value = 0
        prog["color_tex"].value = 1
        prog["mvp"].write(mvp.T.tobytes())  # transpose for column-major
        prog["depth_scale"].value = 1.0
        prog["depth_offset"].value = 0.0
        prog["depth_invert"].value = False
        prog["resolution"].value = (float(dw), float(dh))
        prog["focal_length"].value = focal
        prog["principal"].value = (cx, cy)

    def _render_point_cloud(self, depth_tex, color_tex, mvp, focal, cx, cy, dw, dh):
        self._set_common_uniforms(
            self._pc_prog, depth_tex, color_tex, mvp, focal, cx, cy, dw, dh
        )
        self.ctx.point_size = 2.0
        self._pc_vao.render(moderngl.POINTS, vertices=dw * dh)

    def _render_mesh(self, depth_tex, color_tex, mvp, focal, cx, cy, dw, dh):
        self._build_mesh_index_buffer(dw, dh)
        if self._mesh_vao is None:
            return

        self._set_common_uniforms(
            self._mesh_prog, depth_tex, color_tex, mvp, focal, cx, cy, dw, dh
        )
        self._mesh_vao.render(moderngl.TRIANGLES)

    def _render_frustum(self, mvp: np.ndarray):
        self._frustum_prog["mvp"].write(mvp.T.tobytes())
        self._frustum_prog["line_color"].value = (1.0, 0.8, 0.0)
        self._frustum_vao.render(moderngl.LINES)

    def release(self):
        for res in [
            self._pc_vao, self._mesh_vao, self._mesh_ibo,
            self._frustum_vao, self._frustum_vbo, self._frustum_ibo,
        ]:
            if res is not None:
                res.release()
