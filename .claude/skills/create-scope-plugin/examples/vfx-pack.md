# Complete Example: scope-vfx (VFX Pack Plugin)

This is a real, working Scope plugin that applies GPU-accelerated visual effects to video input. Use it as a reference for the patterns, conventions, and file structure.

## What it does

- **Type**: Video-input pipeline
- **Models**: None (pure PyTorch tensor operations)
- **Effects**: Chromatic aberration + VHS/retro CRT
- **All parameters**: Runtime (live sliders during streaming)

---

## File: pyproject.toml

```toml
[project]
name = "scope-vfx"
version = "0.1.0"
description = "GPU-accelerated visual effects pack for Daydream Scope"
requires-python = ">=3.12"

[project.entry-points."scope"]
scope_vfx = "scope_vfx"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/scope_vfx"]
```

Note: No `[project.dependencies]` — everything is pure PyTorch which Scope provides.

---

## File: src/scope_vfx/__init__.py

```python
from scope.core.plugins.hookspecs import hookimpl


@hookimpl
def register_pipelines(register):
    from .pipeline import VFXPipeline

    register(VFXPipeline)
```

---

## File: src/scope_vfx/schema.py

```python
from pydantic import Field

from scope.core.pipelines.base_schema import BasePipelineConfig, ModeDefaults, ui_field_config


class VFXConfig(BasePipelineConfig):
    """Configuration for the VFX Pack pipeline."""

    pipeline_id = "vfx-pack"
    pipeline_name = "VFX Pack"
    pipeline_description = (
        "GPU-accelerated visual effects: chromatic aberration, VHS/retro CRT, and more"
    )

    supports_prompts = False

    modes = {"video": ModeDefaults(default=True)}

    # --- Chromatic Aberration ---

    chromatic_enabled: bool = Field(
        default=True,
        description="Enable chromatic aberration (RGB channel displacement)",
        json_schema_extra=ui_field_config(order=1, label="Chromatic Aberration"),
    )

    chromatic_intensity: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Strength of the RGB channel displacement (0 = none, 1 = maximum)",
        json_schema_extra=ui_field_config(order=2, label="Intensity"),
    )

    chromatic_angle: float = Field(
        default=0.0,
        ge=0.0,
        le=360.0,
        description="Direction of the channel displacement in degrees",
        json_schema_extra=ui_field_config(order=3, label="Angle"),
    )

    # --- VHS / Retro CRT ---

    vhs_enabled: bool = Field(
        default=False,
        description="Enable VHS / retro CRT effect (scan lines, noise, tracking)",
        json_schema_extra=ui_field_config(order=10, label="VHS / Retro CRT"),
    )

    scan_line_intensity: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Darkness of the scan lines (0 = invisible, 1 = fully black)",
        json_schema_extra=ui_field_config(order=11, label="Scan Lines"),
    )

    scan_line_count: int = Field(
        default=100,
        ge=10,
        le=500,
        description="Number of scan lines across the frame height",
        json_schema_extra=ui_field_config(order=12, label="Line Count"),
    )

    vhs_noise: float = Field(
        default=0.1,
        ge=0.0,
        le=1.0,
        description="Amount of analog noise / film grain",
        json_schema_extra=ui_field_config(order=13, label="Noise"),
    )

    tracking_distortion: float = Field(
        default=0.2,
        ge=0.0,
        le=1.0,
        description="Horizontal tracking distortion (wavy displacement)",
        json_schema_extra=ui_field_config(order=14, label="Tracking"),
    )
```

Key patterns demonstrated:
- `order` values spaced by 10 between groups (1-3 for chromatic, 10-14 for VHS) so new effects slot in cleanly
- `bool` fields create toggles, `float` with `ge`/`le` creates sliders
- All are runtime params (no `is_load_param=True`) so they're live during streaming
- `description` becomes the tooltip text in the UI

---

## File: src/scope_vfx/pipeline.py

```python
from typing import TYPE_CHECKING

import torch

from scope.core.pipelines.interface import Pipeline, Requirements

from .effects import chromatic_aberration, vhs_retro
from .schema import VFXConfig

if TYPE_CHECKING:
    from scope.core.pipelines.base_schema import BasePipelineConfig


class VFXPipeline(Pipeline):
    """GPU-accelerated visual effects pipeline."""

    @classmethod
    def get_config_class(cls) -> type["BasePipelineConfig"]:
        return VFXConfig

    def __init__(self, device: torch.device | None = None, **kwargs):
        self.device = (
            device
            if device is not None
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )

    def prepare(self, **kwargs) -> Requirements:
        """We need exactly one input frame per call."""
        return Requirements(input_size=1)

    def __call__(self, **kwargs) -> dict:
        """Apply enabled effects to input video frames."""
        video = kwargs.get("video")
        if video is None:
            raise ValueError("VFXPipeline requires video input")

        # Stack input frames -> (T, H, W, C) and normalise to [0, 1]
        frames = torch.stack([frame.squeeze(0) for frame in video], dim=0)
        frames = frames.to(device=self.device, dtype=torch.float32) / 255.0

        # --- Effect chain (order matters) ---

        if kwargs.get("chromatic_enabled", True):
            frames = chromatic_aberration(
                frames,
                intensity=kwargs.get("chromatic_intensity", 0.3),
                angle=kwargs.get("chromatic_angle", 0.0),
            )

        if kwargs.get("vhs_enabled", False):
            frames = vhs_retro(
                frames,
                scan_line_intensity=kwargs.get("scan_line_intensity", 0.3),
                scan_line_count=kwargs.get("scan_line_count", 100),
                noise=kwargs.get("vhs_noise", 0.1),
                tracking=kwargs.get("tracking_distortion", 0.2),
            )

        return {"video": frames.clamp(0, 1)}
```

Key patterns demonstrated:
- `get_config_class()` links pipeline to its schema
- `__init__()` only stores the device — no runtime params
- `prepare()` returns `Requirements(input_size=1)` for single-frame processing
- `__call__()` reads ALL params from `kwargs.get()` with defaults matching schema defaults
- Input normalization: `/ 255.0` converts [0,255] -> [0,1]
- Output: `.clamp(0, 1)` ensures valid range
- Effect chain is sequential — each function takes and returns `(T, H, W, C)` tensors

---

## File: src/scope_vfx/effects/chromatic.py

```python
import math

import torch


def chromatic_aberration(
    frames: torch.Tensor,
    intensity: float = 0.3,
    angle: float = 0.0,
) -> torch.Tensor:
    """Displace RGB channels in opposite directions for a chromatic aberration look."""
    if intensity <= 0:
        return frames

    max_shift = int(intensity * 20)
    if max_shift == 0:
        return frames

    rad = math.radians(angle)
    dx = int(round(max_shift * math.cos(rad)))
    dy = int(round(max_shift * math.sin(rad)))

    if dx == 0 and dy == 0:
        return frames

    result = frames.clone()

    # Red channel shifts one direction
    result[..., 0] = torch.roll(frames[..., 0], shifts=(dy, dx), dims=(1, 2))
    # Blue channel shifts the opposite direction
    result[..., 2] = torch.roll(frames[..., 2], shifts=(-dy, -dx), dims=(1, 2))
    # Green channel stays centred

    return result
```

Pattern: Effect functions are pure `(tensor, params) -> tensor`. No state, no classes, no side effects. This makes them composable and testable.

---

## File: src/scope_vfx/effects/vhs.py

```python
import torch


def vhs_retro(
    frames: torch.Tensor,
    scan_line_intensity: float = 0.3,
    scan_line_count: int = 100,
    noise: float = 0.1,
    tracking: float = 0.2,
) -> torch.Tensor:
    """Apply a VHS / retro CRT look."""
    _T, H, W, _C = frames.shape
    result = frames.clone()

    # --- Scan lines ---
    if scan_line_intensity > 0 and scan_line_count > 0:
        rows = torch.arange(H, device=frames.device, dtype=torch.float32)
        wave = torch.sin(rows * (scan_line_count * 3.14159 / H))
        mask = 1.0 - scan_line_intensity * 0.5 * (1.0 - wave)
        result = result * mask.view(1, H, 1, 1)

    # --- Analog noise / film grain ---
    if noise > 0:
        grain = torch.randn_like(result) * (noise * 0.15)
        result = result + grain

    # --- Tracking distortion (GPU-friendly via grid_sample) ---
    if tracking > 0:
        max_shift = tracking * 0.05
        rows_norm = torch.linspace(-1.0, 1.0, H, device=frames.device)
        offsets = max_shift * torch.sin(rows_norm * 6.2832 * 3.0)

        grid_y = torch.linspace(-1.0, 1.0, H, device=frames.device)
        grid_x = torch.linspace(-1.0, 1.0, W, device=frames.device)
        gy, gx = torch.meshgrid(grid_y, grid_x, indexing="ij")

        gx = gx + offsets.view(H, 1)

        grid = torch.stack([gx, gy], dim=-1).unsqueeze(0).expand(result.shape[0], -1, -1, -1)

        result_nchw = result.permute(0, 3, 1, 2)
        result_nchw = torch.nn.functional.grid_sample(
            result_nchw, grid, mode="bilinear", padding_mode="border", align_corners=True
        )
        result = result_nchw.permute(0, 2, 3, 1)

    return result.clamp(0, 1)
```

Pattern: Uses `grid_sample` for spatial displacement instead of per-row Python loops. This runs as a single GPU kernel regardless of resolution.

---

## File: src/scope_vfx/effects/__init__.py

```python
from .chromatic import chromatic_aberration
from .vhs import vhs_retro

__all__ = ["chromatic_aberration", "vhs_retro"]
```

---

## How to add a new effect (extensibility pattern)

1. Create `src/scope_vfx/effects/new_effect.py` with a function: `def my_effect(frames: torch.Tensor, ...) -> torch.Tensor`
2. Add parameters to `schema.py` with `order` values in a new range (e.g., 20-29)
3. Add an `if kwargs.get("effect_enabled"):` block in `pipeline.py`'s `__call__()` effect chain
4. Add import to `effects/__init__.py`
5. Reload plugin in Scope — new UI controls appear automatically
