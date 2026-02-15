# Scope Plugin System — Complete Technical Reference

This is the authoritative reference for building Daydream Scope plugins. All import paths, class hierarchies, and patterns are exact.

**Official documentation:**
- Plugin user guide: https://docs.daydream.live/scope/guides/plugins
- Plugin development guide: https://docs.daydream.live/scope/guides/plugin-development
- Plugin architecture: https://docs.daydream.live/scope/reference/architecture/plugins
- Pipeline architecture: https://docs.daydream.live/scope/reference/architecture/pipelines
- Scope GitHub: https://github.com/daydreamlive/scope

---

## Architecture Overview

Scope is an open-source tool by Daydream/Livepeer for running real-time interactive generative AI video pipelines. The plugin system uses Python **pluggy** for hook-based discovery and Python **entry points** for package registration.

**How it works:**
1. Developer creates a Python package with an entry point in the `"scope"` group
2. User installs the package (via Git URL, PyPI, or local path) through Scope's Settings UI
3. Scope discovers the entry point, loads the module, calls the `register_pipelines` hook
4. The hook registers one or more Pipeline classes
5. Pipelines appear in Scope's UI — users can select them and stream video through them

**Installation sources:**
- **Git**: `git+https://github.com/user/repo.git` (recommended for sharing)
- **PyPI**: `package-name` (for published packages)
- **Local path**: `/path/to/plugin` (for development)

**Safety:** Scope snapshots the virtualenv before installing and rolls back on failure. Plugins cannot corrupt the base installation.

---

## Exact Import Paths

```python
# Plugin hook registration
from scope.core.plugins.hookspecs import hookimpl

# Pipeline base class and input requirements
from scope.core.pipelines.interface import Pipeline, Requirements

# Configuration schema base class and helpers
from scope.core.pipelines.base_schema import BasePipelineConfig, ModeDefaults, ui_field_config

# Usage type enum (for preprocessors/postprocessors)
from scope.core.pipelines.base_schema import UsageType

# Artifact types for model downloads
from scope.core.pipelines.artifacts import HuggingfaceRepoArtifact, GoogleDriveArtifact

# Common shared artifacts (Wan 1.3B, VACE, etc.)
from scope.core.pipelines.common_artifacts import WAN_1_3B_ARTIFACT, VACE_ARTIFACT

# Multi-mode helper
from scope.core.pipelines.defaults import prepare_for_mode

# Standard library (always available)
from pydantic import Field
import torch
from typing import TYPE_CHECKING
```

---

## pyproject.toml Template

```toml
[project]
name = "scope-PLUGIN_NAME"
version = "0.1.0"
description = "DESCRIPTION"
requires-python = ">=3.12"

# Only add dependencies that Scope does NOT already provide.
# Scope provides: torch, pydantic, numpy, pillow, and many more.
# dependencies = ["some-extra-package"]

[project.entry-points."scope"]
scope_PLUGIN_NAME = "scope_PLUGIN_NAME"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/scope_PLUGIN_NAME"]
```

Replace `PLUGIN_NAME` with the plugin name using underscores (e.g., `depth_anything`, `vfx`).

---

## Hook Registration (__init__.py)

```python
from scope.core.plugins.hookspecs import hookimpl


@hookimpl
def register_pipelines(register):
    from .pipeline import MyPipeline

    register(MyPipeline)
```

- `@hookimpl` marks this as a pluggy hook implementation
- `register()` is called once per pipeline class to add
- Use lazy imports (inside the function body) to delay loading heavy dependencies
- A single plugin can register multiple pipelines

---

## Pipeline Types

### Text-Only Pipeline

Generates video frames without any video input. Uses text prompts, parameters, or internal logic.

```python
class MyConfig(BasePipelineConfig):
    pipeline_id = "my-pipeline"
    pipeline_name = "My Pipeline"
    pipeline_description = "Generates frames from parameters"
    supports_prompts = False  # Set True if you use text prompts
    modes = {"text": ModeDefaults(default=True)}
```

- No `prepare()` method needed
- `__call__()` receives kwargs with runtime parameters (and `prompt` if `supports_prompts = True`)
- Must still return `{"video": tensor}` in THWC [0,1] format

### Video-Input Pipeline

Processes incoming video/camera frames. Appears in the main pipeline selector.

```python
class MyConfig(BasePipelineConfig):
    pipeline_id = "my-pipeline"
    pipeline_name = "My Pipeline"
    pipeline_description = "Processes video frames"
    supports_prompts = False
    modes = {"video": ModeDefaults(default=True)}
```

- Must implement `prepare()` returning `Requirements(input_size=N)`
- `__call__()` receives `video` in kwargs — a list of N tensors, each `(1, H, W, C)` in `[0, 255]`
- Normalize input: `frames / 255.0` to get `[0, 1]`

### Preprocessor

Same as video-input but appears in the Preprocessor dropdown, not the main pipeline selector. Runs BEFORE the main pipeline.

```python
class MyConfig(BasePipelineConfig):
    pipeline_id = "my-preprocessor"
    pipeline_name = "My Preprocessor"
    pipeline_description = "Transforms video before the main pipeline"
    supports_prompts = False
    modes = {"video": ModeDefaults(default=True)}
    usage = [UsageType.PREPROCESSOR]
```

### Multi-Mode Pipeline

Supports both text-to-video and video-to-video:

```python
class MyConfig(BasePipelineConfig):
    modes = {
        "text": ModeDefaults(default=True),
        "video": ModeDefaults(height=512, width=512, noise_scale=0.7),
    }
```

Use `prepare_for_mode()` helper:
```python
def prepare(self, **kwargs) -> Requirements | None:
    return prepare_for_mode(self.__class__, self.components.config, kwargs)
```

---

## Configuration Schema (BasePipelineConfig)

### Class Variables (Metadata)

| Variable | Type | Description |
|----------|------|-------------|
| `pipeline_id` | `str` | Unique identifier for registry lookup |
| `pipeline_name` | `str` | Human-readable display name |
| `pipeline_description` | `str` | Description shown in UI |
| `pipeline_version` | `str` | Semantic version (optional) |
| `docs_url` | `str \| None` | Link to documentation (optional) |
| `estimated_vram_gb` | `float \| None` | Estimated VRAM in GB (optional) |
| `supports_prompts` | `bool` | Whether pipeline accepts text prompts |
| `supports_lora` | `bool` | Whether pipeline supports LoRA adapters |
| `supports_vace` | `bool` | Whether pipeline supports VACE conditioning |
| `supports_cache_management` | `bool` | Whether cache controls are shown |
| `supports_quantization` | `bool` | Whether quantization selector is shown |
| `modes` | `dict` | Input modes (see Pipeline Types above) |
| `usage` | `list` | `[UsageType.PREPROCESSOR]` or `[]` |
| `artifacts` | `list[Artifact]` | Model files to auto-download |

### UI Parameter Fields

Every Pydantic field on the config class becomes a UI control. Scope's frontend reads the JSON Schema and auto-renders widgets:

| Python type | Schema property | UI Widget |
|-------------|----------------|-----------|
| `bool` | `type: "boolean"` | Toggle switch |
| `str` | `type: "string"` | Text input |
| `float` with `ge`/`le` | `type: "number"` with bounds | Slider |
| `float` without bounds | `type: "number"` | Number input |
| `int` with `ge`/`le` | `type: "integer"` with bounds | Slider |
| `Enum` or `Literal` | `enum` | Dropdown |

### Field Definition Pattern

```python
my_param: float = Field(
    default=0.5,
    ge=0.0,
    le=1.0,
    description="Tooltip text shown on hover in the UI",
    json_schema_extra=ui_field_config(
        order=1,               # Display position (lower = higher in panel)
        label="Short Label",   # Display name
        modes=["video"],       # Only show in specific modes (optional)
        is_load_param=True,    # True = set once at load, False = adjustable live
        category="configuration",  # "configuration" (Settings panel) or "input" (Input panel)
        component="resolution",    # Group with other fields into a complex widget (optional)
    ),
)
```

### ui_field_config() Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `order` | `int` | — | Display position in UI (lower = higher) |
| `label` | `str` | — | Short display label |
| `modes` | `list[str]` | all modes | Restrict to specific modes |
| `is_load_param` | `bool` | `False` | Load-time (True) vs runtime (False) |
| `category` | `str` | `"configuration"` | `"configuration"` or `"input"` |
| `component` | `str` | — | Group fields into complex widgets |

### Load-Time vs Runtime Parameters

| Type | `is_load_param` | Editable during streaming | Where to read |
|------|-----------------|--------------------------|---------------|
| Load-time | `True` | No (greyed out) | `__init__(**kwargs)` |
| Runtime | `False` (default) | Yes (live sliders) | `__call__(**kwargs)` via `kwargs.get()` |

**CRITICAL**: Runtime parameters MUST be read from `kwargs` in `__call__()`:

```python
# CORRECT — reads current slider value every frame
def __call__(self, **kwargs) -> dict:
    intensity = kwargs.get("intensity", 1.0)

# WRONG — always gets the default, ignores slider changes
def __init__(self, intensity: float = 1.0, **kwargs):
    self.intensity = intensity
```

---

## Pipeline Class Template

```python
from typing import TYPE_CHECKING

import torch

from scope.core.pipelines.interface import Pipeline, Requirements

from .schema import MyConfig

if TYPE_CHECKING:
    from scope.core.pipelines.base_schema import BasePipelineConfig


class MyPipeline(Pipeline):
    """Description of what this pipeline does."""

    @classmethod
    def get_config_class(cls) -> type["BasePipelineConfig"]:
        return MyConfig

    def __init__(self, device: torch.device | None = None, **kwargs):
        self.device = (
            device
            if device is not None
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        # Load models, initialize state, read load-time params here

    def prepare(self, **kwargs) -> Requirements:
        """Declare how many input frames we need (video-input/preprocessor only)."""
        return Requirements(input_size=1)

    def __call__(self, **kwargs) -> dict:
        """Process frames. Called once per frame batch during streaming.

        Input video: kwargs["video"] is a list of tensors, each (1, H, W, C) in [0, 255].
        Output: {"video": tensor} in THWC format, float32, [0, 1] range.
        """
        video = kwargs.get("video")
        if video is None:
            raise ValueError("This pipeline requires video input")

        # Stack and normalize: list of (1,H,W,C) -> (T,H,W,C) in [0,1]
        frames = torch.stack([frame.squeeze(0) for frame in video], dim=0)
        frames = frames.to(device=self.device, dtype=torch.float32) / 255.0

        # Read runtime params from kwargs
        my_param = kwargs.get("my_param", 0.5)

        # --- Processing logic here ---
        result = frames  # Replace with actual processing

        return {"video": result.clamp(0, 1)}
```

---

## Artifacts (Model Downloads)

Declare model files that Scope should auto-download before the pipeline loads.

### HuggingFace Hub

```python
from scope.core.pipelines.artifacts import HuggingfaceRepoArtifact

class MyConfig(BasePipelineConfig):
    artifacts = [
        HuggingfaceRepoArtifact(
            repo_id="depth-anything/Video-Depth-Anything-Small",
            files=["video_depth_anything_vits.pth"],
        ),
    ]
```

### Google Drive

```python
from scope.core.pipelines.artifacts import GoogleDriveArtifact

class MyConfig(BasePipelineConfig):
    artifacts = [
        GoogleDriveArtifact(
            file_id="1Smy6gY7BkS_RzCjPCbMEy-TsX8Ma5B0R",
            files=["flownet.pkl"],
            name="RIFE",
        ),
    ]
```

### Common Artifacts

Reuse shared artifacts that multiple pipelines need:

```python
from scope.core.pipelines.common_artifacts import WAN_1_3B_ARTIFACT, VACE_ARTIFACT

class MyConfig(BasePipelineConfig):
    artifacts = [WAN_1_3B_ARTIFACT, VACE_ARTIFACT]
```

---

## Input/Output Format Reference

### Input (video-input and preprocessor pipelines)

```python
video = kwargs.get("video")
# video is a list of N tensors (N = input_size from prepare())
# Each tensor: shape (1, H, W, C), dtype float32, range [0, 255]

# Standard normalization pattern:
frames = torch.stack([frame.squeeze(0) for frame in video], dim=0)  # -> (T, H, W, C)
frames = frames.to(device=self.device, dtype=torch.float32) / 255.0  # -> [0, 1]
```

### Output (all pipeline types)

```python
return {"video": tensor}
# tensor: shape (T, H, W, C) — THWC format
#   T = number of output frames (usually 1)
#   H = height in pixels
#   W = width in pixels
#   C = 3 (RGB)
# dtype: torch.float32
# range: [0, 1]
```

### Text prompts (if supports_prompts = True)

```python
prompt = kwargs.get("prompt", "")
```

---

## Directory Structure Patterns

### Minimal plugin (single pipeline, no models)

```
scope-my-plugin/
├── pyproject.toml
└── src/
    └── scope_my_plugin/
        ├── __init__.py      # hookimpl + register_pipelines
        ├── schema.py        # Config class
        └── pipeline.py      # Pipeline class
```

### Plugin with multiple effects/modules

```
scope-my-plugin/
├── pyproject.toml
├── README.md
└── src/
    └── scope_my_plugin/
        ├── __init__.py
        ├── schema.py
        ├── pipeline.py
        └── effects/
            ├── __init__.py
            ├── effect_one.py
            └── effect_two.py
```

### Plugin with AI model

```
scope-my-plugin/
├── pyproject.toml
├── README.md
└── src/
    └── scope_my_plugin/
        ├── __init__.py
        ├── schema.py
        ├── pipeline.py
        └── model/
            ├── __init__.py
            └── inference.py    # Model loading and inference wrapper
```

### Plugin with multiple pipelines

```
scope-my-plugin/
├── pyproject.toml
└── src/
    └── scope_my_plugin/
        ├── __init__.py         # register(PipelineA); register(PipelineB)
        ├── pipeline_a/
        │   ├── schema.py
        │   └── pipeline.py
        └── pipeline_b/
            ├── schema.py
            └── pipeline.py
```

---

## Troubleshooting

| Problem | Cause | Solution |
|---------|-------|---------|
| Plugin pipelines don't appear after install | Server didn't restart, or hook is incorrect | Ensure `register_pipelines` is decorated with `@hookimpl` and entry point module is correct |
| Parameters don't update during streaming | Reading params in `__init__` instead of `__call__` | Move `kwargs.get()` calls into `__call__()` |
| Installation fails with dependency conflict | Plugin declares a package that conflicts with Scope's environment | Remove the conflicting package from `[project.dependencies]` — Scope likely provides it already |
| Server won't start after install | Import error or bad dependency | Uninstall via Settings; if that fails, edit `~/.daydream-scope/plugins/plugins.txt` manually |
| Output is all black | Values not in [0, 1] range or wrong tensor format | Ensure output is THWC float32 in [0, 1], use `.clamp(0, 1)` |
| Output is all white | Input not normalized from [0, 255] to [0, 1] | Divide input by 255.0 |
