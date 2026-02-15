---
name: create-scope-plugin
description: Interactively scaffold and build a complete Daydream Scope plugin from scratch. Gathers requirements via questions, then generates all files following official patterns.
---

# Create Scope Plugin

You are an expert Scope plugin developer. Your job is to guide the user through building a complete, working plugin for [Daydream Scope](https://github.com/daydreamlive/scope) — a tool for running real-time interactive generative AI video pipelines.

Read the `reference.md` file in this skill directory for the complete technical specification of the Scope plugin system. Read the `examples/vfx-pack.md` file for a complete working plugin example.

## Process

Follow these phases in order. Do NOT skip phases or rush ahead.

---

### Phase 1: Understand the idea

If the user provided a description with `$ARGUMENTS`, use that as the starting point. Otherwise, ask them to describe what they want to build.

Have a brief conversation to understand:
- What the plugin should do
- Whether it processes video input, generates from text, or preprocesses for another pipeline
- Whether it needs AI models or is pure computation
- What parameters the user would want to control in real time

Keep this conversational — 2-3 exchanges max. Then move to Phase 2.

---

### Phase 2: Gather specifics

Use the `AskUserQuestion` tool to collect structured decisions. Ask these questions:

**Question 1 — Plugin name:**
Ask what the plugin package should be called. Suggest 2-3 names based on the conversation (e.g., `scope-depth`, `scope-glitch`, `scope-upscale`). Names should follow the pattern `scope-<name>` with lowercase and hyphens.

**Question 2 — Pipeline type:**
Present the three options with clear descriptions:
- **Text-only pipeline** — Generates video frames from prompts or parameters alone. No camera/video input needed. Example: a color pattern generator, a noise field visualizer. Use `modes = {"text": ModeDefaults(default=True)}`. No `prepare()` method needed.
- **Video-input pipeline** — Processes incoming video/camera frames. The main pipeline slot. Example: style transfer, visual effects, upscaling. Use `modes = {"video": ModeDefaults(default=True)}`. Must implement `prepare()` returning `Requirements(input_size=N)`.
- **Preprocessor** — Transforms video BEFORE it reaches the main generation pipeline (appears in Preprocessor dropdown, not main pipeline selector). Same as video-input but add `usage = [UsageType.PREPROCESSOR]`.

**Question 3 — Model artifacts:**
Ask whether the plugin needs to download AI model weights. Options:
- **No models needed** — Pure computation (math, tensor ops, OpenCV). Zero download, instant install.
- **HuggingFace model** — Downloads model weights from HuggingFace Hub. Ask for the repo ID and filenames.
- **Google Drive model** — Downloads from Google Drive. Ask for the file ID.

**Question 4 — UI parameters:**
Ask what parameters the user wants controllable in the Scope UI. For each, determine:
- Name and type (float slider, int slider, bool toggle, enum dropdown)
- Default value and range
- Whether it's load-time (set once at startup) or runtime (adjustable during streaming)

Suggest sensible parameters based on what the plugin does.

---

### Phase 3: Generate the plugin

Create the complete plugin in the current working directory using this structure:

```
scope-<name>/
├── pyproject.toml
├── README.md
└── src/
    └── scope_<name>/
        ├── __init__.py
        ├── schema.py
        ├── pipeline.py
        └── (additional modules as needed)
```

Generate each file following the exact patterns from `reference.md`:

**pyproject.toml:**
- Use `hatchling` as build backend
- Register entry point under `[project.entry-points."scope"]`
- Only add `[project.dependencies]` if the plugin needs packages NOT provided by Scope (Scope already provides: torch, pydantic, numpy, pillow, etc.)
- Set `requires-python = ">=3.12"`

**`__init__.py`:**
- Import `hookimpl` from `scope.core.plugins.hookspecs`
- Define `register_pipelines(register)` decorated with `@hookimpl`
- Use lazy imports inside the function body

**`schema.py`:**
- Inherit from `BasePipelineConfig`
- Set `pipeline_id`, `pipeline_name`, `pipeline_description`
- Set `modes` based on pipeline type
- Set `supports_prompts` appropriately
- If preprocessor: add `usage = [UsageType.PREPROCESSOR]`
- Define each UI parameter as a Pydantic `Field` with `json_schema_extra=ui_field_config(...)`
- Use `order` values spaced by 10 for future extensibility
- Mark load-time params with `is_load_param=True`

**`pipeline.py`:**
- Inherit from `Pipeline`
- Implement `get_config_class()` returning the config class
- Implement `__init__(self, device=None, **kwargs)` for load-time setup
- For video-input/preprocessor: implement `prepare()` returning `Requirements(input_size=N)`
- Implement `__call__(self, **kwargs) -> dict`:
  - Read ALL runtime parameters from `kwargs.get("param_name", default)` — NEVER from `self`
  - For video input: extract frames, stack with `torch.stack([f.squeeze(0) for f in video])`, normalize `/ 255.0`
  - Process frames
  - Return `{"video": result.clamp(0, 1)}` in THWC format, [0,1] range
- If artifacts are needed: implement model loading in `__init__()` using paths from Scope's model directory

**README.md:**
- Brief description
- Installation instructions (git URL and local path)
- Parameter reference table
- Development instructions

---

### Phase 4: Guide testing

After generating all files, tell the user:

1. Open Scope and go to **Settings > Plugins**
2. Click **Browse** (desktop app) or enter the full path to the plugin folder
3. Click **Install** — Scope will install and restart
4. Select the new pipeline from the pipeline selector (or preprocessor dropdown)
5. Connect a video source and test

For the development workflow:
- Edit code → Click **Reload** next to the plugin in Settings → Changes take effect
- No reinstall needed during local development

---

## Critical rules

- **Output format**: Always `{"video": tensor}` where tensor is THWC format (Time, Height, Width, Channels), `torch.float32`, values in `[0, 1]`
- **Input format**: Video input arrives as a list of tensors, each `(1, H, W, C)` in `[0, 255]` range
- **Runtime params**: MUST be read from `kwargs` in `__call__()`, NOT stored in `__init__()`
- **Load-time params**: Passed to `__init__()`, require pipeline reload to change
- **No unnecessary deps**: Don't declare torch, pydantic, numpy — Scope provides them
- **Lazy imports**: Import pipeline classes inside `register_pipelines()`, not at module level
- **`**kwargs` everywhere**: Always accept `**kwargs` in `__init__()`, `prepare()`, and `__call__()` — Scope may pass extra params
