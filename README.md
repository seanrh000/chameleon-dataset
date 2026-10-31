# CHAMELEON - Stage One Engine

## Approach

This offline engine generates two layout candidates for each task: a rule-based layout (v0.4) and a locally trained Transformer layout with relation-aware refinement. Both candidates are validated and rendered with the supplied renderer. The deterministic `candidate_quality` heuristic (v3) scores text fit, readability, collisions, element relationships, hierarchy, margins, and foreground coverage, then selects a valid candidate. It also compares image/text prominence with the source and checks conservatively inferred text reading order, allowing vertical-to-horizontal reflow. Transparent image regions are excluded from relevant visual-bound estimates.

The selection score is an internal heuristic, not the official competition score or a calibrated probability of human preference. Small score differences and complex layouts still require visual review. The engine uses general layout rules and does not select results by task filename or sample ID.

Model weights are bundled at `engine/models/relations_v1.pt`. Inference runs on CPU; a GPU and the training datasets are not required. There are no runtime model downloads or API calls. The supplied image assets are reused without regenerating their pixels. Content and font family are retained while permitted layout fields are adjusted.

## Package contents

- `engine/`: runtime code, candidate selector, inference modules, and checkpoint.
- `renderer/`: supplied JSON-to-PNG renderer integration.
- `brand_kit/`: supplied assets and fonts, unchanged.
- `chameleon_design_schema_v0.9.json`: supplied schema, unchanged.
- `input/`: three supplied example tasks for a smoke test.
- `Dockerfile`, `.dockerignore`, `requirements.txt`: container build configuration.
- `MODEL.md`: model provenance and release status.


Training environments, datasets, editors, old engines, and comparison galleries are not included.

## Build

Run from this directory with Docker Desktop / Docker Engine running:

```sh
docker build -t chameleon-final .
```

Internet access is required during the build to install Python packages and system libraries. Runtime is offline. The Dockerfile does not force an architecture; use the organizer's required platform when it is confirmed, for example `docker build --platform linux/amd64 -t chameleon-final .` for amd64.

## Run offline

PowerShell:

```powershell
New-Item -ItemType Directory -Force output | Out-Null
$submissionRoot = (Get-Location).Path
docker run --rm --network none --mount "type=bind,source=$submissionRoot\input,target=/app/input,readonly" --mount "type=bind,source=$submissionRoot\output,target=/app/output" chameleon-final
```

Bash (Linux, WSL, macOS):

```sh
mkdir -p output
docker run --rm --network none \
  --mount "type=bind,source=$(pwd)/input,target=/app/input,readonly" \
  --mount "type=bind,source=$(pwd)/output,target=/app/output" \
  chameleon-final
```

Replace the host input directory with the evaluation task directory as needed. Input JSON files contain `source` and `target_canvas` (and may include `task_id`). The container runs `python -m engine.main` and produces `<input_stem>_output.json` and `<input_stem>_output.png` for each task. Output JSON contains `canvas` and `elements`. The output mount must be writable. Input/output directories are not baked into the container image.

## Optional local execution

With the requirements installed in a compatible Python 3.11 environment, run `python -m engine.main` from this directory. CPU Torch is sufficient. The default interpreter is used for inference; `CHAMELEON_TORCH_PYTHON` is an optional override for a separate compatible interpreter and is not needed in Docker.

To preserve both candidates and their selection reports, use:

```sh
python -m engine.resize_final --input input --output review-output
```

`review-output` must not already exist. This diagnostic command is optional: the standard container entry point also generates and selects between both candidates using v3. Candidate scores in `selection.json` include the scorer version and diagnostics for source-relative prominence and inferred text flow.
