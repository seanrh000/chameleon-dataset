# Engine modules

See [the package README](../README.md) for setup, approach, verification limits, and submission status.

| Module | Purpose |
|---|---|
| `main.py` | Standard task-directory entry point; writes JSON and PNG outputs. |
| `resize.py` | Public resize function, connected to the final selector. |
| `resize_selector_v2.py` | Generates, validates, renders, scores, and selects two candidates. |
| `candidate_quality_v2.py` | Heuristic quality scoring with alpha-aware image bounds. |
| `resize_v0_4.py`, `effects.py` | Active rule-based candidate generator and effects handling. |
| `resize_relations_v2.py`, `relation_refiner_v3.py` | Transformer candidate and relationship refinement. |
| `inference_bridge.py` | Subprocess bridge to the bundled inference package. |
| `chameleon_torch/` | Inference model, feature encoding, geometry, and validation helpers. |
| `models/relations_v1.pt` | Bundled trained checkpoint. |
| `resize_final.py` | Optional diagnostic CLI that preserves candidates and selection records. |
| `RELEASE_MANIFEST.json` | SHA-256 hashes for engine files, excluding the manifest itself. |

Version suffixes identify active components, not unused backups. Both candidate generators are required by the selector. The diagnostic CLI is not required by the standard entry point.
