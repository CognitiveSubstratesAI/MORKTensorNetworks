# Documentation image assets

[`../architecture.md`](../architecture.md) embeds two diagrams from this folder. Save them
here with these exact names:

| File | Diagram |
|---|---|
| `categories.png` | 5-branch category mind-map (Path Algebra / ShardZipper / Tensor Logic / GPU Optimization / HRT) |
| `architecture.png` | 4-layer stack "Bridging Symbolic Path Algebra and GPU Tensor Logic" (Symbolic Summit → Logic Engine → Data Pipeline → Hardware Acceleration) |

Both are source-paper / NotebookLM conceptual diagrams (Goertzel, Oct 2025). They illustrate
intent; the per-component implementation + audit status is the table in `architecture.md`
and the full record in `docs/AUDIT_2026-06-04.md`.

Documenter copies `docs/src/assets/` into the built site, so `![](assets/categories.png)`
resolves on the published GitHub Pages docs.
