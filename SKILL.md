/opt/homebrew/Library/Homebrew/cmd/shellenv.sh: line 21: /bin/ps: Operation not permitted
---
name: paper-figure-splitter
description: Extract figures from scientific PDFs, article webpages, or image files and split composite figures into clean, non-overlapping panels. Use when preparing paper figures for slides, teaching, figure-by-figure explanation, or image datasets; keep intact any panel whose boundary is ambiguous.
---

# Paper Figure Splitter

Turn a paper or figure into traceable, presentation-ready PNG crops. Prefer a conservative intact figure over a confident-looking crop that cuts through content.

## Run the pipeline

Resolve this skill's directory as `SKILL_DIR`, then run:

```bash
uv run --project "$SKILL_DIR" python "$SKILL_DIR/scripts/paper_figure_splitter.py" \
  extract "<PDF-or-URL-or-image>" --output "<output-directory>"
```

If `uv` is unavailable, use a Python 3.10+ environment containing the dependencies from `pyproject.toml`.

The command writes:

- `figures/`: whole figures extracted from the source;
- `panels/<figure-id>/`: clean panel crops;
- `previews/`: annotated originals and contact sheets;
- `manifest.json`: source provenance, pixel bounding boxes, confidence, warnings, and file paths.

For an already extracted figure, use `split` instead of `extract`. For a directory of figures, pass the directory to `extract`.

## Inspect before delivery

Read `manifest.json`, then visually inspect every file in `previews/`. A valid delivery has all of these properties:

- every intended figure is represented;
- panel boxes do not overlap and do not include pixels from neighboring panels;
- panel labels, axes, legends, scale bars, and annotations stay with their panel;
- captions, page headers, body text, and unrelated panels are excluded;
- reading order is top-to-bottom and left-to-right;
- `needs_review` items are either corrected or explicitly reported.

Do not silently accept an automatic crop merely because a file was produced. When a boundary is ambiguous, keep the compound region intact and mark it for review.

## Correct ambiguous figures

For inset panels, shared axes, touching microscopy tiles, dark gutters, or decorative layouts, create a JSON override and rerun:

```json
{
  "boxes": [
    {"label": "a", "x": 12, "y": 18, "width": 840, "height": 620},
    {"label": "b", "x": 874, "y": 18, "width": 840, "height": 620}
  ]
}
```

```bash
uv run --project "$SKILL_DIR" python "$SKILL_DIR/scripts/paper_figure_splitter.py" \
  split "<figure.png>" --output "<output-directory>" --boxes "<boxes.json>"
```

Coordinates are pixels in the original image. Never guess them from a reduced chat preview: inspect the full-resolution annotated preview, determine exact boundaries, rerun, and inspect again.

Read [references/quality-protocol.md](references/quality-protocol.md) when correcting boundaries, handling difficult layouts, or preparing final assets for slides.

## Source rules

- Preserve the source URL or local path and caption in the manifest.
- Do not bypass authentication, paywalls, access controls, or publisher download restrictions.
- Treat extracted figures as source material whose reuse may require permission or attribution; do not imply that extraction changes copyright status.
- Never upscale by default. Preserve native pixels or the configured PDF render DPI.
