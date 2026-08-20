# Paper Figure Splitter

An open-source Codex Skill plus deterministic CLI for extracting scientific-paper figures and splitting composite figures into clean, non-overlapping panels.

The workflow is intentionally conservative: clear whitespace gutters are split automatically; ambiguous layouts remain intact and are flagged for visual review. Exact pixel boxes can be supplied for insets, touching panels, shared axes, and other difficult designs.

## What it handles

- scientific PDFs, including vector figures rendered from the page;
- article webpages with `figure`/`img` assets and direct PDF links;
- individual PNG/JPEG/TIFF/WebP figures or directories of images;
- regular multi-panel grids separated by light gutters;
- manifests, annotated previews, contact sheets, captions, provenance, and quality warnings.

No general-purpose algorithm can infer every semantic panel boundary perfectly. In particular, inset panels and plots with shared axes can be genuinely ambiguous. This project makes those cases visible instead of silently damaging the figure.

## Install as a Codex Skill

Clone the repository into your Codex skills directory:

```bash
git clone https://github.com/dongyingshuai/paper-figure-splitter-pro.git \
  "${CODEX_HOME:-$HOME/.codex}/skills/paper-figure-splitter"
```

Restart Codex if needed. The skill can then be invoked as `$paper-figure-splitter` and may also be selected automatically for figure-extraction tasks.

## CLI usage

With [`uv`](https://docs.astral.sh/uv/):

```bash
uv run python scripts/paper_figure_splitter.py extract paper.pdf --output out
uv run python scripts/paper_figure_splitter.py extract "https://example.org/article" --output out
uv run python scripts/paper_figure_splitter.py split figure.png --output out
```

For an exact correction:

```bash
uv run python scripts/paper_figure_splitter.py split figure.png \
  --output corrected --boxes boxes.json
```

`boxes.json`:

```json
{
  "boxes": [
    {"label": "a", "x": 10, "y": 12, "width": 900, "height": 650},
    {"label": "b", "x": 930, "y": 12, "width": 900, "height": 650}
  ]
}
```

## Output

```text
out/
├── figures/                 whole figures
├── panels/<figure-id>/      panel crops
├── previews/                annotated originals and contact sheets
└── manifest.json            provenance, boxes, confidence, and warnings
```

## Development

```bash
uv sync --dev
uv run pytest
python /path/to/skill-creator/scripts/quick_validate.py .
```

## Copyright and access

The software is MIT-licensed. Figures extracted from papers remain subject to their original copyright and license. The downloader does not bypass logins, paywalls, or access controls.
