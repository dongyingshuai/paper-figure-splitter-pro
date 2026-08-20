# Figure and panel quality protocol

Use this protocol only for visual review or correction of difficult layouts.

## Acceptance checklist

Inspect the annotated original and each exported crop at full resolution.

1. **Coverage** — Compare figure numbers and captions in `manifest.json` with the source. Investigate gaps rather than renumbering around them.
2. **Containment** — Each crop contains its full panel label, axes, tick labels, legend, scale bar, error bars, annotations, and relevant callouts.
3. **Isolation** — No crop contains meaningful marks, labels, axes, or image pixels from a neighboring panel.
4. **Geometry** — Boxes are non-overlapping, inside the original image, and ordered by rows then columns.
5. **Clean edge** — A cut should normally pass through a genuine gutter. If a cut crosses foreground pixels, inspect it closely.
6. **Traceability** — Keep the whole extracted figure, the annotated preview, the crops, and the manifest together.

## Difficult layouts

- **Shared axes or shared legends:** keep the related panels together unless the user explicitly wants duplication or a separate legend crop.
- **Inset panels:** keep the inset with its parent by default. Export it separately only when its border is unambiguous and the parent crop remains useful.
- **Touching microscopy tiles:** use exact manual boxes; do not trim into scale bars or panel letters.
- **Non-white or textured gutters:** automatic whitespace splitting may deliberately keep the figure intact. Use manual boxes.
- **Irregular arrangements:** use the fewest rectangular crops that preserve meaning. Do not force a grid.
- **Panel labels outside image tiles:** expand the box into the surrounding whitespace just enough to retain the label without capturing the neighbor.

## Manual-box procedure

1. Open the full-resolution annotated preview.
2. Locate gutters in original-image pixels using the preview axes or an image viewer.
3. Write boxes as `{label, x, y, width, height}` in a JSON `boxes` array.
4. Rerun `split --boxes`.
5. Reinspect both the new preview and every crop.

If no rectangle can isolate a panel without losing shared context, keep the compound group intact and describe why in the handoff.
