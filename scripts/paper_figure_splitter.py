#!/usr/bin/env python3
"""Extract scientific-paper figures and conservatively split composite panels."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urljoin, urlparse

import cv2
import pymupdf as fitz
import numpy as np
import requests
from bs4 import BeautifulSoup
from PIL import Image, ImageDraw, ImageFont, UnidentifiedImageError


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".webp", ".bmp"}
CAPTION_RE = re.compile(
    r"^\s*((?:extended\s+data\s+|supplementary\s+)?fig(?:ure)?\.?)\s*([sS]?\d+[A-Za-z]?)",
    re.IGNORECASE,
)
USER_AGENT = "paper-figure-splitter/0.1 (+https://github.com/dongyingshuai/paper-figure-splitter)"


@dataclass(frozen=True)
class Box:
    x: int
    y: int
    width: int
    height: int
    label: str = ""

    @property
    def right(self) -> int:
        return self.x + self.width

    @property
    def bottom(self) -> int:
        return self.y + self.height

    def as_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "x": self.x,
            "y": self.y,
            "width": self.width,
            "height": self.height,
        }


@dataclass(frozen=True)
class SplitCandidate:
    axis: str
    start: int
    end: int
    score: float
    gutter_occupancy: float


def panel_label(index: int) -> str:
    label = ""
    value = index
    while True:
        value, remainder = divmod(value, 26)
        label = chr(ord("a") + remainder) + label
        if value == 0:
            return label
        value -= 1


def safe_stem(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-._")
    return value[:80] or "figure"


def read_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"))


def save_rgb(path: Path, rgb: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb).save(path, format="PNG", optimize=True)


def foreground_mask(rgb: np.ndarray) -> np.ndarray:
    """Estimate foreground relative to the image border; fail closed on dark full-bleed art."""
    height, width = rgb.shape[:2]
    band = max(1, min(height, width) // 100)
    border = np.concatenate(
        [
            rgb[:band].reshape(-1, 3),
            rgb[-band:].reshape(-1, 3),
            rgb[:, :band].reshape(-1, 3),
            rgb[:, -band:].reshape(-1, 3),
        ]
    ).astype(np.float32)
    background = np.median(border, axis=0)
    distance = np.linalg.norm(rgb.astype(np.float32) - background, axis=2)
    border_spread = float(np.median(np.linalg.norm(border - background, axis=1)))
    tolerance = max(14.0, min(35.0, border_spread * 3.0 + 8.0))
    mask = distance > tolerance

    # On ordinary white paper, retain faint gray axes and labels.
    if float(background.mean()) > 210:
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        mask |= gray < 244

    coverage = float(mask.mean())
    if coverage < 0.002:
        return np.zeros((height, width), dtype=bool)
    if coverage > 0.97:
        return np.ones((height, width), dtype=bool)

    kernel = np.ones((2, 2), np.uint8)
    return cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, kernel).astype(bool)


def content_box(mask: np.ndarray, padding: int = 8) -> Box:
    height, width = mask.shape
    min_row_ink = max(2, int(width * 0.001))
    min_col_ink = max(2, int(height * 0.001))
    rows = np.flatnonzero(mask.sum(axis=1) >= min_row_ink)
    cols = np.flatnonzero(mask.sum(axis=0) >= min_col_ink)
    if not len(rows) or not len(cols):
        return Box(0, 0, width, height)
    x0 = max(0, int(cols[0]) - padding)
    y0 = max(0, int(rows[0]) - padding)
    x1 = min(width, int(cols[-1]) + padding + 1)
    y1 = min(height, int(rows[-1]) + padding + 1)
    return Box(x0, y0, x1 - x0, y1 - y0)


def _runs(values: np.ndarray) -> Iterable[tuple[int, int]]:
    start: int | None = None
    for index, value in enumerate(values.tolist() + [False]):
        if value and start is None:
            start = index
        elif not value and start is not None:
            yield start, index
            start = None


def best_gutter(
    mask: np.ndarray,
    box: Box,
    *,
    min_panel_fraction: float,
    max_gutter_occupancy: float,
) -> SplitCandidate | None:
    crop = mask[box.y : box.bottom, box.x : box.right]
    candidates: list[SplitCandidate] = []
    for axis in ("vertical", "horizontal"):
        occupancy = crop.mean(axis=0 if axis == "vertical" else 1)
        length = len(occupancy)
        other = crop.shape[0] if axis == "vertical" else crop.shape[1]
        min_side = max(36, int(length * min_panel_fraction))
        min_gutter = max(4, int(length * 0.006))
        low = occupancy <= max_gutter_occupancy
        for start, end in _runs(low):
            gutter = end - start
            if gutter < min_gutter or start < min_side or length - end < min_side:
                continue
            left = crop[:, :start] if axis == "vertical" else crop[:start, :]
            right = crop[:, end:] if axis == "vertical" else crop[end:, :]
            if left.mean() < 0.008 or right.mean() < 0.008:
                continue
            balance = min(start, length - end) / max(start, length - end)
            mean_occupancy = float(occupancy[start:end].mean())
            score = (gutter / length) * 4.0 + balance * 0.15
            score += max(0.0, max_gutter_occupancy - mean_occupancy) * 2.0
            # A gutter should have enough orthogonal extent to be meaningful.
            if other < 60:
                score *= 0.5
            candidates.append(
                SplitCandidate(axis, start, end, score, mean_occupancy)
            )
    return max(candidates, key=lambda candidate: candidate.score, default=None)


def auto_boxes(
    rgb: np.ndarray,
    *,
    max_panels: int = 16,
    min_panel_fraction: float = 0.16,
    max_gutter_occupancy: float = 0.012,
) -> tuple[list[Box], list[float]]:
    mask = foreground_mask(rgb)
    root = content_box(mask)
    leaves: list[Box] = []
    gutter_occupancies: list[float] = []

    def visit(box: Box) -> None:
        if len(leaves) >= max_panels - 1:
            leaves.append(box)
            return
        candidate = best_gutter(
            mask,
            box,
            min_panel_fraction=min_panel_fraction,
            max_gutter_occupancy=max_gutter_occupancy,
        )
        if candidate is None:
            leaves.append(box)
            return
        boundary = (candidate.start + candidate.end) // 2
        gutter_occupancies.append(candidate.gutter_occupancy)
        if candidate.axis == "vertical":
            visit(Box(box.x, box.y, boundary, box.height))
            visit(Box(box.x + boundary, box.y, box.width - boundary, box.height))
        else:
            visit(Box(box.x, box.y, box.width, boundary))
            visit(Box(box.x, box.y + boundary, box.width, box.height - boundary))

    visit(root)

    tightened: list[Box] = []
    for box in leaves:
        local = mask[box.y : box.bottom, box.x : box.right]
        local_box = content_box(local, padding=8)
        tightened.append(
            Box(box.x + local_box.x, box.y + local_box.y, local_box.width, local_box.height)
        )
    tightened.sort(key=lambda box: (round(box.y / max(1, root.height // 20)), box.x, box.y))
    return [Box(b.x, b.y, b.width, b.height, panel_label(i)) for i, b in enumerate(tightened)], gutter_occupancies


def boxes_overlap(first: Box, second: Box) -> bool:
    return not (
        first.right <= second.x
        or second.right <= first.x
        or first.bottom <= second.y
        or second.bottom <= first.y
    )


def load_manual_boxes(path: Path, image_size: tuple[int, int]) -> list[Box]:
    data = json.loads(path.read_text(encoding="utf-8"))
    raw_boxes = data.get("boxes") if isinstance(data, dict) else data
    if not isinstance(raw_boxes, list) or not raw_boxes:
        raise ValueError("boxes JSON must contain a non-empty 'boxes' array")
    width, height = image_size
    boxes: list[Box] = []
    for index, item in enumerate(raw_boxes):
        try:
            box = Box(
                int(item["x"]),
                int(item["y"]),
                int(item["width"]),
                int(item["height"]),
                str(item.get("label") or panel_label(index)),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"invalid box at index {index}: {item!r}") from error
        if box.x < 0 or box.y < 0 or box.width <= 0 or box.height <= 0:
            raise ValueError(f"box {box.label!r} has invalid coordinates")
        if box.right > width or box.bottom > height:
            raise ValueError(f"box {box.label!r} exceeds the {width}x{height} image")
        if any(boxes_overlap(box, previous) for previous in boxes):
            raise ValueError(f"box {box.label!r} overlaps another manual box")
        boxes.append(box)
    return boxes


def annotated_preview(rgb: np.ndarray, boxes: list[Box]) -> Image.Image:
    image = Image.fromarray(rgb).convert("RGB")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=max(12, min(rgb.shape[:2]) // 45))
    colors = ["#ef4444", "#22c55e", "#3b82f6", "#f59e0b", "#a855f7", "#06b6d4"]
    line_width = max(2, min(rgb.shape[:2]) // 300)
    for index, box in enumerate(boxes):
        color = colors[index % len(colors)]
        draw.rectangle((box.x, box.y, box.right - 1, box.bottom - 1), outline=color, width=line_width)
        text = box.label
        text_box = draw.textbbox((box.x, box.y), text, font=font, stroke_width=1)
        draw.rectangle(text_box, fill="white")
        draw.text((box.x, box.y), text, fill=color, font=font, stroke_width=1, stroke_fill="white")
    return image


def contact_sheet(crops: list[tuple[str, Image.Image]]) -> Image.Image:
    thumb_width = 520
    prepared: list[tuple[str, Image.Image]] = []
    for label, crop in crops:
        thumb = crop.copy()
        thumb.thumbnail((thumb_width, 420), Image.Resampling.LANCZOS)
        prepared.append((label, thumb))
    columns = 2 if len(prepared) > 1 else 1
    rows = (len(prepared) + columns - 1) // columns
    cell_width = thumb_width + 30
    cell_height = max(thumb.height for _, thumb in prepared) + 58
    sheet = Image.new("RGB", (columns * cell_width, rows * cell_height), "#f3f4f6")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default(size=18)
    for index, (label, thumb) in enumerate(prepared):
        x = (index % columns) * cell_width + 15
        y = (index // columns) * cell_height + 36
        sheet.paste(thumb, (x, y))
        draw.text((x, 10 + (index // columns) * cell_height), f"panel {label}", fill="black", font=font)
    return sheet


def split_figure(
    image_path: Path,
    output_root: Path,
    figure_id: str,
    *,
    boxes_path: Path | None = None,
    caption: str = "",
    provenance: dict[str, Any] | None = None,
    max_panels: int = 16,
) -> dict[str, Any]:
    rgb = read_rgb(image_path)
    height, width = rgb.shape[:2]
    if boxes_path:
        boxes = load_manual_boxes(boxes_path, (width, height))
        occupancies: list[float] = []
        method = "manual_boxes"
    else:
        boxes, occupancies = auto_boxes(rgb, max_panels=max_panels)
        method = "whitespace_gutters"

    panel_dir = output_root / "panels" / figure_id
    preview_dir = output_root / "previews"
    panel_dir.mkdir(parents=True, exist_ok=True)
    preview_dir.mkdir(parents=True, exist_ok=True)
    for stale_crop in panel_dir.glob(f"{figure_id}_*.png"):
        stale_crop.unlink()

    panels: list[dict[str, Any]] = []
    crops: list[tuple[str, Image.Image]] = []
    for box in boxes:
        crop = Image.fromarray(rgb[box.y : box.bottom, box.x : box.right])
        panel_path = panel_dir / f"{figure_id}_{box.label}.png"
        crop.save(panel_path, format="PNG", optimize=True)
        crops.append((box.label, crop))
        panels.append({**box.as_dict(), "path": str(panel_path.relative_to(output_root))})

    annotated_preview(rgb, boxes).save(preview_dir / f"{figure_id}_annotated.png")
    contact_sheet(crops).save(preview_dir / f"{figure_id}_contact-sheet.png")

    warnings: list[str] = []
    needs_review = False
    if len(boxes) >= max_panels:
        warnings.append(f"panel limit ({max_panels}) reached; the figure may be over-segmented")
        needs_review = True
    if occupancies and max(occupancies) > 0.006:
        warnings.append("one or more cuts use a weak gutter; inspect the annotated preview")
        needs_review = True
    confidence = "manual" if boxes_path else ("high" if len(boxes) > 1 and not needs_review else "conservative")

    return {
        "figure_id": figure_id,
        "whole_figure": str(image_path.relative_to(output_root)),
        "caption": caption.strip(),
        "provenance": provenance or {},
        "width": width,
        "height": height,
        "split_method": method,
        "confidence": confidence,
        "needs_review": needs_review,
        "warnings": warnings,
        "panels": panels,
        "annotated_preview": f"previews/{figure_id}_annotated.png",
        "contact_sheet": f"previews/{figure_id}_contact-sheet.png",
    }


def copy_as_png(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(source) as image:
        image.convert("RGB").save(destination, format="PNG", optimize=True)


def _column_bounds(page: fitz.Page, caption_box: tuple[float, float, float, float]) -> tuple[float, float]:
    width = page.rect.width
    x0, _, x1, _ = caption_box
    margin = width * 0.035
    center = (x0 + x1) / 2
    if x1 <= width * 0.62 and center < width * 0.48:
        return margin, width * 0.505
    if x0 >= width * 0.38 and center > width * 0.52:
        return width * 0.495, width - margin
    return margin, width - margin


def _figure_top(page: fitz.Page, caption_box: tuple[float, float, float, float]) -> float:
    x0, caption_y, x1, _ = caption_box
    candidates: list[float] = []
    for block in page.get_text("blocks"):
        bx0, by0, bx1, by1, text = block[:5]
        clean = " ".join(str(text).split())
        overlap = max(0.0, min(x1, bx1) - max(x0, bx0))
        if by1 >= caption_y - 8 or overlap <= 0 or len(clean) < 90 or CAPTION_RE.match(clean):
            continue
        if by1 - by0 > page.rect.height * 0.18:
            continue
        candidates.append(float(by1 + 4))
    top = max(candidates, default=page.rect.height * 0.035)
    if caption_y - top < 72:
        top = max(page.rect.height * 0.035, caption_y - page.rect.height * 0.42)
    return top


def extract_pdf(pdf_path: Path, figures_dir: Path, dpi: int) -> list[dict[str, Any]]:
    document = fitz.open(pdf_path)
    extracted: list[dict[str, Any]] = []
    figure_number = 0
    for page_index, page in enumerate(document):
        captions = []
        for block in page.get_text("blocks"):
            text = " ".join(str(block[4]).split())
            match = CAPTION_RE.match(text)
            if match:
                captions.append((block[:4], text, match.group(2)))
        for caption_box, caption, source_label in sorted(captions, key=lambda item: item[0][1]):
            x0, x1 = _column_bounds(page, caption_box)
            top = _figure_top(page, (x0, caption_box[1], x1, caption_box[3]))
            clip = fitz.Rect(x0, top, x1, max(top + 1, caption_box[1] - 2))
            pixmap = page.get_pixmap(matrix=fitz.Matrix(dpi / 72, dpi / 72), clip=clip, alpha=False)
            rgb = np.frombuffer(pixmap.samples, dtype=np.uint8).reshape(pixmap.height, pixmap.width, pixmap.n)[..., :3]
            tight = content_box(foreground_mask(rgb), padding=max(8, dpi // 24))
            rgb = rgb[tight.y : tight.bottom, tight.x : tight.right]
            if rgb.shape[0] < 100 or rgb.shape[1] < 100:
                continue
            figure_number += 1
            figure_id = f"figure_{figure_number:03d}"
            destination = figures_dir / f"{figure_id}.png"
            save_rgb(destination, rgb)
            extracted.append(
                {
                    "figure_id": figure_id,
                    "path": destination,
                    "caption": caption,
                    "provenance": {
                        "kind": "pdf_caption_region",
                        "page": page_index + 1,
                        "source_figure_label": source_label,
                        "pdf_clip_points": [round(clip.x0, 2), round(clip.y0, 2), round(clip.x1, 2), round(clip.y1, 2)],
                        "render_dpi": dpi,
                    },
                }
            )

    if extracted:
        return extracted

    # Image-only PDFs and unusual caption typography: retain large embedded assets.
    seen: set[str] = set()
    for page_index, page in enumerate(document):
        for image_info in page.get_images(full=True):
            xref = image_info[0]
            payload = document.extract_image(xref)
            digest = hashlib.sha256(payload["image"]).hexdigest()
            if digest in seen:
                continue
            seen.add(digest)
            try:
                with Image.open(io.BytesIO(payload["image"])) as image:
                    if image.width < 240 or image.height < 180:
                        continue
                    figure_number += 1
                    figure_id = f"figure_{figure_number:03d}"
                    destination = figures_dir / f"{figure_id}.png"
                    image.convert("RGB").save(destination, format="PNG", optimize=True)
            except UnidentifiedImageError:
                continue
            extracted.append(
                {
                    "figure_id": figure_id,
                    "path": destination,
                    "caption": "",
                    "provenance": {"kind": "pdf_embedded_image", "page": page_index + 1, "xref": xref},
                }
            )
    return extracted


def _pick_image_url(tag: Any, base_url: str) -> str | None:
    for attribute in ("data-original", "data-src", "data-lazy-src"):
        if tag.get(attribute):
            return urljoin(base_url, tag[attribute])
    srcset = tag.get("srcset") or tag.get("data-srcset")
    if not srcset and tag.parent and getattr(tag.parent, "name", None) == "picture":
        source = tag.parent.find("source", srcset=True)
        srcset = source.get("srcset") if source else None
    if srcset:
        candidates = [part.strip().split()[0] for part in srcset.split(",") if part.strip()]
        if candidates:
            return urljoin(base_url, candidates[-1])
    if tag.get("src"):
        return urljoin(base_url, tag["src"])
    return None


def _download(session: requests.Session, url: str, max_bytes: int = 80_000_000) -> bytes:
    response = session.get(url, timeout=35, stream=True)
    response.raise_for_status()
    chunks: list[bytes] = []
    size = 0
    for chunk in response.iter_content(1024 * 256):
        size += len(chunk)
        if size > max_bytes:
            raise ValueError(f"download exceeds {max_bytes} bytes: {url}")
        chunks.append(chunk)
    return b"".join(chunks)


def extract_web(url: str, figures_dir: Path, dpi: int) -> tuple[list[dict[str, Any]], list[str]]:
    session = requests.Session()
    session.headers["User-Agent"] = USER_AGENT
    response = session.get(url, timeout=35)
    response.raise_for_status()
    content_type = response.headers.get("content-type", "").lower()
    warnings: list[str] = []
    if "application/pdf" in content_type or urlparse(response.url).path.lower().endswith(".pdf"):
        with tempfile.NamedTemporaryFile(suffix=".pdf") as handle:
            handle.write(response.content)
            handle.flush()
            return extract_pdf(Path(handle.name), figures_dir, dpi), warnings

    soup = BeautifulSoup(response.text, "html.parser")
    tags = list(soup.select("figure img"))
    if not tags:
        tags = list(soup.find_all("img"))
    extracted: list[dict[str, Any]] = []
    seen: set[str] = set()
    for tag in tags:
        image_url = _pick_image_url(tag, response.url)
        if not image_url or image_url.startswith("data:"):
            continue
        try:
            payload = _download(session, image_url)
            digest = hashlib.sha256(payload).hexdigest()
            if digest in seen:
                continue
            with Image.open(io.BytesIO(payload)) as image:
                if image.width < 240 or image.height < 180 or image.width * image.height < 80_000:
                    continue
                seen.add(digest)
                figure_id = f"figure_{len(extracted) + 1:03d}"
                destination = figures_dir / f"{figure_id}.png"
                image.convert("RGB").save(destination, format="PNG", optimize=True)
        except (requests.RequestException, UnidentifiedImageError, OSError, ValueError) as error:
            warnings.append(f"skipped {image_url}: {error}")
            continue
        figure = tag.find_parent("figure")
        caption_tag = figure.find("figcaption") if figure else None
        caption = " ".join(caption_tag.get_text(" ", strip=True).split()) if caption_tag else ""
        extracted.append(
            {
                "figure_id": figure_id,
                "path": destination,
                "caption": caption,
                "provenance": {"kind": "web_image", "page_url": response.url, "image_url": image_url},
            }
        )
    if not extracted:
        warnings.append("no sufficiently large article images were found; try the article's direct PDF URL")
    return extracted, warnings


def extract_local(source: Path, figures_dir: Path) -> list[dict[str, Any]]:
    paths = [source] if source.is_file() else sorted(path for path in source.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES)
    extracted: list[dict[str, Any]] = []
    for index, path in enumerate(paths, start=1):
        if path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        figure_id = f"figure_{index:03d}_{safe_stem(path.stem)}"
        destination = figures_dir / f"{figure_id}.png"
        copy_as_png(path, destination)
        extracted.append(
            {
                "figure_id": figure_id,
                "path": destination,
                "caption": "",
                "provenance": {"kind": "local_image", "source_path": str(path.resolve())},
            }
        )
    return extracted


def prepare_output(output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    (output / "figures").mkdir(exist_ok=True)


def write_manifest(output: Path, source: str, figures: list[dict[str, Any]], warnings: list[str]) -> Path:
    manifest = {
        "schema_version": 1,
        "source": source,
        "figure_count": len(figures),
        "panel_count": sum(len(figure["panels"]) for figure in figures),
        "needs_review": any(figure["needs_review"] for figure in figures),
        "warnings": warnings,
        "figures": figures,
    }
    path = output / "manifest.json"
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def run_extract(args: argparse.Namespace) -> int:
    output = args.output.resolve()
    prepare_output(output)
    source = args.source
    warnings: list[str] = []
    if re.match(r"^https?://", source, re.IGNORECASE):
        extracted, warnings = extract_web(source, output / "figures", args.dpi)
    else:
        source_path = Path(source).expanduser().resolve()
        if not source_path.exists():
            raise FileNotFoundError(source_path)
        if source_path.suffix.lower() == ".pdf":
            extracted = extract_pdf(source_path, output / "figures", args.dpi)
        else:
            extracted = extract_local(source_path, output / "figures")
    results = [
        split_figure(
            item["path"],
            output,
            item["figure_id"],
            caption=item["caption"],
            provenance=item["provenance"],
            max_panels=args.max_panels,
        )
        for item in extracted
    ]
    if not results:
        warnings.append("no figures were extracted")
    manifest = write_manifest(output, source, results, warnings)
    print(f"Extracted {len(results)} figure(s); manifest: {manifest}")
    return 2 if not results else 0


def run_split(args: argparse.Namespace) -> int:
    output = args.output.resolve()
    prepare_output(output)
    source = args.image.expanduser().resolve()
    if not source.exists() or source.suffix.lower() not in IMAGE_SUFFIXES:
        raise ValueError(f"not a supported image: {source}")
    figure_id = safe_stem(source.stem)
    destination = output / "figures" / f"{figure_id}.png"
    copy_as_png(source, destination)
    result = split_figure(
        destination,
        output,
        figure_id,
        boxes_path=args.boxes.expanduser().resolve() if args.boxes else None,
        provenance={"kind": "local_image", "source_path": str(source)},
        max_panels=args.max_panels,
    )
    manifest = write_manifest(output, str(source), [result], [])
    print(f"Wrote {len(result['panels'])} panel(s); manifest: {manifest}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    extract = subparsers.add_parser("extract", help="extract figures and split their panels")
    extract.add_argument("source", help="PDF, article URL, image, or directory of images")
    extract.add_argument("--output", type=Path, required=True)
    extract.add_argument("--dpi", type=int, default=240, help="PDF rendering DPI (default: 240)")
    extract.add_argument("--max-panels", type=int, default=16)
    extract.set_defaults(handler=run_extract)

    split = subparsers.add_parser("split", help="split one already-extracted figure")
    split.add_argument("image", type=Path)
    split.add_argument("--output", type=Path, required=True)
    split.add_argument("--boxes", type=Path, help="JSON file with exact pixel boxes")
    split.add_argument("--max-panels", type=int, default=16)
    split.set_defaults(handler=run_split)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if getattr(args, "dpi", 240) < 96 or getattr(args, "dpi", 240) > 600:
        parser.error("--dpi must be between 96 and 600")
    if args.max_panels < 1 or args.max_panels > 64:
        parser.error("--max-panels must be between 1 and 64")
    try:
        return int(args.handler(args))
    except (OSError, ValueError, requests.RequestException, fitz.FileDataError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
