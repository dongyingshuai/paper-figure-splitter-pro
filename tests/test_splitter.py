import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pymupdf as fitz
import pytest
from PIL import Image, ImageDraw


SCRIPT = Path(__file__).parents[1] / "scripts" / "paper_figure_splitter.py"
SPEC = importlib.util.spec_from_file_location("paper_figure_splitter", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def composite(rows: int, columns: int, panel=(280, 210), gutter=36) -> np.ndarray:
    width = columns * panel[0] + (columns - 1) * gutter + 40
    height = rows * panel[1] + (rows - 1) * gutter + 40
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    index = 0
    for row in range(rows):
        for column in range(columns):
            x = 20 + column * (panel[0] + gutter)
            y = 20 + row * (panel[1] + gutter)
            draw.rectangle((x + 18, y + 22, x + panel[0] - 12, y + panel[1] - 12), outline="black", width=5)
            draw.line((x + 34, y + panel[1] - 45, x + panel[0] - 35, y + 60), fill="#2563eb", width=7)
            draw.text((x + 3, y + 2), chr(ord("a") + index), fill="black")
            index += 1
    return np.asarray(image)


@pytest.mark.parametrize(("rows", "columns", "expected"), [(2, 2, 4), (1, 3, 3)])
def test_regular_composites_split_cleanly(rows, columns, expected):
    boxes, occupancies = MODULE.auto_boxes(composite(rows, columns))
    assert len(boxes) == expected
    assert all(box.width > 100 and box.height > 100 for box in boxes)
    assert all(not MODULE.boxes_overlap(a, b) for i, a in enumerate(boxes) for b in boxes[i + 1 :])
    assert occupancies and max(occupancies) <= 0.012


def test_single_plot_is_not_forced_into_panels():
    image = Image.new("RGB", (640, 440), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((35, 30, 610, 410), outline="black", width=6)
    draw.line((60, 370, 560, 75), fill="red", width=9)
    boxes, _ = MODULE.auto_boxes(np.asarray(image))
    assert len(boxes) == 1


def test_manual_boxes_reject_overlap(tmp_path):
    path = tmp_path / "boxes.json"
    path.write_text(json.dumps({"boxes": [
        {"label": "a", "x": 0, "y": 0, "width": 70, "height": 70},
        {"label": "b", "x": 60, "y": 0, "width": 40, "height": 70},
    ]}))
    with pytest.raises(ValueError, match="overlaps"):
        MODULE.load_manual_boxes(path, (100, 100))


def test_pdf_caption_region_is_extracted(tmp_path):
    pdf_path = tmp_path / "paper.pdf"
    document = fitz.open()
    page = document.new_page(width=600, height=800)
    page.insert_textbox(fitz.Rect(45, 60, 555, 130), "Background text " * 18, fontsize=10)
    page.draw_rect(fitz.Rect(80, 190, 275, 420), color=(0, 0, 0), width=2)
    page.draw_rect(fitz.Rect(325, 190, 520, 420), color=(0, 0, 0), width=2)
    page.insert_text((70, 175), "a", fontsize=14)
    page.insert_text((315, 175), "b", fontsize=14)
    page.insert_textbox(fitz.Rect(50, 455, 550, 500), "Figure 1. A synthetic two-panel result.", fontsize=11)
    document.save(pdf_path)
    figures = MODULE.extract_pdf(pdf_path, tmp_path / "figures", 180)
    assert len(figures) == 1
    assert figures[0]["provenance"]["page"] == 1
    with Image.open(figures[0]["path"]) as image:
        assert image.width > 500 and image.height > 400
