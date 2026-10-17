"""Deterministic frame sampling and timestamped contact sheets."""

from __future__ import annotations

import hashlib
import io
import math
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from evoaction.config import get_settings

FRAME_CACHE_FPS = 10.0  # decoded frame grid: 0.1 s resolution, enough for 0.2 s local inspection


@dataclass
class Sheet:
    jpeg: bytes
    timestamps: list[float]
    tile_px: int
    cols: int
    sha256: str = ""
    meta: dict = field(default_factory=dict)


def _frame_dir(video_path: str, width: int) -> Path:
    key = hashlib.sha256(f"{video_path}|{width}|{FRAME_CACHE_FPS}".encode()).hexdigest()[:16]
    return get_settings().cache_dir / "frames" / key


def decode_frames(video_path: str, width: int) -> list[Path]:
    """Decode the whole video once at FRAME_CACHE_FPS and `width` px; frame k is at t = k / FRAME_CACHE_FPS."""
    d = _frame_dir(video_path, width)
    done = d / ".done"
    if not done.exists():
        d.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["ffmpeg", "-v", "error", "-y", "-i", video_path, "-vf",
             f"fps={FRAME_CACHE_FPS},scale={width}:-2", "-q:v", "3", str(d / "f_%05d.jpg")],
            check=True,
        )
        done.write_text("ok")
    return sorted(d.glob("f_*.jpg"))


def frame_at(frames: list[Path], t: float) -> tuple[Path, float]:
    k = min(max(int(round(t * FRAME_CACHE_FPS)), 0), len(frames) - 1)
    return frames[k], k / FRAME_CACHE_FPS


def uniform_timestamps(duration: float, hz: float, start: float = 0.0) -> list[float]:
    n = int(math.floor((duration - start) * hz + 1e-9)) + 1
    return [round(start + i / hz, 3) for i in range(n) if start + i / hz <= duration + 1e-6]


def _font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # older Pillow
        return ImageFont.load_default()


def _truetype(size: int) -> ImageFont.ImageFont:
    """Macrodata Refiner badge font: Helvetica/Arial/DejaVu Sans at 17, else PIL default."""
    for path in ("/System/Library/Fonts/Helvetica.ttc", "/Library/Fonts/Arial.ttf", "/System/Library/Fonts/Supplemental/Arial.ttf",
                 "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "Helvetica.ttc", "Arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return _font(size)


def build_sheets(
    video_path: str,
    timestamps: list[float],
    tile_px: int = 224,
    frames_per_sheet: int = 20,
    cols: int | None = None,
    label_prefix: str = "",
    badge: str = "default",
    jpeg_quality: int = 85,
) -> list[Sheet]:
    """Tile frames (nearest decoded frame to each timestamp) with burned-in time labels.

    badge="macrodata" reproduces Refiner's timestamped contact sheets: black box (0,0,72,26) at each tile's
    top-left with white "%06.2fs" text at (7,3), font size 17, JPEG quality 95 with 4:2:0 subsampling.
    """
    frames = decode_frames(video_path, tile_px)
    cols = cols or min(5, frames_per_sheet)
    sheets: list[Sheet] = []
    font = _truetype(17) if badge == "macrodata" else _font(max(12, tile_px // 12))
    for start in range(0, len(timestamps), frames_per_sheet):
        chunk = timestamps[start : start + frames_per_sheet]
        picked = [frame_at(frames, t) for t in chunk]
        tiles = [Image.open(p).convert("RGB") for p, _ in picked]
        th = tiles[0].height
        rows = math.ceil(len(tiles) / cols)
        sheet = Image.new("RGB", (cols * tile_px, rows * th), (0, 0, 0))
        draw = ImageDraw.Draw(sheet)
        actual_ts = []
        for i, (tile, (_, t_actual)) in enumerate(zip(tiles, picked)):
            x, y = (i % cols) * tile_px, (i // cols) * th
            sheet.paste(tile, (x, y))
            if badge == "macrodata":
                draw.rectangle((x, y, x + 72, y + 26), fill=(0, 0, 0))
                draw.text((x + 7, y + 3), f"{chunk[i]:06.2f}s", fill=(255, 255, 255), font=font)
            else:
                label = f"{label_prefix}{chunk[i]:.1f}s"
                bbox = draw.textbbox((x + 3, y + 3), label, font=font)
                draw.rectangle((bbox[0] - 2, bbox[1] - 2, bbox[2] + 2, bbox[3] + 2), fill=(0, 0, 0))
                draw.text((x + 3, y + 3), label, fill=(255, 255, 0), font=font)
            actual_ts.append(chunk[i])
        buf = io.BytesIO()
        sheet.save(buf, format="JPEG", quality=jpeg_quality, subsampling=2)
        data = buf.getvalue()
        sheets.append(Sheet(jpeg=data, timestamps=actual_ts, tile_px=tile_px, cols=cols,
                            sha256=hashlib.sha256(data).hexdigest()))
    return sheets
