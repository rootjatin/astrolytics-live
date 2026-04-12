
#!/usr/bin/env python3
"""
fits_mast_clip_creator_color.py

Create source-aware GIF/MP4 clips from:
- image sequences
- local videos
- local FITS files
- MAST observations (via astroquery.mast.Observations)

This version adds false-color rendering for single-band FITS/images.

Examples
--------
# 1) Local FITS -> false-color GIF + MP4
python3 fits_mast_clip_creator_color.py \
  --source-mode fits \
  --input-fits data/ngc6503.fits \
  --out-dir outputs \
  --color-mode false_color \
  --palette bluegold

# 2) MAST target lookup -> FITS -> false-color GIF + MP4
python3 fits_mast_clip_creator_color.py \
  --source-mode mast_fits \
  --mast-target-name "NGC 6503" \
  --mast-collection HST \
  --mast-radius "0.005 deg" \
  --out-dir outputs \
  --color-mode false_color \
  --palette nebula

# 3) Keep grayscale output
python3 fits_mast_clip_creator_color.py \
  --source-mode mast_fits \
  --mast-target-name "NGC 6503" \
  --mast-collection HST \
  --out-dir outputs \
  --color-mode gray
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageOps

try:
    import imageio.v2 as imageio
except Exception as exc:  # pragma: no cover
    raise SystemExit("Missing dependency: imageio. Install with: pip install imageio") from exc

try:
    from astropy.io import fits
except Exception:  # pragma: no cover
    fits = None

try:
    from astroquery.mast import Observations
except Exception:  # pragma: no cover
    Observations = None


# ---------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------
@dataclass
class LoadedSource:
    frames: list[Image.Image]
    info: dict[str, Any]
    label: str
    fits_header: dict[str, Any] | None = None
    mast_meta: dict[str, Any] | None = None
    mast_manifest: list[dict[str, Any]] | None = None


# ---------------------------------------------------------------------
# General helpers
# ---------------------------------------------------------------------
def eprint(*args: Any) -> None:
    print(*args, file=sys.stderr)


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def slugify(text: str) -> str:
    text = str(text).strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "clip"


def join_nonempty(parts: Iterable[Any], sep: str = " ") -> str:
    vals = [str(p).strip() for p in parts if p is not None and str(p).strip()]
    return sep.join(vals)


def natural_key(path: Path) -> list[Any]:
    s = str(path)
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


def parse_size(size_text: str) -> tuple[int, int]:
    m = re.fullmatch(r"\s*(\d+)\s*x\s*(\d+)\s*", size_text.lower())
    if not m:
        raise ValueError(f"Invalid size format: {size_text!r}. Use WIDTHxHEIGHT, e.g. 1280x720")
    return int(m.group(1)), int(m.group(2))


def parse_rgb(text: str) -> tuple[int, int, int]:
    parts = [int(x.strip()) for x in text.split(",")]
    if len(parts) != 3:
        raise ValueError(f"Invalid RGB value: {text!r}. Use R,G,B.")
    if any(not (0 <= p <= 255) for p in parts):
        raise ValueError(f"RGB values must be in 0..255: {text!r}")
    return tuple(parts)  # type: ignore[return-value]


def parse_percentiles(text: str) -> tuple[float, float]:
    vals = [float(x.strip()) for x in text.split(",")]
    if len(vals) != 2:
        raise ValueError("--fits-percentiles must be LOW,HIGH")
    lo, hi = vals
    if hi <= lo:
        raise ValueError("--fits-percentiles requires HIGH > LOW")
    return lo, hi


def size_from_preset(preset: str) -> tuple[int, int]:
    mapping = {
        "landscape_hd": (1280, 720),
        "landscape_fullhd": (1920, 1080),
        "square": (1080, 1080),
        "portrait": (1080, 1920),
        "original": (-1, -1),
    }
    if preset not in mapping:
        raise ValueError(f"Unknown size preset: {preset}")
    return mapping[preset]


def maybe_bounce(frames: list[Image.Image], enabled: bool) -> list[Image.Image]:
    if not enabled or len(frames) < 2:
        return frames
    return frames + frames[-2:0:-1]


def fit_to_canvas(img: Image.Image, size: tuple[int, int], background: tuple[int, int, int]) -> Image.Image:
    img = img.convert("RGB")
    out = Image.new("RGB", size, background)
    contained = ImageOps.contain(img, size)
    x = (size[0] - contained.width) // 2
    y = (size[1] - contained.height) // 2
    out.paste(contained, (x, y))
    return out


def trim_words(text: str, max_words: int) -> str:
    words = str(text).split()
    return " ".join(words[:max_words])


# ---------------------------------------------------------------------
# False-color palettes
# ---------------------------------------------------------------------
PALETTES: dict[str, dict[str, tuple[int, int, int]]] = {
    "bluegold": {
        "black": (0, 0, 20),
        "mid": (0, 120, 255),
        "white": (255, 220, 180),
    },
    "nebula": {
        "black": (10, 0, 0),
        "mid": (180, 30, 30),
        "white": (255, 240, 200),
    },
    "emerald": {
        "black": (0, 10, 0),
        "mid": (0, 180, 120),
        "white": (240, 255, 240),
    },
    "purple": {
        "black": (10, 0, 20),
        "mid": (140, 60, 220),
        "white": (255, 230, 255),
    },
}


def get_palette_colors(
    palette_name: str,
    black_override: str | None,
    mid_override: str | None,
    white_override: str | None,
) -> tuple[tuple[int, int, int], tuple[int, int, int], tuple[int, int, int]]:
    palette = PALETTES[palette_name]
    black = parse_rgb(black_override) if black_override else palette["black"]
    mid = parse_rgb(mid_override) if mid_override else palette["mid"]
    white = parse_rgb(white_override) if white_override else palette["white"]
    return black, mid, white


# ---------------------------------------------------------------------
# Title / text generation
# ---------------------------------------------------------------------
def build_process_label(source_mode: str, frame_count: int) -> str:
    if source_mode == "fits":
        return f"Rendered {frame_count} frame(s) from FITS data"
    if source_mode == "mast_fits":
        return f"Queried MAST, downloaded FITS, rendered {frame_count} frame(s)"
    if source_mode == "video":
        return f"Sampled {frame_count} frame(s) from source video"
    return f"Loaded {frame_count} image frame(s) from sequence"


def generate_text_bundle(
    source_label: str,
    source_mode: str,
    frame_count: int,
    subject_description: str | None,
    process_steps: str | None,
    fits_header: dict[str, Any] | None,
    mast_meta: dict[str, Any] | None,
) -> dict[str, str]:
    hdr = fits_header or {}
    mast = mast_meta or {}

    object_name = hdr.get("OBJECT") or mast.get("target_name")
    telescope = hdr.get("TELESCOP") or mast.get("obs_collection")
    instrument = hdr.get("INSTRUME") or mast.get("instrument_name")
    filt = hdr.get("FILTER")
    date_obs = hdr.get("DATE-OBS")

    base_subject = (
        subject_description
        or join_nonempty([object_name, telescope, instrument, filt], " ")
        or source_label
    )
    process_label = process_steps or build_process_label(source_mode, frame_count)

    title = trim_words(join_nonempty([base_subject, "|", process_label], " "), 20)
    what_shown = join_nonempty(
        [
            f"Source: {source_label}.",
            f"Subject: {base_subject}." if base_subject else "",
            f"Telescope/collection: {telescope}." if telescope else "",
            f"Instrument: {instrument}." if instrument else "",
            f"Filter: {filt}." if filt else "",
            f"Date observed: {date_obs}." if date_obs else "",
        ],
        " ",
    )
    description = join_nonempty(
        [
            what_shown,
            f"Process involved: {process_label}.",
            "This export was generated automatically from the source data and named from its provenance.",
        ],
        " ",
    )
    return {
        "title": title,
        "what_video_shows": what_shown,
        "process_involved": process_label,
        "description": description,
    }


# ---------------------------------------------------------------------
# Image sequence / video loading
# ---------------------------------------------------------------------
def list_input_images(input_dir: Path, file_glob: str) -> list[Path]:
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
    files = sorted(input_dir.glob(file_glob), key=natural_key)
    if not files:
        raise FileNotFoundError(f"No images found in {input_dir} matching {file_glob!r}")
    return files


def load_image_sequence(
    input_dir: Path,
    file_glob: str,
    color_mode: str,
    black: tuple[int, int, int],
    mid: tuple[int, int, int],
    white: tuple[int, int, int],
) -> LoadedSource:
    files = list_input_images(input_dir, file_glob)
    frames: list[Image.Image] = []
    for path in files:
        gray = Image.open(path).convert("L")
        frames.append(colorize_gray_image(gray, color_mode, black, mid, white))
    label = f"{input_dir.name}_{slugify(file_glob.replace('*', 'seq'))}"
    info = {
        "count": len(frames),
        "files": [str(p) for p in files],
        "color_mode": color_mode,
    }
    return LoadedSource(frames=frames, info=info, label=label)


def load_video(
    video_path: Path,
    start_at: float,
    max_seconds: float | None,
    target_sample_fps: float,
    max_frames: int,
    color_mode: str,
    black: tuple[int, int, int],
    mid: tuple[int, int, int],
    white: tuple[int, int, int],
) -> LoadedSource:
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    reader = imageio.get_reader(video_path)
    meta = reader.get_meta_data()
    native_fps = float(meta.get("fps", 24.0) or 24.0)
    start_frame = max(0, int(start_at * native_fps))
    stride = max(1, int(round(native_fps / max(target_sample_fps, 1e-6))))

    frames: list[Image.Image] = []
    try:
        for idx, arr in enumerate(reader):
            if idx < start_frame:
                continue
            if max_seconds is not None and ((idx - start_frame) / native_fps) > max_seconds:
                break
            if (idx - start_frame) % stride != 0:
                continue

            img = Image.fromarray(arr)
            if color_mode == "gray":
                frames.append(img.convert("RGB"))
            else:
                gray = img.convert("L")
                frames.append(colorize_gray_image(gray, color_mode, black, mid, white))

            if len(frames) >= max_frames:
                break
    finally:
        reader.close()

    if not frames:
        raise RuntimeError("No frames extracted from video")

    info = {
        "video_path": str(video_path),
        "native_fps": native_fps,
        "sample_fps": native_fps / stride,
        "count": len(frames),
        "color_mode": color_mode,
    }
    return LoadedSource(frames=frames, info=info, label=video_path.stem)


# ---------------------------------------------------------------------
# FITS loading
# ---------------------------------------------------------------------
def require_astropy() -> None:
    if fits is None:
        raise RuntimeError("Missing dependency: astropy. Install with: pip install astropy")


def normalize_to_u8(frame: np.ndarray, percentiles: tuple[float, float]) -> np.ndarray:
    x = np.asarray(frame, dtype=np.float64)
    finite = np.isfinite(x)

    if not finite.any():
        lo, hi = 0.0, 1.0
        x = np.zeros_like(x, dtype=np.float64)
    else:
        vals = x[finite]
        lo, hi = np.percentile(vals, percentiles)
        if not np.isfinite(lo):
            lo = float(np.nanmin(vals))
        if not np.isfinite(hi):
            hi = float(np.nanmax(vals))
        if hi <= lo:
            hi = lo + 1.0

    x = np.nan_to_num(x, nan=lo, posinf=hi, neginf=lo)
    x = np.clip((x - lo) / (hi - lo + 1e-12), 0, 1)
    return (x * 255).astype(np.uint8)


def colorize_gray_image(
    gray: Image.Image,
    color_mode: str,
    black: tuple[int, int, int],
    mid: tuple[int, int, int],
    white: tuple[int, int, int],
) -> Image.Image:
    gray = gray.convert("L")
    if color_mode == "gray":
        return gray.convert("RGB")
    return ImageOps.colorize(gray, black=black, mid=mid, white=white)


def gray_to_rgb_pil(
    u8: np.ndarray,
    color_mode: str = "false_color",
    black: tuple[int, int, int] = (0, 0, 20),
    mid: tuple[int, int, int] = (0, 120, 255),
    white: tuple[int, int, int] = (255, 220, 180),
) -> Image.Image:
    gray = Image.fromarray(u8, mode="L")
    return colorize_gray_image(gray, color_mode, black, mid, white)


def detect_frame_axis(arr: np.ndarray) -> int | None:
    arr = np.asarray(arr)
    if arr.ndim < 3:
        return None
    candidate_axes = [ax for ax in range(arr.ndim - 2) if arr.shape[ax] > 1]
    if not candidate_axes:
        return None
    candidate_axes.sort(key=lambda ax: arr.shape[ax])
    return candidate_axes[0]


def iter_fits_frames(arr: np.ndarray, frame_axis: int | None = None) -> Iterable[np.ndarray]:
    arr = np.squeeze(np.asarray(arr))

    if arr.ndim == 2:
        yield arr
        return

    if arr.ndim < 2:
        raise ValueError(f"FITS data is not image-like: shape={arr.shape}")

    if frame_axis is None:
        frame_axis = detect_frame_axis(arr)

    if frame_axis is None:
        while arr.ndim > 2:
            arr = arr[0]
        yield arr
        return

    moved = np.moveaxis(arr, frame_axis, 0)
    ny, nx = moved.shape[-2], moved.shape[-1]
    moved = moved.reshape(moved.shape[0], -1, ny, nx)
    for i in range(moved.shape[0]):
        yield moved[i, 0]


def derive_fits_label(header: dict[str, Any], path: Path) -> str:
    parts = []
    for key in ("OBJECT", "TELESCOP", "INSTRUME", "FILTER", "DETECTOR"):
        value = header.get(key)
        if value:
            parts.append(str(value))
    if not parts:
        parts.append(path.stem)
    return slugify("_".join(parts[:4]))


def hdu_has_image_data(hdu: Any) -> bool:
    data = getattr(hdu, "data", None)
    if data is None:
        return False
    arr = np.asarray(data)
    return arr.ndim >= 2 and arr.size > 0


def pick_image_hdu(hdul: Any, requested_hdu_index: int | None) -> int:
    if requested_hdu_index is not None:
        if requested_hdu_index < 0 or requested_hdu_index >= len(hdul):
            raise IndexError(f"Requested HDU {requested_hdu_index} out of range for file with {len(hdul)} HDUs")
        if hdu_has_image_data(hdul[requested_hdu_index]):
            return requested_hdu_index
        raise ValueError(f"HDU {requested_hdu_index} exists but has no usable image data")

    sci_candidates: list[int] = []
    for idx, hdu in enumerate(hdul):
        name = str(getattr(hdu, "name", "")).upper()
        if name == "SCI" and hdu_has_image_data(hdu):
            sci_candidates.append(idx)
    if sci_candidates:
        return sci_candidates[0]

    for idx, hdu in enumerate(hdul):
        if hdu_has_image_data(hdu):
            return idx

    raise ValueError("No image data HDU found in FITS file")


def load_fits(
    fits_path: Path,
    hdu_index: int | None,
    frame_axis: int | None,
    percentiles: tuple[float, float],
    max_frames: int,
    color_mode: str,
    black: tuple[int, int, int],
    mid: tuple[int, int, int],
    white: tuple[int, int, int],
) -> LoadedSource:
    require_astropy()

    fits_path = Path(fits_path)
    if not fits_path.exists():
        raise FileNotFoundError(f"FITS file not found: {fits_path}")

    with fits.open(fits_path, memmap=False) as hdul:
        actual_hdu_index = pick_image_hdu(hdul, hdu_index)
        data = hdul[actual_hdu_index].data
        header = dict(hdul[actual_hdu_index].header)

    assert data is not None

    frames: list[Image.Image] = []
    for idx, frame in enumerate(iter_fits_frames(data, frame_axis=frame_axis)):
        if idx >= max_frames:
            break
        u8 = normalize_to_u8(frame, percentiles)
        frames.append(
            gray_to_rgb_pil(
                u8,
                color_mode=color_mode,
                black=black,
                mid=mid,
                white=white,
            )
        )

    if not frames:
        raise RuntimeError("No renderable frames found in FITS data")

    info = {
        "fits_path": str(fits_path),
        "shape": tuple(np.asarray(np.squeeze(data)).shape),
        "frame_count": len(frames),
        "hdu_index": actual_hdu_index,
        "color_mode": color_mode,
    }
    return LoadedSource(
        frames=frames,
        info=info,
        label=derive_fits_label(header, fits_path),
        fits_header=header,
    )


# ---------------------------------------------------------------------
# MAST helpers
# ---------------------------------------------------------------------
def require_astroquery() -> None:
    if Observations is None:
        raise RuntimeError("Missing dependency: astroquery. Install with: pip install astroquery")


def row_to_dict(row: Any) -> dict[str, Any]:
    if row is None:
        return {}
    out: dict[str, Any] = {}
    try:
        names = list(row.colnames)
    except Exception:
        try:
            names = list(row.keys())
        except Exception:
            return out
    for name in names:
        value = row[name]
        out[str(name)] = value.item() if hasattr(value, "item") else value
    return out


def derive_mast_label(row_dict: dict[str, Any], fallback: str = "mast_clip") -> str:
    text = join_nonempty(
        [
            row_dict.get("target_name"),
            row_dict.get("obs_collection"),
            row_dict.get("instrument_name"),
            row_dict.get("obs_id"),
        ],
        "_",
    )
    return slugify(text or fallback)


def filter_obs_table_python_side(obs: Any, collection: str | None, obs_id: str | None) -> Any:
    filtered = obs
    if collection and "obs_collection" in filtered.colnames:
        filtered = filtered[filtered["obs_collection"] == collection]
    if obs_id and "obs_id" in filtered.colnames:
        filtered = filtered[filtered["obs_id"] == obs_id]
    if "intentType" in filtered.colnames:
        filtered = filtered[filtered["intentType"] == "science"]
    if "dataRights" in filtered.colnames:
        filtered = filtered[filtered["dataRights"] == "PUBLIC"]
    return filtered


def fetch_observations(
    target_name: str | None,
    region: str | None,
    radius: str,
    obs_id: str | None,
    collection: str | None,
) -> Any:
    require_astroquery()

    if target_name:
        try:
            return Observations.query_criteria(
                object_name=target_name,
                radius=radius,
                obs_collection=collection if collection else None,
                obs_id=obs_id if obs_id else None,
                intentType="science",
                dataRights="PUBLIC",
            )
        except Exception:
            obs = Observations.query_object(target_name, radius=radius)
            return filter_obs_table_python_side(obs, collection=collection, obs_id=obs_id)

    if region:
        try:
            return Observations.query_criteria(
                coordinates=region,
                radius=radius,
                obs_collection=collection if collection else None,
                obs_id=obs_id if obs_id else None,
                intentType="science",
                dataRights="PUBLIC",
            )
        except Exception:
            obs = Observations.query_region(region, radius=radius)
            return filter_obs_table_python_side(obs, collection=collection, obs_id=obs_id)

    criteria: dict[str, Any] = {"intentType": "science", "dataRights": "PUBLIC"}
    if obs_id:
        criteria["obs_id"] = obs_id
    if collection:
        criteria["obs_collection"] = collection
    if len(criteria) == 2:
        raise ValueError(
            "For mast_fits mode, provide --mast-target-name, --mast-region, "
            "or at least one of --mast-obs-id / --mast-collection."
        )
    return Observations.query_criteria(**criteria)


def product_score(row_dict: dict[str, Any]) -> tuple[int, int, int]:
    product_type = str(row_dict.get("productType", "")).lower()
    description = str(row_dict.get("description", "")).lower()
    calib_level = int(row_dict.get("calib_level", 0) or 0)

    score_science = 1 if product_type == "science" else 0
    score_calibrated = 1 if any(k in description for k in ("calibrated", "drizzled", "science")) else 0
    return (score_science, score_calibrated, calib_level)


def normalize_manifest_table(manifest_table: Any) -> list[dict[str, Any]]:
    manifest: list[dict[str, Any]] = []
    for row in manifest_table:
        d: dict[str, Any] = {}
        for name in row.colnames:
            value = row[name]
            d[str(name)] = value.item() if hasattr(value, "item") else value
        manifest.append(d)
    return manifest


def mast_download_best_fits(
    target_name: str | None,
    region: str | None,
    radius: str,
    obs_id: str | None,
    collection: str | None,
    download_dir: Path,
) -> tuple[Path, dict[str, Any], list[dict[str, Any]], str]:
    require_astroquery()
    ensure_dir(download_dir)

    obs = fetch_observations(target_name, region, radius, obs_id, collection)
    if len(obs) == 0:
        raise RuntimeError("No MAST observations matched the query")

    filtered = filter_obs_table_python_side(obs, collection=collection, obs_id=obs_id)
    if len(filtered) == 0:
        raise RuntimeError("Observations were found, but none matched the requested filters")

    for i in range(min(len(filtered), 50)):
        row = filtered[i]

        try:
            products = Observations.get_unique_product_list(row)
        except Exception:
            continue
        if len(products) == 0:
            continue

        try:
            fits_products = Observations.filter_products(
                products, extension="fits", productType="SCIENCE"
            )
            if len(fits_products) == 0:
                fits_products = Observations.filter_products(products, extension="fits")
        except Exception:
            fits_products = products

        if len(fits_products) == 0:
            continue

        row_dicts = [row_to_dict(prod) for prod in fits_products]
        order = sorted(range(len(row_dicts)), key=lambda idx: product_score(row_dicts[idx]), reverse=True)

        for idx in order[:10]:
            candidate = fits_products[idx:idx + 1]
            manifest_table = Observations.download_products(candidate, download_dir=str(download_dir))
            manifest = normalize_manifest_table(manifest_table)
            complete = [m for m in manifest if str(m.get("Status", "")).upper() == "COMPLETE"]
            if not complete:
                continue

            local_path = Path(str(complete[0]["Local Path"]))
            best_row = row_to_dict(row)
            label = derive_mast_label(best_row, fallback=local_path.stem)
            return local_path, best_row, manifest, label

    raise RuntimeError(
        "Matched observations, but none yielded a downloadable FITS product. "
        "Try a smaller radius, a different collection, or a specific obs_id."
    )


# ---------------------------------------------------------------------
# Overlays / export
# ---------------------------------------------------------------------
def add_title_card(
    frames: list[Image.Image],
    title: str,
    size: tuple[int, int],
    background: tuple[int, int, int],
    text_color: tuple[int, int, int],
    seconds: float,
    fps: float,
) -> list[Image.Image]:
    n = max(1, int(round(seconds * fps)))
    card = Image.new("RGB", size, background)
    draw = ImageDraw.Draw(card)
    text = title.strip() or "Untitled Clip"
    margin = max(24, size[0] // 20)
    draw.multiline_text((margin, margin), text, fill=text_color, spacing=8)
    return [card.copy() for _ in range(n)] + frames


def add_footer_text(
    frames: list[Image.Image],
    text: str,
    text_color: tuple[int, int, int],
    padding: int = 16,
) -> list[Image.Image]:
    if not text.strip():
        return frames

    out: list[Image.Image] = []
    for frame in frames:
        img = frame.copy()
        img_rgba = img.convert("RGBA")
        draw = ImageDraw.Draw(img_rgba)
        bbox = draw.multiline_textbbox((padding, padding), text)
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        x = padding
        y = img.height - h - padding

        overlay = Image.new("RGBA", (w + padding * 2, h + padding), (0, 0, 0, 140))
        img_rgba.alpha_composite(overlay, (x - padding // 2, y - padding // 2))

        draw = ImageDraw.Draw(img_rgba)
        draw.multiline_text((x, y), text, fill=text_color, spacing=4)
        out.append(img_rgba.convert("RGB"))
    return out


def build_provenance_report(metadata: dict[str, Any]) -> str:
    lines = [f"Source mode: {metadata.get('source_mode')}", f"Source label: {metadata.get('source_label')}", ""]
    if metadata.get("fits_header"):
        lines.append("FITS header summary:")
        for k, v in metadata["fits_header"].items():
            lines.append(f"  {k}: {v}")
        lines.append("")
    if metadata.get("mast_meta"):
        lines.append("MAST metadata:")
        for k, v in metadata["mast_meta"].items():
            lines.append(f"  {k}: {v}")
        lines.append("")
    lines += [
        "What the video shows:",
        metadata["text_bundle"]["what_video_shows"],
        "",
        "Process involved:",
        metadata["text_bundle"]["process_involved"],
        "",
        "Generated title:",
        metadata["text_bundle"]["title"],
    ]
    return "\n".join(lines).rstrip() + "\n"


def save_outputs(
    frames: Sequence[Image.Image],
    out_dir: Path,
    base_name: str,
    export_gif: bool,
    export_mp4: bool,
    gif_fps: float,
    mp4_fps: float,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    ensure_dir(out_dir)
    arrays = [np.asarray(f.convert("RGB")) for f in frames]
    result: dict[str, Any] = {
        "gif_path": None,
        "mp4_path": None,
        "thumbnail_path": None,
        "metadata_json": None,
        "title_txt": None,
        "description_txt": None,
        "provenance_txt": None,
    }

    if export_gif:
        gif_path = out_dir / f"{base_name}.gif"
        imageio.mimsave(gif_path, arrays, fps=gif_fps)
        result["gif_path"] = str(gif_path)

    if export_mp4:
        mp4_path = out_dir / f"{base_name}.mp4"
        try:
            imageio.mimsave(mp4_path, arrays, fps=mp4_fps)
        except Exception as exc:
            eprint(f"Warning: MP4 export failed: {exc}")
            eprint("Tip: install/enable ffmpeg support for imageio if GIF works but MP4 does not.")
        else:
            result["mp4_path"] = str(mp4_path)

    thumb = out_dir / f"{base_name}_thumb.jpg"
    frames[0].save(thumb, quality=92)
    result["thumbnail_path"] = str(thumb)

    meta_json = out_dir / f"{base_name}_metadata.json"
    meta_json.write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
    result["metadata_json"] = str(meta_json)

    title_txt = out_dir / f"{base_name}_title.txt"
    title_txt.write_text(str(metadata["text_bundle"]["title"]) + "\n", encoding="utf-8")
    result["title_txt"] = str(title_txt)

    description_txt = out_dir / f"{base_name}_description.txt"
    description_txt.write_text(str(metadata["text_bundle"]["description"]) + "\n", encoding="utf-8")
    result["description_txt"] = str(description_txt)

    provenance_txt = out_dir / f"{base_name}_provenance.txt"
    provenance_txt.write_text(build_provenance_report(metadata), encoding="utf-8")
    result["provenance_txt"] = str(provenance_txt)

    return result


# ---------------------------------------------------------------------
# Argument parsing / pipeline
# ---------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Create source-aware GIF/MP4 clips from images, video, FITS, or MAST FITS."
    )
    p.add_argument("--source-mode", required=True, choices=["images", "video", "fits", "mast_fits"])

    # Images
    p.add_argument("--input-dir", type=Path, default=Path("input_frames"))
    p.add_argument("--file-glob", default="*.png")

    # Video
    p.add_argument("--input-video", type=Path, default=Path("input_video.mp4"))
    p.add_argument("--start-at-seconds", type=float, default=0.0)
    p.add_argument("--max-seconds", type=float, default=8.0)
    p.add_argument("--video-sample-fps", type=float, default=8.0)

    # FITS
    p.add_argument("--input-fits", type=Path, default=Path("data/source.fits"))
    p.add_argument("--fits-hdu-index", type=int, default=None)
    p.add_argument("--fits-frame-axis", type=int, default=None)
    p.add_argument("--fits-percentiles", default="1,99.5", help="low,high percentiles for display scaling")

    # MAST
    p.add_argument("--mast-target-name", default=None)
    p.add_argument("--mast-region", default=None, help='Sky coordinates string for query_region, e.g. "10.684 41.269"')
    p.add_argument("--mast-radius", default="0.02 deg")
    p.add_argument("--mast-obs-id", default=None)
    p.add_argument("--mast-collection", default=None)
    p.add_argument("--mast-download-dir", type=Path, default=Path("mast_downloads"))

    # Output
    p.add_argument("--out-dir", type=Path, default=Path("outputs"))
    p.add_argument("--max-frames", type=int, default=120)
    p.add_argument("--bounce", action="store_true")
    p.add_argument("--export-gif", action="store_true")
    p.add_argument("--export-mp4", action="store_true")
    p.add_argument("--gif-fps", type=float, default=10.0)
    p.add_argument("--mp4-fps", type=float, default=12.0)

    # Layout
    p.add_argument(
        "--size-preset",
        default="landscape_hd",
        choices=["landscape_hd", "landscape_fullhd", "square", "portrait", "original"],
    )
    p.add_argument("--custom-size", default=None, help="Override size preset with WIDTHxHEIGHT")
    p.add_argument("--background", default="0,0,0", help="RGB background as R,G,B")
    p.add_argument("--text-color", default="255,255,255", help="RGB text color as R,G,B")

    # Text / metadata
    p.add_argument("--subject-description", default=None)
    p.add_argument("--process-steps", default=None)
    p.add_argument("--add-title-card", action="store_true")
    p.add_argument("--title-card-seconds", type=float, default=1.5)
    p.add_argument("--overlay-footer", action="store_true")

    # Color
    p.add_argument("--color-mode", choices=["gray", "false_color"], default="false_color")
    p.add_argument("--palette", choices=sorted(PALETTES.keys()), default="bluegold")
    p.add_argument("--black-color", default=None, help="Override low-intensity color as R,G,B")
    p.add_argument("--mid-color", default=None, help="Override mid-intensity color as R,G,B")
    p.add_argument("--white-color", default=None, help="Override high-intensity color as R,G,B")

    return p


def determine_target_size(size_preset: str, custom_size: str | None, first_frame: Image.Image) -> tuple[int, int]:
    if custom_size:
        return parse_size(custom_size)
    size = size_from_preset(size_preset)
    if size == (-1, -1):
        return (first_frame.width, first_frame.height)
    return size


def load_source(
    args: argparse.Namespace,
    percentiles: tuple[float, float],
    color_mode: str,
    black: tuple[int, int, int],
    mid: tuple[int, int, int],
    white: tuple[int, int, int],
) -> LoadedSource:
    if args.source_mode == "images":
        return load_image_sequence(args.input_dir, args.file_glob, color_mode, black, mid, white)

    if args.source_mode == "video":
        return load_video(
            args.input_video,
            start_at=args.start_at_seconds,
            max_seconds=args.max_seconds,
            target_sample_fps=args.video_sample_fps,
            max_frames=args.max_frames,
            color_mode=color_mode,
            black=black,
            mid=mid,
            white=white,
        )

    if args.source_mode == "fits":
        return load_fits(
            args.input_fits,
            hdu_index=args.fits_hdu_index,
            frame_axis=args.fits_frame_axis,
            percentiles=percentiles,
            max_frames=args.max_frames,
            color_mode=color_mode,
            black=black,
            mid=mid,
            white=white,
        )

    downloaded_fits, mast_meta, mast_manifest, source_label = mast_download_best_fits(
        target_name=args.mast_target_name,
        region=args.mast_region,
        radius=args.mast_radius,
        obs_id=args.mast_obs_id,
        collection=args.mast_collection,
        download_dir=args.mast_download_dir,
    )
    loaded = load_fits(
        downloaded_fits,
        hdu_index=args.fits_hdu_index,
        frame_axis=args.fits_frame_axis,
        percentiles=percentiles,
        max_frames=args.max_frames,
        color_mode=color_mode,
        black=black,
        mid=mid,
        white=white,
    )
    loaded.mast_meta = mast_meta
    loaded.mast_manifest = mast_manifest
    if source_label not in {"mast_clip", "clip"}:
        loaded.label = source_label
    return loaded


def main() -> int:
    args = build_parser().parse_args()

    if not args.export_gif and not args.export_mp4:
        args.export_gif = True
        args.export_mp4 = True

    out_dir = ensure_dir(args.out_dir)
    bg = parse_rgb(args.background)
    text_color = parse_rgb(args.text_color)
    percentiles = parse_percentiles(args.fits_percentiles)

    black, mid, white = get_palette_colors(
        args.palette,
        args.black_color,
        args.mid_color,
        args.white_color,
    )

    source = load_source(
        args,
        percentiles,
        color_mode=args.color_mode,
        black=black,
        mid=mid,
        white=white,
    )
    source.frames = maybe_bounce(source.frames[: args.max_frames], args.bounce)

    if not source.frames:
        raise RuntimeError("No frames available after loading")

    target_size = determine_target_size(args.size_preset, args.custom_size, source.frames[0])
    prepared_frames = [fit_to_canvas(frame, target_size, bg) for frame in source.frames]

    text_bundle = generate_text_bundle(
        source_label=source.label,
        source_mode=args.source_mode,
        frame_count=len(prepared_frames),
        subject_description=args.subject_description,
        process_steps=args.process_steps,
        fits_header=source.fits_header,
        mast_meta=source.mast_meta,
    )

    if args.add_title_card:
        prepared_frames = add_title_card(
            prepared_frames,
            title=text_bundle["title"],
            size=target_size,
            background=bg,
            text_color=text_color,
            seconds=args.title_card_seconds,
            fps=max(args.mp4_fps, args.gif_fps),
        )

    if args.overlay_footer:
        prepared_frames = add_footer_text(prepared_frames, text_bundle["title"], text_color=text_color)

    base_name = slugify(source.label)
    fits_header_summary = None
    if source.fits_header:
        keys = ["OBJECT", "TELESCOP", "INSTRUME", "FILTER", "DATE-OBS"]
        fits_header_summary = {k: source.fits_header.get(k) for k in keys if k in source.fits_header}

    metadata: dict[str, Any] = {
        "source_mode": args.source_mode,
        "source_label": source.label,
        "output_base_name": base_name,
        "frame_count": len(prepared_frames),
        "target_size": list(target_size),
        "source_info": source.info,
        "fits_header": fits_header_summary,
        "mast_meta": source.mast_meta,
        "mast_manifest": source.mast_manifest,
        "text_bundle": text_bundle,
        "settings": {
            "export_gif": args.export_gif,
            "export_mp4": args.export_mp4,
            "gif_fps": args.gif_fps,
            "mp4_fps": args.mp4_fps,
            "bounce": args.bounce,
            "fits_percentiles": list(percentiles),
            "title_card": args.add_title_card,
            "overlay_footer": args.overlay_footer,
            "color_mode": args.color_mode,
            "palette": args.palette,
            "black_color": list(black),
            "mid_color": list(mid),
            "white_color": list(white),
        },
    }

    output_paths = save_outputs(
        frames=prepared_frames,
        out_dir=out_dir,
        base_name=base_name,
        export_gif=args.export_gif,
        export_mp4=args.export_mp4,
        gif_fps=args.gif_fps,
        mp4_fps=args.mp4_fps,
        metadata=metadata,
    )

    print("\nDone.")
    print(f"Source label : {source.label}")
    print(f"Frame count  : {len(prepared_frames)}")
    print(f"Output base  : {base_name}")
    for key, value in output_paths.items():
        print(f"{key:14s}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
