# color_utils.py
from PIL import Image
import numpy as np
from typing import List, Dict
from matplotlib.colors import CSS4_COLORS  # stable mapping of CSS color names -> hex

def _hex_to_rgb_tuple(hex_code: str) -> tuple:
    hex_code = hex_code.strip()
    if not hex_code.startswith("#"):
        hex_code = f"#{hex_code}"
    return (
        int(hex_code[1:3], 16),
        int(hex_code[3:5], 16),
        int(hex_code[5:7], 16),
    )

def _rgb_to_hex(rgb: tuple) -> str:
    r, g, b = rgb
    return f"#{r:02x}{g:02x}{b:02x}"

def _nearest_css_name(hex_code: str) -> str:
    """
    Find the nearest CSS color name (using Matplotlib's CSS4 color list)
    by Euclidean distance in RGB space.
    """
    target = np.array(_hex_to_rgb_tuple(hex_code), dtype=float)
    best_name = None
    best_dist = float("inf")

    for name, hx in CSS4_COLORS.items():
        rgb = np.array(_hex_to_rgb_tuple(hx), dtype=float)
        d = np.linalg.norm(rgb - target)
        if d < best_dist:
            best_name, best_dist = name, d

    # Title-case for nicer display (e.g., 'lightblue' -> 'Lightblue')
    return best_name.replace("-", " ").title() if best_name else "Unknown"

def extract_dominant_colors(image: Image.Image, k: int = 8) -> List[Dict[str, object]]:
    """
    Extract dominant colors from an image (no external deps).
    Uses PIL adaptive quantization to get up to k dominant colors.

    Returns a list of dicts: [{'hex': '#rrggbb', 'name': 'Color Name', 'pct': 0.0-1.0}, ...]
    """
    # Convert to RGB and quantize to k colors
    img = image.convert("RGB")
    quantized = img.convert("P", palette=Image.ADAPTIVE, colors=max(1, k))

    # Get palette (RGB triplets) and color counts
    palette = quantized.getpalette()  # flat list [r0,g0,b0, r1,g1,b1, ...]
    color_counts = quantized.getcolors()  # list of (count, palette_index)

    if not color_counts:
        # Fallback: just return the average color
        arr = np.array(img)
        rgb_mean = tuple(np.mean(arr.reshape(-1, 3), axis=0).astype(int))
        return [{
            "hex": _rgb_to_hex(rgb_mean),
            "name": _nearest_css_name(_rgb_to_hex(rgb_mean)),
            "pct": 1.0
        }]

    total = sum(count for count, _ in color_counts)

    # Build list of colors with percentages
    results = []
    for count, idx in color_counts:
        base = idx * 3
        rgb = (palette[base], palette[base + 1], palette[base + 2])
        hex_code = _rgb_to_hex(rgb)
        pct = count / total
        results.append({
            "hex": hex_code,
            "name": _nearest_css_name(hex_code),
            "pct": float(pct)
        })

    # Sort by pct descending
    results.sort(key=lambda x: x["pct"], reverse=True)

    # Limit to top k (quantize may produce fewer than k)
    return results[:k]
