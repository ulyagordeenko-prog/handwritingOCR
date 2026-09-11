"""
Render the window background from the Figma design, once, at several scales.

design/main.svg is the design exported from Figma at the window's size. Its
static parts -- the frosted background, glass panels, button faces with their
labels, the confidence legend, title-bar ornaments -- are drawn straight from
it, so the app matches the design to the pixel without needing the design's
fonts: Figma exported every label as an outline.

Three things in the export are placeholders for content that changes, and are
removed before rendering so the app can draw the real thing in their place:
the note in the middle of the title bar ("по середине будет имя файла ..."),
the "100 %" zoom figure, and the two sample scrollbar thumbs.

Rendered at the common Windows display scales rather than at start-up: a
render takes 2-7 s, and the resulting PNGs are 0.1-0.4 MB each, so shipping
them costs less than making every launch wait -- and needs no SVG renderer on
the user's machine.

    python scripts/build_ui_assets.py
"""
import io
import os
import re
import xml.etree.ElementTree as ET

SVG_NS = "http://www.w3.org/2000/svg"
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SOURCE = os.path.join(ROOT, "design", "main.svg")
OUT_DIR = os.path.join(ROOT, "src", "ocr_project", "assets", "ui")

SCALES = (0.85, 1.0, 1.25, 1.5, 1.75, 2.0)
WIDTH, HEIGHT = 1280, 760

# Placeholders, identified by where they sit. Bounding boxes are in design
# pixels, generous by a couple of pixels either way.
PLACEHOLDER_PATHS = [
    ("title-bar note", (505, 10, 922, 31)),
    ("zoom figure",    (44, 732, 79, 744)),
]
PLACEHOLDER_RECTS = [
    ("vertical thumb",   {"x": "617", "y": "638.5", "width": "10", "height": "39"}),
    ("horizontal thumb", {"transform": "rotate(90 581 712.5)"}),
]


def _path_bbox(d: str):
    tokens = re.findall(r"[A-Za-z]|-?\d*\.?\d+(?:e-?\d+)?", d)
    xs, ys, cmd, i = [], [], None, 0
    while i < len(tokens):
        t = tokens[i]
        if t.isalpha():
            cmd, i = t, i + 1
            continue
        if cmd == "H":
            xs.append(float(t)); i += 1
        elif cmd == "V":
            ys.append(float(t)); i += 1
        elif cmd in ("M", "L", "C", "Q", "S", "T"):
            xs.append(float(tokens[i])); ys.append(float(tokens[i + 1])); i += 2
        else:
            i += 1
    return (min(xs), min(ys), max(xs), max(ys)) if xs and ys else None


def _inside(box, area):
    return (box[0] >= area[0] and box[1] >= area[1]
            and box[2] <= area[2] and box[3] <= area[3])


def strip_placeholders(svg_text: str) -> str:
    ET.register_namespace("", SVG_NS)
    root = ET.fromstring(svg_text)
    parent = {c: p for p in root.iter() for c in p}
    removed = []

    for el in list(root.iter(f"{{{SVG_NS}}}path")):
        box = _path_bbox(el.get("d", ""))
        if not box:
            continue
        for name, area in PLACEHOLDER_PATHS:
            if _inside(box, area):
                parent[el].remove(el)
                removed.append(name)

    for el in list(root.iter(f"{{{SVG_NS}}}rect")):
        for name, attrs in PLACEHOLDER_RECTS:
            if all(el.get(k) == v for k, v in attrs.items()):
                parent[el].remove(el)
                removed.append(name)

    expected = {n for n, _ in PLACEHOLDER_PATHS} | {n for n, _ in PLACEHOLDER_RECTS}
    missing = expected - set(removed)
    if missing:
        # A re-exported design with the placeholders moved would otherwise
        # render them baked into the background, under the live versions.
        raise SystemExit(f"не найдены заглушки в макете: {sorted(missing)}")
    print("убраны заглушки:", ", ".join(removed))
    return ET.tostring(root, encoding="unicode")


def main() -> int:
    import resvg_py
    from PIL import Image

    svg = strip_placeholders(open(SOURCE, encoding="utf-8").read())
    os.makedirs(OUT_DIR, exist_ok=True)
    for scale in SCALES:
        w, h = round(WIDTH * scale), round(HEIGHT * scale)
        png = bytes(resvg_py.svg_to_bytes(svg_string=svg, width=w, height=h))
        image = Image.open(io.BytesIO(png)).convert("RGBA")
        path = os.path.join(OUT_DIR, f"bg@{scale:.2f}.png")
        image.save(path, optimize=True)
        print(f"  {os.path.basename(path)}  {w}x{h}  {os.path.getsize(path) / 1e6:.2f} МБ")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
