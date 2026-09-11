"""
Render the window background from the Figma design, once, at several scales.

The design comes as two exports:

  design/window.svg   the whole window, 1280 x 760 -- frosted background and
                      the title bar with the HANDWRITER mark and window
                      controls. Only those two are taken from it.
  design/content.svg  the working area -- toolbar, legend, photo and text
                      panels with their scrollbars and zoom. This is the part
                      that gets reworked, and it can be re-exported on its own.

The two are composed into one picture, the working area placed where the
window's own working area was. That spot is found, not assumed: the Open photo
pill is located in both files and the difference is the offset.

Static parts are drawn straight from the design, so the app matches it to the
pixel without needing its fonts -- Figma exports every label as an outline.
Placeholders for content that changes are removed first, so the app can draw
the live version in their place: the note in the title bar, the "100 %"
figures and the sample scrollbar thumbs. The build fails if any of them is
not where expected, rather than baking one in under its live replacement.

Rendered at the common Windows display scales rather than at start-up: a
render takes 2-7 s, and the resulting PNGs are 0.1-0.3 MB each, so shipping
them costs less than making every launch wait -- and needs no SVG renderer on
the user's machine.

    python scripts/build_ui_assets.py
"""
import io
import os
import re
import xml.etree.ElementTree as ET

SVG = "http://www.w3.org/2000/svg"
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
WINDOW_SVG = os.path.join(ROOT, "design", "window.svg")
CONTENT_SVG = os.path.join(ROOT, "design", "content.svg")
OUT_DIR = os.path.join(ROOT, "src", "ocr_project", "assets", "ui")

SCALES = (0.85, 1.0, 1.25, 1.5, 1.75, 2.0)
WIDTH, HEIGHT = 1280, 760
TITLE_BAR_BOTTOM = 41          # window.svg: everything below this is its old working area

WINDOW_PLACEHOLDER_PATHS = [("title-bar note", (505, 10, 922, 31))]
# content.svg coordinates
CONTENT_PLACEHOLDER_PATHS = [
    ("photo zoom figure", (34, 668, 91, 687)),
    ("text zoom figure", (654, 668, 711, 687)),
]
THUMB_SIZE = ("10", "39")      # every sample thumb in the design is 10 x 39
THUMBS_EXPECTED = 4


def tag(el):
    return el.tag.replace(f"{{{SVG}}}", "")


def path_bbox(d):
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


def inside(box, area):
    return box[0] >= area[0] and box[1] >= area[1] and box[2] <= area[2] and box[3] <= area[3]


def top_y(el):
    """Top edge of an element, good enough to tell title bar from working area."""
    t = tag(el)
    if t == "path":
        box = path_bbox(el.get("d", ""))
        return box[1] if box else 0.0
    if t in ("rect", "mask"):
        transform = el.get("transform", "")
        nums = [float(n) for n in re.findall(r"-?\d*\.?\d+", transform)]
        if transform.startswith("matrix") and len(nums) == 6:
            return nums[5]
        if transform.startswith("rotate") and len(nums) == 3:
            return nums[2]
        return float(el.get("y", "0"))
    if t == "g":
        kids = [k for k in el if tag(k) in ("rect", "path", "g")]
        return min((top_y(k) for k in kids), default=0.0)
    return 0.0


def pill_origin(root):
    """(x, y) of the first 150 x 45 pill -- the Open photo button."""
    for el in root.iter(f"{{{SVG}}}rect"):
        if el.get("width") == "150" and el.get("height") == "45":
            return float(el.get("x", "0")), float(el.get("y", "0"))
    raise SystemExit("в макете не найдена кнопка Open photo (150 x 45)")


def strip(root, paths, thumbs_expected=0):
    parent = {c: p for p in root.iter() for c in p}
    removed = []
    for el in list(root.iter(f"{{{SVG}}}path")):
        box = path_bbox(el.get("d", ""))
        for name, area in paths:
            if box and inside(box, area):
                parent[el].remove(el)
                removed.append(name)
    thumbs = 0
    if thumbs_expected:
        for el in list(root.iter(f"{{{SVG}}}rect")):
            if (el.get("width"), el.get("height")) == THUMB_SIZE:
                parent[el].remove(el)
                thumbs += 1
    missing = {n for n, _ in paths} - set(removed)
    if missing or thumbs != thumbs_expected:
        # A re-exported design with a placeholder moved would otherwise render
        # it baked into the background, under the live version.
        raise SystemExit(f"не найдены заглушки: {sorted(missing)}; "
                         f"ползунков {thumbs} из {thumbs_expected}")
    return removed + [f"ползунков-примеров: {thumbs}"] * bool(thumbs_expected)


def compose():
    ET.register_namespace("", SVG)
    window = ET.fromstring(open(WINDOW_SVG, encoding="utf-8").read())
    content = ET.fromstring(open(CONTENT_SVG, encoding="utf-8").read())

    # where the working area goes: the offset between the two Open photo pills,
    # measured before window.svg's own working area is cleared away
    wx, wy = pill_origin(window)
    cx, cy = pill_origin(content)
    dx, dy = wx - cx, wy - cy

    report = strip(window, WINDOW_PLACEHOLDER_PATHS)
    # keep window.svg's background and title bar, drop its old working area
    kept = 0
    for el in list(window):
        if tag(el) == "defs":
            continue
        if tag(el) == "g" and el.get("clip-path"):
            kept += 1                    # the frosted background
            continue
        if top_y(el) >= TITLE_BAR_BOTTOM:
            window.remove(el)
        else:
            kept += 1
    report += strip(content, CONTENT_PLACEHOLDER_PATHS, THUMBS_EXPECTED)

    area = ET.SubElement(window, f"{{{SVG}}}g", {"transform": f"translate({dx} {dy})"})
    for el in list(content):
        if tag(el) == "defs":
            defs = window.find(f"{{{SVG}}}defs")
            for d in el:
                defs.append(d)
        else:
            area.append(el)
    print(f"рабочая область встала со сдвигом ({dx}, {dy}); от окна оставлено элементов: {kept}")
    print("убраны заглушки:", ", ".join(report))
    return ET.tostring(window, encoding="unicode")


def main() -> int:
    import resvg_py
    from PIL import Image

    svg = compose()
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
