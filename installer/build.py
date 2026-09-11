"""
Build Handwriter.exe: the icon from the design's feather, then PyInstaller.

    python installer/build.py

Needs resvg-py and pyinstaller in the environment it runs in -- build-time
tools only; neither is an app dependency. Output: Handwriter.exe in the repo
root, which is what people download.
"""
import io
import os
import re
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SVG_NS = "http://www.w3.org/2000/svg"
FEATHER_BOX = (30, 4, 63, 36)       # where the feather sits in window.svg's title bar
ICON = os.path.join(HERE, "handwriter.ico")


def feather_path() -> str:
    """The quill beside HANDWRITER, found by where it sits in the title bar."""
    root = ET.fromstring(open(os.path.join(ROOT, "design", "window.svg"), encoding="utf-8").read())
    for el in root.iter(f"{{{SVG_NS}}}path"):
        nums = [float(n) for n in re.findall(r"-?\d*\.?\d+", el.get("d", ""))]
        xs, ys = nums[0::2], nums[1::2]
        if xs and ys and min(xs) >= FEATHER_BOX[0] and max(xs) <= FEATHER_BOX[2] \
                and min(ys) >= FEATHER_BOX[1] and max(ys) <= FEATHER_BOX[3]:
            return el.get("d")
    raise SystemExit("перо не найдено в design/window.svg")


def build_icon():
    """The feather on a light rounded tile. Black on transparent would vanish
    on Windows' dark taskbar; the tile keeps it legible on either."""
    import resvg_py
    from PIL import Image

    d = feather_path()
    # the feather spans roughly 32..61 x 6..34 in the design; centre it on the tile
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="256" height="256" viewBox="0 0 256 256">
  <defs>
    <linearGradient id="g" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0" stop-color="#fbfbfb"/><stop offset="1" stop-color="#d9d9d9"/>
    </linearGradient>
  </defs>
  <rect x="8" y="8" width="240" height="240" rx="52" fill="url(#g)" stroke="#bdbdbd" stroke-width="3"/>
  <g transform="translate(128 128) scale(6.2) translate(-46.5 -20)">
    <path d="{d}" fill="#141414"/>
  </g>
</svg>"""
    png = bytes(resvg_py.svg_to_bytes(svg_string=svg, width=256, height=256))
    image = Image.open(io.BytesIO(png)).convert("RGBA")
    image.save(ICON, sizes=[(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)])
    image.save(os.path.join(HERE, "handwriter_preview.png"))
    print("иконка:", ICON)


def build_exe():
    work = os.path.join(HERE, "_build")
    subprocess.check_call([
        sys.executable, "-m", "PyInstaller", "--noconfirm", "--clean", "--onefile",
        "--noconsole", "--name", "Handwriter", "--icon", ICON,
        "--add-data", f"{ICON}{os.pathsep}.",
        "--distpath", os.path.join(work, "dist"), "--workpath", os.path.join(work, "work"),
        "--specpath", work, os.path.join(HERE, "launcher.py"),
    ])
    out = os.path.join(ROOT, "Handwriter.exe")
    shutil.copy2(os.path.join(work, "dist", "Handwriter.exe"), out)
    print(f"готово: {out}  ({os.path.getsize(out) / 1e6:.1f} МБ)")


if __name__ == "__main__":
    build_icon()
    if "--icon-only" not in sys.argv:
        build_exe()
