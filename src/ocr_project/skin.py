"""
The window's look, taken from the Figma design (design/window.svg for the
background and title bar, design/content.svg for the working area).

Every coordinate here is in window pixels -- the window's logical size is
1280 x 760 -- read straight out of the SVG exports rather than measured off a
picture. The working area's coordinates are content.svg's own plus the offset
it is placed at, (24.5, 55), which the asset build finds by matching the
Open photo pill in both files. The app multiplies them by the display scale.

The background, glass panels, button faces with their labels and the legend
are pre-rendered from the design by scripts/build_ui_assets.py; the app lays
live widgets over them at these positions.
"""
from __future__ import annotations

import os

from PIL import Image, ImageDraw, ImageEnhance

ASSET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "ui")
# 0.85 is for small screens: a 1366x768 display has about 688 px of height
# above its taskbar, and 760 does not fit in that even at 100%.
SCALES = (0.85, 1.0, 1.25, 1.5, 1.75, 2.0)
WIDTH, HEIGHT = 1280, 760

# window chrome -------------------------------------------------------------
WINDOW_RADIUS = 15
TITLE_BAR = (0, 0, 1280, 40)
# the placeholder note sat between the logo and the window controls
STATUS_CENTER = (712, 20)
STATUS_MAX_WIDTH = 800
MINIMIZE_HIT = (1134, 4, 52, 32)
CLOSE_HIT = (1214, 3, 40, 34)

# toolbar: (x, y, w, h) of each pill as drawn, and its corner radius --------
BUTTON_RADIUS = 15
BUTTONS = {
    "open":   (28.5, 55, 150, 45),
    "save":   (203.5, 55, 150, 45),
    "copy":   (378.5, 55, 150, 45),
    "rotate": (553.5, 55, 56, 45),
}

# The photo and text panels are built alike: a 570 x 570 glass panel, a
# vertical scrollbar beside it, a horizontal one under it, and a zoom below
# that -- "+" on the left, "-" on the right, the figure between them.
# "track" is where a thumb travels, between the two arrows; "back" and "fwd"
# are the arrows' hit areas, a little larger than the 12 px circles drawn.

# photo panel ---------------------------------------------------------------
PHOTO_PANEL = (29, 115, 570, 570)
PHOTO_VIEW = (31, 117, 566, 566)        # inside the panel's 1 px white edge
VSCROLL = {
    "track": (616.5, 130.5, 10, 539),
    "back":  (613.5, 114.5, 16, 16),    # up arrow
    "fwd":   (613.5, 669.5, 16, 16),    # down arrow
}
HSCROLL = {
    "track": (44, 702.5, 539, 10),
    "back":  (28, 699.5, 16, 16),       # left arrow
    "fwd":   (583, 699.5, 16, 16),      # right arrow
}
ZOOM_IN_HIT = (29.5, 721, 24, 24)       # the "+" circle
ZOOM_OUT_HIT = (118.5, 721, 24, 24)     # the "-" circle
ZOOM_LABEL_CENTER = (87.15, 732.5)

# text panel ----------------------------------------------------------------
TEXT_PANEL = (649, 115, 570, 570)
TEXT_VIEW = (651, 117, 566, 566)
TEXT_VSCROLL = {
    "track": (1236.5, 130.5, 10, 539),
    "back":  (1233.5, 114.5, 16, 16),
    "fwd":   (1233.5, 669.5, 16, 16),
}
TEXT_HSCROLL = {
    "track": (664, 702.5, 539, 10),
    "back":  (648, 699.5, 16, 16),
    "fwd":   (1203, 699.5, 16, 16),
}
TEXT_ZOOM_IN_HIT = (649.5, 721, 24, 24)
TEXT_ZOOM_OUT_HIT = (738.5, 721, 24, 24)
TEXT_ZOOM_LABEL_CENTER = (707.15, 732.5)
# the "100 %" figures are drawn ~13 px tall in the design
ZOOM_FONT_PX = 19

# The reading-mode switch, on the empty stretch of the bottom bar and ending
# level with the right edge of the photo panel (anchor "e").
READ_MODE_RIGHT = (597, 732.5)

TEXT_COLOR = "#1c1c1c"
STATUS_COLOR = "#141414"

# Line tints for the transcript, taken from the legend's own pills -- Low is
# drawn in #FF7F7F and Medium in #FFEE7F -- mixed into the panel's grey so a
# tinted line reads as the same colour as its pill.
LOW_TINT, MEDIUM_TINT = (255, 127, 127), (255, 238, 127)
TINT_STRENGTH = 0.32

# The box drawn around the piece of the photo being read. Deep red rather
# than one of the legend's tints: it lies on a photograph, where a pale
# colour disappears against paper, and it must not read as a confidence mark.
SELECT_COLOR = "#8c1c2b"


def pick_scale(dpi_scale: float, screen_w: int, screen_h: int,
               margin_w: int = 0, margin_h: int = 0) -> float:
    """The rendered scale nearest the display's, stepped down until the window
    fits the screen. A 1920x1080 laptop at 150% would otherwise get a
    1920x1140 window whose bottom is under the taskbar -- and a fixed window
    cannot be resized to recover it."""
    ranked = sorted(SCALES, key=lambda s: abs(s - dpi_scale))
    nearest = ranked[0]
    for s in sorted(SCALES, reverse=True):
        if s > nearest:
            continue
        if WIDTH * s <= screen_w - margin_w and HEIGHT * s <= screen_h - margin_h:
            return s
    return SCALES[0]


def load_background(scale: float) -> Image.Image:
    return Image.open(os.path.join(ASSET_DIR, f"bg@{scale:.2f}.png")).convert("RGBA")


def scaled(rect, scale: float):
    """Design (x, y, w, h) -> integer pixel box (left, top, right, bottom)."""
    x, y, w, h = rect
    return (round(x * scale), round(y * scale),
            round((x + w) * scale), round((y + h) * scale))


def mask(size, shape: str, radius: float) -> Image.Image:
    """Where a hover or press may show: the pill or the circle, not the square
    around it. Brightening the whole crop would leave a faint rectangle on the
    glass around every rounded button."""
    w, h = size
    m = Image.new("L", size, 0)
    draw = ImageDraw.Draw(m)
    if shape == "circle":
        draw.ellipse((0, 0, w - 1, h - 1), fill=255)
    else:
        draw.rounded_rectangle((0, 0, w - 1, h - 1), radius=radius, fill=255)
    return m


def button_states(crop: Image.Image, shape: str, radius: float) -> dict:
    """normal / hover / pressed / disabled faces for one button."""
    m = mask(crop.size, shape, radius)
    white = Image.new("RGBA", crop.size, (255, 255, 255, 255))
    looks = {
        "normal": crop,
        "hover": ImageEnhance.Brightness(crop).enhance(1.07),
        "pressed": ImageEnhance.Brightness(crop).enhance(0.9),
        "disabled": Image.blend(crop, white, 0.5),
    }
    return {k: Image.composite(v, crop, m) for k, v in looks.items()}


def thumb_image(background: Image.Image, box, radius: float) -> Image.Image:
    """A scrollbar thumb in the design's style -- white at 15% with a white
    edge, fully rounded -- composited over the background it sits on."""
    l, t, r, b = box
    under = background.crop(box).convert("RGBA")
    overlay = Image.new("RGBA", under.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    draw.rounded_rectangle((0, 0, r - l - 1, b - t - 1), radius=radius,
                           fill=(255, 255, 255, 38), outline=(255, 255, 255, 255), width=1)
    # A touch of shade so the thumb can be found at all: drawn exactly as the
    # design has it, white on a white track, it was all but invisible once the
    # app was running, and a scrollbar nobody can see does not tell anyone
    # there is more below.
    shade = Image.new("RGBA", under.size, (0, 0, 0, 0))
    ImageDraw.Draw(shade).rounded_rectangle((0, 0, r - l - 1, b - t - 1), radius=radius,
                                            outline=(110, 110, 110, 150), width=1)
    return Image.alpha_composite(Image.alpha_composite(under, shade), overlay)


def mean_color(background: Image.Image, box) -> tuple[int, int, int]:
    """Average colour of a region -- the flat stand-in for the glass behind a
    widget that cannot draw an image under itself."""
    r, g, b = background.crop(box).convert("RGB").resize((1, 1), Image.BOX).getpixel((0, 0))
    return r, g, b


def tint(base: tuple[int, int, int], color: tuple[int, int, int]) -> str:
    k = TINT_STRENGTH
    r, g, b = (round(base[i] * (1 - k) + color[i] * k) for i in range(3))
    return f"#{r:02x}{g:02x}{b:02x}"


def hex_color(rgb) -> str:
    return "#%02x%02x%02x" % tuple(rgb)


def window_shape(background: Image.Image, key: tuple[int, int, int]) -> Image.Image:
    """Flatten the rounded, partly transparent background onto the colour
    Windows will treat as see-through. Hard threshold rather than blending:
    a blended edge pixel is part key colour, and a key colour that is only
    nearly exact shows as a coloured fringe round the corners."""
    alpha = background.getchannel("A")
    solid = alpha.point(lambda a: 255 if a >= 128 else 0)
    flat = Image.new("RGB", background.size, key)
    flat.paste(background.convert("RGB"), (0, 0), solid)
    return flat
