"""
Desktop app: photo -> text.

The page sits on the left and its transcription on the right, the way a
person proofreads. Two engines feed it: the line recognizer, which cuts the
page into lines and tints each by how sure it was, and Qwen3-VL, which
reads the whole photo in one pass -- more accurate, but it needs a card
with about 8 GB, so the app uses it only where one is present. Text is
editable, so fixing a mistake and exporting a clean transcript is one flow.

The window is the Figma design (design/window.svg and design/content.svg):
its frosted background, glass panels, button faces and legend are
pre-rendered images (see skin.py), and the live parts -- photo, text, status,
zoom figures, scroll thumbs -- are laid over them at the design's own
coordinates.
"""
from __future__ import annotations

import ctypes
import os
import queue
import threading
import tkinter as tk
import tkinter.font as tkfont
from tkinter import filedialog, messagebox

import cv2
import numpy as np
from PIL import Image, ImageTk

from . import page_reader, skin
from .recognizer import RecognizedLine, Recognizer

# Confidence below this is treated as "look at this one".
LOW_CONFIDENCE = 0.80
MEDIUM_CONFIDENCE = 0.90

ZOOM_MIN, ZOOM_MAX, ZOOM_STEP = 1.0, 8.0, 1.25
SELECT_HINT = "Обведите участок мышью — прочитаю только его"
# Text zooms by type size. It may go below 100% -- smaller type shows more of
# a long transcript at once -- where the photo never goes below fitting.
TEXT_ZOOM_MIN, TEXT_ZOOM_MAX = 0.6, 3.0
TEXT_PX = 17

# Painted where the window should be see-through -- outside its rounded
# corners. Magenta, because the design is greys and a key colour that turned
# up inside the window would punch a hole in it.
KEY = (255, 0, 255)


def _enable_crisp_text():
    """Tell Windows the app handles display scaling itself. Without this a
    laptop at 125-150% scaling stretches the whole window as a bitmap and
    every letter comes out blurred."""
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        pass    # not Windows, or too old a Windows -- blurry is still usable


class SkinButton:
    """A button that is a region of the pre-rendered design. The face in the
    background image is the button; this lays hover, pressed and disabled
    versions of that same face over it."""

    def __init__(self, app, rect, command, shape="round", radius=skin.BUTTON_RADIUS):
        self.app, self.command = app, command
        box = skin.scaled(rect, app.s)
        crop = app._bg_pil.crop(box)
        faces = skin.button_states(crop, shape, radius * app.s)
        self.faces = {k: ImageTk.PhotoImage(v.convert("RGB")) for k, v in faces.items()}
        self.item = app.ui.create_image(box[0], box[1], anchor="nw", image=self.faces["normal"])
        self.enabled, self.hover, self.pressed = True, False, False
        ui = app.ui
        ui.tag_bind(self.item, "<Enter>", self._enter)
        ui.tag_bind(self.item, "<Leave>", self._leave)
        ui.tag_bind(self.item, "<ButtonPress-1>", self._press)
        ui.tag_bind(self.item, "<ButtonRelease-1>", self._release)

    def set_enabled(self, on: bool):
        self.enabled = bool(on)
        self._refresh()

    def _refresh(self):
        if not self.enabled:
            face = "disabled"
        elif self.pressed and self.hover:
            face = "pressed"
        elif self.hover:
            face = "hover"
        else:
            face = "normal"
        self.app.ui.itemconfigure(self.item, image=self.faces[face])
        self.app.ui.configure(cursor="hand2" if self.hover and self.enabled else "")

    def _enter(self, _):
        self.hover = True
        self._refresh()

    def _leave(self, _):
        self.hover = self.pressed = False
        self._refresh()

    def _press(self, _):
        self.pressed = True
        self._refresh()

    def _release(self, _):
        fire = self.pressed and self.hover and self.enabled
        self.pressed = False
        self._refresh()
        if fire:
            self.command()


class ThinScrollbar:
    """The design's scrollbar: a slim rounded thumb on a track that is part of
    the background image, with the arrow circles at either end clickable.
    Speaks the same set()/command protocol as a ttk.Scrollbar."""

    def __init__(self, app, orient, spec, command):
        self.app, self.orient, self.command = app, orient, command
        self.track = skin.scaled(spec["track"], app.s)
        self.item, self._image, self._span = None, None, (0.0, 1.0)
        self._grab = None
        for key, step in (("back", -1), ("fwd", 1)):
            if key in spec:
                SkinButton(app, spec[key], lambda n=step: command("scroll", n, "units"),
                           shape="circle")

    def _extent(self):
        l, t, r, b = self.track
        return (l, r) if self.orient == "horizontal" else (t, b)

    def set(self, first, last):
        first, last = float(first), float(last)
        self._span = (first, last)
        ui = self.app.ui
        if last - first >= 0.999:          # nothing to scroll: no thumb
            if self.item is not None:
                ui.itemconfigure(self.item, state="hidden")
            return
        start, end = self._extent()
        length = end - start
        size = max(round(24 * self.app.s), round(length * (last - first)))
        pos = start + round((length - size) * first / max(1e-6, 1 - (last - first)))
        l, t, r, b = self.track
        box = (pos, t, pos + size, b) if self.orient == "horizontal" else (l, pos, r, pos + size)
        thickness = (b - t) if self.orient == "horizontal" else (r - l)
        self._image = ImageTk.PhotoImage(
            skin.thumb_image(self.app._bg_pil, box, thickness / 2).convert("RGB"))
        if self.item is None:
            self.item = ui.create_image(box[0], box[1], anchor="nw", image=self._image)
            ui.tag_bind(self.item, "<ButtonPress-1>", self._press)
            ui.tag_bind(self.item, "<B1-Motion>", self._drag)
        else:
            ui.coords(self.item, box[0], box[1])
            ui.itemconfigure(self.item, image=self._image, state="normal")
        self._thumb = (box[0], box[2]) if self.orient == "horizontal" else (box[1], box[3])

    def _coord(self, event):
        return event.x if self.orient == "horizontal" else event.y

    def _press(self, event):
        self._grab = self._coord(event) - self._thumb[0]

    def _drag(self, event):
        start, end = self._extent()
        size = self._thumb[1] - self._thumb[0]
        first, last = self._span
        travel = max(1, (end - start) - size)
        frac = (self._coord(event) - self._grab - start) / travel
        frac = min(1.0, max(0.0, frac)) * (1 - (last - first))
        self.command("moveto", frac)


class HandwritingApp(tk.Tk):
    def __init__(self):
        _enable_crisp_text()
        super().__init__()
        self.title("Handwriter")
        self._setup_window()

        self.recognizer: Recognizer | None = None
        self.page_reader: page_reader.PageReader | None = None
        # Whole-page reading where the card can take it, lines everywhere else.
        # The design has no switch for this, and the only sensible choice is
        # the hardware's anyway.
        self.whole_page = page_reader.enough_vram()
        self._page_image = None
        self._page_name = ""
        self._busy = False
        self._region = None          # the piece of the page to read, in image pixels
        self._selecting = False
        self._select_from = (0, 0)
        # set once the person has agreed to whole-page reading on a card too
        # small for it, which puts part of the model on the processor
        self._slow_whole_page = False

        # Worker threads must not touch tkinter (not even via .after());
        # they post messages here and the main thread drains the queue.
        self._events: queue.Queue[tuple] = queue.Queue()

        self._build_ui()
        self.status.set("Загрузка модели…")
        threading.Thread(target=self._load_model, daemon=True).start()
        self.after(50, self._drain_events)

    # -- window ---------------------------------------------------------------

    def _setup_window(self):
        dpi = self.winfo_fpixels("1i") / 96.0          # 1.0 at 100%, 1.5 at 150%
        sw, sh = self.winfo_screenwidth(), self.winfo_screenheight()
        self.s = skin.pick_scale(dpi, sw, sh, margin_w=round(20 * dpi), margin_h=round(80 * dpi))
        self.W, self.H = round(skin.WIDTH * self.s), round(skin.HEIGHT * self.s)
        x = max(0, (sw - self.W) // 2)
        y = max(0, (sh - self.H) // 2 - round(20 * dpi))
        self.geometry(f"{self.W}x{self.H}+{x}+{y}")
        self.resizable(False, False)
        # The design draws its own title bar, so Windows' is removed.
        self.overrideredirect(True)
        self.configure(bg=skin.hex_color(KEY))
        try:
            self.attributes("-transparentcolor", skin.hex_color(KEY))
        except tk.TclError:
            pass
        self.after(10, self._show_in_taskbar)

    def _hwnd(self):
        return ctypes.windll.user32.GetParent(self.winfo_id())

    def _show_in_taskbar(self, attempts: int = 40):
        """A window without Windows' title bar also loses its taskbar button
        and its place in Alt+Tab. Marking it an app window brings both back.

        It has to wait for the window to exist. Scheduled straight after
        construction, this ran before Tk had created the real window -- the
        layout build in between is heavy -- so the handle it got was 0 and the
        style change went nowhere, silently. It now retries until there is a
        handle, and checks that the change took."""
        GWL_EXSTYLE, WS_EX_APPWINDOW, WS_EX_TOOLWINDOW = -20, 0x00040000, 0x00000080
        try:
            self.update_idletasks()
            user32 = ctypes.windll.user32
            hwnd = self._hwnd()
            if not hwnd:
                if attempts:
                    self.after(50, self._show_in_taskbar, attempts - 1)
                return
            style = user32.GetWindowLongW(hwnd, GWL_EXSTYLE)
            user32.SetWindowLongW(hwnd, GWL_EXSTYLE, (style & ~WS_EX_TOOLWINDOW) | WS_EX_APPWINDOW)
            self.withdraw()
            self.after(10, self._reappear)
        except Exception:
            pass    # not Windows: no taskbar button to restore

    def _reappear(self):
        self.deiconify()
        self.focus_force()

    def _minimize(self):
        # Tk refuses to iconify a window without a title bar; Windows will.
        try:
            ctypes.windll.user32.ShowWindow(self._hwnd(), 6)      # SW_MINIMIZE
        except Exception:
            self.overrideredirect(False)
            self.iconify()

    def _drag_start(self, event):
        if event.y > skin.TITLE_BAR[3] * self.s:
            self._drag = None
            return
        self._drag = (event.x_root - self.winfo_x(), event.y_root - self.winfo_y())

    def _drag_move(self, event):
        if getattr(self, "_drag", None):
            self.geometry(f"+{event.x_root - self._drag[0]}+{event.y_root - self._drag[1]}")

    # -- layout ---------------------------------------------------------------

    def _build_ui(self):
        s = self.s
        self._bg_pil = skin.load_background(s)
        self.ui = tk.Canvas(self, width=self.W, height=self.H, highlightthickness=0,
                            bd=0, bg=skin.hex_color(KEY))
        self.ui.place(x=0, y=0)
        self._bg_photo = ImageTk.PhotoImage(skin.window_shape(self._bg_pil, KEY))
        bg = self.ui.create_image(0, 0, anchor="nw", image=self._bg_photo)
        self.ui.tag_bind(bg, "<ButtonPress-1>", self._drag_start)
        self.ui.tag_bind(bg, "<B1-Motion>", self._drag_move)

        # status, where the design's note in the title bar was
        self.status = tk.StringVar(value="")
        self._status_font = tkfont.Font(family="Segoe UI", size=-round(16 * s))
        cx, cy = skin.STATUS_CENTER
        self._status_item = self.ui.create_text(cx * s, cy * s, text="", anchor="center",
                                                fill=skin.STATUS_COLOR, font=self._status_font)
        self.ui.tag_bind(self._status_item, "<ButtonPress-1>", self._drag_start)
        self.ui.tag_bind(self._status_item, "<B1-Motion>", self._drag_move)
        self.status.trace_add("write", lambda *_: self._render_status())

        SkinButton(self, skin.MINIMIZE_HIT, self._minimize, radius=8)
        SkinButton(self, skin.CLOSE_HIT, self.destroy, radius=8)

        self.buttons = {
            "open": SkinButton(self, skin.BUTTONS["open"], self.on_open),
            "save": SkinButton(self, skin.BUTTONS["save"], self.on_save),
            "copy": SkinButton(self, skin.BUTTONS["copy"], self.on_copy),
            "rotate": SkinButton(self, skin.BUTTONS["rotate"], self.on_rotate),
        }

        self._build_photo_view()
        self._build_text_view()
        self._update_buttons()

    def _build_photo_view(self):
        s = self.s
        l, t, r, b = skin.scaled(skin.PHOTO_VIEW, s)
        self.canvas = tk.Canvas(self.ui, width=r - l, height=b - t, highlightthickness=0, bd=0)
        self.ui.create_window(l, t, anchor="nw", window=self.canvas)
        # The glass the photo lies on. A canvas cannot see through to its
        # parent, so the design's panel is drawn into it and kept pinned to
        # the view while the page scrolls over it.
        self._photo_glass = ImageTk.PhotoImage(self._bg_pil.crop((l, t, r, b)).convert("RGB"))
        self._glass_item = self.canvas.create_image(0, 0, anchor="nw", image=self._photo_glass)

        self.vscroll = ThinScrollbar(self, "vertical", skin.VSCROLL, self.canvas.yview)
        self.hscroll = ThinScrollbar(self, "horizontal", skin.HSCROLL, self.canvas.xview)
        self.canvas.configure(yscrollcommand=self._on_yscroll, xscrollcommand=self._on_xscroll)

        self.zoom_in = SkinButton(self, skin.ZOOM_IN_HIT, lambda: self._zoom_by(ZOOM_STEP),
                                  shape="circle")
        self.zoom_out = SkinButton(self, skin.ZOOM_OUT_HIT, lambda: self._zoom_by(1 / ZOOM_STEP),
                                   shape="circle")
        self._zoom_font = ("Segoe UI", -round(skin.ZOOM_FONT_PX * s))
        zx, zy = skin.ZOOM_LABEL_CENTER
        self._zoom_item = self.ui.create_text(zx * s, zy * s, text="100 %", anchor="center",
                                              fill=skin.TEXT_COLOR, font=self._zoom_font)

        # Which way the page is read, on the empty stretch of the bottom bar.
        # The hardware picks the first mode, but both are offered everywhere:
        # whole-page reads more accurately, line-by-line marks the lines it
        # doubts, and only the person reading knows which they need now.
        mx, my = skin.READ_MODE_RIGHT
        self._mode_item = self.ui.create_text(mx * s, my * s, text="", anchor="e",
                                              fill=skin.TEXT_COLOR, font=self._zoom_font)
        self.ui.tag_bind(self._mode_item, "<Button-1>", self._toggle_mode)
        self.ui.tag_bind(self._mode_item, "<Enter>",
                         lambda e: self.ui.configure(cursor="hand2"))
        self.ui.tag_bind(self._mode_item, "<Leave>", lambda e: self.ui.configure(cursor=""))
        self._update_mode_label()

        self._page_photo = None
        self._zoom = ZOOM_MIN            # 1.0 = the whole page fits the panel
        self._offset = (0, 0)
        self._shown = (1, 1)

        # Ctrl + wheel zooms around the cursor; the plain wheel scrolls, and
        # shift + wheel scrolls sideways -- the conventions of image viewers.
        self.canvas.bind("<Control-MouseWheel>", self._on_wheel_zoom)
        self.canvas.bind("<MouseWheel>",
                         lambda e: self.canvas.yview_scroll(-e.delta // 120, "units"))
        self.canvas.bind("<Shift-MouseWheel>",
                         lambda e: self.canvas.xview_scroll(-e.delta // 120, "units"))
        # drag to move around an enlarged page; double-click to fit it again
        self.canvas.bind("<ButtonPress-1>", self._on_press)
        self.canvas.bind("<B1-Motion>", self._on_motion)
        self.canvas.bind("<ButtonRelease-1>", self._on_release)
        self.canvas.bind("<Double-Button-1>", self._on_double_click)
        self.canvas.bind("<Enter>", self._hint_on)
        self.canvas.bind("<Leave>", self._hint_off)
        # Keyboard zoom acts on whichever panel has the focus: typing in the
        # transcript and pressing Ctrl+plus should enlarge the text, not the
        # photo on the other side of the window.
        for key in ("<Control-plus>", "<Control-equal>", "<Control-KP_Add>"):
            self.bind(key, lambda e: self._zoom_key(ZOOM_STEP))
        for key in ("<Control-minus>", "<Control-KP_Subtract>"):
            self.bind(key, lambda e: self._zoom_key(1 / ZOOM_STEP))
        self.bind("<Control-0>", lambda e: self._zoom_key(None))
        self._update_zoom_controls()

    def _build_text_view(self):
        s = self.s
        l, t, r, b = skin.scaled(skin.TEXT_VIEW, s)
        base = skin.mean_color(self._bg_pil, skin.scaled(skin.TEXT_PANEL, s))
        self._text_zoom = 1.0
        self._text_font = tkfont.Font(family="Georgia", size=-round(TEXT_PX * s))
        # A Text widget cannot show an image under its lines, so it takes the
        # panel's average colour, and stays hidden until there is text: the
        # empty panel is then the design's glass exactly.
        #
        # No wrapping: one line of transcript is one line of the page, and the
        # design gives the text a horizontal scrollbar of its own, which would
        # have nothing to scroll if long lines folded -- once the type is
        # enlarged, they run past the edge and the scrollbar reaches them.
        self.text = tk.Text(self.ui, wrap=tk.NONE, font=self._text_font,
                            bg=skin.hex_color(base), fg=skin.TEXT_COLOR, bd=0,
                            highlightthickness=0, padx=round(14 * s), pady=round(12 * s),
                            undo=True, insertbackground=skin.TEXT_COLOR,
                            selectbackground="#b9c7d8", spacing1=round(2 * s))
        self._text_window = self.ui.create_window(l, t, anchor="nw", window=self.text,
                                                  width=r - l, height=b - t, state="hidden")
        # Confidence is shown by tinting whole lines in place, so the text
        # stays one editable block you can select and copy across.
        self.text.tag_configure("low", background=skin.tint(base, skin.LOW_TINT))
        self.text.tag_configure("medium", background=skin.tint(base, skin.MEDIUM_TINT))

        self.tvscroll = ThinScrollbar(self, "vertical", skin.TEXT_VSCROLL, self.text.yview)
        self.thscroll = ThinScrollbar(self, "horizontal", skin.TEXT_HSCROLL, self.text.xview)
        self.text.configure(yscrollcommand=self.tvscroll.set, xscrollcommand=self.thscroll.set)

        self.text_zoom_in = SkinButton(self, skin.TEXT_ZOOM_IN_HIT,
                                       lambda: self._set_text_zoom(self._text_zoom * ZOOM_STEP),
                                       shape="circle")
        self.text_zoom_out = SkinButton(self, skin.TEXT_ZOOM_OUT_HIT,
                                        lambda: self._set_text_zoom(self._text_zoom / ZOOM_STEP),
                                        shape="circle")
        zx, zy = skin.TEXT_ZOOM_LABEL_CENTER
        self._text_zoom_item = self.ui.create_text(zx * s, zy * s, text="100 %", anchor="center",
                                                   fill=skin.TEXT_COLOR, font=self._zoom_font)
        self.text.bind("<Control-MouseWheel>", self._on_text_wheel_zoom)
        self._update_text_zoom_controls()

    def _on_text_wheel_zoom(self, event):
        self._set_text_zoom(self._text_zoom * (ZOOM_STEP if event.delta > 0 else 1 / ZOOM_STEP))
        return "break"          # or the Text widget also scrolls by the same turn

    def _set_text_zoom(self, zoom):
        zoom = min(TEXT_ZOOM_MAX, max(TEXT_ZOOM_MIN, zoom))
        # snap onto 100% when a step lands next to it, so zooming out and back
        # in returns to exactly the designed size rather than 99% or 101%
        if abs(zoom - 1.0) < 0.02:
            zoom = 1.0
        self._text_zoom = zoom
        self._text_font.configure(size=-round(TEXT_PX * self.s * zoom))
        self._update_text_zoom_controls()

    def _update_text_zoom_controls(self):
        self.ui.itemconfigure(self._text_zoom_item, text=f"{self._text_zoom * 100:.0f} %")
        shown = self.ui.itemcget(self._text_window, "state") != "hidden"
        self.text_zoom_in.set_enabled(shown and self._text_zoom < TEXT_ZOOM_MAX - 1e-6)
        self.text_zoom_out.set_enabled(shown and self._text_zoom > TEXT_ZOOM_MIN + 1e-6)

    def _zoom_key(self, factor):
        """Keyboard zoom: the text when it has the focus, else the photo.
        factor None means back to the designed size."""
        if self.focus_get() is self.text:
            self._set_text_zoom(1.0 if factor is None else self._text_zoom * factor)
        elif factor is None:
            self._set_zoom(ZOOM_MIN)
        else:
            self._zoom_by(factor)

    def _render_status(self):
        """Fit the status into the space between the logo and the window
        controls, cutting it with an ellipsis rather than letting it run
        under either."""
        text = self.status.get()
        limit = skin.STATUS_MAX_WIDTH * self.s
        if self._status_font.measure(text) > limit:
            while text and self._status_font.measure(text + "…") > limit:
                text = text[:-1]
            text += "…"
        self.ui.itemconfigure(self._status_item, text=text)

    def _update_buttons(self):
        ready = self.recognizer is not None and not self._busy
        has_text = bool(self._current_text()) and not self._busy
        self.buttons["open"].set_enabled(ready)
        self.buttons["save"].set_enabled(has_text)
        self.buttons["copy"].set_enabled(has_text)
        self.buttons["rotate"].set_enabled(ready and self._page_image is not None)
        # the mode switch greys out while a page is being read
        self.ui.itemconfigure(self._mode_item, fill=skin.TEXT_COLOR if ready else "#8a8a8a")

    # -- photo view -----------------------------------------------------------

    def _on_yscroll(self, first, last):
        self.vscroll.set(first, last)
        self._pin_glass()

    def _on_xscroll(self, first, last):
        self.hscroll.set(first, last)
        self._pin_glass()

    def _pin_glass(self):
        self.canvas.coords(self._glass_item, self.canvas.canvasx(0), self.canvas.canvasy(0))
        self.canvas.tag_lower(self._glass_item)

    def _update_zoom_controls(self):
        self.ui.itemconfigure(self._zoom_item, text=f"{self._zoom * 100:.0f} %")
        has_page = self._page_image is not None
        self.zoom_out.set_enabled(has_page and self._zoom > ZOOM_MIN)
        self.zoom_in.set_enabled(has_page and self._zoom < ZOOM_MAX)
        # crosshair where a drag draws a box, the hand where it moves the photo
        self.canvas.configure(cursor="fleur" if self._zoom > ZOOM_MIN
                              else ("crosshair" if has_page else ""))

    def _zoom_by(self, factor, anchor=None):
        self._set_zoom(self._zoom * factor, anchor)

    def _set_zoom(self, zoom, anchor=None):
        """Change magnification, keeping the point under `anchor` (canvas
        window coordinates) where it was -- the centre of the view if none."""
        if self._page_image is None:
            return
        zoom = min(ZOOM_MAX, max(ZOOM_MIN, zoom))
        if abs(zoom - self._zoom) < 1e-6:
            return
        if anchor is None:
            anchor = (self.canvas.winfo_width() / 2, self.canvas.winfo_height() / 2)
        ax, ay = anchor
        # where the anchor sits on the page, as a fraction of the shown image
        ox, oy = self._offset
        sw, sh = self._shown
        fx = (self.canvas.canvasx(ax) - ox) / max(1, sw)
        fy = (self.canvas.canvasy(ay) - oy) / max(1, sh)

        self._zoom = zoom
        self._redraw_page()

        ox, oy = self._offset
        sw, sh = self._shown
        total_w = max(self.canvas.winfo_width(), sw)
        total_h = max(self.canvas.winfo_height(), sh)
        self.canvas.xview_moveto(max(0.0, (ox + fx * sw - ax) / total_w))
        self.canvas.yview_moveto(max(0.0, (oy + fy * sh - ay) / total_h))

    def _on_wheel_zoom(self, event):
        self._zoom_by(ZOOM_STEP if event.delta > 0 else 1 / ZOOM_STEP, (event.x, event.y))

    # -- reading one piece of the page ----------------------------------------
    #
    # A stamp in a passport is a few lines of writing inside a large photograph
    # of a book. Reading the whole frame spends the model's attention on the
    # cover, the fingers and the printed form around it; boxing the stamp hands
    # it the writing and nothing else.

    def _on_press(self, event):
        self.canvas.scan_mark(event.x, event.y)
        # At the size that fits the panel there is nothing to drag the photo
        # around by, so a drag there means "read this bit". Magnified, a drag
        # still moves the photo and Shift draws the box instead.
        self._selecting = (self._page_image is not None and not self._busy
                           and (self._zoom <= ZOOM_MIN + 1e-6 or bool(event.state & 0x0001)))
        self._select_from = (self.canvas.canvasx(event.x), self.canvas.canvasy(event.y))

    def _on_motion(self, event):
        if not self._selecting:
            self.canvas.scan_dragto(event.x, event.y, gain=1)
            return
        x0, y0 = self._select_from
        self._draw_box(x0, y0, self.canvas.canvasx(event.x), self.canvas.canvasy(event.y))

    def _on_release(self, event):
        if not self._selecting:
            return
        self._selecting = False
        x0, y0 = self._select_from
        region = self._to_image_box(x0, y0, self.canvas.canvasx(event.x), self.canvas.canvasy(event.y))
        if region is None:              # a click, or a box too small to hold writing
            self.canvas.delete("region")
            self._draw_region()
            return
        self._region = region
        self._draw_region()
        self._start_recognition()

    def _on_double_click(self, _event):
        """Back to the whole page, and back to the size that fits."""
        self._set_zoom(ZOOM_MIN)
        if self._region is None or self._busy:
            # mid-read the box stays: it is what is being read right now
            return
        self._region = None
        self.canvas.delete("region")
        self._start_recognition()

    def _to_image_box(self, x0, y0, x1, y1):
        """Canvas coordinates to page pixels, or None if what was drawn is too
        small to be a selection rather than a stray click."""
        if self._page_image is None:
            return None
        h, w = self._page_image.shape[:2]
        ox, oy = self._offset
        sw, sh = self._shown
        left, right = sorted((x0, x1))
        top, bottom = sorted((y0, y1))
        box = [int((left - ox) / sw * w), int((top - oy) / sh * h),
               int((right - ox) / sw * w), int((bottom - oy) / sh * h)]
        if box[2] - box[0] < 24 or box[3] - box[1] < 12:
            return None
        # A hand-drawn box lands on the writing, not around it, so it is given
        # a little air: otherwise the tails of "у" and "р" are cut off and the
        # model has to guess at half a letter.
        # Kept small on purpose: at 2% of a page-wide box the margin reached
        # into the line above and read half of it as well.
        pad = min(12, max(4, int(0.01 * max(box[2] - box[0], box[3] - box[1]))))
        return (max(0, box[0] - pad), max(0, box[1] - pad),
                min(w, box[2] + pad), min(h, box[3] + pad))

    def _draw_box(self, x0, y0, x1, y1):
        self.canvas.delete("region")
        self.canvas.create_rectangle(x0, y0, x1, y1, outline=skin.SELECT_COLOR,
                                     width=2, dash=(5, 3), tags="region")

    def _draw_region(self):
        """Redraw the box from page pixels, so it stays on the same writing
        when the photo is magnified or moved."""
        self.canvas.delete("region")
        if self._region is None or self._page_image is None:
            return
        h, w = self._page_image.shape[:2]
        ox, oy = self._offset
        sw, sh = self._shown
        x0, y0, x1, y1 = self._region
        self._draw_box(ox + x0 / w * sw, oy + y0 / h * sh, ox + x1 / w * sw, oy + y1 / h * sh)

    def _hint_on(self, _event):
        """Say what a drag over the photo does -- there is no room in the
        design for a caption, and the gesture is otherwise invisible."""
        if self._page_image is None or self._busy:
            return
        self._hint_under = self.status.get()
        self.status.set(SELECT_HINT)

    def _hint_off(self, _event):
        if self.status.get() == SELECT_HINT:      # only if nothing else was said meanwhile
            self.status.set(getattr(self, "_hint_under", ""))

    def _redraw_page(self):
        """Show the page at the current magnification. At 100% it fits the
        panel and sits centred; above that it overflows and scrolls."""
        image = self._page_image
        if image is None:
            return
        avail_w = max(self.canvas.winfo_width(), 1)
        avail_h = max(self.canvas.winfo_height(), 1)
        h, w = image.shape[:2]
        scale = min(avail_w / w, avail_h / h) * self._zoom
        if scale <= 0:
            return
        new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
        # cv2 rather than PIL's Lanczos: this runs on every zoom step, and a
        # 12-megapixel photo enlarged 8x has to redraw without a visible stall
        interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
        rgb = cv2.cvtColor(cv2.resize(image, (new_w, new_h), interpolation=interp),
                           cv2.COLOR_BGR2RGB)

        # keep a reference or tkinter garbage-collects the image away
        self._page_photo = ImageTk.PhotoImage(Image.fromarray(rgb))
        ox, oy = max(0, (avail_w - new_w) // 2), max(0, (avail_h - new_h) // 2)
        self._offset, self._shown = (ox, oy), (new_w, new_h)
        self.canvas.delete("page")
        self.canvas.create_image(ox, oy, anchor="nw", image=self._page_photo, tags="page")
        self.canvas.configure(scrollregion=(0, 0, max(avail_w, new_w), max(avail_h, new_h)))
        self._pin_glass()
        self._draw_region()
        self._update_zoom_controls()

    # -- models ---------------------------------------------------------------

    def _drain_events(self):
        try:
            while True:
                kind, payload = self._events.get_nowait()
                if kind == "model_ready":
                    self._model_ready(payload)
                elif kind == "model_failed":
                    self._model_failed(payload)
                elif kind == "recognized":
                    self._show_results(*payload)
                elif kind == "recognize_failed":
                    self._recognize_failed(*payload)
                elif kind == "status":
                    self.status.set(payload)
                elif kind == "progress":
                    self._update_progress(*payload)
        except queue.Empty:
            pass
        self.after(50, self._drain_events)

    def _load_model(self):
        try:
            recognizer = Recognizer()
        except Exception as exc:  # surfacing the real reason beats a silent dead button
            self._events.put(("model_failed", exc))
            return
        self._events.put(("model_ready", recognizer))

    def _model_ready(self, recognizer: Recognizer):
        self.recognizer = recognizer
        where = "GPU" if recognizer.device == "cuda" else "CPU"
        mode = "страница целиком" if self.whole_page else "по строкам"
        self.status.set(f"Готово ({mode}, {where}). Откройте фото страницы.")
        self._update_buttons()

        # Explain the fall back to CPU once, up front -- otherwise the only
        # symptom is that every page takes several minutes for no visible
        # reason.
        if getattr(recognizer, "device_warning", None):
            messagebox.showwarning("Видеокарта не используется", recognizer.device_warning,
                                   parent=self)

    def _model_failed(self, exc: Exception):
        self.status.set("Не удалось загрузить модель")
        messagebox.showerror("Ошибка загрузки модели", str(exc), parent=self)

    # -- actions --------------------------------------------------------------

    def on_open(self):
        path = filedialog.askopenfilename(
            parent=self, title="Выберите фото страницы",
            filetypes=[("Изображения", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff"),
                       ("Все файлы", "*.*")],
        )
        if not path:
            return
        image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            messagebox.showerror("Не открылось", "Этот файл не удалось прочитать как изображение.",
                                 parent=self)
            return
        self._page_image, self._page_name = image, os.path.basename(path)
        self._region = None
        self._show_page()
        self._start_recognition()

    def _update_mode_label(self):
        self.ui.itemconfigure(self._mode_item,
                              text="чтение: целиком" if self.whole_page else "чтение: по строкам")

    def _toggle_mode(self, _event=None):
        """Switch between reading the page whole and reading it line by line.

        Whole page reads more accurately; line by line colours the lines it
        doubts and needs a fraction of the memory. The hardware chooses at
        startup, but neither mode is hidden: on a card too small for the big
        model it still runs with part of the work on the processor, which is
        slow enough to be worth saying out loud first.
        """
        if self._busy or self.recognizer is None:
            return
        want_whole = not self.whole_page
        if want_whole and not self._slow_whole_page and not page_reader.enough_vram():
            if not page_reader.can_split():
                messagebox.showinfo(
                    "Чтение страницы целиком",
                    "На этом компьютере такой режим не получится: нужна видеокарта "
                    "NVIDIA и не меньше 20 ГБ оперативной памяти.\n\n"
                    "Приложение продолжит читать по строкам — этот режим ещё и "
                    "подсвечивает строки, в которых сомневается.", parent=self)
                return
            if not messagebox.askyesno(
                    "Чтение страницы целиком",
                    "Видеокарты для этого режима не хватает, но прочитать можно: "
                    "модель поместится в оперативную память, а в видеокарту будет "
                    "подгружаться по частям.\n\n"
                    "Это медленно: выделенный участок — несколько минут, целая "
                    "страница — десятки минут, и компьютер всё это время сильно "
                    "загружен.\n\nВключить такой режим?", parent=self):
                return
            self._slow_whole_page = True
        self.whole_page = want_whole
        self._update_mode_label()
        if self._page_image is not None:
            self._start_recognition()

    def on_rotate(self):
        """A quarter turn clockwise, then read again: a phone often saves a
        page on its side, and deskewing corrects a few degrees, not ninety."""
        if self._page_image is None or self._busy:
            return
        self._page_image = cv2.rotate(self._page_image, cv2.ROTATE_90_CLOCKWISE)
        self._region = None          # its coordinates belong to the old orientation
        self._show_page()
        self._start_recognition()

    def _show_page(self):
        self.canvas.delete("region")
        self._zoom = ZOOM_MIN
        self.canvas.xview_moveto(0)
        self.canvas.yview_moveto(0)
        self._redraw_page()

    def _start_recognition(self):
        self.text.delete("1.0", tk.END)
        self.ui.itemconfigure(self._text_window, state="hidden")
        self._update_text_zoom_controls()
        self._busy = True
        self._update_buttons()
        region = self._region
        self.status.set(f"{self._page_name}: " + (
            "читаю выделенный участок…" if region
            else ("читаю страницу целиком…" if self.whole_page else "ищу строки…")))
        # The mode and the selection are read here, on the main thread, and
        # handed to the worker with a copy of what it must read: worker threads
        # must not touch tkinter, nor a page a rotation could replace under them.
        image = self._page_image
        if region:
            x0, y0, x1, y1 = region
            image = image[y0:y1, x0:x1]
        threading.Thread(target=self._recognize,
                         args=(image.copy(), self._page_name, self.whole_page, bool(region)),
                         daemon=True).start()

    def _recognize(self, image, name: str, whole: bool, region: bool = False):
        try:
            if whole:
                lines = self._read_whole_page(image, region)
            else:
                lines = self.recognizer.recognize_page(
                    image, progress=lambda done, total: self._events.put(("progress", (done, total)))
                )
        except Exception as exc:
            self._events.put(("recognize_failed", (exc, whole)))
            return
        self._events.put(("recognized", (name, lines)))

    def _read_whole_page(self, image, region: bool = False):
        """Qwen3-VL reads the page in one pass, so there is no per-line
        confidence to colour by -- every line comes back unmarked."""
        if self.page_reader is None:
            self._events.put(("status", "Загружаю модель чтения страницы "
                                        "(в первый раз качается ~6 ГБ)…"))
            self.page_reader = page_reader.PageReader(offload=self._slow_whole_page)
        def progress(done, total):
            self._events.put(("progress", (done, total)))

        read = self.page_reader.read_region if region else self.page_reader.read_page
        texts = read(image, progress=progress)
        return [
            RecognizedLine(index=i, bbox=(0, 0, 0, 0), text=t, confidence=1.0, image=None)
            for i, t in enumerate(texts)
        ]

    def status_from_worker(self, message: str):
        self._events.put(("status", message))

    def _update_progress(self, done: int, total: int):
        if self._region:
            self.status.set(f"{self._page_name}: читаю выделенный участок…")
        elif self.whole_page:
            self.status.set(f"{self._page_name}: читаю страницу целиком…")
        else:
            self.status.set(f"{self._page_name}: строка {done} из {total}")

    def _recognize_failed(self, exc: Exception, whole: bool):
        self._busy = False
        if whole and "out of memory" in str(exc).lower():
            # There is no switch to flip by hand any more, so the app flips it:
            # the rest of this session reads by lines, which needs a fraction
            # of the memory, and this page is read again that way.
            self.whole_page = False
            self._update_mode_label()
            self.status.set("Не хватило видеопамяти — читаю по строкам")
            self._start_recognition()
            return
        self.status.set("Ошибка распознавания")
        self._update_buttons()
        messagebox.showerror("Ошибка распознавания", str(exc), parent=self)

    def _show_results(self, filename: str, lines):
        self._busy = False
        if not lines:
            self.status.set(f"{filename}: строки не найдены")
            self._update_buttons()
            return

        flagged = sum(1 for line in lines if line.confidence < MEDIUM_CONFIDENCE)
        self.text.delete("1.0", tk.END)
        for i, line in enumerate(lines, start=1):
            self.text.insert(tk.END, line.text + "\n")
            if line.confidence < LOW_CONFIDENCE:
                self.text.tag_add("low", f"{i}.0", f"{i}.end")
            elif line.confidence < MEDIUM_CONFIDENCE:
                self.text.tag_add("medium", f"{i}.0", f"{i}.end")
        self.ui.itemconfigure(self._text_window, state="normal")
        self._update_text_zoom_controls()

        what = "участок" if self._region else filename
        if self.whole_page:
            tuned = getattr(self.page_reader, "adapter_loaded", False)
            model = "дообученная модель" if tuned else "базовая модель"
            self.status.set(f"{what}: строк {len(lines)} ({model})")
        else:
            self.status.set(f"{what}: строк {len(lines)}, требуют проверки {flagged}")
        self._update_buttons()

    def _current_text(self) -> str:
        return self.text.get("1.0", tk.END).strip()

    def on_copy(self):
        self.clipboard_clear()
        self.clipboard_append(self._current_text())
        lines = len(self._current_text().splitlines())
        self.status.set(f"Скопировано строк: {lines}")

    def on_save(self):
        path = filedialog.asksaveasfilename(
            parent=self, title="Сохранить распознанный текст",
            defaultextension=".txt", filetypes=[("Текстовый файл", "*.txt")],
        )
        if not path:
            return
        with open(path, "w", encoding="utf-8") as f:
            f.write(self._current_text() + "\n")
        self.status.set(f"Сохранено: {os.path.basename(path)}")


def main():
    app = HandwritingApp()
    # Handwriter.exe waits for this line to know the window opened, rather
    # than guessing at how long start-up takes. Under pythonw with no log
    # attached, stdout is None and print is a no-op.
    print("HANDWRITER_READY", flush=True)
    app.mainloop()


if __name__ == "__main__":
    main()
