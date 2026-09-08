"""
Desktop app: photo -> text.

The page sits on the left and its transcription on the right, the way a
person proofreads. Two engines feed it: the line recognizer, which cuts the
page into lines and tints each by how sure it was, and Qwen3-VL, which
reads the whole photo in one pass -- more accurate, but it needs a card
with about 8 GB. Text is editable, so fixing a mistake and exporting a
clean transcript is one flow.
"""
from __future__ import annotations

import os
import queue
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import cv2
import numpy as np
from PIL import Image, ImageTk

from . import page_reader
from .recognizer import RecognizedLine, Recognizer

# Confidence below this is treated as "look at this one".
LOW_CONFIDENCE = 0.80
MEDIUM_CONFIDENCE = 0.90

COLOR_LOW = "#ffd9d9"
COLOR_MEDIUM = "#fff3cd"
COLOR_OK = "#ffffff"



def confidence_color(confidence: float) -> str:
    if confidence < LOW_CONFIDENCE:
        return COLOR_LOW
    if confidence < MEDIUM_CONFIDENCE:
        return COLOR_MEDIUM
    return COLOR_OK


class HandwritingApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Распознавание рукописного текста")
        self.geometry("1150x800")

        self.recognizer: Recognizer | None = None
        self.page_reader: page_reader.PageReader | None = None

        # Worker threads must not touch tkinter (not even via .after());
        # they post messages here and the main thread drains the queue.
        self._events: queue.Queue[tuple] = queue.Queue()

        self._build_toolbar()
        self._build_body()

        self.status.set("Загрузка модели…")
        threading.Thread(target=self._load_model, daemon=True).start()
        self.after(50, self._drain_events)

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
                    self._recognize_failed(payload)
                elif kind == "status":
                    self.status.set(payload)
                elif kind == "progress":
                    self._update_progress(*payload)
        except queue.Empty:
            pass
        self.after(50, self._drain_events)

    def _build_toolbar(self):
        bar = ttk.Frame(self, padding=(10, 8))
        bar.pack(fill=tk.X)

        self.open_btn = ttk.Button(bar, text="Открыть фото", command=self.on_open, state=tk.DISABLED)
        self.open_btn.pack(side=tk.LEFT)

        self.save_btn = ttk.Button(bar, text="Сохранить текст", command=self.on_save, state=tk.DISABLED)
        self.save_btn.pack(side=tk.LEFT, padx=(8, 0))

        self.copy_btn = ttk.Button(bar, text="Копировать всё", command=self.on_copy, state=tk.DISABLED)
        self.copy_btn.pack(side=tk.LEFT, padx=(8, 0))

        # Whole-page mode reads the photo in one go and is more accurate,
        # but needs a card with ~8 GB; the line mode runs anywhere.
        self.whole_page = tk.BooleanVar(value=page_reader.enough_vram())
        self.mode_check = ttk.Checkbutton(
            bar, text="читать страницу целиком (точнее)", variable=self.whole_page
        )
        self.mode_check.pack(side=tk.LEFT, padx=(16, 0))
        if not page_reader.enough_vram():
            self.mode_check.state(["disabled"])

        self.status = tk.StringVar(value="")
        ttk.Label(bar, textvariable=self.status).pack(side=tk.LEFT, padx=(16, 0))

        # A page takes roughly two minutes; without this the window just
        # sits there looking hung.
        self.progress = ttk.Progressbar(bar, mode="determinate", length=160)
        self.progress.pack(side=tk.LEFT, padx=(16, 0))
        self.progress.stop()
        self.progress.pack_forget()

        legend = ttk.Frame(bar)
        legend.pack(side=tk.RIGHT)
        ttk.Label(legend, text="уверенность:").pack(side=tk.LEFT, padx=(0, 6))
        for color, text in ((COLOR_LOW, "низкая"), (COLOR_MEDIUM, "средняя"), (COLOR_OK, "высокая")):
            swatch = tk.Label(legend, text=f" {text} ", bg=color, relief=tk.SOLID, borderwidth=1)
            swatch.pack(side=tk.LEFT, padx=2)

    def _build_body(self):
        # Page on the left, its transcription on the right -- the line strips
        # this used to show one under another mirrored how the recognizer
        # works internally, not how anyone wants to proofread a page.
        panes = ttk.PanedWindow(self, orient=tk.HORIZONTAL)
        panes.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))

        left = ttk.Frame(panes)
        self.canvas = tk.Canvas(left, background="#f5f5f5", highlightthickness=0)
        canvas_scroll = ttk.Scrollbar(left, orient=tk.VERTICAL, command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=canvas_scroll.set)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        canvas_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        panes.add(left, weight=1)

        right = ttk.Frame(panes)
        self.text = tk.Text(right, wrap=tk.WORD, font=("Segoe UI", 11),
                            padx=8, pady=8, undo=True)
        text_scroll = ttk.Scrollbar(right, orient=tk.VERTICAL, command=self.text.yview)
        self.text.configure(yscrollcommand=text_scroll.set)
        self.text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        text_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        panes.add(right, weight=1)

        # Confidence is shown by tinting whole lines in place, so the text
        # stays one editable block you can select and copy across.
        self.text.tag_configure("low", background=COLOR_LOW)
        self.text.tag_configure("medium", background=COLOR_MEDIUM)

        self._page_photo = None
        self.canvas.bind("<Configure>", lambda e: self._redraw_page())

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
        adapter = "дообученная модель" if recognizer.adapter_loaded else "базовая модель"
        self.status.set(f"Готово ({adapter}, {where}). Откройте фото страницы.")
        self.open_btn.configure(state=tk.NORMAL)

        # Explain the fall back to CPU once, up front -- otherwise the only
        # symptom is that every page takes several minutes for no visible
        # reason.
        if getattr(recognizer, "device_warning", None):
            messagebox.showwarning("Видеокарта не используется", recognizer.device_warning)

    def _model_failed(self, exc: Exception):
        self.status.set("Не удалось загрузить модель")
        messagebox.showerror("Ошибка загрузки модели", str(exc))

    def on_open(self):
        path = filedialog.askopenfilename(
            title="Выберите фото страницы",
            filetypes=[("Изображения", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff"), ("Все файлы", "*.*")],
        )
        if not path:
            return

        self.text.delete("1.0", tk.END)
        data = np.fromfile(path, dtype=np.uint8)
        self._page_image = cv2.imdecode(data, cv2.IMREAD_COLOR)
        self._redraw_page()

        self.status.set(
            "Читаю страницу целиком…" if self.whole_page.get() else "Поиск строк…"
        )
        self.open_btn.configure(state=tk.DISABLED)
        self.save_btn.configure(state=tk.DISABLED)
        self.copy_btn.configure(state=tk.DISABLED)
        self.progress.pack(side=tk.LEFT, padx=(16, 0))
        self.progress.configure(value=0, maximum=100)
        threading.Thread(target=self._recognize, args=(path,), daemon=True).start()

    def _recognize(self, path: str):
        try:
            if self.whole_page.get():
                lines = self._read_whole_page(path)
            else:
                lines = self.recognizer.recognize_file(
                    path, progress=lambda done, total: self._events.put(("progress", (done, total)))
                )
        except Exception as exc:
            self._events.put(("recognize_failed", exc))
            return
        self._events.put(("recognized", (os.path.basename(path), lines)))

    def _read_whole_page(self, path: str):
        """Qwen3-VL reads the page in one pass, so there is no per-line
        confidence to colour by -- every line comes back unmarked."""
        if self.page_reader is None:
            self._events.put(("progress", (0, 1)))
            self.status_from_worker("Загружаю модель чтения страницы (в первый раз качается ~6 ГБ)…")
            self.page_reader = page_reader.PageReader()
        self._events.put(("progress", (0, 1)))
        texts = self.page_reader.read_file(path)
        self._events.put(("progress", (1, 1)))
        return [
            RecognizedLine(index=i, bbox=(0, 0, 0, 0), text=t, confidence=1.0, image=None)
            for i, t in enumerate(texts)
        ]

    def status_from_worker(self, message: str):
        self._events.put(("status", message))

    def _update_progress(self, done: int, total: int):
        # Whole-page mode has nothing to count -- one call in, one answer
        # out -- so it gets a marching bar instead of a filling one.
        if self.whole_page.get():
            if str(self.progress["mode"]) != "indeterminate":
                self.progress.configure(mode="indeterminate")
                self.progress.start(12)
            return
        if str(self.progress["mode"]) != "determinate":
            self.progress.stop()
            self.progress.configure(mode="determinate")
        self.progress.configure(value=done, maximum=max(total, 1))
        self.status.set(f"Распознавание… строка {done} из {total}")

    def _recognize_failed(self, exc: Exception):
        self.progress.stop()
        self.progress.pack_forget()
        self.status.set("Ошибка распознавания")
        self.open_btn.configure(state=tk.NORMAL)
        if "out of memory" in str(exc).lower():
            messagebox.showerror(
                "Не хватило видеопамяти",
                "Чтение страницы целиком не поместилось в видеокарту.\n\n"
                "Снимите галочку «читать страницу целиком» и откройте фото "
                "заново — построчный режим требует намного меньше памяти.",
            )
        else:
            messagebox.showerror("Ошибка распознавания", str(exc))

    def _show_results(self, filename: str, lines):
        self.progress.stop()
        self.progress.pack_forget()
        if not lines:
            self.status.set(f"{filename}: строки не найдены")
            self.open_btn.configure(state=tk.NORMAL)
            return

        flagged = sum(1 for line in lines if line.confidence < MEDIUM_CONFIDENCE)
        whole = self.whole_page.get()
        self.text.delete("1.0", tk.END)
        for i, line in enumerate(lines, start=1):
            self.text.insert(tk.END, line.text + "\n")
            if line.confidence < LOW_CONFIDENCE:
                self.text.tag_add("low", f"{i}.0", f"{i}.end")
            elif line.confidence < MEDIUM_CONFIDENCE:
                self.text.tag_add("medium", f"{i}.0", f"{i}.end")

        if whole:
            self.status.set(f"{filename}: строк {len(lines)} (страница целиком)")
        else:
            self.status.set(f"{filename}: строк {len(lines)}, требуют проверки {flagged}")
        self.open_btn.configure(state=tk.NORMAL)
        self.save_btn.configure(state=tk.NORMAL)
        self.copy_btn.configure(state=tk.NORMAL)

    def _redraw_page(self):
        """Fit the page into the left pane, keeping its proportions."""
        image = getattr(self, "_page_image", None)
        if image is None:
            return
        avail_w = max(self.canvas.winfo_width(), 1)
        avail_h = max(self.canvas.winfo_height(), 1)
        h, w = image.shape[:2]
        scale = min(avail_w / w, avail_h / h)
        if scale <= 0:
            return
        pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        pil = pil.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.LANCZOS)

        # keep a reference or tkinter garbage-collects the image away
        self._page_photo = ImageTk.PhotoImage(pil)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor="nw", image=self._page_photo)
        self.canvas.configure(scrollregion=(0, 0, pil.width, pil.height))

    def _current_text(self) -> str:
        return self.text.get("1.0", tk.END).strip()

    def on_copy(self):
        self.clipboard_clear()
        self.clipboard_append(self._current_text())
        lines = len(self._current_text().splitlines())
        self.status.set(f"Скопировано строк: {lines}")

    def on_save(self):
        path = filedialog.asksaveasfilename(
            title="Сохранить распознанный текст",
            defaultextension=".txt",
            filetypes=[("Текстовый файл", "*.txt")],
        )
        if not path:
            return

        content = self._current_text()
        with open(path, "w", encoding="utf-8") as f:
            f.write(content + "\n")
        self.status.set(f"Сохранено: {os.path.basename(path)}")


def main():
    app = HandwritingApp()
    app.mainloop()


if __name__ == "__main__":
    main()
