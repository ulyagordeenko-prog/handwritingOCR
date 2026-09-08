"""
Desktop app: photo -> lines -> text.

Layout mirrors how a person proofreads: every recognized line sits directly
under the actual image strip it came from, and lines the model was unsure
about are tinted so the eye lands on them first. Text fields are editable,
so correcting a mistake and exporting a clean transcript is one flow.
"""
from __future__ import annotations

import os
import queue
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import cv2
from PIL import Image, ImageTk

from .recognizer import Recognizer

# Confidence below this is treated as "look at this one".
LOW_CONFIDENCE = 0.80
MEDIUM_CONFIDENCE = 0.90

COLOR_LOW = "#ffd9d9"
COLOR_MEDIUM = "#fff3cd"
COLOR_OK = "#ffffff"

MAX_STRIP_WIDTH = 900


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
        self.line_widgets: list[tuple[tk.Text, float]] = []
        self._photo_refs: list[ImageTk.PhotoImage] = []

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

        self.status = tk.StringVar(value="")
        ttk.Label(bar, textvariable=self.status).pack(side=tk.LEFT, padx=(16, 0))

        # A page takes roughly two minutes; without this the window just
        # sits there looking hung.
        self.progress = ttk.Progressbar(bar, mode="determinate", length=160)
        self.progress.pack(side=tk.LEFT, padx=(16, 0))
        self.progress.pack_forget()

        legend = ttk.Frame(bar)
        legend.pack(side=tk.RIGHT)
        ttk.Label(legend, text="уверенность:").pack(side=tk.LEFT, padx=(0, 6))
        for color, text in ((COLOR_LOW, "низкая"), (COLOR_MEDIUM, "средняя"), (COLOR_OK, "высокая")):
            swatch = tk.Label(legend, text=f" {text} ", bg=color, relief=tk.SOLID, borderwidth=1)
            swatch.pack(side=tk.LEFT, padx=2)

    def _build_body(self):
        container = ttk.Frame(self)
        container.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))

        self.canvas = tk.Canvas(container, background="#f5f5f5", highlightthickness=0)
        scrollbar = ttk.Scrollbar(container, orient=tk.VERTICAL, command=self.canvas.yview)
        self.inner = ttk.Frame(self.canvas)

        self.inner.bind(
            "<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )
        self.canvas.create_window((0, 0), window=self.inner, anchor="nw")
        self.canvas.configure(yscrollcommand=scrollbar.set)

        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)

    def _on_mousewheel(self, event):
        self.canvas.yview_scroll(int(-event.delta / 120), "units")

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

        for child in self.inner.winfo_children():
            child.destroy()
        self.line_widgets.clear()
        self._photo_refs.clear()

        self.status.set("Поиск строк…")
        self.open_btn.configure(state=tk.DISABLED)
        self.save_btn.configure(state=tk.DISABLED)
        self.copy_btn.configure(state=tk.DISABLED)
        self.progress.pack(side=tk.LEFT, padx=(16, 0))
        self.progress.configure(value=0, maximum=100)
        threading.Thread(target=self._recognize, args=(path,), daemon=True).start()

    def _recognize(self, path: str):
        try:
            lines = self.recognizer.recognize_file(
                path, progress=lambda done, total: self._events.put(("progress", (done, total)))
            )
        except Exception as exc:
            self._events.put(("recognize_failed", exc))
            return
        self._events.put(("recognized", (os.path.basename(path), lines)))

    def _update_progress(self, done: int, total: int):
        self.progress.configure(value=done, maximum=max(total, 1))
        self.status.set(f"Распознавание… строка {done} из {total}")

    def _recognize_failed(self, exc: Exception):
        self.progress.pack_forget()
        self.status.set("Ошибка распознавания")
        self.open_btn.configure(state=tk.NORMAL)
        messagebox.showerror("Ошибка распознавания", str(exc))

    def _show_results(self, filename: str, lines):
        self.progress.pack_forget()
        if not lines:
            self.status.set(f"{filename}: строки не найдены")
            self.open_btn.configure(state=tk.NORMAL)
            return

        flagged = sum(1 for line in lines if line.confidence < MEDIUM_CONFIDENCE)
        for line in lines:
            self._add_line_row(line)

        self.status.set(
            f"{filename}: строк {len(lines)}, требуют проверки {flagged}"
        )
        self.open_btn.configure(state=tk.NORMAL)
        self.save_btn.configure(state=tk.NORMAL)
        self.copy_btn.configure(state=tk.NORMAL)

    def _current_text(self) -> str:
        return "\n".join(widget.get("1.0", tk.END).strip() for widget, _ in self.line_widgets)

    def on_copy(self):
        self.clipboard_clear()
        self.clipboard_append(self._current_text())
        self.status.set(f"Скопировано строк: {len(self.line_widgets)}")

    def _add_line_row(self, line):
        row = ttk.Frame(self.inner, padding=(0, 6))
        row.pack(fill=tk.X, expand=True, anchor="w")

        rgb = cv2.cvtColor(line.image, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        if pil.width > MAX_STRIP_WIDTH:
            ratio = MAX_STRIP_WIDTH / pil.width
            pil = pil.resize((MAX_STRIP_WIDTH, max(1, int(pil.height * ratio))), Image.LANCZOS)
        photo = ImageTk.PhotoImage(pil)
        self._photo_refs.append(photo)

        tk.Label(row, image=photo, borderwidth=1, relief=tk.SOLID).pack(anchor="w")

        entry_row = ttk.Frame(row)
        entry_row.pack(fill=tk.X, anchor="w", pady=(3, 0))

        tk.Label(entry_row, text=f"{line.confidence * 100:.0f}%", width=5, anchor="e").pack(
            side=tk.LEFT, padx=(0, 6)
        )

        text = tk.Text(
            entry_row,
            height=1,
            wrap=tk.NONE,
            background=confidence_color(line.confidence),
            font=("Segoe UI", 11),
        )
        text.insert("1.0", line.text)
        text.pack(side=tk.LEFT, fill=tk.X, expand=True)

        self.line_widgets.append((text, line.confidence))

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
