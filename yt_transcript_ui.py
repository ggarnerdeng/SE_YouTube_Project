#!/usr/bin/env python3
from __future__ import annotations

import re
import threading
import tkinter as tk
from tkinter import ttk, messagebox, filedialog

from youtube_transcript_api import YouTubeTranscriptApi


VIDEO_ID_RE = re.compile(r"[A-Za-z0-9_-]{11}")


def extract_video_id(url_or_id: str) -> str:
    s = (url_or_id or "").strip()
    if not s:
        raise ValueError("Please paste a YouTube URL (or a video id).")

    # If user pasted just the 11-char ID
    if VIDEO_ID_RE.fullmatch(s):
        return s

    # Common URL patterns
    patterns = [
        r"(?:v=)([A-Za-z0-9_-]{11})",                 # watch?v=ID
        r"(?:youtu\.be/)([A-Za-z0-9_-]{11})",         # youtu.be/ID
        r"(?:/shorts/)([A-Za-z0-9_-]{11})",           # /shorts/ID
        r"(?:/live/)([A-Za-z0-9_-]{11})",             # /live/ID
        r"(?:/embed/)([A-Za-z0-9_-]{11})",            # /embed/ID
    ]
    for pat in patterns:
        m = re.search(pat, s)
        if m:
            return m.group(1)

    raise ValueError("Could not extract a video id from that input.")


def fetch_transcript_text(video_id: str, lang: str | None = None) -> str:
    api = YouTubeTranscriptApi()

    # v1.x returns a list of objects with .text
    if lang:
        segments = api.fetch(video_id, languages=[lang])
    else:
        segments = api.fetch(video_id)

    lines = []
    for seg in segments:
        txt = (getattr(seg, "text", "") or "").replace("\n", " ").strip()
        if txt:
            lines.append(txt)

    return "\n".join(lines)


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("YouTube Transcript Extractor")
        self.geometry("900x650")
        self.minsize(720, 520)

        self.url_var = tk.StringVar()
        self.lang_var = tk.StringVar(value="en")  # default; user can clear it

        self._build_ui()

    def _build_ui(self):
        pad = 10

        top = ttk.Frame(self)
        top.pack(fill="x", padx=pad, pady=(pad, 0))

        ttk.Label(top, text="YouTube URL or Video ID:").grid(row=0, column=0, sticky="w")
        url_entry = ttk.Entry(top, textvariable=self.url_var)
        url_entry.grid(row=1, column=0, columnspan=4, sticky="ew", pady=(4, 0))
        url_entry.focus()

        ttk.Label(top, text="Language (optional, e.g. en). Clear to auto:").grid(row=2, column=0, sticky="w", pady=(10, 0))
        lang_entry = ttk.Entry(top, textvariable=self.lang_var, width=10)
        lang_entry.grid(row=3, column=0, sticky="w", pady=(4, 0))

        self.fetch_btn = ttk.Button(top, text="Fetch Transcript", command=self.on_fetch)
        self.fetch_btn.grid(row=3, column=1, sticky="w", padx=(10, 0))

        self.save_btn = ttk.Button(top, text="Save As…", command=self.on_save, state="disabled")
        self.save_btn.grid(row=3, column=2, sticky="w", padx=(10, 0))

        self.clear_btn = ttk.Button(top, text="Clear", command=self.on_clear)
        self.clear_btn.grid(row=3, column=3, sticky="w", padx=(10, 0))

        top.columnconfigure(0, weight=1)

        self.status = tk.StringVar(value="Ready.")
        status_bar = ttk.Label(self, textvariable=self.status, anchor="w")
        status_bar.pack(fill="x", padx=pad, pady=(8, 0))

        # Transcript box
        mid = ttk.Frame(self)
        mid.pack(fill="both", expand=True, padx=pad, pady=pad)

        self.text = tk.Text(mid, wrap="word")
        self.text.pack(side="left", fill="both", expand=True)

        scroll = ttk.Scrollbar(mid, command=self.text.yview)
        scroll.pack(side="right", fill="y")
        self.text.configure(yscrollcommand=scroll.set)

    def set_busy(self, busy: bool):
        self.fetch_btn.configure(state=("disabled" if busy else "normal"))
        self.clear_btn.configure(state=("disabled" if busy else "normal"))
        # Save enabled only if transcript exists
        if busy:
            self.save_btn.configure(state="disabled")
        else:
            if self.text.get("1.0", "end-1c").strip():
                self.save_btn.configure(state="normal")

    def on_clear(self):
        self.text.delete("1.0", "end")
        self.save_btn.configure(state="disabled")
        self.status.set("Cleared.")

    def on_fetch(self):
        url_or_id = self.url_var.get().strip()
        lang = self.lang_var.get().strip() or None

        try:
            video_id = extract_video_id(url_or_id)
        except Exception as e:
            messagebox.showerror("Invalid input", str(e))
            return

        self.set_busy(True)
        self.status.set("Fetching transcript…")

        def worker():
            try:
                transcript = fetch_transcript_text(video_id, lang=lang)
                if not transcript.strip():
                    raise RuntimeError("Transcript fetched, but it was empty.")
                self._ui_success(transcript)
            except Exception as e:
                self._ui_error(e)

        threading.Thread(target=worker, daemon=True).start()

    def _ui_success(self, transcript: str):
        def run():
            self.text.delete("1.0", "end")
            self.text.insert("1.0", transcript)
            self.status.set("Done.")
            self.set_busy(False)
            self.save_btn.configure(state="normal")
        self.after(0, run)

    def _ui_error(self, err: Exception):
        def run():
            self.status.set("Error.")
            self.set_busy(False)
            messagebox.showerror("Failed to fetch transcript", str(err))
        self.after(0, run)

    def on_save(self):
        content = self.text.get("1.0", "end-1c")
        if not content.strip():
            messagebox.showinfo("Nothing to save", "Transcript box is empty.")
            return

        path = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text file", "*.txt"), ("All files", "*.*")],
            title="Save transcript as…",
        )
        if not path:
            return

        with open(path, "w", encoding="utf-8") as f:
            f.write(content + "\n")

        self.status.set(f"Saved: {path}")


if __name__ == "__main__":
    # nicer ttk look on Windows
    try:
        import ctypes
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        pass

    app = App()
    app.mainloop()
