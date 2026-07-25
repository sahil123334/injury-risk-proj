"""
launcher.py

The startup picker window shown before the frame loop starts. Explains
what the app does in plain terms, then lets you choose "record live"
vs "analyze an existing video file," and if recording, which camera to
use. Visual design lives in ui_style.py so this window and
session_summary.py read as one app.

If you close the window without picking anything, `show_launcher()`
returns None and main.py exits quietly rather than falling back to
some default -- an explicit choice is required to start.
"""

import tkinter as tk
from dataclasses import dataclass
from tkinter import filedialog, ttk
from typing import Optional

import config
from camera_utils import discover_cameras
from tk_utils import center_and_raise
import ui_style as s

VIDEO_FILETYPES = [
    ("Video files", "*.mp4 *.mov *.m4v *.avi *.mkv"),
    ("All files", "*.*"),
]


@dataclass
class LaunchChoice:
    mode: str  # "record" or "upload"
    camera_index: Optional[int] = None
    video_path: Optional[str] = None


def show_launcher() -> Optional[LaunchChoice]:
    result: Optional[LaunchChoice] = None
    cameras = discover_cameras()

    root = tk.Tk()
    s.init_window(root, config.PRODUCT_NAME)

    outer = tk.Frame(root, bg=s.PAGE_BG, padx=32, pady=28)
    outer.pack()

    # ---------- header ----------
    header = tk.Frame(outer, bg=s.PAGE_BG)
    header.pack(fill="x")

    s.draw_brand_mark(header).pack(side="left", padx=(0, 14))

    title_box = tk.Frame(header, bg=s.PAGE_BG)
    title_box.pack(side="left", fill="x", expand=True)
    s.heading(title_box, config.PRODUCT_NAME).pack(anchor="w")
    s.subtitle(title_box, "Squat form and fatigue-proxy feedback from real-time pose tracking.").pack(
        anchor="w", pady=(2, 0)
    )

    s.disclaimer(outer, "Prototype -- not a medical assessment.").pack(anchor="w", pady=(10, 18))

    s.section_title(outer, "Injury Report").pack(anchor="w", pady=(0, 4))
    s.subtitle(outer, "Choose how you'd like to get feedback:", wraplength=420).pack(anchor="w", pady=(0, 16))

    # ---------- record card ----------
    record_card = s.card(outer)
    record_card.pack(fill="x", pady=(0, 14))

    s.card_heading(record_card, "Record live").pack(anchor="w")
    s.card_body(
        record_card, "Use your webcam for real-time rep counting and movement-quality feedback as you lift."
    ).pack(anchor="w", pady=(4, 12))

    camera_var = tk.StringVar()
    if cameras:
        options = [f"[{c.index}] {c.name} ({c.resolution[0]}x{c.resolution[1]})" for c in cameras]
        camera_var.set(options[0])
        row = tk.Frame(record_card, bg=s.CARD_BG)
        row.pack(anchor="w", fill="x", pady=(0, 14))
        tk.Label(row, text="Camera", font=(s.FONT, 10), bg=s.CARD_BG, fg=s.TEXT_MUTED).pack(side="left", padx=(0, 10))
        ttk.Combobox(row, textvariable=camera_var, values=options, state="readonly", width=36).pack(side="left")
    else:
        tk.Label(record_card, text="No camera detected.", font=(s.FONT, 10), bg=s.CARD_BG, fg=s.CRITICAL).pack(
            anchor="w", pady=(0, 14)
        )

    def choose_record():
        nonlocal result
        index = int(camera_var.get().split("]")[0].strip("[")) if cameras else 0
        result = LaunchChoice(mode="record", camera_index=index)
        root.destroy()

    record_btn = s.primary_button(record_card, "Start Recording", choose_record, enabled=bool(cameras))
    record_btn.pack(anchor="w")

    # ---------- upload card ----------
    upload_card = s.card(outer)
    upload_card.pack(fill="x")

    s.card_heading(upload_card, "Upload a video").pack(anchor="w")
    s.card_body(
        upload_card, "Analyze footage you've already recorded and get the exact same rep-by-rep feedback."
    ).pack(anchor="w", pady=(4, 14))

    def choose_upload():
        nonlocal result
        path = filedialog.askopenfilename(title="Choose a video to analyze", filetypes=VIDEO_FILETYPES)
        if path:
            result = LaunchChoice(mode="upload", video_path=path)
            root.destroy()
        # Cancelling the file dialog just returns to the launcher window.

    s.secondary_button(upload_card, "Choose Video File...", choose_upload).pack(anchor="w")

    root.protocol("WM_DELETE_WINDOW", root.destroy)
    center_and_raise(root)
    root.mainloop()

    return result
