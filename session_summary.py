"""
session_summary.py

A small native window shown the moment a session ends (stop button,
'q', or an uploaded video finishing), so "see your results" happens
inside the app itself rather than a browser tab quietly opening in the
background. From here you explicitly choose to open the full HTML
report, rather than it popping up unannounced. Visual design lives in
ui_style.py so this window and launcher.py read as one app.
"""

import tkinter as tk
from typing import Callable, Optional

import config
from tk_utils import center_and_raise
import ui_style as s


def show_session_summary(
    *,
    duration_sec: float,
    rep_count: int,
    overall_label: Optional[str],
    report_path: Optional[str],
    on_view_report: Callable[[str], None],
) -> None:
    root = tk.Tk()
    s.init_window(root, f"{config.PRODUCT_NAME} -- Session Complete")

    outer = tk.Frame(root, bg=s.PAGE_BG, padx=32, pady=28)
    outer.pack()

    header = tk.Frame(outer, bg=s.PAGE_BG)
    header.pack(fill="x")

    s.draw_brand_mark(header).pack(side="left", padx=(0, 14))

    title_box = tk.Frame(header, bg=s.PAGE_BG)
    title_box.pack(side="left", fill="x", expand=True)
    s.heading(title_box, "Session complete").pack(anchor="w")
    s.subtitle(title_box, "Fatigue/movement-quality proxy -- not a medical assessment.").pack(
        anchor="w", pady=(2, 0)
    )

    # ---------- stat tiles ----------
    stats = tk.Frame(outer, bg=s.PAGE_BG)
    stats.pack(fill="x", pady=(22, 20))

    def stat_tile(parent, label, value, value_color=s.TEXT_PRIMARY):
        tile = tk.Frame(parent, bg=s.CARD_BG, highlightbackground=s.BORDER, highlightthickness=1, padx=16, pady=12)
        tk.Label(tile, text=label, font=(s.FONT, 9), bg=s.CARD_BG, fg=s.TEXT_SECONDARY).pack(anchor="w")
        tk.Label(tile, text=value, font=(s.FONT, 16, "bold"), bg=s.CARD_BG, fg=value_color).pack(anchor="w")
        return tile

    stat_tile(stats, "Duration", f"{duration_sec:.0f}s").pack(side="left", padx=(0, 8), fill="x", expand=True)
    stat_tile(stats, "Reps completed", f"{rep_count}").pack(side="left", padx=8, fill="x", expand=True)
    if overall_label:
        color = s.STATUS_COLORS.get(overall_label, s.TEXT_PRIMARY)
        stat_tile(stats, "Final status", overall_label, color).pack(side="left", padx=(8, 0), fill="x", expand=True)

    # ---------- actions ----------
    button_row = tk.Frame(outer, bg=s.PAGE_BG)
    button_row.pack(fill="x")

    if report_path:
        def view_report():
            on_view_report(report_path)
            root.destroy()

        s.primary_button(button_row, "View Full Report", view_report).pack(side="left")
    else:
        s.subtitle(outer, "No session data was recorded.").pack(anchor="w", pady=(0, 14))

    s.secondary_button(button_row, "Close", root.destroy).pack(side="left", padx=(10, 0))

    root.protocol("WM_DELETE_WINDOW", root.destroy)
    center_and_raise(root)
    root.mainloop()
