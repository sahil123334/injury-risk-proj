"""
ui_style.py

Shared visual design for the two native windows in this app
(launcher.py, session_summary.py), so they read as one product instead
of two separately hacked-together tkinter scripts. Same color values as
report_template.html's light palette, so the native windows and the
browser report feel like the same app.

Deliberately built on plain `tk` widgets with explicit colors, not
`ttk` + a custom theme: the `clam` theme was tried first for nicer
native-looking hover states, but on this machine's Tk runtime it
occupies layout space and is clickable while painting invisibly --
worse than plain, reliable `tk` widgets with colors set directly.
"""

import tkinter as tk
from typing import Callable, Optional

PAGE_BG = "#f9f9f7"
CARD_BG = "#ffffff"
BORDER = "#e1e0d9"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
TEXT_MUTED = "#898781"
ACCENT = "#2a78d6"
ACCENT_ACTIVE = "#1c5cab"
SECONDARY_BG = "#ececea"
SECONDARY_ACTIVE = "#dedede"
GOOD = "#0ca30c"
WARNING = "#9a6a00"
CRITICAL = "#d03b3b"

STATUS_COLORS = {"GREEN": GOOD, "YELLOW": WARNING, "RED": CRITICAL}

FONT = "TkDefaultFont"  # the system UI font (San Francisco on macOS)


def init_window(root: tk.Tk, title: str) -> None:
    root.title(title)
    root.resizable(False, False)
    root.configure(bg=PAGE_BG)


def card(parent: tk.Widget) -> tk.Frame:
    return tk.Frame(parent, bg=CARD_BG, highlightbackground=BORDER, highlightthickness=1, padx=18, pady=16)


def heading(parent: tk.Widget, text: str, bg: str = PAGE_BG, size: int = 19) -> tk.Label:
    return tk.Label(parent, text=text, font=(FONT, size, "bold"), bg=bg, fg=TEXT_PRIMARY)


def subtitle(parent: tk.Widget, text: str, bg: str = PAGE_BG, wraplength: int = 340) -> tk.Label:
    return tk.Label(
        parent, text=text, font=(FONT, 11), bg=bg, fg=TEXT_SECONDARY,
        wraplength=wraplength, justify="left",
    )


def disclaimer(parent: tk.Widget, text: str, wraplength: int = 420) -> tk.Label:
    return tk.Label(
        parent, text=text, font=(FONT, 9), bg=PAGE_BG, fg=TEXT_MUTED,
        wraplength=wraplength, justify="left",
    )


def card_heading(parent: tk.Widget, text: str) -> tk.Label:
    return tk.Label(parent, text=text, font=(FONT, 13, "bold"), bg=CARD_BG, fg=TEXT_PRIMARY)


def card_body(parent: tk.Widget, text: str, wraplength: int = 380) -> tk.Label:
    return tk.Label(
        parent, text=text, font=(FONT, 10), bg=CARD_BG, fg=TEXT_SECONDARY,
        wraplength=wraplength, justify="left",
    )


def section_title(parent: tk.Widget, text: str, bg: str = PAGE_BG) -> tk.Label:
    return tk.Label(parent, text=text, font=(FONT, 15, "bold"), bg=bg, fg=TEXT_PRIMARY)


def _bind_button_behavior(widget: tk.Label, bg: str, active_bg: str, command: Callable[[], None]) -> None:
    widget.bind("<Enter>", lambda _e: widget.configure(bg=active_bg))
    widget.bind("<Leave>", lambda _e: widget.configure(bg=bg))
    widget.bind("<Button-1>", lambda _e: command())


def primary_button(parent: tk.Widget, text: str, command: Callable[[], None], enabled: bool = True) -> tk.Label:
    """
    A Label styled and behaving like a button, not a real tk.Button.
    On this machine's Tk, plain Buttons render as small native-gray
    boxes regardless of the bg/fg passed to them (a known macOS Tk
    quirk) -- Labels don't have that restriction, so a Label with click
    and hover bindings is what actually shows the intended color.
    """
    bg = ACCENT if enabled else "#a9c6e8"
    btn = tk.Label(
        parent, text=text, bg=bg, fg="white", font=(FONT, 12, "bold"),
        padx=24, pady=13, cursor=("hand2" if enabled else "arrow"),
    )
    if enabled:
        _bind_button_behavior(btn, bg, ACCENT_ACTIVE, command)
    return btn


def secondary_button(parent: tk.Widget, text: str, command: Callable[[], None], enabled: bool = True) -> tk.Label:
    bg = SECONDARY_BG
    btn = tk.Label(
        parent, text=text, bg=bg, fg=TEXT_PRIMARY, font=(FONT, 12),
        padx=24, pady=13, cursor=("hand2" if enabled else "arrow"),
    )
    if enabled:
        _bind_button_behavior(btn, bg, SECONDARY_ACTIVE, command)
    return btn


def draw_brand_mark(parent: tk.Widget, bg: str = PAGE_BG, size: int = 52) -> tk.Canvas:
    """
    A small drawn badge -- a bent hip/knee/ankle line, echoing what the
    app actually measures -- instead of a generic title-only header.
    Plain Canvas drawing, so it's unaffected by any widget theme issues.
    """
    canvas = tk.Canvas(parent, width=size, height=size, bg=bg, highlightthickness=0)

    pad = 3
    canvas.create_oval(pad, pad, size - pad, size - pad, fill=ACCENT, outline="")

    cx, cy = size / 2, size / 2
    hip = (cx - 9, cy - 11)
    knee = (cx + 7, cy + 1)
    ankle = (cx - 5, cy + 13)

    canvas.create_line(*hip, *knee, fill="white", width=3, capstyle="round")
    canvas.create_line(*knee, *ankle, fill="white", width=3, capstyle="round")
    for x, y in (hip, knee, ankle):
        r = 2.6
        canvas.create_oval(x - r, y - r, x + r, y + r, fill="white", outline="")

    return canvas
