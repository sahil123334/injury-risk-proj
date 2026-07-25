"""
tk_utils.py

Shared helper for the two tkinter windows in this app (launcher.py,
session_summary.py). On macOS, a freshly created Tk window doesn't
always come to the front when launched from a terminal -- it can open
behind whatever's currently focused, which looks exactly like "nothing
happened." This forces it to the front and centers it, without relying
on `tk::PlaceWindow` (a Tk 8.6+ command that isn't available on every
Tk build, including some still in use on macOS).
"""

import tkinter as tk


def center_and_raise(root: tk.Tk) -> None:
    root.update_idletasks()
    width = root.winfo_reqwidth()
    height = root.winfo_reqheight()
    x = (root.winfo_screenwidth() - width) // 2
    y = (root.winfo_screenheight() - height) // 3
    root.geometry(f"{width}x{height}+{x}+{y}")

    # Briefly force "always on top" to guarantee the window is visible above
    # the terminal, then release it so it behaves like a normal window.
    root.lift()
    root.attributes("-topmost", True)
    root.after(300, lambda: root.attributes("-topmost", False))
    root.focus_force()
