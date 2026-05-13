import tkinter as tk
from tkinter import ttk


class AppFonts:
    def __init__(self):
        self.bold        = ("Arial",  9, "bold")
        self.label       = ("Arial", 10, "bold")
        self.button      = ("Arial", 10, "bold")
        self.value       = ("Arial",  9)
        self.value_large = ("Arial", 11, "bold")
        self.info_label  = ("Arial", 10, "bold")
        self.rules_text  = ("Arial",  9, "bold")
        self.agg_text    = ("Arial",  9, "bold")
        self.title       = ("Arial", 12, "bold")
        self.progress    = ("Arial", 12, "bold")
        self.welcome     = ("Arial", 14, "bold")


def make_section(parent, title, fonts):
    frame = ttk.LabelFrame(parent, padding=4)
    frame.configure(labelwidget=tk.Label(frame, text=title, font=fonts.bold))
    return frame


def make_labelled_entry(parent, text, row, col_label,
                        padx_label, fonts, entry_width=12):
    tk.Label(parent, text=text, font=fonts.label).grid(
        row=row, column=col_label, padx=padx_label, pady=4, sticky="w")
    entry = ttk.Entry(parent, width=entry_width)
    entry.grid(row=row, column=col_label + 1, padx=(0, 10), pady=4)
    return entry


def make_text_box(parent, height, width, font):
    widget = tk.Text(parent, height=height, width=width,
                     wrap="word", font=font)
    widget.pack(fill="x")
    return widget


def make_info_row(parent, label_text, row, fonts, anchor="w"):
    tk.Label(parent, text=label_text, font=fonts.info_label,
             anchor="nw" if anchor == "nw" else "w"
             ).grid(row=row, column=0,
                    sticky="nw" if anchor == "nw" else "w",
                    pady=(4, 2))
    value_lbl = tk.Label(parent, text="", font=fonts.value,
                         justify="left", wraplength=350, anchor="w")
    value_lbl.grid(row=row, column=1, sticky="w", pady=(4, 2))
    return value_lbl


def configure_progress_style(style):
    style.theme_use('default')
    style.configure("Custom.Horizontal.TProgressbar",
                    troughcolor="white", background="#165673",
                    thickness=16, bordercolor="white")
    style.configure("Bold.TButton", font=("Arial", 10, "bold"))