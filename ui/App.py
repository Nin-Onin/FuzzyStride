import numpy as np
import tkinter as tk
from tkinter import messagebox, ttk
from PIL import Image, ImageTk

from FuzzyModel import FuzzyModel
from FuzzyEngine import FuzzyEngine, validate_input
from ui.Widgets import (
    AppFonts, make_section, make_labelled_entry,
    make_text_box, make_info_row, configure_progress_style,
)
from ui.Plots import MembershipPlots, DefuzzPlot

class FuzzyStrideApp:

    WINDOW_SIZE   = "1400x780"
    APP_TITLE     = "FuzzyStride"
    SECTION_TITLE = "Smart Running Adviser - Training Status Evaluation"
    LOGO_PATH     = "Assets/fuzzyStride-logo.png"
    BRAND_COLOR   = "#165673"

    def __init__(self):
        self.model  = FuzzyModel()
        self.engine = FuzzyEngine(self.model)
        self.root = tk.Tk()
        self.root.title(self.APP_TITLE)
        self.root.geometry(self.WINDOW_SIZE)
        self.root.resizable(True, True)
        self.fonts = AppFonts()
        self._style = ttk.Style()
        configure_progress_style(self._style)
        self._set_icon()
        self._build_splash()

    def run(self):
        self.root.mainloop()


    ICO_PATH = "Assets/fuzzyStride-logo.ico"

    def _set_icon(self):
        try:
            # iconbitmap with .ico is the most reliable method on Windows
            self.root.iconbitmap(self.ICO_PATH)
        except Exception as e:
            print(f"Icon error: {e}")

    def _build_splash(self):
        self._splash = tk.Frame(self.root, bg="white")
        self._splash.pack(fill="both", expand=True)

        logo_img = Image.open(self.LOGO_PATH).resize(
            (400, 400), Image.Resampling.LANCZOS)
        self._logo_photo = ImageTk.PhotoImage(logo_img)

        center = tk.Frame(self._splash, bg="white")
        center.place(relx=0.5, rely=0.4, anchor="center")

        tk.Label(center, text=self.SECTION_TITLE,
                 font=self.fonts.welcome,
                 bg="white", fg=self.BRAND_COLOR).pack(pady=(0, 8))
        tk.Label(center, image=self._logo_photo,
                 bg="white").pack(pady=(0, 8))
        tk.Label(center, text="L o a d i n g . . . .",
                 font=self.fonts.progress,
                 bg="white", fg=self.BRAND_COLOR).pack(pady=(4, 4))

        self._progress = ttk.Progressbar(
            center, orient="horizontal", mode="determinate",
            length=350, style="Custom.Horizontal.TProgressbar")
        self._progress.pack(pady=(0, 8))

        self._progress_value = 0
        self._tick_progress()

    def _tick_progress(self):
        self._progress_value += 2
        self._progress['value'] = self._progress_value
        if self._progress_value < 100:
            self.root.after(60, self._tick_progress)
        else:
            self._splash.destroy()
            self._build_main_ui()

    def _build_main_ui(self):
        # Title bar
        self._build_title_bar()

        # Single frame — all three sections packed directly
        self._main_frame = ttk.Frame(self.root)
        self._main_frame.pack(fill="both", expand=True, padx=6, pady=4)

        self._build_fuzzification_section()
        self._build_inference_section()
        self._build_defuzzification_section()

    def _build_title_bar(self):
        bar = ttk.Frame(self.root)
        bar.pack(fill="x", pady=(6, 2))

        tk.Label(bar, text=self.SECTION_TITLE,
                 font=self.fonts.title, anchor="w").pack(side="left", padx=8)

        btn_frame = ttk.Frame(bar)
        btn_frame.pack(side="right", padx=8)
        ttk.Button(btn_frame, text="Evaluate", style="Bold.TButton",
                   command=self._on_evaluate).pack(side="left", padx=4)
        ttk.Button(btn_frame, text="Clear", style="Bold.TButton",
                   command=self._on_clear).pack(side="left", padx=4)

    # Fuzzufication
    def _build_fuzzification_section(self):
        section = make_section(self._main_frame, "Fuzzification", self.fonts)
        section.pack(fill="x", padx=4, pady=2)

        row = ttk.Frame(section)
        row.pack(fill="x", pady=4)

        for col in range(6):
            row.columnconfigure(col, weight=1)

        self._entry_hr = make_labelled_entry(
            row, "Heart Rate (100-190 bpm):",
            row=0, col_label=0, padx_label=(8, 4), fonts=self.fonts)

        self._entry_pc = make_labelled_entry(
            row, "Pacing (3.0-9.0 min/km):",
            row=0, col_label=2, padx_label=(8, 4), fonts=self.fonts)

        self._entry_dist = make_labelled_entry(
            row, "Distance (0-42 km):",
            row=0, col_label=4, padx_label=(8, 4), fonts=self.fonts)

        self._mf_plots = MembershipPlots(section, self.model)

    # Inference
    def _build_inference_section(self):
        section = make_section(
            self._main_frame, "Inference / Rule Evaluation", self.fonts)
        section.pack(fill="x", padx=4, pady=2)

        content = ttk.Frame(section)
        content.pack(fill="both", expand=True)

        rules_col = ttk.Frame(content)
        rules_col.pack(side="left", fill="both", expand=True, padx=4)
        tk.Label(rules_col, text="Activated Rules",
                 font=self.fonts.label).pack(anchor="w", pady=(0, 2))
        self._rules_text = make_text_box(
            rules_col, height=4, width=60, font=self.fonts.rules_text)

        agg_col = ttk.Frame(content)
        agg_col.pack(side="left", fill="both", expand=True, padx=4)
        tk.Label(agg_col, text="Aggregated Outputs",
                 font=self.fonts.label).pack(anchor="w", pady=(0, 2))
        self._agg_text = make_text_box(
            agg_col, height=4, width=60, font=self.fonts.agg_text)

    # Defuzzification
    def _build_defuzzification_section(self):
        section = make_section(
            self._main_frame, "Defuzzification", self.fonts)
        section.pack(fill="x", padx=4, pady=2)

        content = ttk.Frame(section)
        content.pack(fill="x", expand=True)

        graph_frame = ttk.Frame(content)
        graph_frame.pack(side="left", fill="both",
                         expand=True, padx=(10, 6), pady=4)
        self._defuzz_plot = DefuzzPlot(graph_frame, self.model)

        info_frame = ttk.Frame(content)
        info_frame.pack(side="left", fill="both",
                        expand=True, padx=(6, 10), pady=4)

        self._lbl_crisp  = make_info_row(info_frame, "Crisp Value:",    0, self.fonts)
        self._lbl_status = make_info_row(info_frame, "Final Status:",   1, self.fonts)
        self._lbl_advice = make_info_row(info_frame, "Recommendation:", 2, self.fonts,
                                         anchor="nw")

    # Callbacks
    def _on_evaluate(self):
        try:
            hr   = validate_input(self._entry_hr.get(),   100, 190, "Heart Rate")
            pc   = validate_input(self._entry_pc.get(),   3.0, 9.0, "Pacing")
            dist = validate_input(self._entry_dist.get(), 0,   42,  "Distance")
        except ValueError as exc:
            messagebox.showerror("Input Error", str(exc))
            return
        result = self.engine.evaluate(hr, pc, dist)
        self._update_ui(result)

    def _on_clear(self):
        for entry in (self._entry_hr, self._entry_pc, self._entry_dist):
            entry.delete(0, tk.END)
        for lbl in (self._lbl_crisp, self._lbl_status, self._lbl_advice):
            lbl.config(text="")
        self._rules_text.delete(1.0, tk.END)
        self._agg_text.delete(1.0, tk.END)
        self._mf_plots.clear_markers()
        self._defuzz_plot.reset()


    def _update_ui(self, result):
        self._update_result_labels(result)
        self._update_rules_text(result)
        self._update_agg_text(result)
        self._mf_plots.update_markers(
            result.heart_rate, result.pacing, result.distance)
        self._defuzz_plot.update(result)

    def _update_result_labels(self, result):
        f = self.fonts.value_large
        self._lbl_crisp.config( text=f"{result.crisp_value:.2f}", font=f)
        self._lbl_status.config(text=result.status,               font=f)
        self._lbl_advice.config(text=result.recommendation,       font=f)

    def _update_rules_text(self, result):
        self._rules_text.delete(1.0, tk.END)
        self._rules_text.tag_configure("r", font=self.fonts.rules_text)
        for rule in result.activated_rules:
            self._rules_text.insert(tk.END, "  " + rule + "\n", "r")

    def _update_agg_text(self, result):
        self._agg_text.delete(1.0, tk.END)
        self._agg_text.tag_configure("r", font=self.fonts.agg_text)
        for key, val in result.agg_values.items():
            self._agg_text.insert(
                tk.END,
                f"  {key.capitalize()} Max Membership: {np.max(val):.4f}\n", "r")