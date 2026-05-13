import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from FuzzyModel import FuzzyModel
from FuzzyEngine import EvaluationResult


class MembershipPlots:
    """Three input membership-function charts (Heart Rate, Pacing, Distance)."""

    def __init__(self, parent, model: FuzzyModel):
        self.model = model
        self.fig, (self.ax_hr, self.ax_pc, self.ax_dist) = \
            plt.subplots(1, 3, figsize=(14, 2.2))
        self.fig.subplots_adjust(left=0.05, right=0.98, top=0.88,
                                 bottom=0.22, wspace=0.35)
        self._draw_base_mfs()

        self.canvas = FigureCanvasTkAgg(self.fig, master=parent)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill="x", pady=2)

    def _draw_base_mfs(self):
        specs = [
            (self.ax_hr,   self.model.heart_rate, "Heart Rate MF", "Heart Rate (bpm)"),
            (self.ax_pc,   self.model.pacing,     "Pacing MF",     "Pacing (min/km)"),
            (self.ax_dist, self.model.distance,   "Distance MF",   "Distance (km)"),
        ]
        for ax, var, title, xlabel in specs:
            for label, mf in var.terms.items():
                ax.plot(var.universe, mf.mf, label=label, linewidth=1.2)
            ax.set_title(title, fontsize=8, weight="bold")
            ax.set_xlabel(xlabel, fontsize=7)
            ax.set_ylabel("Degree", fontsize=7)
            ax.set_ylim(0, 1)
            ax.tick_params(labelsize=6)
            ax.grid(True, alpha=0.3)
            ax.legend(loc='upper right', fontsize=6)

    def update_markers(self, hr: float, pc: float, dist: float):
        self._remove_markers()
        kw = dict(color='red', linestyle='--', linewidth=1.5, alpha=0.8)
        self.ax_hr.axvline(x=hr,    **kw)
        self.ax_pc.axvline(x=pc,    **kw)
        self.ax_dist.axvline(x=dist, **kw)
        self.canvas.draw()

    def clear_markers(self):
        self._remove_markers()
        self.canvas.draw()

    def _remove_markers(self):
        for ax in (self.ax_hr, self.ax_pc, self.ax_dist):
            for line in [l for l in ax.lines if l.get_linestyle() == '--']:
                line.remove()


class DefuzzPlot:
    def __init__(self, parent, model: FuzzyModel):
        self.model = model
        self.fig, self.ax = plt.subplots(figsize=(5, 2.2))
        self.fig.subplots_adjust(left=0.1, right=0.97,
                                 top=0.88, bottom=0.22)
        self._draw_base_mfs()

        self.canvas = FigureCanvasTkAgg(self.fig, master=parent)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

    def _draw_base_mfs(self):
        for label, mf in self.model.training_status.terms.items():
            self.ax.plot(self.model.training_status.universe, mf.mf,
                         label=label.capitalize(), linewidth=1.5)
        self._style_axis()

    def _style_axis(self):
        self.ax.set_title("Defuzzified Output (Training Status)",
                          fontsize=8, weight="bold")
        self.ax.set_xlabel("Training Intensity Level (0–1)", fontsize=7)
        self.ax.set_ylabel("Membership Degree",              fontsize=7)
        self.ax.set_xlim(0, 1)
        self.ax.set_ylim(0, 1.05)
        self.ax.tick_params(labelsize=6)
        self.ax.legend(loc='upper right', fontsize=6)
        self.ax.grid(True, linestyle="--", alpha=0.4)

    def update(self, result: EvaluationResult):
        self.ax.clear()
        for label, mf in self.model.training_status.terms.items():
            self.ax.plot(self.model.training_status.universe, mf.mf,
                         label=label.capitalize(), linewidth=1.5)

        self.ax.fill_between(self.model.training_status.universe,
                             result.aggregated_mf,
                             color="#165673", alpha=0.3)

        cv = result.crisp_value
        self.ax.axvline(x=cv, color='red', linestyle='--',
                        linewidth=1.5, alpha=0.8)
        self.ax.text(cv + 0.02, 0.88, f"Crisp = {cv:.2f}",
                     color='red', fontsize=7, weight="bold")
        self.ax.text(cv + 0.02, 0.75, f"→ {result.status}",
                     color='#165673', fontsize=7, weight="bold")
        self._style_axis()
        self.ax.set_title("Defuzzified Output: Training Status",
                          fontsize=8, weight="bold")
        self.canvas.draw()

    def reset(self):
        self.ax.clear()
        self._draw_base_mfs()
        self.canvas.draw()