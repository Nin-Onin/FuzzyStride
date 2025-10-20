import matplotlib.pyplot as plt
import numpy as np
import skfuzzy as fuzz
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from PIL import Image, ImageTk  # for logo display
from tkinter import font as tkfont
from tkinter import messagebox, ttk
from skfuzzy import control as ctrl

# Fuzzy Variables
heart_rate = ctrl.Antecedent(np.arange(100, 190.1, 0.1), 'heart_rate')
pacing = ctrl.Antecedent(np.arange(3.0, 9.1, 0.1), 'pacing')
distance = ctrl.Antecedent(np.arange(0, 42.1, 0.1), 'distance')
training_status = ctrl.Consequent(np.arange(0, 1.01, 0.01), 'training_status')

# Membership Functions - Fuzzy Sets
# Input Variable - Heart Rate (bpm)
heart_rate['low'] = fuzz.trimf(heart_rate.universe, [100, 120, 135])
heart_rate['moderate'] = fuzz.trimf(heart_rate.universe, [130, 145, 165])
heart_rate['high'] = fuzz.trimf(heart_rate.universe, [160, 175, 190])

# Input Variable - Pacing (min/km)
pacing['fast'] = fuzz.trimf(pacing.universe, [3.0, 3.8, 4.6])
pacing['moderate'] = fuzz.trimf(pacing.universe, [4.4, 5.6, 6.8])
pacing['slow'] = fuzz.trimf(pacing.universe, [6.4, 7.8, 9.0])

# Input Variable - Distance (km)
distance['short'] = fuzz.trimf(distance.universe, [0, 4, 10])
distance['medium'] = fuzz.trimf(distance.universe, [8, 18, 28])
distance['long'] = fuzz.trimf(distance.universe, [25, 35, 42])

# Output Variable - Training Status
training_status['undertraining'] = fuzz.trimf(training_status.universe, [0.0, 0.2, 0.5])
training_status['normal']        = fuzz.trimf(training_status.universe, [0.3, 0.55, 0.8])
training_status['overtraining']  = fuzz.trimf(training_status.universe, [0.6, 0.8, 1.0])

# Fuzzy Rules
rules = [
    ctrl.Rule(distance['short'] & pacing['slow'] & heart_rate['low'], training_status['undertraining']),
    ctrl.Rule(distance['short'] & pacing['slow'] & heart_rate['moderate'], training_status['undertraining']),
    ctrl.Rule(distance['short'] & pacing['slow'] & heart_rate['high'], training_status['normal']),
    ctrl.Rule(distance['short'] & pacing['moderate'] & heart_rate['low'], training_status['undertraining']),
    ctrl.Rule(distance['short'] & pacing['moderate'] & heart_rate['moderate'], training_status['normal']),
    ctrl.Rule(distance['short'] & pacing['moderate'] & heart_rate['high'], training_status['normal']),
    ctrl.Rule(distance['short'] & pacing['fast'] & heart_rate['low'], training_status['normal']),
    ctrl.Rule(distance['short'] & pacing['fast'] & heart_rate['moderate'], training_status['normal']),
    ctrl.Rule(distance['short'] & pacing['fast'] & heart_rate['high'], training_status['overtraining']),
    ctrl.Rule(distance['medium'] & pacing['slow'] & heart_rate['low'], training_status['undertraining']),
    ctrl.Rule(distance['medium'] & pacing['slow'] & heart_rate['moderate'], training_status['normal']),
    ctrl.Rule(distance['medium'] & pacing['slow'] & heart_rate['high'], training_status['normal']),
    ctrl.Rule(distance['medium'] & pacing['moderate'] & heart_rate['low'], training_status['normal']),
    ctrl.Rule(distance['medium'] & pacing['moderate'] & heart_rate['moderate'], training_status['normal']),
    ctrl.Rule(distance['medium'] & pacing['moderate'] & heart_rate['high'], training_status['overtraining']),
    ctrl.Rule(distance['medium'] & pacing['fast'] & heart_rate['low'], training_status['normal']),
    ctrl.Rule(distance['medium'] & pacing['fast'] & heart_rate['moderate'], training_status['overtraining']),
    ctrl.Rule(distance['medium'] & pacing['fast'] & heart_rate['high'], training_status['overtraining']),
    ctrl.Rule(distance['long'] & pacing['slow'] & heart_rate['low'], training_status['normal']),
    ctrl.Rule(distance['long'] & pacing['slow'] & heart_rate['moderate'], training_status['normal']),
    ctrl.Rule(distance['long'] & pacing['slow'] & heart_rate['high'], training_status['overtraining']),
    ctrl.Rule(distance['long'] & pacing['moderate'] & heart_rate['low'], training_status['normal']),
    ctrl.Rule(distance['long'] & pacing['moderate'] & heart_rate['moderate'], training_status['overtraining']),
    ctrl.Rule(distance['long'] & pacing['moderate'] & heart_rate['high'], training_status['overtraining']),
    ctrl.Rule(distance['long'] & pacing['fast'] & heart_rate['low'], training_status['normal']),
    ctrl.Rule(distance['long'] & pacing['fast'] & heart_rate['moderate'], training_status['overtraining']),
    ctrl.Rule(distance['long'] & pacing['fast'] & heart_rate['high'], training_status['overtraining']),
]
training_ctrl = ctrl.ControlSystem(rules)
training_sim = ctrl.ControlSystemSimulation(training_ctrl)

def validate_and_convert_input(value_str, min_val, max_val, param_name):
    if not value_str.strip():
        raise ValueError(f"{param_name} cannot be empty")
    try:
        value = float(value_str.strip())
    except ValueError:
        raise ValueError(f"{param_name} must be a number (integer or decimal)")

    if not (min_val <= value <= max_val):
        raise ValueError(f"{param_name} must be between {min_val} and {max_val}")
    return value

root = tk.Tk()
root.title("FuzzyStride")
root.geometry("2000x1025")
app_icon = ImageTk.PhotoImage(file="Assets/fuzzyStride-logo.png")
root.iconphoto(True, app_icon)
bold_font = tkfont.Font(weight="bold")
label_font = tkfont.Font(family="Arial", size=14, weight="bold")
button_font = tkfont.Font(family="Arial", size=14, weight="bold")
welcome_frame = tk.Frame(root, width=2000, height=1025, bg="white")
welcome_frame.pack(fill="both", expand=True)
logo_path = "Assets/fuzzyStride-logo.png"
logo_image = Image.open(logo_path)
logo_image = logo_image.resize((500, 500), Image.Resampling.LANCZOS)
logo_photo = ImageTk.PhotoImage(logo_image)
center_frame = tk.Frame(welcome_frame, bg="white")
center_frame.place(relx=0.5, rely=0.4, anchor="center")
welcome_text = tk.Label(center_frame, text="Smart Running Adviser - Training Status Evaluation", font=("Arial", 18, "bold"), bg="white", fg="#165673")
welcome_text.pack(side="top", pady=(0, 10))
logo_label = tk.Label(center_frame, image=logo_photo, bg="white")
logo_label.pack(side="top", pady=(0, 10))
progress_label = tk.Label(center_frame, text="L o a d i n g . . . .", font=("Arial", 16, "bold"), bg="white", fg="#165673")
progress_label.pack(side="top", pady=(5, 5))
style = ttk.Style()
style.theme_use('default')
style.configure("Custom.Horizontal.TProgressbar", troughcolor="white", background="#165673", thickness=20, bordercolor="white")
progress_bar = ttk.Progressbar(center_frame, orient="horizontal", mode="determinate", length=400, style="Custom.Horizontal.TProgressbar")
progress_bar.pack(side="top", pady=(0, 10))

progress_value = 0
def update_progress():
    global progress_value
    progress_value += 2
    progress_bar['value'] = progress_value
    if progress_value < 100:
        root.after(60, update_progress)
    else:
        show_main_interface()

def show_main_interface():
    welcome_frame.destroy()
    title_frame.pack(fill="x", pady=10)
    main_frame.pack(fill="both", expand=True, padx=10, pady=10)

update_progress()
title_frame = ttk.Frame(root)
style.configure("Bold.TButton", font=button_font)
title_label = tk.Label(title_frame, text="Smart Running Adviser - Training Status Evaluation", font=("Arial", 16, "bold"), anchor="w")
title_label.pack(side="left", padx=10)
button_frame = ttk.Frame(title_frame)
button_frame.pack(side="right", padx=10)
btn_eval = ttk.Button(button_frame, text="Evaluate", style="Bold.TButton")
btn_eval.pack(side="left", padx=5)
btn_clear = ttk.Button(button_frame, text="Clear", style="Bold.TButton")
btn_clear.pack(side="left", padx=5)
main_frame = ttk.Frame(root)

# Fuzzification Section
fuzz_frame = ttk.LabelFrame(main_frame, padding=10)
fuzz_frame.configure(labelwidget=tk.Label(fuzz_frame, text="Fuzzification", font=bold_font))
fuzz_frame.pack(side="top", fill="x", padx=5, pady=5)
input_frame = ttk.Frame(fuzz_frame)
input_frame.pack(fill="x", pady=10)

# Heart Rate Input
tk.Label(input_frame, text="Enter Heart Rate (100-190 bpm):", font=label_font).grid(row=0, column=0, padx=(100, 5), pady=10, sticky="w")
entry_hr = ttk.Entry(input_frame, width=15)
entry_hr.grid(row=0, column=1, padx=(0, 20), pady=10)

# Pacing Input
tk.Label(input_frame, text="Enter Pacing (3.0-9.0 min/km):", font=label_font).grid(row=0, column=2, padx=(200, 3), pady=10, sticky="w")
entry_pc = ttk.Entry(input_frame, width=15)
entry_pc.grid(row=0, column=3, padx=(0, 20), pady=10)

# Distance Input
tk.Label(input_frame, text="Enter Distance (0-42 km):", font=label_font).grid(row=0, column=4, padx=(210, 3), pady=10, sticky="w")
entry_dist = ttk.Entry(input_frame, width=15)
entry_dist.grid(row=0, column=5, padx=(0, 20), pady=10)

# Membership Function Graphs
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(21, 3))
plt.tight_layout(pad=3)

for label, mf in heart_rate.terms.items():
    ax1.plot(heart_rate.universe, mf.mf, label=label)
    ax1.set_title("Heart Rate Membership Function")
    ax1.set_xlabel("Heart Rate (bpm)")
    ax1.set_ylabel("Membership Degree")
    ax1.set_ylim(0, 1)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper right')

for label, mf in pacing.terms.items():
    ax2.plot(pacing.universe, mf.mf, label=label)
    ax2.set_title("Pacing Membership Function")
    ax2.set_xlabel("Pacing (min/km)")
    ax2.set_ylabel("Membership Degree")
    ax2.set_ylim(0, 1)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper right')

for label, mf in distance.terms.items():
    ax3.plot(distance.universe, mf.mf, label=label)
    ax3.set_title("Distance Membership Function")
    ax3.set_xlabel("Distance (km)")
    ax3.set_ylabel("Membership Degree")
    ax3.set_ylim(0, 1)
    ax3.grid(True, alpha=0.3)
    ax3.legend(loc='upper right')

canvas = FigureCanvasTkAgg(fig, master=fuzz_frame)
canvas.draw()
canvas.get_tk_widget().pack(fill="both", expand=True, pady=10)

# Inference Rule Evaluation Section
inference_frame = ttk.LabelFrame(main_frame, padding=10)
inference_frame.configure(labelwidget=tk.Label(inference_frame, text="Inference / Rule Evaluation", font=bold_font))
inference_frame.pack(side="top", fill="x", padx=5, pady=5)
inference_content = ttk.Frame(inference_frame)
inference_content.pack(fill="both", expand=True)
rules_frame = ttk.Frame(inference_content)
rules_frame.pack(side="left", fill="both", expand=True, padx=5)
tk.Label(rules_frame, text="Activated Rules", font=label_font, anchor="w").pack(anchor="w", pady=(0, 5))
rules_text = tk.Text(rules_frame, height=5, width=70, wrap="word", font=("Arial", 12))
rules_text.pack(fill="x")
agg_frame = ttk.Frame(inference_content)
agg_frame.pack(side="left", fill="both", expand=True, padx=5)
tk.Label(agg_frame, text="Aggregated Outputs", font=label_font, anchor="w").pack(anchor="w", pady=(0, 5))
agg_outputs_text = tk.Text(agg_frame, height=5, width=70, wrap="word", font=("Arial", 13, "bold"))
agg_outputs_text.pack(fill="x")

# Defuzzification Section
defuzz_frame = ttk.LabelFrame(main_frame, padding=10)
defuzz_frame.configure(labelwidget=tk.Label(defuzz_frame, text="Defuzzification", font=bold_font))
defuzz_frame.pack(side="top", fill="x", padx=5, pady=5)
defuzz_content = ttk.Frame(defuzz_frame)
defuzz_content.pack(fill="x", expand=True)
defuzz_graph_frame = ttk.Frame(defuzz_content)
defuzz_graph_frame.pack(side="left", fill="both", expand=True, padx=(20, 10), pady=10)

fig_defuzz, ax_defuzz = plt.subplots(figsize=(7, 3))
for label, mf in training_status.terms.items():
    ax_defuzz.plot(training_status.universe, mf.mf, label=label)
    ax_defuzz.set_title("Defuzzified Output (Training Status)")
    ax_defuzz.set_xlabel("Output Value")
    ax_defuzz.set_ylabel("Membership Degree")
    ax_defuzz.legend(loc='upper right')
    ax_defuzz.grid(True, alpha=0.3)

canvas_defuzz = FigureCanvasTkAgg(fig_defuzz, master=defuzz_graph_frame)
canvas_defuzz.draw()
canvas_defuzz.get_tk_widget().pack(fill="both", expand=True)
defuzz_info_frame = ttk.Frame(defuzz_content)
defuzz_info_frame.pack(side="left", fill="both", expand=True, padx=(10, 20), pady=10)
info_label_font = tkfont.Font(family="Arial", size=14, weight="bold")
value_font = tkfont.Font(family="Arial", size=13)
tk.Label(defuzz_info_frame, text="Crisp Value:", font=info_label_font, anchor="w").grid(row=0, column=0, sticky="w", pady=(10, 5))
crisp_val_label = tk.Label(defuzz_info_frame, text="", font=value_font, anchor="w")
crisp_val_label.grid(row=0, column=1, sticky="w", pady=(10, 5))
tk.Label(defuzz_info_frame, text="Final Status:", font=info_label_font, anchor="w").grid(row=1, column=0, sticky="w", pady=(10, 5))
status_label = tk.Label(defuzz_info_frame, text="", font=value_font, anchor="w")
status_label.grid(row=1, column=1, sticky="w", pady=(10, 5))
tk.Label(defuzz_info_frame, text="Recommendation:", font=info_label_font, anchor="nw", justify="left").grid(row=2, column=0, sticky="nw", pady=(10, 5))
recommendation_label = tk.Label(defuzz_info_frame, text="", font=value_font, justify="left", wraplength=400, anchor="w")
recommendation_label.grid(row=2, column=1, sticky="w", pady=(10, 5))

# Function Evaluations
def plot_input_marker(hr_val, pc_val, dist_val):
    for ax in [ax1, ax2, ax3]:
        lines_to_remove = [line for line in ax.lines if line.get_linestyle() == '--']
        for line in lines_to_remove:
            line.remove()
    ax1.axvline(x=hr_val, color='red', linestyle='--', linewidth=2, alpha=0.8)
    ax2.axvline(x=pc_val, color='red', linestyle='--', linewidth=2, alpha=0.8)
    ax3.axvline(x=dist_val, color='red', linestyle='--', linewidth=2, alpha=0.8)
    canvas.draw()

def evaluate():
    try:
        hr = validate_and_convert_input(entry_hr.get(), 100, 190, "Heart Rate")
        pc = validate_and_convert_input(entry_pc.get(), 3.0, 9.0, "Pacing")
        dist = validate_and_convert_input(entry_dist.get(), 0, 42, "Distance")
        rule_defs = [
            # Short Distance
            ('short', 'slow', 'low', 'undertraining'),
            ('short', 'slow', 'moderate', 'undertraining'),
            ('short', 'slow', 'high', 'normal'),
            ('short', 'moderate', 'low', 'undertraining'),
            ('short', 'moderate', 'moderate', 'normal'),
            ('short', 'moderate', 'high', 'normal'),
            ('short', 'fast', 'low', 'normal'),
            ('short', 'fast', 'moderate', 'normal'),
            ('short', 'fast', 'high', 'overtraining'),
            ('medium', 'slow', 'low', 'undertraining'),
            ('medium', 'slow', 'moderate', 'normal'),
            ('medium', 'slow', 'high', 'normal'),
            ('medium', 'moderate', 'low', 'normal'),
            ('medium', 'moderate', 'moderate', 'normal'),
            ('medium', 'moderate', 'high', 'overtraining'),
            ('medium', 'fast', 'low', 'normal'),
            ('medium', 'fast', 'moderate', 'overtraining'),
            ('medium', 'fast', 'high', 'overtraining'),
            ('long', 'slow', 'low', 'normal'),
            ('long', 'slow', 'moderate', 'normal'),
            ('long', 'slow', 'high', 'overtraining'),
            ('long', 'moderate', 'low', 'normal'),
            ('long', 'moderate', 'moderate', 'overtraining'),
            ('long', 'moderate', 'high', 'overtraining'),
            ('long', 'fast', 'low', 'normal'),
            ('long', 'fast', 'moderate', 'overtraining'),
            ('long', 'fast', 'high', 'overtraining')
        ]
        agg_values = {
            "undertraining": np.zeros_like(training_status.universe),
            "normal": np.zeros_like(training_status.universe),
            "overtraining": np.zeros_like(training_status.universe)
        }

        # Calculate membership degrees
        activated_rules = []
        for idx, (dist_term, pace_term, hr_term, out_term) in enumerate(rule_defs, start=1):
            deg_dist = fuzz.interp_membership(distance.universe, distance[dist_term].mf, dist)
            deg_pace = fuzz.interp_membership(pacing.universe, pacing[pace_term].mf, pc)
            deg_hr   = fuzz.interp_membership(heart_rate.universe, heart_rate[hr_term].mf, hr)
            firing_strength = np.fmin(np.fmin(deg_dist, deg_pace), deg_hr)

            if firing_strength > 0:
                rule_text = (
                    "Rule " + str(idx) + ": IF Distance is " + dist_term.capitalize() + " AND Pacing is " + pace_term.capitalize() + " AND Heart Rate is " + hr_term.capitalize() + " THEN Training Status is " + out_term.capitalize()
                )
                activated_rules.append(rule_text)

            term_mf = training_status[out_term].mf
            clipped = np.fmin(firing_strength, term_mf)
            agg_values[out_term] = np.fmax(agg_values[out_term], clipped)
            aggregated_mf = np.fmax(
            agg_values['undertraining'],
            np.fmax(agg_values['normal'], agg_values['overtraining'])
        )

        # Defuzzification
        crisp_result = fuzz.defuzz(training_status.universe, aggregated_mf, 'centroid')
        deg_under = fuzz.interp_membership(training_status.universe, training_status['undertraining'].mf, crisp_result)
        deg_norm = fuzz.interp_membership(training_status.universe, training_status['normal'].mf, crisp_result)
        deg_over = fuzz.interp_membership(training_status.universe, training_status['overtraining'].mf, crisp_result)
        degrees = {"Undertraining": deg_under, "Normal": deg_norm, "Overtraining": deg_over}
        status = max(degrees, key=degrees.get)

        advice_map = {
            "Undertraining": "Increase intensity or distance gradually.",
            "Normal": "Maintain current training regimen.",
            "Overtraining": "Rest or reduce intensity to prevent injury."
        }
        advice = advice_map[status]
        larger_font = tkfont.Font(family="Arial", size=16)
        crisp_val_label.config(text=f"{crisp_result:.2f}", font=larger_font)
        status_label.config(text=status, font=larger_font)
        recommendation_label.config(text=advice, font=larger_font)

        ax_defuzz.clear()
        for label, mf in training_status.terms.items():
            ax_defuzz.plot(training_status.universe, mf.mf, label=label.capitalize(), linewidth=2)
        ax_defuzz.fill_between(training_status.universe, aggregated_mf, color="#165673", alpha=0.3)
        ax_defuzz.axvline(x=crisp_result, color='red', linestyle='--', linewidth=2, alpha=0.8)
        ax_defuzz.text(crisp_result + 0.02, 0.9, "Crisp = " + str(round(crisp_result, 2)), color='red', fontsize=12, weight="bold")
        ax_defuzz.text(crisp_result + 0.02, 0.8, "→ " + str(status), color='#165673', fontsize=12, weight="bold")
        ax_defuzz.set_title("Defuzzified Output: Training Status", fontsize=13, weight="bold")
        ax_defuzz.set_xlabel("Training Intensity Level (0–1 Scale)", fontsize=11)
        ax_defuzz.set_ylabel("Membership Degree", fontsize=11)
        ax_defuzz.set_xlim(0, 1)
        ax_defuzz.set_ylim(0, 1.05)
        ax_defuzz.legend(loc='upper right', fontsize=10)
        ax_defuzz.grid(True, linestyle="--", alpha=0.4)
        canvas_defuzz.draw()

        plot_input_marker(hr, pc, dist)
        rules_text.delete(1.0, tk.END)
        rules_text.tag_configure("bold_rule", font=("Arial", 12, "bold"))
        for rule in activated_rules:
            rules_text.insert(tk.END, " " * 10 + rule + "\n", "bold_rule")
        agg_outputs_text.delete(1.0, tk.END)
        for key, val in agg_values.items():
            max_val = np.max(val)
            agg_outputs_text.insert(tk.END, " " * 20 + key.capitalize() + " Max Membership: " + str(max_val) + "\n", "bold_rule")
    except ValueError as ve:
        messagebox.showerror("Input Error", str(ve))

def clear():
    entry_hr.delete(0, tk.END)
    entry_pc.delete(0, tk.END)
    entry_dist.delete(0, tk.END)
    crisp_val_label.config(text="")
    status_label.config(text="")
    recommendation_label.config(text="")
    rules_text.delete(1.0, tk.END)
    agg_outputs_text.delete(1.0, tk.END)
    for ax in [ax1, ax2, ax3]:
        lines_to_remove = [line for line in ax.lines if line.get_linestyle() == '--']
        for line in lines_to_remove:
            line.remove()
    ax_defuzz.clear()
    for label, mf in training_status.terms.items():
        ax_defuzz.plot(training_status.universe, mf.mf, label=label.capitalize(), linewidth=2)
    ax_defuzz.set_title("Defuzzified Output: Training Status", fontsize=13, weight="bold")
    ax_defuzz.set_xlabel("Training Intensity Level (0–1 Scale)", fontsize=11)
    ax_defuzz.set_ylabel("Membership Degree", fontsize=11)
    ax_defuzz.set_xlim(0, 1)
    ax_defuzz.set_ylim(0, 1.05)
    ax_defuzz.legend(loc='upper right', fontsize=10)
    ax_defuzz.grid(True, linestyle="--", alpha=0.4)
    canvas.draw()
    canvas_defuzz.draw()

btn_eval.config(command=evaluate)
btn_clear.config(command=clear)
root.mainloop()
