# FuzzyStride: Smart Running Adviser
 
## 📖 About
FuzzyStride is a smart running adviser developed as Laboratory Exercise - 3 in CSci 141 Intelligent Systems. It evaluates a runner's training status using fuzzy logic by processing inputs such as heart rate (bpm), pacing (min/km), and distance (km) through a Mamdani Fuzzy Inference Engine, then delivers a training status — Undertraining, Normal, or Overtraining — along with a recommendation using Python's Tkinter GUI.

## ✨ Features/Demo
#### Welcome Interface
<img src="./Assets/Welcome_Interface.png" alt="LS" width="600" height="400">

#### Main User Interface
<img src="./Assets/Main_UI_Interface.png" alt="LS" width="600" height="400">

#### Input
<img src="./Assets/Input.png" alt="LS" width="600" height="400">

#### Fuzzification
<img src="./Assets/Fuzzification.png" alt="LS" width="600" height="400">

#### Rules Evaluation
<img src="./Assets/Rules_Evaluation.png" alt="LS" width="600" height="400">

#### Defuzzification
<img src="./Assets/Defuzzification.png" alt="LS" width="600" height="400">

#### Overview
<img src="./Assets/Main_UI_Overview.png" alt="LS" width="600" height="400">

## 🛠️ Tech Stack
* Python 3.x <br>
* Tkinter / ttk <br>
* NumPy <br>
* Matplotlib <br>
* scikit-fuzzy <br>
* Pillow (PIL) <br>

## ⚙️ Getting Started
### Prerequisites
* Python 3.x <br>
* pip <br>
### Installation & Run
1. Clone the repository <br>
   &nbsp;&nbsp; `git clone https://github.com/Nin-Onin/FuzzyStride.git` <br>
   &nbsp;&nbsp; `cd FuzzyStride`
2. Install the required dependencies <br>
   &nbsp;&nbsp; `pip install tk` <br>
   &nbsp;&nbsp; `pip install numpy` <br>
   &nbsp;&nbsp; `pip install matplotlib` <br>
   &nbsp;&nbsp; `pip install scikit-fuzzy` <br>
   &nbsp;&nbsp; `pip install pillow`
3. Run the application <br>
   &nbsp;&nbsp; `python FuzzyStride.py`

## 🚀 Usage
1. Enter your **Heart Rate** (100–190 bpm), **Pacing** (3.0–9.0 min/km), and **Distance** (0–42 km).
2. Click **Evaluate** to run the fuzzy inference and view results.
3. Click **Clear** to reset all inputs.
### Fuzzy Rules Summary
| Output | Condition |
|---|---|
| Undertraining | Short distance + Slow pacing + Low/Moderate heart rate |
| Normal Training | Balanced combination of distance, pacing, and heart rate |
| Overtraining | Fast pacing or Long distance + High heart rate |
 
### Membership Functions
| Variable | Terms | Range |
|---|---|---|
| Heart Rate (bpm) | Low / Moderate / High | 100 – 190 |
| Pacing (min/km) | Fast / Moderate / Slow | 3.0 – 9.0 |
| Distance (km) | Short / Medium / Long | 1 – 42 |
| Training Status | Undertraining / Normal / Overtraining | 0.0 – 1.0 |
 
## 👤 Author
**Niño M. Austria**
- Course: CSci 141 – Intelligent Systems
- GitHub: [@Nin-Onin](https://github.com/Nin-Onin)
