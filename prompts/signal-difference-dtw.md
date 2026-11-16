When your model captures the correct qualitative shape (e.g., the right number of peaks, rise times, and oscillations) but suffers from a **time delay or phase shift**, standard point-by-point comparison (like Euclidean distance) fails miserably. It will penalize your model heavily just because a peak occurs slightly later in time.

**Dynamic Time Warping (DTW)** solves this by "warping" the time axis non-linearly to find the optimal alignment between your experimental and simulation curves.

---

### Prerequisites

To use DTW in Python, you will need the `fastdtw` package along with `scipy` and `numpy`:

```bash
pip install fastdtw scipy numpy matplotlib

```

---

### Python Implementation: DTW Alignment and Visualization

Here is a complete, ready-to-run script showing how to compute DTW and visualize how the time steps map to each other:

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw

# --- 1. Generate Signals with a Time/Phase Shift ---
t = np.linspace(0, 10, 100)
y_exp = np.sin(t)                  # Experimental data
y_sim = np.sin(t - 0.8)            # Simulation with a time delay (phase shift)

# --- 2. Compute DTW Distance and Optimal Warping Path ---
# fastdtw compares the two sequences and returns the total distance and the index mapping path
distance, path = fastdtw(y_sim, y_exp, dist=euclidean)

print(f"Total DTW Distance: {distance:.4f}")

# --- 3. Visualize the Alignment ---
plt.figure(figsize=(10, 5))

# Plot the original signals
plt.plot(t, y_exp, label='Experimental Signal', color='black', linewidth=2)
plt.plot(t, y_sim, label='Simulation (Time-Shifted)', color='royalblue', linewidth=2, linestyle='--')

# Draw grey connection lines showing how DTW maps simulation points to experimental points
# (We sample every 4th point in the path to avoid visual clutter)
for sim_idx, exp_idx in path[::4]:
    plt.plot([t[sim_idx], t[exp_idx]], [y_sim[sim_idx], y_exp[exp_idx]], 
             color='gray', linestyle=':', alpha=0.6)

plt.title(f"Dynamic Time Warping Alignment (DTW Distance: {distance:.2f})")
plt.xlabel("Time")
plt.ylabel("Value")
plt.legend(loc='upper right')
plt.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()

```

---

### How to Interpret the Output

1. **The Warping Path (`path`)**: The algorithm returns a list of index pairs `(sim_index, exp_index)`. These pairs tell you which point in the simulation corresponds to which point in the experiment after accounting for the time shift.
2. **The Dotted Grey Lines**: In the visualization, these lines show the "stretch" or alignment. If your model output is just delayed, you will see parallel diagonal shift lines.
3. **DTW Distance**: Unlike a standard error metric that spikes during phase shifts, a low DTW distance indicates that *once aligned*, the shapes match closely.
