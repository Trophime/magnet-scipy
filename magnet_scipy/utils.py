import numpy as np


def exp_metrics(t, exp_time, data, exp_data):
    # Calculate and display comparison metrics if time ranges overlap
    t_min = max(float(t.min()), float(exp_time.min()))
    t_max = min(float(t.max()), float(exp_time.max()))

    if t_max > t_min:
        # Interpolate both datasets to common time grid for comparison

        common_time = np.linspace(t_min, t_max, 200)
        computed_interp = np.interp(common_time, t, data)
        exp_interp = np.interp(common_time, exp_time, exp_data)

        # Calculate RMS difference and MAE
        rms_diff = np.sqrt(np.mean((computed_interp - exp_interp) ** 2))
        mae_diff = np.mean(np.abs(computed_interp - exp_interp))

    return rms_diff, mae_diff


class fake_sol:
    def __init__(self, t, y):
        self.t = t
        self.y = y
