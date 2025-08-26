import pytest
import numpy as np
import pandas as pd

from magnet_scipy.csv_utils import (
    create_function_from_csv,
    create_multi_column_function_from_csv,
)


def test_scipy_csv_functions():
    """Test all Scipy CSV function capabilities"""

    # Create test data
    time = np.linspace(0, 5, 400)
    current = 2.0 + 0.5 * np.sin(2 * np.pi * time)
    voltage = 1.5 * current  # + 0.2 * np.random.randn(len(time))

    # Save test CSV files
    df_1d = pd.DataFrame({"time": time, "current": current, "voltage": voltage})
    df_1d.to_csv("test_1d.csv", index=False)

    print("Testing Scipy CSV functions:")

    # Test 1D function
    func_1d, x_data, y_data = create_function_from_csv("test_1d.csv", "time", "current")
    test_val_1d = func_1d(2.5)
    assert test_val_1d == pytest.approx(2.0 + 0.5 * np.sin(2 * np.pi * 2.5), 1.0e-5)
    print(f"✓ 1D function evaluation at t=2.5: {test_val_1d:.4f}")

    # Test multi-column function
    func_multi, x_multi, y_multi = create_multi_column_function_from_csv(
        "test_1d.csv", "time", ["current", "voltage"]
    )
    test_val_multi = func_multi(2.5)
    print(
        f"✓ Multi-column function evaluation at t=2.5: {test_val_multi} (type={type(test_val_multi)})"
    )

    print("All tests passed!")
