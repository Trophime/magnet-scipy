"""
Tests for Scipy CSV utilities functionality.
"""

import pytest
import numpy as np
import pandas as pd
import tempfile
from pathlib import Path

from magnet_scipy.csv_utils import (
    create_function_from_csv,
    create_2d_function_from_csv,
    create_multi_column_function_from_csv,
    create_parametric_function_from_csv,
)


class TestCreateFunctionFromCSV:
    """Test 1D Scipy function creation from CSV."""

    @pytest.fixture
    def simple_csv_data(self, test_data_dir):
        """Create simple CSV data for testing."""
        x = np.linspace(0, 10, 21)
        y = 2 * x + 3  # Linear function

        df = pd.DataFrame({"x_values": x, "y_values": y})
        csv_path = test_data_dir / "simple_data.csv"
        df.to_csv(csv_path, index=False)
        return str(csv_path)

    @pytest.fixture
    def nonlinear_csv_data(self, test_data_dir):
        """Create nonlinear CSV data for testing."""
        x = np.linspace(0, 2 * np.pi, 50)
        y = np.sin(x) + 0.5 * np.cos(2 * x)  # Nonlinear function

        df = pd.DataFrame({"time": x, "signal": y})
        csv_path = test_data_dir / "nonlinear_data.csv"
        df.to_csv(csv_path, index=False)
        return str(csv_path)

    def test_linear_interpolation(self, simple_csv_data):
        """Test linear interpolation functionality."""
        scipy_fun, x_data, y_data = create_function_from_csv(
            simple_csv_data, "x_values", "y_values", method="linear"
        )

        # Test at data points
        for i in range(len(x_data)):
            result = scipy_fun(float(x_data[i]))
            expected = float(y_data[i])
            assert abs(result - expected) < 1e-6

        # Test interpolation between points
        x_mid = (float(x_data[5]) + float(x_data[6])) / 2
        y_mid = scipy_fun(x_mid)
        expected_mid = (float(y_data[5]) + float(y_data[6])) / 2
        assert abs(y_mid - expected_mid) < 1e-6

    def test_nearest_interpolation(self, simple_csv_data):
        """Test nearest neighbor interpolation."""
        scipy_fun, x_data, y_data = create_function_from_csv(
            simple_csv_data, "x_values", "y_values", method="nearest"
        )

        # Test at data points
        for i in range(len(x_data)):
            result = scipy_fun(float(x_data[i]))
            expected = float(y_data[i])
            assert abs(result - expected) < 1e-6

        # Test between points (should snap to nearest)
        x_mid = (float(x_data[5]) + float(x_data[6])) / 2
        y_mid = scipy_fun(x_mid)

        # Should be one of the neighboring values
        assert (
            abs(y_mid - float(y_data[5])) < 1e-6 or abs(y_mid - float(y_data[6])) < 1e-6
        )

    def test_extrapolation_behavior(self, simple_csv_data):
        """Test behavior outside data range."""
        scipy_fun, x_data, y_data = create_function_from_csv(
            simple_csv_data, "x_values", "y_values", method="linear"
        )

        # Test below range
        x_min = float(np.min(x_data))
        y_below = scipy_fun(x_min - 1.0)
        assert np.isfinite(y_below)

        # Test above range
        x_max = float(np.max(x_data))
        y_above = scipy_fun(x_max + 1.0)
        assert np.isfinite(y_above)

    def test_Scipy_compatibility(self, nonlinear_csv_data):
        """Test Scipy JIT compilation compatibility."""
        import scipy

        scipy_func, _, _ = create_function_from_csv(
            nonlinear_csv_data, "time", "signal", method="linear"
        )

        result = scipy_func(3.14)
        assert np.isfinite(result)
        assert isinstance(result, np.ndarray) or isinstance(result, float)

    def test_vectorized_input(self, simple_csv_data):
        """Test function with vectorized input."""
        scipy_fun, x_data, _ = create_function_from_csv(
            simple_csv_data, "x_values", "y_values", method="linear"
        )

        # Test with array input
        x_test = np.array([1.0, 3.0, 5.0, 7.0])
        y_test = np.array([scipy_fun(x) for x in x_test])

        assert len(y_test) == len(x_test)
        assert np.all(np.isfinite(y_test))

    def test_invalid_method(self, simple_csv_data):
        """Test error handling for invalid interpolation method."""
        with pytest.raises(ValueError, match="Unknown method"):
            create_function_from_csv(
                simple_csv_data, "x_values", "y_values", method="invalid"
            )

    def test_missing_columns(self, test_data_dir):
        """Test error handling for missing columns."""
        # Create CSV without required columns
        df = pd.DataFrame({"wrong_x": [1, 2, 3], "wrong_y": [4, 5, 6]})
        csv_path = test_data_dir / "missing_columns.csv"
        df.to_csv(csv_path, index=False)

        with pytest.raises(KeyError):
            create_function_from_csv(
                str(csv_path), "x_values", "y_values", method="linear"
            )


class TestCreate2DScipyFunctionFromCSV:
    """Test 2D Scipy function creation from CSV."""

    @pytest.fixture
    def grid_csv_data(self, test_data_dir):
        """Create 2D grid CSV data for testing."""
        x_vals = np.linspace(0, 5, 10)
        y_vals = np.linspace(0, 3, 8)

        data = []
        for x in x_vals:
            for y in y_vals:
                z = x**2 + y**2  # Simple 2D function
                data.append({"x_coord": x, "y_coord": y, "z_value": z})

        df = pd.DataFrame(data)
        csv_path = test_data_dir / "grid_data.csv"
        df.to_csv(csv_path, index=False)
        return str(csv_path)

    def test_2d_linear_interpolation(self, grid_csv_data):
        """Test 2D linear interpolation."""
        Scipy_func_2d, x_data, y_data, z_data = create_2d_function_from_csv(
            grid_csv_data, "x_coord", "y_coord", "z_value", method="linear"
        )

        # Test at a few points
        test_points = [(1.0, 1.0), (2.5, 1.5), (4.0, 2.0)]

        for x_test, y_test in test_points:
            result = Scipy_func_2d(x_test, y_test)
            assert np.isfinite(result)
            assert isinstance(result, (np.ndarray, float))

    def test_2d_nearest_interpolation(self, grid_csv_data):
        """Test 2D nearest neighbor interpolation."""
        scipy_func_2d, _, _, _ = create_2d_function_from_csv(
            grid_csv_data, "x_coord", "y_coord", "z_value", method="nearest"
        )

        result = scipy_func_2d(2.0, 1.5)
        assert np.isfinite(result)

    def test_2d_Scipy_compatibility(self, grid_csv_data):
        """Test 2D function Scipy compatibility."""
        import scipy

        scipy_func_2d, _, _, _ = create_2d_function_from_csv(
            grid_csv_data, "x_coord", "y_coord", "z_value", method="linear"
        )

        result = scipy_func_2d(2.0, 1.0)
        assert np.isfinite(result)

    def test_2d_invalid_method(self, grid_csv_data):
        """Test error handling for invalid 2D method."""
        with pytest.raises(ValueError, match="Unknown method"):
            create_2d_function_from_csv(
                grid_csv_data, "x_coord", "y_coord", "z_value", method="invalid"
            )


class TestCreateMultiColumnScipyFunction:
    """Test multi-column Scipy function creation."""

    @pytest.fixture
    def multi_output_csv_data(self, test_data_dir):
        """Create multi-output CSV data."""
        x = np.linspace(0, 10, 20)
        y1 = np.sin(x)
        y2 = np.cos(x)
        y3 = x**2

        df = pd.DataFrame({"input": x, "output1": y1, "output2": y2, "output3": y3})
        csv_path = test_data_dir / "multi_output.csv"
        df.to_csv(csv_path, index=False)
        return str(csv_path)

    def test_multi_column_linear(self, multi_output_csv_data):
        """Test multi-column linear interpolation."""
        scipy_func_multi, x_data, y_data = create_multi_column_function_from_csv(
            multi_output_csv_data,
            "input",
            ["output1", "output2", "output3"],
            method="linear",
        )

        # Test at a point
        result = scipy_func_multi(5.0)

        assert isinstance(result, np.ndarray)
        assert len(result) == 3  # Should have 3 outputs
        assert np.all(np.isfinite(result))

    def test_multi_column_nearest(self, multi_output_csv_data):
        """Test multi-column nearest interpolation."""
        Scipy_func_multi, _, _ = create_multi_column_function_from_csv(
            multi_output_csv_data, "input", ["output1", "output2"], method="nearest"
        )

        result = Scipy_func_multi(3.0)
        assert len(result) == 2
        assert np.all(np.isfinite(result))

    def test_multi_column_Scipy_compatibility(self, multi_output_csv_data):
        """Test multi-column Scipy compatibility."""
        import scipy

        scipy_func_multi, _, _ = create_multi_column_function_from_csv(
            multi_output_csv_data,
            "input",
            ["output1", "output2", "output3"],
            method="linear",
        )

        result = scipy_func_multi(4.0)
        assert len(result) == 3
        assert np.all(np.isfinite(result))

    def test_multi_column_vectorized(self, multi_output_csv_data):
        """Test multi-column function with vectorized inputs."""
        Scipy_func_multi, _, _ = create_multi_column_function_from_csv(
            multi_output_csv_data, "input", ["output1", "output2"], method="linear"
        )

        # Test multiple inputs
        x_test = np.array([1.0, 3.0, 5.0])

        results = []
        for x in x_test:
            results.append(Scipy_func_multi(x))

        results = np.array(results)
        assert results.shape == (3, 2)  # 3 inputs, 2 outputs each
        assert np.all(np.isfinite(results))


class TestCreateParametricScipyFunction:
    """Test parametric Scipy function creation."""

    @pytest.fixture
    def parametric_csv_data(self, test_data_dir):
        """Create parametric CSV data."""
        data = []

        # Different parameter values
        params = [1.0, 2.0, 3.0]
        x_vals = np.linspace(0, 5, 15)

        for param in params:
            for x in x_vals:
                y = param * x + 0.5 * param**2  # Parametric function
                data.append({"parameter": param, "x_input": x, "y_output": y})

        df = pd.DataFrame(data)
        csv_path = test_data_dir / "parametric_data.csv"
        df.to_csv(csv_path, index=False)
        return str(csv_path)

    def test_parametric_linear_interpolation(self, parametric_csv_data):
        """Test parametric linear interpolation."""
        Scipy_func_param, param_data, x_data, y_data = (
            create_parametric_function_from_csv(
                parametric_csv_data, "parameter", "x_input", "y_output", method="linear"
            )
        )

        # Test at specific parameter and x values
        result = Scipy_func_param(2.0, 1.5)  # x=2.0, param=1.5
        assert np.isfinite(result)
        assert isinstance(result, (np.ndarray, float))

    def test_parametric_nearest_interpolation(self, parametric_csv_data):
        """Test parametric nearest interpolation."""
        Scipy_func_param, _, _, _ = create_parametric_function_from_csv(
            parametric_csv_data, "parameter", "x_input", "y_output", method="nearest"
        )

        result = Scipy_func_param(1.5, 2.5)  # x=1.5, param=2.5
        assert np.isfinite(result)

    def test_parametric_Scipy_compatibility(self, parametric_csv_data):
        """Test parametric function Scipy compatibility."""
        import scipy

        scipy_func_param, _, _, _ = create_parametric_function_from_csv(
            parametric_csv_data, "parameter", "x_input", "y_output", method="linear"
        )

        result = scipy_func_param(3.0, 2.0)
        assert np.isfinite(result)

    def test_parametric_invalid_method(self, parametric_csv_data):
        """Test error handling for invalid parametric method."""
        with pytest.raises(ValueError, match="Unknown method"):
            create_parametric_function_from_csv(
                parametric_csv_data,
                "parameter",
                "x_input",
                "y_output",
                method="invalid",
            )


@pytest.mark.unit
class TestCSVUtilsEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_csv_file(self, test_data_dir):
        """Test handling of empty CSV files."""
        # Create empty CSV
        df = pd.DataFrame()
        csv_path = test_data_dir / "empty.csv"
        df.to_csv(csv_path, index=False)

        with pytest.raises((ValueError, KeyError, IndexError)):
            create_function_from_csv(str(csv_path), "x", "y", method="linear")

    def test_single_point_csv(self, test_data_dir):
        """Test handling of CSV with single data point."""
        df = pd.DataFrame({"x": [1.0], "y": [2.0]})
        csv_path = test_data_dir / "single_point.csv"
        df.to_csv(csv_path, index=False)

        scipy_fun, _, _ = create_function_from_csv(
            str(csv_path), "x", "y", method="nearest"
        )

        # Should return the single value for any input
        result = scipy_fun(5.0)
        assert abs(result - 2.0) < 1e-6

    def test_unsorted_data(self, test_data_dir):
        """Test handling of unsorted CSV data."""
        # Create unsorted data
        x = np.array([5, 1, 3, 2, 4])
        y = x * 2  # Simple relationship

        df = pd.DataFrame({"x_vals": x, "y_vals": y})
        csv_path = test_data_dir / "unsorted.csv"
        df.to_csv(csv_path, index=False)

        scipy_fun, x_data, y_data = create_function_from_csv(
            str(csv_path), "x_vals", "y_vals", method="linear"
        )

        # Data should be sorted internally
        assert np.all(x_data[:-1] <= x_data[1:])  # Should be sorted

        # Function should still work correctly
        result = scipy_fun(2.5)
        assert np.isfinite(result)

    def test_duplicate_x_values(self, test_data_dir):
        """Test handling of duplicate x values."""
        # Create data with duplicate x values
        x = np.array([1, 2, 2, 3, 4])  # Duplicate x=2
        y = np.array([2, 4, 5, 6, 8])  # Different y values for x=2

        df = pd.DataFrame({"x_vals": x, "y_vals": y})
        csv_path = test_data_dir / "duplicates.csv"
        df.to_csv(csv_path, index=False)

        # Should handle duplicates (possibly by taking last value or averaging)
        scipy_fun, _, _ = create_function_from_csv(
            str(csv_path), "x_vals", "y_vals", method="linear"
        )

        result = scipy_fun(2.0)
        assert np.isfinite(result)

    def test_nan_values(self, test_data_dir):
        """Test handling of NaN values in CSV."""
        x = np.array([1, 2, 3, 4, 5])
        y = np.array([2, 4, np.nan, 8, 10])  # Contains NaN

        df = pd.DataFrame({"x_vals": x, "y_vals": y})
        csv_path = test_data_dir / "with_nan.csv"
        df.to_csv(csv_path, index=False)

        # Should either handle NaN or raise appropriate error
        try:
            scipy_fun, _, _ = create_function_from_csv(
                str(csv_path), "x_vals", "y_vals", method="linear"
            )

            # If it succeeds, function should still work at non-NaN points
            result = scipy_fun(1.0)
            assert np.isfinite(result)

        except (ValueError, RuntimeError):
            # It's acceptable to fail with NaN values
            pass

    def test_nonexistent_file(self):
        """Test handling of nonexistent CSV file."""
        with pytest.raises(FileNotFoundError):
            create_function_from_csv("nonexistent_file.csv", "x", "y", method="linear")

    def test_malformed_csv(self, test_data_dir):
        """Test handling of malformed CSV file."""
        # Create malformed CSV
        malformed_content = "x,y\n1,2\n3,4,extra_column\n5,6"
        csv_path = test_data_dir / "malformed.csv"

        with open(csv_path, "w") as f:
            f.write(malformed_content)

        # Pandas should handle this gracefully or raise appropriate error
        try:
            scipy_fun, _, _ = create_function_from_csv(
                str(csv_path), "x", "y", method="linear"
            )
            # If it succeeds, test that it works
            result = scipy_fun(2.0)
            assert np.isfinite(result)

        except (pd.errors.Error, ValueError):
            # It's acceptable to fail with malformed CSV
            pass


@pytest.mark.integration
class TestCSVUtilsIntegration:
    """Integration tests with real simulation use cases."""

    def test_resistance_function_integration(self, sample_resistance_csv):
        """Test integration with resistance function use case."""
        # This mimics how resistance CSV is used in RLCircuitPID
        Scipy_func_2d, current_data, temp_data, resistance_data = (
            create_2d_function_from_csv(
                sample_resistance_csv,
                "current",
                "temperature",
                "resistance",
                method="linear",
            )
        )

        # Test that it behaves as expected for circuit simulation
        test_currents = [0.0, 50.0, 100.0, 150.0]
        test_temperature = 30.0

        resistances = []
        for current in test_currents:
            R = Scipy_func_2d(current, test_temperature)
            resistances.append(float(R))
            assert R > 0  # Resistance should be positive
            assert np.isfinite(R)

        # Resistance should generally increase with current (for typical model)
        # This depends on the specific resistance model used in the test data

    def test_reference_function_integration(self, sample_reference_csv):
        """Test integration with reference current use case."""
        # This mimics how reference CSV is used in RLCircuitPID
        scipy_fun, time_data, current_data = create_function_from_csv(
            sample_reference_csv, "time", "current", method="linear"
        )

        # Test time series evaluation
        test_times = np.linspace(float(time_data.min()), float(time_data.max()), 50)

        for t in test_times:
            i_ref = scipy_fun(t)
            assert np.isfinite(i_ref)
            assert float(i_ref) >= 0  # Current should be non-negative

    @pytest.mark.slow
    def test_performance_with_large_dataset(self, test_data_dir):
        """Test performance with larger datasets."""
        # Create larger dataset
        n_points = 1000
        x = np.linspace(0, 100, n_points)
        y = np.sin(x) + 0.1 * np.random.randn(n_points)

        df = pd.DataFrame({"x": x, "y": y})
        csv_path = test_data_dir / "large_dataset.csv"
        df.to_csv(csv_path, index=False)

        # Should handle large dataset efficiently
        scipy_fun, _, _ = create_function_from_csv(
            str(csv_path), "x", "y", method="linear"
        )

        # Test multiple evaluations (simulating time series)
        test_x = np.linspace(10, 90, 100)

        import time

        start_time = time.time()

        for x_val in test_x:
            result = scipy_fun(x_val)
            assert np.isfinite(result)

        end_time = time.time()

        # Should complete in reasonable time (less than 1 second for this size)
        assert (end_time - start_time) < 1.0


class TestCSVUtilsErrorRecovery:
    """Test error recovery and robustness."""

    def test_corrupted_csv_graceful_handling(self, test_data_dir):
        """Test graceful handling of slightly corrupted CSV."""
        # Create CSV with some problematic rows
        content = """x,y
1.0,2.0
2.0,4.0
3.0,corrupted_value
4.0,8.0
5.0,10.0"""

        csv_path = test_data_dir / "corrupted.csv"
        with open(csv_path, "w") as f:
            f.write(content)

        # Should either handle gracefully or raise clear error
        try:
            scipy_fun, _, _ = create_function_from_csv(
                str(csv_path), "x", "y", method="linear"
            )

            # If successful, test that it works for valid data
            result = scipy_fun(1.5)
            assert np.isfinite(result)

        except (ValueError, pd.errors.Error) as e:
            # Acceptable to fail with clear error message
            assert "corrupted" in str(e).lower() or "convert" in str(e).lower()

    def test_extreme_values_handling(self, test_data_dir):
        """Test handling of extreme numerical values."""
        # Create data with extreme values
        x = np.array([1e-10, 1e-5, 1.0, 1e5, 1e10])
        y = np.array([1e-15, 1e-8, 1.0, 1e8, 1e15])

        df = pd.DataFrame({"x": x, "y": y})
        csv_path = test_data_dir / "extreme_values.csv"
        df.to_csv(csv_path, index=False)

        scipy_fun, _, _ = create_function_from_csv(
            str(csv_path), "x", "y", method="linear"
        )

        # Test interpolation with extreme values
        test_points = [1e-8, 1.0, 1e8]

        for x_test in test_points:
            result = scipy_fun(x_test)
            # Result should be finite (not NaN or inf)
            assert np.isfinite(result)

    def test_unicode_handling(self, test_data_dir):
        """Test handling of CSV files with unicode characters."""
        # Create CSV with unicode column names
        x = np.linspace(0, 5, 10)
        y = x**2

        df = pd.DataFrame({"time_μs": x, "signal_Ω": y})  # Unicode characters
        csv_path = test_data_dir / "unicode.csv"
        df.to_csv(csv_path, index=False, encoding="utf-8")

        # Should handle unicode column names
        scipy_fun, _, _ = create_function_from_csv(
            str(csv_path), "time_μs", "signal_Ω", method="linear"
        )

        result = scipy_fun(2.5)
        assert np.isfinite(result)

    def test_mixed_data_types(self, test_data_dir):
        """Test handling of CSV with mixed data types."""
        # Create CSV with mixed types
        data = {
            "x": [1.0, 2.0, 3.0, 4.0, 5.0],
            "y": [2.0, 4.0, 6.0, 8.0, 10.0],
            "category": ["A", "B", "A", "C", "B"],  # String column
            "flag": [True, False, True, False, True],  # Boolean column
        }

        df = pd.DataFrame(data)
        csv_path = test_data_dir / "mixed_types.csv"
        df.to_csv(csv_path, index=False)

        # Should work with numeric columns
        scipy_fun, _, _ = create_function_from_csv(
            str(csv_path), "x", "y", method="linear"
        )

        result = scipy_fun(2.5)
        assert np.isfinite(result)
        assert abs(result - 5.0) < 1e-6  # Should interpolate correctly


class TestCSVUtilsComplexScenarios:
    """Test complex real-world scenarios."""

    def test_time_series_with_gaps(self, test_data_dir):
        """Test time series data with gaps."""
        # Create time series with gaps
        t1 = np.linspace(0, 5, 50)
        t2 = np.linspace(8, 12, 40)  # Gap from 5 to 8
        t3 = np.linspace(15, 20, 30)  # Gap from 12 to 15

        time = np.concatenate([t1, t2, t3])
        signal = np.sin(time) + 0.1 * time

        df = pd.DataFrame({"time": time, "signal": signal})
        csv_path = test_data_dir / "time_series_gaps.csv"
        df.to_csv(csv_path, index=False)

        scipy_fun, _, _ = create_function_from_csv(
            str(csv_path), "time", "signal", method="linear"
        )

        # Test interpolation in gaps (extrapolation)
        result_in_gap = scipy_fun(6.5)  # In gap between 5 and 8
        assert np.isfinite(result_in_gap)

        # Test normal interpolation
        result_normal = scipy_fun(2.5)  # In dense region
        assert np.isfinite(result_normal)

    def test_noisy_experimental_data(self, test_data_dir):
        """Test with noisy experimental-like data."""
        # Simulate noisy experimental data
        n_points = 200
        x = np.sort(np.random.uniform(0, 10, n_points))  # Irregular spacing

        # Underlying function with noise
        y_true = 2 * x + 5 * np.sin(x)
        noise = 0.5 * np.random.randn(n_points)
        y = y_true + noise

        df = pd.DataFrame({"input": x, "output": y})
        csv_path = test_data_dir / "noisy_experimental.csv"
        df.to_csv(csv_path, index=False)

        scipy_fun, _, _ = create_function_from_csv(
            str(csv_path), "input", "output", method="linear"
        )

        # Test that interpolation works despite noise
        test_points = np.linspace(1, 9, 20)
        results = [scipy_fun(x_val) for x_val in test_points]

        assert all(np.isfinite(r) for r in results)

        # Results should be roughly in the expected range
        assert min(results) > min(y) - 5  # Reasonable bounds
        assert max(results) < max(y) + 5

    def test_multi_physics_simulation_data(self, test_data_dir):
        """Test with multi-physics simulation-like data."""
        # Simulate resistance data as function of current and temperature
        currents = np.linspace(0, 100, 25)
        temperatures = np.linspace(20, 80, 20)

        data = []
        for T in temperatures:
            for I in currents:
                # Realistic resistance model
                R0 = 1.2
                alpha = 0.004  # Temperature coefficient
                beta = 0.0001  # Current coefficient
                R = R0 * (1 + alpha * (T - 25) + beta * I)

                data.append({"current_A": I, "temperature_C": T, "resistance_ohm": R})

        df = pd.DataFrame(data)
        csv_path = test_data_dir / "resistance_model.csv"
        df.to_csv(csv_path, index=False)

        # Test 2D function creation
        Scipy_func_2d, _, _, _ = create_2d_function_from_csv(
            str(csv_path),
            "current_A",
            "temperature_C",
            "resistance_ohm",
            method="linear",
        )

        # Test realistic operating points
        test_points = [
            (25.0, 30.0),  # Low current, low temp
            (50.0, 45.0),  # Medium current, medium temp
            (75.0, 60.0),  # High current, high temp
        ]

        for I_test, T_test in test_points:
            R_test = Scipy_func_2d(I_test, T_test)

            # Should be physically reasonable
            assert 1.0 < float(R_test) < 2.0  # Reasonable resistance range
            assert np.isfinite(R_test)

        # Test that resistance increases with temperature and current
        R_low = Scipy_func_2d(10.0, 25.0)
        R_high_temp = Scipy_func_2d(10.0, 50.0)  # Same current, higher temp
        R_high_current = Scipy_func_2d(50.0, 25.0)  # Same temp, higher current

        assert float(R_high_temp) > float(R_low)  # Higher temp = higher resistance
        assert float(R_high_current) > float(
            R_low
        )  # Higher current = higher resistance

    def test_control_system_reference_signals(self, test_data_dir):
        """Test with complex control system reference signals."""
        # Create complex reference signal
        t = np.linspace(0, 10, 500)

        # Multi-step reference with smooth transitions
        ref = np.zeros_like(t)
        ref[t >= 1.0] = 10.0  # Step to 10
        ref[t >= 3.0] = 25.0  # Step to 25
        ref[t >= 5.0] = 50.0 + 20.0 * np.sin(
            2 * np.pi * 0.5 * (t[t >= 5.0] - 5.0)
        )  # Sinusoidal
        ref[t >= 8.0] = 15.0  # Step back down

        # Add some simple smoothing (without scipy dependency)
        # Simple moving average smoothing
        window_size = 5
        ref_smooth = np.copy(ref)
        for i in range(window_size, len(ref) - window_size):
            ref_smooth[i] = np.mean(ref[i - window_size : i + window_size])

        df = pd.DataFrame({"time_s": t, "reference_A": ref_smooth})
        csv_path = test_data_dir / "control_reference.csv"
        df.to_csv(csv_path, index=False)

        scipy_fun, _, _ = create_function_from_csv(
            str(csv_path), "time_s", "reference_A", method="linear"
        )

        # Test that we can evaluate at arbitrary times
        test_times = [0.5, 2.0, 4.0, 6.5, 9.0]

        references = []
        for t_test in test_times:
            ref_val = scipy_fun(t_test)
            references.append(float(ref_val))
            assert np.isfinite(ref_val)
            assert ref_val >= 0  # Current should be non-negative

        # Should show the step changes
        assert references[0] < 5.0  # Before first step
        assert 5.0 < references[1] < 15.0  # After first step
        assert 20.0 < references[2] < 30.0  # After second step
        assert references[3] > 30.0  # In sinusoidal region
        assert 10.0 < references[4] < 20.0  # After final step

    @pytest.mark.slow
    def test_high_frequency_sampling(self, test_data_dir):
        """Test with high-frequency sampled data."""
        # High frequency data (simulating fast ADC sampling)
        fs = 1000  # 1 kHz sampling (reduced for faster tests)
        duration = 0.1  # 100 ms
        t = np.linspace(0, duration, int(fs * duration))

        # Signal with multiple frequency components
        signal = (
            2.0 * np.sin(2 * np.pi * 50 * t)  # 50 Hz component
            + 0.5 * np.sin(2 * np.pi * 200 * t)  # 200 Hz component
            + 0.02 * np.random.randn(len(t))
        )  # Noise (reduced amplitude)

        df = pd.DataFrame({"time": t, "voltage": signal})
        csv_path = test_data_dir / "high_freq_data.csv"
        df.to_csv(csv_path, index=False)

        scipy_fun, _, _ = create_function_from_csv(
            str(csv_path), "time", "voltage", method="linear"
        )

        # Test interpolation at intermediate points
        t_interp = np.linspace(0.01, 0.09, 50)  # Reduced number of points

        voltages = []
        for t_val in t_interp:
            v = scipy_fun(t_val)
            voltages.append(float(v))
            assert np.isfinite(v)

        # Should preserve signal characteristics
        voltages = np.array(voltages)
        assert np.std(voltages) > 0.1  # Should have reasonable variation
        assert abs(np.mean(voltages)) < 1.0  # Should be roughly zero-mean


class TestCSVUtilsUtilities:
    """Test utility functions and helper methods."""

    def test_function_serialization(self, simple_csv_data):
        """Test that created functions can be used in different contexts."""
        scipy_fun, _, _ = create_function_from_csv(
            simple_csv_data, "x_values", "y_values", method="linear"
        )

        # Test that function can be called multiple times
        results = []
        for i in range(10):
            result = scipy_fun(float(i))
            results.append(result)

        assert len(results) == 10
        assert all(np.isfinite(r) for r in results)

    def test_function_closure_behavior(self, simple_csv_data):
        """Test that functions properly capture data in closure."""
        Scipy_func1, _, _ = create_function_from_csv(
            simple_csv_data, "x_values", "y_values", method="linear"
        )

        # Create another function from different data
        x2 = np.array([1, 2, 3, 4])
        y2 = np.array([10, 20, 30, 40])  # Different slope
        df2 = pd.DataFrame({"x": x2, "y": y2})

        csv_path2 = Path(simple_csv_data).parent / "different_data.csv"
        df2.to_csv(csv_path2, index=False)

        Scipy_func2, _, _ = create_function_from_csv(
            str(csv_path2), "x", "y", method="linear"
        )

        # Functions should give different results
        test_x = 2.5
        result1 = Scipy_func1(test_x)
        result2 = Scipy_func2(test_x)

        assert result1 != result2  # Should be different
        assert np.isfinite(result1)
        assert np.isfinite(result2)

    def test_data_range_properties(self, nonlinear_csv_data):
        """Test that data range information is preserved."""
        scipy_fun, x_data, y_data = create_function_from_csv(
            nonlinear_csv_data, "time", "signal", method="linear"
        )

        # Check data properties
        assert len(x_data) == len(y_data)
        assert len(x_data) > 0

        # Data should be sorted
        assert np.all(x_data[:-1] <= x_data[1:])

        # All data should be finite
        assert np.all(np.isfinite(x_data))
        assert np.all(np.isfinite(y_data))

    def test_interpolation_consistency(self, simple_csv_data):
        """Test that linear and nearest give consistent results at data points."""
        Scipy_func_linear, x_data, y_data = create_function_from_csv(
            simple_csv_data, "x_values", "y_values", method="linear"
        )

        Scipy_func_nearest, _, _ = create_function_from_csv(
            simple_csv_data, "x_values", "y_values", method="nearest"
        )

        # At data points, both methods should give same result
        for i in range(0, len(x_data), 3):  # Test every 3rd point
            x_test = float(x_data[i])

            result_linear = Scipy_func_linear(x_test)
            result_nearest = Scipy_func_nearest(x_test)

            assert np.allclose(result_linear, result_nearest, rtol=1e-5)


class TestCSVUtilsRegressionTests:
    """Regression tests for specific issues that might arise."""

    def test_single_value_interpolation(self, test_data_dir):
        """Regression test: ensure single-point data doesn't crash."""
        df = pd.DataFrame({"x": [5.0], "y": [10.0]})
        csv_path = test_data_dir / "single_point_regression.csv"
        df.to_csv(csv_path, index=False)

        scipy_fun, _, _ = create_function_from_csv(
            str(csv_path), "x", "y", method="nearest"
        )

        # Should return the single value for any input
        test_inputs = [-10.0, 0.0, 5.0, 100.0]
        for x_test in test_inputs:
            result = scipy_fun(x_test)
            assert np.allclose(result, 10.0)

    def test_identical_x_values_handling(self, test_data_dir):
        """Regression test: handle duplicate x values gracefully."""
        # Data with some duplicate x values
        x = np.array([1, 2, 2, 2, 3, 4])  # Multiple x=2 values
        y = np.array([1, 2, 3, 4, 5, 6])  # Different y values

        df = pd.DataFrame({"x": x, "y": y})
        csv_path = test_data_dir / "duplicate_x_regression.csv"
        df.to_csv(csv_path, index=False)

        # Should handle duplicates (typically by taking the last value)
        scipy_fun, x_data, y_data = create_function_from_csv(
            str(csv_path), "x", "y", method="linear"
        )

        # Should still create a valid function
        result = scipy_fun(2.0)
        assert np.isfinite(result)

        # Data arrays should be well-formed
        assert np.all(np.isfinite(x_data))
        assert np.all(np.isfinite(y_data))

    def test_very_small_dataset(self, test_data_dir):
        """Regression test: handle very small datasets."""
        # Minimal viable dataset
        df = pd.DataFrame({"input": [0.0, 1.0], "output": [0.0, 1.0]})
        csv_path = test_data_dir / "minimal_dataset.csv"
        df.to_csv(csv_path, index=False)

        scipy_fun, _, _ = create_function_from_csv(
            str(csv_path), "input", "output", method="linear"
        )

        # Should work for interpolation
        result_middle = scipy_fun(0.5)
        assert np.allclose(result_middle, 0.5)  # Linear interpolation

        # Should work for extrapolation
        result_extrap = scipy_fun(2.0)
        assert np.isfinite(result_extrap)

    def test_large_value_ranges(self, test_data_dir):
        """Regression test: handle large value ranges without overflow."""
        # Data spanning many orders of magnitude
        x = np.logspace(-5, 5, 20)  # 10^-5 to 10^5
        y = x**0.5  # Square root relationship

        df = pd.DataFrame({"x": x, "y": y})
        csv_path = test_data_dir / "large_range.csv"
        df.to_csv(csv_path, index=False)

        scipy_fun, _, _ = create_function_from_csv(
            str(csv_path), "x", "y", method="linear"
        )

        # Test at various scales
        test_points = [1e-3, 1.0, 1e3]

        for x_test in test_points:
            result = scipy_fun(x_test)
            assert np.isfinite(result)
            assert result > 0  # Should be positive for sqrt function

    def test_zero_and_negative_values(self, test_data_dir):
        """Regression test: handle zero and negative values properly."""
        x = np.array([-5, -2, 0, 2, 5])
        y = np.array([-10, -4, 0, 4, 10])  # Linear relationship

        df = pd.DataFrame({"x": x, "y": y})
        csv_path = test_data_dir / "negative_values.csv"
        df.to_csv(csv_path, index=False)

        scipy_fun, _, _ = create_function_from_csv(
            str(csv_path), "x", "y", method="linear"
        )

        # Test around zero
        test_points = [-1.0, 0.0, 1.0]
        expected_results = [-2.0, 0.0, 2.0]

        for x_test, expected in zip(test_points, expected_results):
            result = scipy_fun(x_test)
            assert np.allclose(result, expected, rtol=1e-5)


# Mark all tests with csv marker for easy filtering
pytestmark = pytest.mark.csv
