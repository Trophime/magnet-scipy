import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from typing import Tuple, Callable, List


def create_function_from_csv(
    csv_file_path: str, x_column: str, y_column: str, method: str = "linear"
) -> Tuple[Callable, np.ndarray, np.ndarray]:
    """
    Create a Scipy function from tabulated CSV data.

    Args:
        csv_file_path: Path to the CSV file
        x_column: Name of the x-axis column (independent variable)
        y_column: Name of the y-axis column (dependent variable)
        method: Interpolation method ('linear', 'nearest', or 'cubic')

    Returns:
        Tuple of (scipy_function, x_data, y_data)
    """

    # Load CSV data
    df = pd.read_csv(csv_file_path)

    # Extract and sort data by x values
    df_sorted = df.sort_values(x_column)
    x_data = df_sorted[x_column].values
    y_data = df_sorted[y_column].values

    scipy_interp = interp1d(x_data, y_data, kind=method)

    return scipy_interp, x_data, y_data


def create_2d_function_from_csv(
    csv_file_path: str,
    x_column: str,
    y_column: str,
    z_column: str,
    method: str = "linear",
) -> Tuple[Callable, np.ndarray, np.ndarray, np.ndarray]:
    """
    Create a 2D Scipy function from tabulated CSV data for z = f(x, y).

    Handles gridded data where X and Y columns contain duplicate values
    representing a regular grid structure.

    Args:
        csv_file_path: Path to the CSV file
        x_column: Name of the first independent variable column
        y_column: Name of the second independent variable column
        z_column: Name of the dependent variable column
        method: Interpolation method ('linear' or 'nearest')

    Returns:
        Tuple of (scipy_function, x_unique, y_unique, z_grid)
        where z_grid has shape (len(y_unique), len(x_unique))
    """

    # Load CSV data
    df = pd.read_csv(csv_file_path)

    # Extract data
    x_data = df[x_column].values
    y_data = df[y_column].values
    z_data = df[z_column].values

    # Get unique x and y values and sort them
    x_unique = np.array(np.unique(x_data))
    y_unique = np.array(np.unique(y_data))

    # Reconstruct the grid
    # Create a mapping from (x,y) pairs to z values
    nx, ny = len(x_unique), len(y_unique)

    # Fill the grid - handle potential floating point precision issues
    z_grid_array = np.full(
        (ny, nx), np.nan
    )  # Initialize with NaN to detect missing values

    for i, (x_val, y_val, z_val) in enumerate(zip(x_data, y_data, z_data)):
        # Find closest indices to handle floating point precision
        x_idx = np.argmin(np.abs(x_unique - x_val))
        y_idx = np.argmin(np.abs(y_unique - y_val))

        # Verify we're close enough (tolerance for floating point comparison)
        if (
            np.abs(x_unique[x_idx] - x_val) < 1e-10
            and np.abs(y_unique[y_idx] - y_val) < 1e-10
        ):
            z_grid_array[y_idx, x_idx] = z_val
        else:
            print(
                f"Warning: Point ({x_val}, {y_val}) doesn't match grid structure exactly"
            )
            z_grid_array[y_idx, x_idx] = z_val  # Use it anyway

    # Check for missing values
    if np.any(np.isnan(z_grid_array)):
        missing_count = np.sum(np.isnan(z_grid_array))
        print(f"Warning: {missing_count} grid points are missing data")
        # Fill with nearest neighbor or interpolation if needed
        # For now, replace NaN with 0
        z_grid_array[np.isnan(z_grid_array)] = 0.0

    z_grid = np.array(z_grid_array, dtype=np.float32)

    if method == "linear":

        def scipy_function_2d(x, y):
            """2D bilinear interpolation on regular grid"""
            # Clamp to grid bounds
            x_clamped = np.clip(x, x_unique[0], x_unique[-1])
            y_clamped = np.clip(y, y_unique[0], y_unique[-1])

            # Find the grid cell containing the point
            x_idx = np.searchsorted(x_unique, x_clamped, side="right") - 1
            y_idx = np.searchsorted(y_unique, y_clamped, side="right") - 1

            # Clamp indices to valid range
            x_idx = np.clip(x_idx, 0, nx - 2)  # -2 because we need x_idx+1
            y_idx = np.clip(y_idx, 0, ny - 2)

            # Get the four corner points of the grid cell
            x0, x1 = x_unique[x_idx], x_unique[x_idx + 1]
            y0, y1 = y_unique[y_idx], y_unique[y_idx + 1]

            # Get the four z values at the corners
            z00 = z_grid[y_idx, x_idx]  # (x0, y0)
            z10 = z_grid[y_idx, x_idx + 1]  # (x1, y0)
            z01 = z_grid[y_idx + 1, x_idx]  # (x0, y1)
            z11 = z_grid[y_idx + 1, x_idx + 1]  # (x1, y1)

            # Compute interpolation weights
            dx = x1 - x0
            dy = y1 - y0

            # Handle degenerate cases (single point or line)
            wx = np.where(np.abs(dx) > 1e-10, (x_clamped - x0) / dx, 0.5)
            wy = np.where(np.abs(dy) > 1e-10, (y_clamped - y0) / dy, 0.5)

            # Bilinear interpolation
            z_interp = (
                z00 * (1 - wx) * (1 - wy)
                + z10 * wx * (1 - wy)
                + z01 * (1 - wx) * wy
                + z11 * wx * wy
            )

            return z_interp

    elif method == "nearest":

        def scipy_function_2d(x, y):
            """2D nearest neighbor interpolation on regular grid"""
            # Find nearest grid indices
            x_idx = np.argmin(np.abs(x_unique - x))
            y_idx = np.argmin(np.abs(y_unique - y))

            return z_grid[y_idx, x_idx]

    else:
        raise ValueError(f"Unknown method: {method}. Use 'linear' or 'nearest'")

    return scipy_function_2d, x_unique, y_unique, z_grid


def create_multi_column_function_from_csv(
    csv_file_path: str, x_column: str, y_columns: List[str], method: str = "linear"
) -> Tuple[Callable, np.ndarray, np.ndarray]:
    """
    Create a Scipy function that returns multiple outputs from CSV data.

    Args:
        csv_file_path: Path to the CSV file
        x_column: Name of the x-axis column (independent variable)
        y_columns: List of column names for dependent variables
        method: Interpolation method ('linear' or 'nearest')

    Returns:
        Tuple of (scipy_function, x_data, y_data_array)
        where scipy_function returns an array of interpolated values
    """

    # Load CSV data
    df = pd.read_csv(csv_file_path)

    # Extract and sort data by x values
    df_sorted = df.sort_values(x_column)
    x_data = np.array(df_sorted[x_column].values)

    # Stack all y columns into a 2D array
    y_data_list = []
    for col in y_columns:
        y_data_list.append(df_sorted[col].values)
    y_data = np.array(y_data_list).T  # Shape: (n_points, n_outputs)

    def scipy_function_multi(x):
        """Multi-output linear interpolation function"""
        # Vectorize interpolation across all columns
        results = []
        for i in range(len(y_columns)):
            results.append(interp1d(x_data, y_data[:, i], kind=method))
        return results

    return scipy_function_multi, x_data, y_data


def create_parametric_function_from_csv(
    csv_file_path: str,
    param_column: str,
    x_column: str,
    y_column: str,
    method: str = "linear",
) -> Tuple[Callable, np.ndarray, np.ndarray, np.ndarray]:
    """
    Create a parametric Scipy function from CSV data: y = f(x, param).

    Args:
        csv_file_path: Path to the CSV file
        param_column: Name of the parameter column
        x_column: Name of the x-axis column
        y_column: Name of the y-axis column (dependent variable)
        method: Interpolation method ('linear' or 'nearest')

    Returns:
        Tuple of (scipy_function, param_data, x_data, y_data)
        where scipy_function takes (x, param) and returns interpolated y
    """

    # Load CSV data
    df = pd.read_csv(csv_file_path)

    # Extract data
    param_data = df[param_column].values
    x_data = df[x_column].values
    y_data = df[y_column].values

    # Stack coordinates for 2D interpolation
    points = np.column_stack([x_data, param_data])

    if method == "linear":

        def scipy_function_param(x, param):
            """Parametric interpolation function"""
            query_point = np.array([x, param])

            # Use inverse distance weighting
            distances = np.linalg.norm(points - query_point, axis=1)
            epsilon = 1e-10
            weights = 1.0 / (distances + epsilon)
            weights = weights / np.sum(weights)

            return np.sum(weights * y_data)

    elif method == "nearest":

        def scipy_function_param(x, param):
            """Parametric nearest neighbor interpolation"""
            query_point = np.array([x, param])
            distances = np.linalg.norm(points - query_point, axis=1)
            nearest_idx = np.argmin(distances)
            return y_data[nearest_idx]

    else:
        raise ValueError(f"Unknown method: {method}. Use 'linear' or 'nearest'")

    return scipy_function_param, param_data, x_data, y_data
