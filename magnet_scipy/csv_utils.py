from pathlib import Path
from typing import Callable, List, Tuple

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

INTERP1D_KINDS = (
    "linear",
    "nearest",
    "nearest-up",
    "zero",
    "slinear",
    "quadratic",
    "cubic",
    "previous",
    "next",
)


def _read_csv(csv_file_path: str) -> pd.DataFrame:
    """Read a CSV file with auto-detected separator, checking first that it exists and is not empty."""
    path = Path(csv_file_path)
    if not path.is_file():
        raise FileNotFoundError(f"CSV file not found: {csv_file_path}")
    with path.open() as f:
        if not any(line.strip() for line in f):
            raise ValueError(f"CSV file is empty: {csv_file_path}")
    return pd.read_csv(path, sep=None, engine="python")


def _make_interp1d(x_data: np.ndarray, y_data: np.ndarray, method: str) -> interp1d:
    """Build an interp1d along axis 0 that holds the edge values outside the x range."""
    if method not in INTERP1D_KINDS:
        raise ValueError(f"Unknown method: {method}. Use one of {INTERP1D_KINDS}")
    return interp1d(
        x_data,
        y_data,
        kind=method,
        axis=0,
        bounds_error=False,
        fill_value=(y_data[0], y_data[-1]),
    )


def create_function_from_csv(
    csv_file_path: str, x_column: str, y_column: str, method: str = "linear"
) -> Tuple[Callable, np.ndarray, np.ndarray]:
    """Create a SciPy interpolation function from tabulated CSV data.

    Parameters
    ----------
    csv_file_path : str
        Path to the CSV file; the separator is auto-detected.
    x_column : str
        Name of the independent variable column.
    y_column : str
        Name of the dependent variable column.
    method : str, default "linear"
        Interpolation kind passed to :class:`scipy.interpolate.interp1d`
        (one of :data:`INTERP1D_KINDS`, e.g. ``"linear"``, ``"nearest"``,
        ``"cubic"``).

    Returns
    -------
    function : scipy.interpolate.interp1d
        Interpolant y = f(x). Outside the x range it holds the first/last
        y value (no extrapolation).
    x_data : numpy.ndarray
        x values, sorted ascending.
    y_data : numpy.ndarray
        Corresponding y values.

    Raises
    ------
    FileNotFoundError
        If ``csv_file_path`` does not exist.
    ValueError
        If ``method`` is not a supported kind or the file is empty.
    """
    # Load CSV data
    df = _read_csv(csv_file_path)

    # Extract and sort data by x values
    df_sorted = df.sort_values(x_column)
    x_data = df_sorted[x_column].values
    y_data = df_sorted[y_column].values

    scipy_interp = _make_interp1d(x_data, y_data, method)

    return scipy_interp, x_data, y_data


def create_piecewise_linear_derivative(
    x_data: np.ndarray, y_data: np.ndarray
) -> Callable:
    """Build an exact derivative function for a piecewise-linear interpolant.

    The interpolant is the one produced by
    ``create_function_from_csv(..., method="linear")``.

    Parameters
    ----------
    x_data : numpy.ndarray
        Sorted independent variable values (e.g. time [s]).
    y_data : numpy.ndarray
        Corresponding dependent variable values (e.g. current [A]).

    Returns
    -------
    callable
        Maps x (scalar or array) to the local slope dy/dx. Out-of-range x is
        clamped to the nearest edge segment's slope.

    Notes
    -----
    The derivative of a piecewise-linear function is piecewise CONSTANT: it
    equals the segment slope between consecutive data points. This looks it
    up directly via binary search in O(log n), instead of estimating it with
    a finite difference (which needs two interpolation calls and gives wrong
    results within `dt` of a breakpoint, where it straddles the kink).
    """
    slopes = np.diff(y_data) / np.diff(x_data)
    n_segments = len(slopes)

    def derivative(x):
        idx = np.searchsorted(x_data, x, side="right") - 1
        idx = np.clip(idx, 0, n_segments - 1)
        return slopes[idx]

    return derivative


def create_2d_function_from_csv(
    csv_file_path: str,
    x_column: str,
    y_column: str,
    z_column: str,
    method: str = "linear",
) -> Tuple[Callable, np.ndarray, np.ndarray, np.ndarray]:
    """Create a 2D interpolation function z = f(x, y) from tabulated CSV data.

    Handles gridded data where X and Y columns contain duplicate values
    representing a regular grid structure.

    Parameters
    ----------
    csv_file_path : str
        Path to the CSV file; the separator is auto-detected.
    x_column : str
        Name of the first independent variable column.
    y_column : str
        Name of the second independent variable column.
    z_column : str
        Name of the dependent variable column.
    method : {"linear", "nearest"}, default "linear"
        Bilinear or nearest-neighbour interpolation on the grid.

    Returns
    -------
    function : callable
        ``function(x, y)`` returning the interpolated z. With ``"linear"``,
        points outside the grid are clamped to its bounds.
    x_unique : numpy.ndarray
        Sorted unique x values.
    y_unique : numpy.ndarray
        Sorted unique y values.
    z_grid : numpy.ndarray
        Grid values (float32), shape ``(len(y_unique), len(x_unique))``.
        Missing grid points are filled with 0.

    Raises
    ------
    FileNotFoundError
        If ``csv_file_path`` does not exist.
    ValueError
        If ``method`` is not ``"linear"`` or ``"nearest"``, or the file is empty.
    """

    # Load CSV data
    df = _read_csv(csv_file_path)

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
    """Create interpolation functions for several columns of CSV data.

    Parameters
    ----------
    csv_file_path : str
        Path to the CSV file; the separator is auto-detected.
    x_column : str
        Name of the independent variable column.
    y_columns : list of str
        Names of the dependent variable columns.
    method : str, default "linear"
        Interpolation kind passed to :class:`scipy.interpolate.interp1d`
        (one of :data:`INTERP1D_KINDS`).

    Returns
    -------
    function : callable
        ``function(x)`` returning a :class:`~numpy.ndarray` of the
        interpolated values, shape ``(len(y_columns),)`` for scalar ``x``.
        Outside the x range it holds the first/last row (no extrapolation).
    x_data : numpy.ndarray
        x values, sorted ascending.
    y_data : numpy.ndarray
        y values, shape ``(n_points, len(y_columns))``.

    Raises
    ------
    FileNotFoundError
        If ``csv_file_path`` does not exist.
    ValueError
        If ``method`` is not a supported kind or the file is empty.
    """

    # Load CSV data
    df = _read_csv(csv_file_path)

    # Extract and sort data by x values
    df_sorted = df.sort_values(x_column)
    x_data = np.array(df_sorted[x_column].values)

    # Stack all y columns into a 2D array
    y_data_list = []
    for col in y_columns:
        y_data_list.append(df_sorted[col].values)
    y_data = np.array(y_data_list).T  # Shape: (n_points, n_outputs)

    scipy_function_multi = _make_interp1d(x_data, y_data, method)

    return scipy_function_multi, x_data, y_data


def create_parametric_function_from_csv(
    csv_file_path: str,
    param_column: str,
    x_column: str,
    y_column: str,
    method: str = "linear",
) -> Tuple[Callable, np.ndarray, np.ndarray, np.ndarray]:
    """Create a parametric interpolation function y = f(x, param) from CSV data.

    Parameters
    ----------
    csv_file_path : str
        Path to the CSV file; the separator is auto-detected.
    param_column : str
        Name of the parameter column.
    x_column : str
        Name of the independent variable column.
    y_column : str
        Name of the dependent variable column.
    method : {"linear", "nearest"}, default "linear"
        ``"linear"`` uses inverse-distance weighting over all points;
        ``"nearest"`` returns the value of the closest point.

    Returns
    -------
    function : callable
        ``function(x, param)`` returning the interpolated y (scalar inputs).
    param_data : numpy.ndarray
        Parameter values, in file order.
    x_data : numpy.ndarray
        x values, in file order.
    y_data : numpy.ndarray
        y values, in file order.

    Raises
    ------
    FileNotFoundError
        If ``csv_file_path`` does not exist.
    ValueError
        If ``method`` is not ``"linear"`` or ``"nearest"``, or the file is empty.
    """

    # Load CSV data
    df = _read_csv(csv_file_path)

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
