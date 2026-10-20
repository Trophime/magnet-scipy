import numpy as np
from typing import Tuple, List, Dict, Optional
from dataclasses import dataclass


@dataclass
class PIDParams:
    """Data class to hold PID parameters for a specific current region"""

    Kp: float
    Ki: float
    Kd: float

    def __post_init__(self):
        """Validate PID parameters"""
        if self.Kp < 0 or self.Ki < 0 or self.Kd < 0:
            raise ValueError("PID parameters must be non-negative")


@dataclass
class RegionConfig:
    """Configuration for a current region with PID parameters and threshold"""

    params: PIDParams
    threshold: Optional[float] = (
        None  # Upper bound for this region (None for highest region)
    )

    def __post_init__(self):
        if self.threshold is not None and self.threshold < 0:
            raise ValueError("Threshold must be non-negative")


class PIDController:
    """
    Dictionary-based Adaptive PID Controller with flexible region definitions

    Manages PID parameters using a dictionary of named regions, each with
    their own parameters and threshold values.
    """

    def __init__(self, regions: Optional[Dict[str, RegionConfig]] = None):
        """Initialize adaptive PID controller with named regions.

        Parameters
        ----------
        regions : dict of str to RegionConfig, optional
            Region name to configuration, ordered from lowest to highest
            current. Thresholds [A] must be ascending and exactly one region
            (the highest) has ``threshold=None``. If None, default
            low/medium/high regions with thresholds 60 and 800 A are used.

        Raises
        ------
        ValueError
            If ``regions`` is empty, thresholds are not ascending, or not
            exactly one region has ``threshold=None``.

        Examples
        --------
        >>> regions = {
        ...     "low": RegionConfig(PIDParams(10.0, 5.0, 0.1), threshold=60.0),
        ...     "medium": RegionConfig(PIDParams(15.0, 8.0, 0.05), threshold=800.0),
        ...     "high": RegionConfig(PIDParams(25.0, 12.0, 0.02)),
        ... }
        >>> controller = PIDController(regions)
        """

        if regions is None:
            # Create default regions
            regions = {
                "low": RegionConfig(PIDParams(10.0, 5.0, 0.1), threshold=60.0),
                "medium": RegionConfig(PIDParams(15.0, 8.0, 0.05), threshold=800.0),
                "high": RegionConfig(PIDParams(25.0, 12.0, 0.02)),
            }

        self.regions = regions
        self._validate_regions()

        self._setup_arrays()

    def _validate_regions(self):
        """Validate region configuration"""
        if not self.regions:
            raise ValueError("At least one region must be defined")

        # Check that exactly one region has no threshold (the highest)
        no_threshold_count = sum(
            1 for config in self.regions.values() if config.threshold is None
        )
        if no_threshold_count != 1:
            raise ValueError(
                "Exactly one region must have no threshold (the highest region)"
            )

        # Check that thresholds are in ascending order
        thresholds = [
            config.threshold
            for config in self.regions.values()
            if config.threshold is not None
        ]
        if thresholds != sorted(thresholds):
            raise ValueError("Region thresholds must be in ascending order")

    def _setup_arrays(self):
        """Setup Scipy-compatible arrays for efficient region selection"""
        # Create ordered lists of thresholds and parameters
        self.region_names = list(self.regions.keys())

        # Extract thresholds (use large value for region without threshold)
        thresholds = []
        kp_values = []
        ki_values = []
        kd_values = []

        for name in self.region_names:
            config = self.regions[name]
            # Use a very large threshold for the region without threshold
            threshold = config.threshold if config.threshold is not None else 1e10
            thresholds.append(threshold)
            kp_values.append(config.params.Kp)
            ki_values.append(config.params.Ki)
            kd_values.append(config.params.Kd)

        # Convert to numpy arrays for efficient computation
        self.thresholds = np.array(thresholds)
        self.kp_values = np.array(kp_values)
        self.ki_values = np.array(ki_values)
        self.kd_values = np.array(kd_values)

    def get_pid_parameters(self, i_ref: float) -> Tuple[float, float, float]:
        """Get PID gains for the region selected by the reference current.

        The region is the first one whose threshold exceeds ``|i_ref|``.

        Parameters
        ----------
        i_ref : float
            Reference current [A].

        Returns
        -------
        tuple of float
            ``(Kp, Ki, Kd)`` [V/A, V/(A·s), V·s/A] for the operating region.
        """
        abs_i_ref = np.abs(i_ref)

        # Find the first region where current is below threshold
        # This creates a boolean mask and uses argmax to find first True
        below_threshold = abs_i_ref < self.thresholds
        region_idx = np.argmax(below_threshold)

        # Extract parameters using the region index
        Kp = self.kp_values[region_idx]
        Ki = self.ki_values[region_idx]
        Kd = self.kd_values[region_idx]

        return Kp, Ki, Kd

    def get_current_region_name(self, i_ref: float) -> str:
        """Get the operating region name for a reference current.

        Parameters
        ----------
        i_ref : float
            Reference current [A].

        Returns
        -------
        str
            Name of the operating region.
        """
        abs_i_ref = np.abs(i_ref)
        below_threshold = abs_i_ref < self.thresholds
        region_idx = int(np.argmax(below_threshold))
        return self.region_names[region_idx]

    def get_region_index(self, i_ref):
        """Get the operating region index for a reference current.

        Array-friendly counterpart of :meth:`get_current_region_name`.

        Parameters
        ----------
        i_ref : float or numpy.ndarray
            Reference current [A].

        Returns
        -------
        int or numpy.ndarray
            Index of the operating region in :attr:`region_names`.
        """
        abs_i_ref = np.abs(i_ref)
        below_threshold = abs_i_ref < self.thresholds
        return np.argmax(below_threshold)

    def add_region(self, name: str, config: RegionConfig):
        """Add a new region to the controller.

        Parameters
        ----------
        name : str
            Name of the new region.
        config : RegionConfig
            Configuration of the new region.

        Raises
        ------
        ValueError
            If the region already exists or the resulting configuration is
            invalid.
        """
        if name in self.regions:
            raise ValueError(f"Region '{name}' already exists")

        self.regions[name] = config
        self._validate_regions()
        self._setup_arrays()

    def remove_region(self, name: str):
        """Remove a region from the controller.

        Parameters
        ----------
        name : str
            Name of the region to remove.

        Raises
        ------
        ValueError
            If the region does not exist, is the last one, or the resulting
            configuration is invalid.
        """
        if name not in self.regions:
            raise ValueError(f"Region '{name}' does not exist")

        if len(self.regions) <= 1:
            raise ValueError("Cannot remove the last region")

        del self.regions[name]
        self._validate_regions()
        self._setup_arrays()

    def update_region(self, name: str, config: RegionConfig):
        """Update an existing region's configuration.

        Parameters
        ----------
        name : str
            Name of the region to update.
        config : RegionConfig
            New configuration.

        Raises
        ------
        ValueError
            If the region does not exist or the resulting configuration is
            invalid.
        """
        if name not in self.regions:
            raise ValueError(f"Region '{name}' does not exist")

        self.regions[name] = config
        self._validate_regions()
        self._setup_arrays()

    def get_region_names(self) -> List[str]:
        """Get list of all region names"""
        return list(self.regions.keys())

    def get_thresholds(self) -> Dict[str, Optional[float]]:
        """Get all thresholds as a dictionary.

        Returns
        -------
        dict of str to float or None
            Region name to threshold [A]; None for the highest region.
        """
        return {name: config.threshold for name, config in self.regions.items()}

    def print_summary(self):
        """Print a summary of all regions and their configurations"""
        print("\n=== PID Controller Configuration ===")
        print(f"Number of regions: {len(self.regions)}")
        print("\nRegions (in order):")

        for region_name, config in self.regions.items():
            threshold_str = (
                f"< {config.threshold}"
                if config.threshold is not None
                else ">= previous"
            )
            print(
                f"  {region_name:10s}: |I_ref| {threshold_str:12s} -> "
                f"Kp={config.params.Kp:6.2f}, Ki={config.params.Ki:6.2f}, Kd={config.params.Kd:6.4f}"
            )

    def __repr__(self) -> str:
        """String representation of the PID controller"""
        return f"PIDController(regions={len(self.regions)}, names={list(self.regions.keys())})"


def create_default_pid_controller() -> PIDController:
    """Create a PID controller with standard low/medium/high regions.

    Returns
    -------
    PIDController
        Controller with thresholds 60 and 800 A.
    """
    regions = {
        "low": RegionConfig(PIDParams(10.0, 5.0, 0.1), threshold=60.0),
        "medium": RegionConfig(PIDParams(15.0, 8.0, 0.05), threshold=800.0),
        "high": RegionConfig(PIDParams(25.0, 12.0, 0.02)),
    }
    return PIDController(regions)


def create_adaptive_pid_controller(
    # Low current gains (more aggressive for small signals)
    Kp_low: float = 20.0,
    Ki_low: float = 15.0,
    Kd_low: float = 0.1,
    # Medium current gains (balanced)
    Kp_medium: float = 12.0,
    Ki_medium: float = 8.0,
    Kd_medium: float = 0.05,
    # High current gains (more conservative for stability)
    Kp_high: float = 8.0,
    Ki_high: float = 5.0,
    Kd_high: float = 0.02,
    # Thresholds
    low_threshold: float = 60.0,
    high_threshold: float = 800.0,
) -> PIDController:
    """Create a low/medium/high adaptive PID controller (backward compatibility).

    Parameters
    ----------
    Kp_low, Ki_low, Kd_low : float
        Gains for ``|i_ref| < low_threshold`` [V/A, V/(A·s), V·s/A].
    Kp_medium, Ki_medium, Kd_medium : float
        Gains for ``low_threshold <= |i_ref| < high_threshold``.
    Kp_high, Ki_high, Kd_high : float
        Gains for ``|i_ref| >= high_threshold``.
    low_threshold : float, default 60.0
        Upper bound of the low region [A].
    high_threshold : float, default 800.0
        Upper bound of the medium region [A].

    Returns
    -------
    PIDController
        Configured controller.
    """
    regions = {
        "low": RegionConfig(PIDParams(Kp_low, Ki_low, Kd_low), threshold=low_threshold),
        "medium": RegionConfig(
            PIDParams(Kp_medium, Ki_medium, Kd_medium), threshold=high_threshold
        ),
        "high": RegionConfig(PIDParams(Kp_high, Ki_high, Kd_high)),
    }
    return PIDController(regions)


def create_custom_pid_controller(
    region_configs: Dict[str, Tuple[Tuple[float, float, float], Optional[float]]],
) -> PIDController:
    """Create a custom PID controller with arbitrary regions.

    Parameters
    ----------
    region_configs : dict of str to tuple
        Mapping of region name to ``((Kp, Ki, Kd), threshold)``, with gains
        Kp [V/A], Ki [V/(A·s)], Kd [V·s/A] and threshold [A] on ``|i_ref|``.
        Thresholds must be ascending; exactly one region (the highest) has
        ``threshold=None``.

    Returns
    -------
    PIDController
        Configured controller.

    Raises
    ------
    ValueError
        If thresholds are not ascending or not exactly one region has
        ``threshold=None``.

    Examples
    --------
    >>> configs = {
    ...     "startup": ((50.0, 30.0, 0.2), 10.0),
    ...     "normal": ((20.0, 15.0, 0.1), 100.0),
    ...     "overload": ((5.0, 2.0, 0.01), None),
    ... }
    >>> controller = create_custom_pid_controller(configs)
    """
    regions = {}
    for name, ((Kp, Ki, Kd), threshold) in region_configs.items():
        regions[name] = RegionConfig(PIDParams(Kp, Ki, Kd), threshold)

    return PIDController(regions)
