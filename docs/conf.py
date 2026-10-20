"""Sphinx configuration for magnet-scipy documentation."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import magnet_scipy  # noqa: E402

# -- Project information -----------------------------------------------------

project = "magnet-scipy"
author = magnet_scipy.__author__
copyright = f"2025, {author}"
version = release = magnet_scipy.__version__

# -- General configuration ---------------------------------------------------

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
]

# .rst is needed for the stub pages autosummary generates.
source_suffix = {".md": "markdown", ".rst": "restructuredtext"}
root_doc = "index"
exclude_patterns = ["_build"]

# -- MyST --------------------------------------------------------------------

myst_enable_extensions = [
    "amsmath",
    "colon_fence",
    "deflist",
    "dollarmath",
]
# Generate anchors for h1-h3 so in-page links like ``[x](#section)`` resolve.
myst_heading_anchors = 3

# -- autodoc / autosummary / napoleon ----------------------------------------

autosummary_generate = True
autodoc_default_options = {
    "members": True,
    "show-inheritance": True,
}
autodoc_typehints = "description"

napoleon_google_docstring = False
napoleon_numpy_docstring = True
# Resolve short type names used in docstrings to their canonical module, since
# some classes are also re-exported from the package or duplicated elsewhere.
napoleon_preprocess_types = True
napoleon_type_aliases = {
    "RLCircuitPID": "~magnet_scipy.rlcircuitpid.RLCircuitPID",
    "CoupledRLCircuitsPID": "~magnet_scipy.coupled_circuits.CoupledRLCircuitsPID",
    "PIDController": "~magnet_scipy.pid_controller.PIDController",
    "RegionConfig": "~magnet_scipy.pid_controller.RegionConfig",
    "SimulationResult": "~magnet_scipy.simulation_types.SimulationResult",
}

# -- intersphinx -------------------------------------------------------------

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
}

# -- HTML output -------------------------------------------------------------

html_theme = "sphinx_rtd_theme"
html_title = f"{project} {release}"
