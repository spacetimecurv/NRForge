"""NRForge — a lightweight toolkit for numerical relativity initial data and evolution.

Two subpackages:

- :mod:`nrforge.utils` — constants, derived binary parameters, EOS helpers and
  phenomenological fits. Everything here is re-exported at the top level, so
  ``from nrforge import lambda_tilde`` and
  ``from nrforge.utils import lambda_tilde`` are equivalent.
- :mod:`nrforge.initialdata` — interfaces to the initial data solvers, e.g.
  :class:`~nrforge.initialdata.elliptica.Elliptica`.
"""

import importlib as _importlib

from .utils import (
    G_SI,
    c_SI,
    Msun_SI,
    lambda_tilde,
    chi_effective,
    compose_get_M_from_MR,
    compose_get_R_from_MR,
    load_elliptica_table,
    calculate_enthalpy_bounds,
    normalized_R_ISCO,
    BHNS_remnant_mass_model_Foucart2018,
)
from . import utils

try:  # Installed package: read the version from the metadata.
    from importlib.metadata import version as _version

    __version__ = _version("nrforge")
except Exception:  # pragma: no cover - running from a source checkout.
    __version__ = "0.0.0+unknown"

__ascii_art__ = r"""
.__   __. .______       ._______ ______ ._____   ______.______
|  \ |  | |   _  \      |  ____|/  __  \|  _  \ / _____| _____|
|   \|  | |  |_)  |     | |____|  |  |  | |_)  | /  ___| |____
|  . `  | |      /      |  ____|  |  |  |     /| | |__ | _____|
|  |\   | |  |\  \----. | |    |  `--'  | |\  \| \___/ | |____
|__| \__| | _| `._____| |_|     \______/|_| \__\\_____/|______|
"""

__quote__ = "~You can never solve a problem on the level on which it was created.~"


def banner():
    """Print the NRForge banner, its version and the motto."""
    print(__ascii_art__)
    print(f"  version {__version__}")
    print(f"  {__quote__}")


# Lazily imported names: the module they live in, keyed by attribute name.
_LAZY_MODULES = {"initialdata"}
_LAZY_ATTRS = {
    "Elliptica": ".initialdata.elliptica",
    "get_elliptica_bhns_template": ".initialdata.templates",
    "get_elliptica_bhns_user_params_example": ".initialdata.templates",
    "get_iterated_bh_mass": ".initialdata.templates"
}


def __getattr__(name):
    if name in _LAZY_MODULES:
        module = _importlib.import_module(f".{name}", __name__)
        globals()[name] = module
        return module
    if name in _LAZY_ATTRS:
        module = _importlib.import_module(_LAZY_ATTRS[name], __name__)
        attr = getattr(module, name)
        globals()[name] = attr
        return attr
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | _LAZY_MODULES | set(_LAZY_ATTRS))


__all__ = [
    # Subpackages.
    "utils",
    "initialdata",
    # Constants.
    "G_SI",
    "c_SI",
    "Msun_SI",
    # Derived binary variables.
    "lambda_tilde",
    "chi_effective",
    # EOS utilities.
    "compose_get_M_from_MR",
    "compose_get_R_from_MR",
    "load_elliptica_table",
    "calculate_enthalpy_bounds",
    # Fits.
    "normalized_R_ISCO",
    "BHNS_remnant_mass_model_Foucart2018",
    # Initial data solvers.
    "Elliptica",
    # Parameter-file templates.
    "get_elliptica_bhns_template",
    "get_elliptica_bhns_user_params_example",
    "get_iterated_bh_mass",
    # Meta.
    "banner",
    "__version__",
]
