"""Shared utilities: constants, derived variables, EOS helpers and fits.

- :mod:`~nrforge.utils.constants` — physical constants in <selective> units.
- :mod:`~nrforge.utils.variables` — derived binary parameters.
- :mod:`~nrforge.utils.eos_utils` — equation-of-state utilities.
- :mod:`~nrforge.utils.fits` — phenomenological fits from the literature.

Every name below is also re-exported from the top-level :mod:`nrforge`.
"""

from .constants import G_SI, c_SI, Msun_SI
from .variables import lambda_tilde, chi_effective
from .eos_utils import compose_get_M_from_MR, compose_get_R_from_MR
from .fits import normalized_R_ISCO, BHNS_remnant_mass_model_Foucart2018

from . import constants
from . import variables
from . import eos_utils
from . import fits

__all__ = [
    "G_SI",
    "c_SI",
    "Msun_SI",
    "lambda_tilde",
    "chi_effective",
    "compose_get_M_from_MR",
    "compose_get_R_from_MR",
    "normalized_R_ISCO",
    "BHNS_remnant_mass_model_Foucart2018",
    "constants",
    "variables",
    "eos_utils",
    "fits",
]
