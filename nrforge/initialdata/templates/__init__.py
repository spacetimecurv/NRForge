"""Parameter-file templates for the initial data solvers.

One module per solver:

- :mod:`~nrforge.initialdata.templates.elliptica_templates` — Elliptica
  parameter dictionaries, handed out by :func:`get_elliptica_bhns_template`.

Every name below is also re-exported from :mod:`nrforge.initialdata` and from
the top-level :mod:`nrforge`.
"""

from .elliptica_templates import get_elliptica_bhns_template, get_elliptica_bhns_user_params_example, \
                                 get_iterated_bh_mass

from . import elliptica_templates

__all__ = [
    "get_elliptica_bhns_template",
    "get_elliptica_bhns_user_params_example",
    "get_iterated_bh_mass",
    "elliptica_templates",
]
