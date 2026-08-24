"""Interfaces to the initial data solvers.

Currently supported:

- :mod:`~nrforge.initialdata.elliptica` — the Elliptica solver, wrapped by
  :class:`~nrforge.initialdata.elliptica.Elliptica`.

Parameter-file templates for every solver live in
:mod:`~nrforge.initialdata.templates`.
"""

import importlib as _importlib

_MODULES = {"elliptica", "templates"}
_ATTRS = {"Elliptica": ".elliptica", "get_elliptica_bhns_template": ".templates",
          "get_elliptica_bhns_user_params_example": ".templates"}

__all__ = ["elliptica", "Elliptica", "templates", "get_elliptica_bhns_template",
           "get_elliptica_bhns_user_params_example", "get_iterated_bh_mass"]


def __getattr__(name):
    if name in _MODULES:
        module = _importlib.import_module(f".{name}", __name__)
        globals()[name] = module
        return module
    if name in _ATTRS:
        module = _importlib.import_module(_ATTRS[name], __name__)
        attr = getattr(module, name)
        globals()[name] = attr
        return attr
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(__all__)
