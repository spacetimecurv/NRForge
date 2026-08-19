"""Interfaces to the initial data solvers.

Currently supported:

- :mod:`~nrforge.initialdata.elliptica` — the Elliptica solver, wrapped by
  :class:`~nrforge.initialdata.elliptica.Elliptica`.
"""

import importlib as _importlib

_MODULES = {"elliptica"}
_ATTRS = {"Elliptica": ".elliptica"}

__all__ = ["elliptica", "Elliptica"]


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
