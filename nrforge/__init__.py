#!/usr/bin/env python

from . import utils, initialdata

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
