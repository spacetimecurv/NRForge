#!/usr/bin/env python
from setuptools import setup, find_packages
import re
import sys

__ascii_art__ = r"""
.__   __. .______       ._______ ______ ._____   ______.______
|  \ |  | |   _  \      |  ____|/  __  \|  _  \ / _____| _____|
|   \|  | |  |_)  |     | |____|  |  |  | |_)  | /  ___| |____
|  . `  | |      /      |  ____|  |  |  |     /| | |__ | _____|
|  |\   | |  |\  \----. | |    |  `--'  | |\  \| \___/ | |____
|__| \__| | _| `._____| |_|     \______/|_| \__\\_____/|______|
"""

print(__ascii_art__)
print("~You can never solve a problem on the level on which it was created.~")

# Read version from pyproject.toml
ini = open('pyproject.toml').read()
vrs = r"^version = ['\"]([^'\"]*)['\"]"
mo  = re.search(vrs, ini, re.M)
version = mo.group(1)

setup(
    name='nrforge',
    version=version,
    description='Numerical Relativity Forge',
    author='OS',
    author_email='oliver.steppohn@uni-jena.de',
    url = 'https://github.com/spacetimecurv/NRForge.git',
    packages = find_packages(),
    requires = ['numpy', 'matplotlib'],
)

