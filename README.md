# NRForge
A lightweight tool to generate initial data for various systems and solvers.
Adapted and modified from https://git.tpi.uni-jena.de/agonzalez/nrtools.

## Installation

```bash
git clone https://github.com/spacetimecurv/NRForge.git
cd NRForge
pip install -e .        # editable: edits to the source are picked up immediately
```

This makes `nrforge` importable from anywhere in that environment.

## Usage

```python
import nrforge

nrforge.banner()

# Utilities live in their modules.
lam = nrforge.utils.variables.lambda_tilde(q=1.2, lam1=400.0, lam2=900.0)
chi = nrforge.utils.variables.chi_effective(q=1.2, chi1=0.1, chi2=-0.05)

# Solver interfaces likewise.
ell = nrforge.initialdata.elliptica.Elliptica(path="/path/to/run", mode="analyze",
                                              print_info=True)
ell.convergence(plot=True, save=True)
```

Importing `nrforge` pulls in every subpackage, so any module is reachable through
the dotted path. Import the names directly for shorter call sites:

```python
from nrforge.utils.variables import lambda_tilde, chi_effective
from nrforge.initialdata.elliptica import Elliptica
```

Subpackages:

| Module | Contents |
| --- | --- |
| `nrforge.utils.constants` | physical constants |
| `nrforge.utils.variables` | derived binary variables |
| `nrforge.utils.eos_utils` | CompOSE / Elliptica EOS helpers |
| `nrforge.utils.fits` | phenomenological fits from the literature |
| `nrforge.initialdata.elliptica` | interface to the Elliptica solver (`Elliptica`) |
| `nrforge.initialdata.templates.elliptica_templates` | Elliptica parameter-file templates |
