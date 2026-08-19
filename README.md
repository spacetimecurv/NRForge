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

# Utilities are re-exported at the top level.
lam = nrforge.lambda_tilde(q=1.2, lam1=400.0, lam2=900.0)
chi = nrforge.chi_effective(q=1.2, chi1=0.1, chi2=-0.05)

# Solver interfaces are imported on first access.
ell = nrforge.Elliptica(print_info=True)
ell.locate_initial_data("/path/to/run")
ell.convergence(plot=True, save=True)
```

Subpackages:

| Module | Contents |
| --- | --- |
| `nrforge.utils` | constants, derived binary variables, CompOSE EOS helpers, fits |
| `nrforge.initialdata` | interfaces to initial data solvers (`Elliptica`) |
