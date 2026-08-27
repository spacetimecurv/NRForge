# NRForge
A lightweight tool to generate initial data for various systems and solvers and write parameter files for evolution codes from templates.

## Installation

```bash
git clone --recurse-submodules https://github.com/spacetimecurv/NRForge.git
cd NRForge
pip install -e .
```

This makes `nrforge` importable from anywhere in that environment.

## Submodules

Two external tools live under `external/` as git submodules:

| Submodule | Contents |
| --- | --- |
| `external/EOSympOSE` | fetches CompOSE equations of state and converts them for NR codes and initial data solvers (wraps `PyCompOSE`, a nested submodule) |
| `external/batchtools` | sets up and manages simulation directories and batch scripts on HPC clusters |

Neither is needed to import `nrforge`, but the EOS tables and batch scripts they
produce feed the pipelines above. If the repository is already checked out, or the nested
`PyCompOSE` is missing, pull them in afterwards:

```bash
git submodule update --init --recursive
```

Installing the EOS tools alongside `nrforge` is then a single call:

```bash
pip install -e . -e external/EOSympOSE/external/PyCompOSE -e external/EOSympOSE
```

Both repositories document themselves in `external/EOSympOSE/README.md` and
`external/batchtools/README.md`.

## Usage

```python
import nrforge

nrforge.banner()

# Utilities live in their modules.
lam = nrforge.utils.variables.lambda_tilde(q=1.2, lam1=400.0, lam2=900.0)
chi = nrforge.utils.variables.chi_effective(q=1.2, chi1=0.1, chi2=-0.05)

# Solver interfaces likewise.
ell = nrforge.initialdata.elliptica.Elliptica(path="/path/to/id", mode="analyze",
                                              print_info=True)
ell.convergence(plot=True, save=True)

# Once the initial data is done, write the evolution parameter file.
evo = nrforge.simulation.evolution.Evolution("AthenaK", "Elliptica", "BHNS")
evo.write_parfile(params=params, ev_path="/path/to/evo", id_path="/path/to/id")
```

Importing `nrforge` pulls in every subpackage, so any module is reachable through
the dotted path. Import the names directly for shorter call sites:

```python
from nrforge.utils.variables import lambda_tilde, chi_effective
from nrforge.initialdata.elliptica import Elliptica
from nrforge.simulation.evolution import Evolution
```

Subpackages:

| Module | Contents |
| --- | --- |
| `nrforge.utils.constants` | physical constants |
| `nrforge.utils.variables` | derived binary variables |
| `nrforge.utils.eos_utils` | CompOSE / Elliptica EOS helpers |
| `nrforge.utils.fits` | phenomenological fits from the literature |
| `nrforge.utils.style` | terminal formatting helpers |
| `nrforge.initialdata.elliptica` | interface to the Elliptica solver (`Elliptica`) |
| `nrforge.initialdata.templates` | Elliptica parameter-file templates and example parameters |
| `nrforge.simulation.evolution` | interface to the evolution codes (`Evolution`) |
| `nrforge.simulation.templates` | evolution parameter-file templates and resolution configs |

Initial data and evolution are meant to be chained: `Elliptica` creates and monitors
the initial data, `Evolution` then picks up that directory, builds the run directory,
copies the initial data over and writes the evolution parameter file from a `jinja2`
template. Currently supported is `AthenaK` with `Elliptica` initial data for BHNS
binaries. The details live in the READMEs of the subpackages:

| README | Contents |
| --- | --- |
| `nrforge/initialdata/README.md` | the initial data solvers and how to run them |
| `nrforge/initialdata/templates/README.md` | Elliptica templates and their parameters |
| `nrforge/simulation/templates/athenak/README.md` | the AthenaK pipeline, its parameters and resolutions |
