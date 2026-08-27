# Initial data solvers

NRForge supports various initial data solvers, although currently only `Elliptica` is implemented. Each solvers supports its own class that makes automated pipelines for initial data generation ambiguous.

## Elliptica

For the initial data solver [Elliptica](https://github.com/rashti-alireza/Elliptica) the class `Elliptica` is implemented in `elliptica.py`. Once `Elliptica` is installed on the respective machine, NRForge can be used to create and monitor Elliptica initial data. The mode 'create' is used for creation and a simple script could be

```python
import json
from nrforge.initialdata.elliptica import Elliptica, get_iterated_bh_mass
from nrforge.utils.eos_utils import load_elliptica_table, calculate_enthalpy_bounds

mass = get_iterated_bh_mass(mass=4.3, sequence=[50, 50, 50, 20], gap=0.5)
rows = load_elliptica_table(path='data/DD2_eos.txt', geo=False)
h_floor, h_ceil = calculate_enthalpy_bounds(rows=rows, margin=5e-4, ceil_trim=0, print_info=False)

with open("../initialdata/templates/elliptica/bhns_params.json", "r") as file:
  user_params = json.load(file)

user_params['bh_irreducible_mass'] = mass
user_params['ns_h_floor']         = h_floor
user_params['ns_h_ceil']          = h_ceil
elliptica = Elliptica(path='/path/to/ID',
                      mode='create', logs_path=None, print_info=False, user_params=user_params, system='BHNS')
elliptica.write_bashfile(bashname="BHNS.slurm", cluster="ARA", id_exe="/path/to/Elliptica/Exe/elliptica")
elliptica.submit_job()
```

How the class in 'create' mode works is, that we first specify a path where the initial data should be created and pass that to `path`. With `print_info` we can print useful information about `Elliptica` and it's system solvers as well as some build instructions. Since currently only black hole-neutron star binaries are supported, we can only pass 'BHNS' to `system`. `logs_path` can be ignored in 'create' mode. Lastly, the most important parameter is `user_params` which takes a dictionary with user parameters from the user and later passes that internally to the `jinja2` templating engine. The production level template can be found under `initialdata/templates/elliptica/elliptica_bhns.in.j2`. For the parameters, the user can take the example .json file under `initialdata/templates/elliptica/bhns_params.json` where the most important parameters about the binary can be set. The equation-of-state can be created with EOSympOSE via

```python
options = {"eos_name": "DD2",
           "output_dir": "/path/to/dir",
           "elliptica": True,
           "elliptica_format": "compose",
           "eos_cold": True}
eos_path = eosympose.run(**options)
```

This tells EOSympOSE to fetch the DD2 table from the CompOSE database and write it into the format `Elliptica` expects. Other tables can be easily added to the config file in EOSympOSE (see the README.md there). What usually helps with stability, especially when using equation-of-state tables with `Elliptica` is to iterate over the irreducible mass of the black hole, which is why we can explicitely set an iteration sequence in the .json file or compute it with `get_iterated_bh_mass` as in the snippet above. Furthermore, depending on the equation-of-state the enthalpy floor might be different, and we can use the function `calculate_enthalpy_bounds` to calculate the enthalpy bounds from the example equation of state given in `tests/data`. These are best practices and do not have to be followed. If desired we can also write a slurm file and submit the job directly. Currently only supported is `ARA`, but an extension to other machines is trivial. Once everything worked, we will have an initial data folder with naming based on the binary parameters, a parameter file and a slurm file. The run can now start!

To monitor the run, we can switch to 'analyze' mode via

```python
elliptica = Elliptica(path='/path/to/ID',
                     mode='analyze', logs_path=None, print_info=False, user_params=None)

elliptica.initial_data_progress(print_timing=True)
elliptica.convergence(plot=False, save=False)
```
This will first try to find the per-resolution directories, parameter files, log-files (slurm; if at a different location, they can be located via `logs_path`), check the status/accuracy, create a convergence plot and give an estimated remaining time based on the log-files. For more detailed time estimates one can call `initial_data_progress` which details how fast each resolution will be.

Once the status of the initial data object switches to `Done`, the checkpoint and properties file will be moved from the highest-resolution directory to the parent level. An extension to other binary systems is trivial and planned for the future. One just has to locate the `self.system` blocks and add the respective system logic.
