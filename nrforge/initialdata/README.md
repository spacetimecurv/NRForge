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
elliptica = Elliptica.create(path='/path/to/ID', user_params=user_params, system='BHNS')
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
elliptica = Elliptica.analyze(path='/path/to/ID')
elliptica.initial_data_progress(print_timing=True)
elliptica.convergence(plot=False, save=False)
```

This will first try to find the per-resolution directories, parameter files, log-files (slurm; if at a different location, they can be located via `logs_path`), check the status/accuracy, create a convergence plot and give an estimated remaining time based on the log-files. For more detailed time estimates one can call `initial_data_progress` which details how fast each resolution will be.

Once the status of the initial data object switches to `Done`, the checkpoint and properties file will be moved from the highest-resolution directory to the parent level. An extension to other binary systems is trivial and planned for the future. One just has to locate the `self.system` blocks and add the respective system logic.

Other options include the 'free' mode, where all the utility of the `Elliptica` class can be used. This is really useful with the built-in diagnostics readers that enable per-iteration analysis and plots of the running/finished initial data creation. This can be used for instance via:

```python
elliptica = Elliptica.free(path='/path/to/ID')
elliptica.read_diagnostics()
elliptica.plot_constraints_per_patch_group()
elliptica.plot_constraints_per_patch_group(groups={
  "left_NS":          ("#D55E00", "NS"),
  "left_NS_around":   ("#E69F00", "NS surroundings"),
  "right_BH_around":  ("#009E73", "BH surroundings"),
  "left_central_box": ("#CC79A7", "central box"),
  "left_outermost":   ("#56B4E9", "outermost (left)"),
  "right_outermost":  ("#0072B2", "outermost (right)"),
})
elliptica.plot_target_system_convergence()
```

This will call the 'free' classmethod of `Elliptica`, read the `Diagnostics_00` of Elliptica per resolution directory that was found when loading the constructor and then create plots of the constraints per SCDD patch by either taking the default groupings or some specified groups. A plot of the convergence towards the target values of the `<system>_properties.txt` file can also be generated with only selected targets or all of them indicated with the `_current` suffix.

The last option is the 'compare' mode, which interfaces the `EllipticaComparison` class that loads N `Elliptica` objects in 'free' mode, where N is the number of initial data directories. This is particularly useful for comparing initial data of the same system but with different convergence parameters or just for comparison in general. It can be used via:

```python
cmp = Elliptica.compare(
  paths  = ["/path/to/ID1", "/path/to/ID2"],
  labels = ["Label I", "Label II"],
)

cmp.plot_target_system_convergence(quantities=[
  "BH_irreducible_mass", "BH_chi_z", "NS_baryonic_mass"
], output_dir=base)

for run in cmp:
  run.plot_constraints_per_patch_group()

# Create a comparison LaTeX table.
keep = [
  "BH_x_CM", "BH_y_CM", "BH_z_CM",
  "BH_min_radius", "BH_Komar_mass", "BH_irreducible_mass_current",
  "BH_Christodoulou_mass_current", "BH_chi_z_current",
  "NS_x_CM", "NS_y_CM", "NS_z_CM", "NS_TOV_radius",
  "NS_TOV_ADM_mass", "NS_baryonic_mass_current", "NS_rho0_center",
  "BHNS_x_CM", "BHNS_y_CM", "BHNS_z_CM",
  "BHNS_ADM_mass", "BHNS_angular_velocity", "BHNS_binding_energy"
]

cmp.latex_comparison_table(
  keep=keep,
  output_dir="/home/no96soq/athenak/runs/PROJECT_largescale/initial_data/bhns/convergence_tuning/production/comparison_results",
  diffs="both"
)
```

This will locate and load the per-resolution directories of each specified path, plot the target convergence comparison for some selected quantities, plot the per-patch constraints of each run via the `Elliptica` methods, and create a LaTeX comparison table for either all `<system>_properties.txt` parameters or only for some selected ones. For multiple runs a reference run can be selected in the function for the computed differences. Differences can be either shown as absolute, relative, relative and absolute, or via min/max spread. They can also be neglected via 'none'.
