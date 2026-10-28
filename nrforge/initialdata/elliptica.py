####################################################
#          Class for Elliptica ID solver           #
####################################################

# Built-in libraries.
import sys
import re
import warnings
from collections import defaultdict
from collections.abc import Iterator
import statistics
import subprocess
import json
from datetime import datetime

# Third-party libraries.
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from jinja2 import Environment, FileSystemLoader
from pathlib import Path
from matplotlib.lines import Line2D
import matplotlib.transforms as mtransforms
import matplotlib.colors as mcolors

# NRForge libraries.
from nrforge.utils.variables import get_id_gw_frequency_Hz, get_id_gw_frequency_Hz_22
from nrforge.utils.style import _style, _fmt, _BOLD, _DIM, _CYAN


# ==============================================================================
# Parfile write helper functions.
# ==============================================================================

# Get a iterated BH mass (needed for BH mass ramp, which
# improves convergence).
def get_iterated_bh_mass(mass: float, sequence: list, gap: float) -> str:
  """
  Computes an iterated mass string for a Elliptica parfile
  that gives more stability for the convergence.

  Parameters:
  mass    (float): target mass that will be iterated to.
  sequence (list): list with the iterations per iterated mass.
  gap     (float): gap between iterated masses.

  Returns:
  String holding the iteration sequence to the target mass.
  """
  string = ""
  length = len(sequence)

  for i, iter in enumerate(sequence):
    string += str((mass + ((length-i)*gap)))
    string += f"(x{sequence[i]})"
    string += f"->"

  string += str(mass)
  return string


# ==============================================================================
# Plotting helper functions.
# ==============================================================================

# Color palette for plotting.
_PALETTE = ["#D55E00", "#E69F00", "#009E73", "#CC79A7",
            "#56B4E9", "#0072B2", "#F0E442", "#000000"]


def _target_symbol(key: str) -> str:
  """
  From an Elliptica key in _properties.txt, extract the key that matches
  _TARGET_SYMBOLS and get the symbol for plotting.

  Parameters:
  key (str): such as 'BH_irreducible_mass'.

  Returns:
  String of the symbol for plotting.
  """
  obj, name = key.split("_", 1)
  sym       = _TARGET_SYMBOLS.get(name, r"\mathrm{" + name.replace("_", r"\_") + "}")
  return sym + r"^\mathrm{" + obj + "}"


def _format_target_axes(axs: np.ndarray, quantities: list, last_iter: int) -> None:
  """
  Format the target axes for the system target convergence plot.

  Parameters:
  axs: axes object.
  quantities (list): list with the target quantities.
  last_iter   (int): last iteration for the current resolution.

  Returns:
  None.
  """
  for i, key in enumerate(quantities):
    sym = _target_symbol(key)
    axs[0,i].set_ylabel(f"${sym}$", fontsize=16)
    axs[1,i].set_ylabel(rf"$|\Delta {sym}| / {sym}$", fontsize=16)
    axs[1,i].set_xlabel("Iterations", fontsize=16)
    axs[0,i].tick_params(labelbottom=False)

  for ax in axs.flat:
    ax.set_xlim(0, last_iter)


def _save_figure(fig: plt.Figure, out_dir: Path, out_file: str, name: str) -> None:
  """
  Wrapper to save a plotting figure.

  Parameters:
  fig: Figure object.
  out_dir (Path): pathlib Path object pointing to the output directory.
  out_file (str): Output file name.
  name     (str): Name of the plot for CMD formatting.

  Returns:
  None.
  """
  path = Path(out_dir) / out_file
  fig.savefig(path, dpi=150)
  print(f"  {_style("$", _DIM)} Saved {name} plot to: {_style(path, _CYAN)}")
  plt.close(fig)


# Constraint keys from _0D.txt output and their plotting labels.
_CONSTRAINTS = {
  "ham1|L2":    r"$||\mathcal{H}||_2$",
  "mom1_U0|L2": r"$||\mathcal{M}_x||_2$",
  "mom1_U1|L2": r"$||\mathcal{M}_y||_2$",
  "mom1_U2|L2": r"$||\mathcal{M}_z||_2$",
}

# Target symbols for the system target convergence plot.
_TARGET_SYMBOLS = {
  "irreducible_mass":   r"M_\mathrm{irr}",
  "Christodoulou_mass": r"M_\mathrm{chr}",
  "baryonic_mass":      r"M_\mathrm{b}",
  "ADM_mass":           r"M_\mathrm{ADM}",
  "Jz_ADM":             r"J_z",
  "chi_x":              r"\chi_x",
  "chi_y":              r"\chi_y",
  "chi_z":              r"\chi_z",
}


# ==============================================================================
# Elliptica class helper functions.
# ==============================================================================

# Resolution directories regular expression.
_RES_DIR = re.compile(r"_(\d+)x(\d+)x(\d+)_\d+$")


def _has_resolution_dirs(path: Path) -> bool:
  """
  Check whether a path has resolution directories with
  the _RES_DIR regular expression.

  Parameters:
  path (Path): pathlib Path object.

  Returns:
  True if resolution directories found.
  """
  return path.is_dir() and any(d.is_dir() and _RES_DIR.search(d.name) for d in path.iterdir())


def _resolution_n(key: str) -> int:
  """
  Get resolution integer from string like '22x22x22', which
  are the per-resolution access keys.

  Parameters:
  key (str): like '22x22x22'.

  Returns:
  Integer, i.e. '22' from '22x22x22'.
  """
  return int(key.split("x")[0])


def _infer_system(name: str) -> str:
  """
  Infer the system from the specified name.

  Parameters:
  name (str): run directory name.

  Returns:
  System like 'BHNS' if name contains 'BH' and 'NS'.
  """
  return "BHNS" if "BH" in name and "NS" in name else None


# ==============================================================================
# Elliptica class.
# ==============================================================================
class Elliptica:
  """
  This class groups useful utility for the analysis of initial
    data produced by the Elliptica solver. Some of the capabilities
    are:
      - Estimate the runtime per resolution for an ongoing run.
      - Check the run status.
      - Check the convergence on specified domain patches.
      - Write a parameter file from template.
      - Write slurm submission scripts for specific machines.
      - Print useful information about Elliptica (compilation etc.).
      - ...
  """
  def __init__(self, system: str = None, print_info: bool = False) -> None:
    """
    Sets up an empty instance. Use one of the alternative constructors
    to locate or create initial data:
    - Elliptica.free(path):        locates the initial data and per-resolution directories;
                                   all other functions can then be used freely.
    - Elliptica.analyze(path):     runs the full analysis pipeline on initial data
                                   created with Elliptica.create.
    - Elliptica.create(path, ...): creates an initial data run directory and parfile.
    - Elliptica.compare(path, ...): compare N initial data runs of the same system.

    Parameters:
    system       (str): the binary system (currently only 'BHNS').
    print_info  (bool): prints useful information about Elliptica.

    Returns:
    None.
    """
    if print_info: self.print_information()

    self.simname         = None
    self.system          = system
    self.user_params     = None
    self.path            = None
    self.parent_path     = None
    self.resolutions     = None
    self.parfile_path    = None
    self.par_dic         = None
    self.logs_path       = None
    self.properties_path = None
    self.prop_dic        = None
    self.diag_dict       = None
    self.bash_path       = None
    self.json_path       = None
    self.status          = None


  @classmethod
  def free(cls, path: str, system: str = None, print_info: bool = False) -> "Elliptica":
    """
    Locates the initial data and the per-resolution directories under the
    given path. The path can point to the directory holding the per-resolution
    directories or to its parent. All analysis functions can then be used freely.

    Parameters:
    path         (str): path to the initial data.
    system       (str): the binary system; inferred from the name if None.
    print_info  (bool): prints useful information about Elliptica.

    Returns:
    Elliptica instance.
    """
    obj = cls(system=system, print_info=print_info)
    obj.locate_initial_data(path)  # locate ID
    obj.resolution_dirs()          # find per-resolution directories
    return obj


  @classmethod
  def analyze(cls, path: str, logs_path: str = None, system: str = None,
              print_info: bool = False) -> "Elliptica":
    """
    Locates the initial data under the given path, i.e. searches for
    per-resolution directories in the specified directories, parses the
    resolutions, locates a parfile, and locates logfiles. If logs_path is
    None it will search for the log files inside of path, else it will
    take the specified path if the logs files are stored elsewhere.
    Furthermore, the status is checked, the ETA estimated and a convergence
    plot created.

    Parameters:
    path         (str): path to the initial data created with Elliptica.create.
    logs_path    (str): path to the logs file if they do not exist on the ID level.
    system       (str): the binary system; inferred from the run name or, failing
                        that, from the <system>_properties.txt file if None.
    print_info  (bool): prints useful information about Elliptica.

    Returns:
    Elliptica instance.
    """
    obj = cls(system=system, print_info=print_info)
    name  = Path(path).name
    width = len(name) + 4
    print(f'╭{"─" * width}╮')
    print(f'│  {_style(name, _BOLD)}{" " * (width - 2 - len(name))}│')
    print(f'╰{"─" * width}╯')
    print(f"Assuming initial data exists under {_style(path, _CYAN)}...")
    print(_style("Searching for output...", _BOLD))
    obj.locate_initial_data(path)  # locate ID
    obj.resolution_dirs()          # find per-resolution directories
    obj.locate_parfile()           # locate parfile
    obj.read_parfile()             # read the parfile
    obj.locate_logfiles(logs_path) # locate log-files
    obj.locate_properties()        # locate <system>_properties.txt
    obj.read_properties()          # read <system>_properties.txt
    obj.read_diagnostics()
    obj.check_status()             # check status of ID
    obj.check_accuracy()           # check accuracy
    print(f"  {_style("$", _DIM)} Remaining time (estimated): {_fmt(obj.initial_data_progress(print_timing=False))}")
    obj.convergence(plot=True, save=True) # plot convergence
    obj.plot_constraints_per_patch_group()
    obj.plot_target_system_convergence()
    obj.write_metadata()           # write metadata about the binary
    obj.move_data()                # move the data to the parent directory
    return obj


  @classmethod
  def create(cls, path: str, user_params: dict, system: str = 'BHNS',
             print_info: bool = False) -> "Elliptica":
    """
    Creates a initial data run directory and a parameter file based on
    some specified user parameters for the given binary. A Bash script
    can also be written and submitted.

    Parameters:
    path         (str): path where the initial data is created.
    user_params (dict): dictionary with the binary parameters.
    system       (str): for which system to create initial data (currently only 'BHNS').
    print_info  (bool): prints useful information about Elliptica.

    Returns:
    Elliptica instance.
    """
    if system != 'BHNS':
      raise SystemExit('System has to be "BHNS" (currently only supported with "create")!')

    obj = cls(system=system, print_info=print_info)
    obj.user_params = user_params

    # Make the parfile.
    print(_style("Creating initial data setup...", _BOLD))
    obj.make_parfile(path)
    return obj


  @classmethod
  def compare(cls, paths: list, labels: list = None, system: str = None) -> "EllipticaComparison":
    """
    Compare N runs of Elliptica initial data with each other. Here, 'compare'
    is a wrapper around EllipticaComparison which handles the plotting etc.

    Parameters:
    paths       (list): paths where the initial data is created.
    labels      (list): list of labels for the different runs.
    system       (str): for which system to create initial data (currently only 'BHNS').

    Returns:
    EllipticaComparison instance.
    """
    runs = [cls.free(p, system=system) for p in paths]
    for run in runs:
      run.read_diagnostics()

    return EllipticaComparison(runs, labels)


  # ==============================================================================
  # Mover.
  # ==============================================================================

  # Move the data to the parent folder, once done.
  def move_data(self) -> None:
    """
    Moves the data from the highest resolution directory to
    the parent directory.
    """
    if self.status == 'Done':
      keys    = list(self.resolutions.keys())
      hr_path = self.resolutions[keys[-1]]
      subprocess.run(['cp', hr_path / "checkpoint.dat", self.parent_path])
      subprocess.run(['cp', self.properties_path, self.parent_path])

      print(f"  {_style("$", _DIM)} Copied the highest-resolution results to the parent folder...")


  # ==============================================================================
  # Writer.
  # ==============================================================================

  # Create a parfile from the template.
  def make_parfile(self, path: str) -> None:
    """
    Create a initial data run directory and parameter file.

    Parameters:
    path (str): path to where the parameter file should be written.
    """
    # Create the directory.
    params = self.user_params
    BASE_DIR = Path(__file__).resolve().parent
    TEMPLATE_DIR = BASE_DIR / "templates" / "elliptica"
    obj1, obj2 = params["project"].split("_")[0], \
                 params["project"].split("_")[1]
    m1, m2     = 0.0, 0.0
    if self.system == 'BHNS':
      template_name = "elliptica_bhns.in.j2"
      if "->" in str(params['bh_irreducible_mass']): # iterated mass
        m1     = float(params['bh_irreducible_mass'].split("->")[-1])
      else:
        m1     = float(params['bh_irreducible_mass'])
      m2       = float(params['ns_baryonic_mass'])
      s1       = np.sqrt(float(params['bh_chix'])**2 + float(params['bh_chiy'])**2 \
                        + float(params['bh_chiz'])**2)
      s2       = np.sqrt(float(params['ns_omegax'])**2 + float(params['ns_omegay'])**2 \
                              + float(params['ns_omegaz'])**2)
      sep      = params['separation']
      self.simname = params['ns_eos_name'] + f"_M{obj1}-" + str(round(m1,1)) + f"_s{obj1}-" \
                    + str(round(s1,2)) + f"--M{obj2}-" + str(round(m2,2)) + f"_s{obj2}-" + str(round(s2,2)) \
                    + "--d" + str(sep)
      self.path    = Path(path) / self.simname
      try:
        self.path.mkdir()
        print(f"  {_style("$", _DIM)} Directory created under {_style(self.path, _CYAN)}")
      except FileExistsError:
        print(f"  {_style("$", _DIM)} Directory exists already under {_style(self.path, _CYAN)}")

    # Write the parfile.
    simpath = self.path / f"{self.simname}.par"
    env = Environment(loader=FileSystemLoader(TEMPLATE_DIR))
    template = env.get_template(template_name)
    rendered = template.render(**params)

    # Write the parameter file.
    with open(simpath, "w") as f:
      f.write(rendered)
    print(f"  {_style("$", _DIM)} Parfile written to {_style(simpath, _CYAN)}")


  # Write a slurm script.
  def write_bashfile(self, bashname: str, cluster: str, id_exe: str) -> None:
    """
    Writes a bash script for the specified cluster and under
    the specified name.

    Parameters:
    bashname (str): name of the bashfile.
    cluster  (str): name of the cluster to run on.
    id_exe   (str): path to the Elliptica executable.
    """
    cluster_list = ["ARA"]
    if cluster not in cluster_list:
      raise SystemExit(f"Currently supported clusters are: {cluster_list}")

    # Cluster specifics.
    if cluster == "ARA":
      cluster_setup = {
        "partition": "s_standard",
        "cpus"     : "36",
        "wallclock": "8-00:00:00",
        "memory"   : "180G",
        "modules"  : ["intel/oneapi/latest", "mpi/openmpi/5.0.2/gcc"]
      }

    bash_path = self.path / bashname
    bss = open(bash_path, 'a')
    bss.write('#!/bin/bash\n')
    bss.write(f'#SBATCH --job-name={self.simname}\n')
    bss.write('#SBATCH --output=slurm-%j.out\n')
    bss.write('#SBATCH --error=slurm-%j.err\n')
    bss.write(f'#SBATCH --partition={cluster_setup['partition']}\n')
    bss.write('#SBATCH --nodes=1\n')
    bss.write('#SBATCH --ntasks=1\n')
    bss.write(f'#SBATCH --cpus-per-task={cluster_setup['cpus']}\n')
    bss.write(f'#SBATCH --time={cluster_setup['wallclock']}\n')
    bss.write(f'#SBATCH --mem={cluster_setup['memory']}\n')
    bss.write('\n# ==============================\n')
    bss.write('# Modules\n')
    bss.write('# ==============================\n\n')
    bss.write('module purge\n')
    for mod in cluster_setup['modules']:
      bss.write(f'module load {mod}\n')
    bss.write('\n# ==============================\n')
    bss.write('# Job\n')
    bss.write('# ==============================\n\n')
    bss.write('export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK\n')
    bss.write(f'cd {self.path}\n\n')
    bss.write(f"{id_exe} -nt $SLURM_CPUS_PER_TASK {self.simname}.par")
    bss.close()

    print(f"  {_style("$", _DIM)} Wrote bash script to {_style(bash_path, _CYAN)}")
    self.bash_path = bash_path


  # Submit the job.
  def submit_job(self) -> None:
    """
    Submit the job using the written slurm file.
    """
    if not self.bash_path:
      raise SystemExit("Slurm file has to be written before submitting!")

    res = subprocess.run(['sbatch', self.bash_path], capture_output=True, text=True)
    print(f"  {_style("$", _DIM)} {res.stdout}")


  # ==============================================================================
  # Initial data.
  # ==============================================================================

  # Locate the initial data.
  def locate_initial_data(self, path: str) -> None:
    """
    Small wrapper to locate created or running initial data output.
    It needs to point to the directory holding the sub-directories
      with the respective resolutions.

    Parameters:
    path (str): Absolute path to the directory holding the resolution sub-directories.

    Returns:
    None; sets the 'path' class member
    """
    path = Path(path)
    if not path.exists():
      raise ValueError(f'The following path does not exist: {path}')
    elif not any(path.iterdir()):
      raise ValueError(f'Directory is empty!')

    if self.simname is not None and _has_resolution_dirs(path / f'{self.simname}_00'):
      self.path = path / f'{self.simname}_00'
    elif _has_resolution_dirs(path):
      self.path = path
    else:
      cands = [d for d in sorted(path.iterdir()) if _has_resolution_dirs(d)]
      if len(cands) != 1:
        raise SystemExit(f"Could not uniquely locate per-resolution directories under {path}!")
      self.path = cands[0]

    self.parent_path = self.path.parent
    if self.simname is None:
      self.simname = self.path.name.removesuffix('_00')
    if self.system is None:
      self.system = _infer_system(self.simname)

    print(f'  {_style("$", _DIM)} Set initial data output path to: {_style(self.path, _CYAN)}')


  # ==============================================================================
  # Properties.
  # ==============================================================================

  # Locate the <system>_properties.txt file.
  def locate_properties(self) -> None:
    """
    Reads an existing <system>_properties.txt file from the
    highest resolution directory existing.
    """
    if not self.resolutions:
      raise SystemExit("Locate per-resolution directories first with resolution_dirs()!")

    # Locate the properties file.
    keys = list(self.resolutions.keys())
    self.properties_path = None
    for file in self.resolutions[keys[-1]].glob("*_properties.txt"):
      self.properties_path = file
      print(f"  {_style("$", _DIM)} Found properties under: {_style(self.properties_path, _CYAN)}")

    if self.properties_path is None:
      raise SystemExit("Properties file not found in the highest-resolution folder." \
                       "Check the folder!")

    if self.system is None:
      self.system = self.properties_path.name.removesuffix("_properties.txt")


  # Read the metadata from properties.
  def read_properties(self) -> None:
    """
    Reads metadata into a dictionary from the (final) <system>_properties.txt.
    """
    if not self.properties_path:
      raise SystemExit("Locate the <system>_properties.txt first with locate_properties()!")

    dic = {}
    with open (self.properties_path) as file:
      dic['simname'] = self.simname
      for i, line in enumerate(file):
        if i < 3 or len(line.strip()) == 0:
          continue
        else:
          k, v = line.strip().split('=')
          dic[k.strip()] = v.strip()

    file.close()
    self.prop_dic = dic
    print(f"  {_style("$", _DIM)} Read properties and stored metadata...")


  # Check the accuracy from the diagnostics.
  def check_accuracy(self) -> None:
    """
    Compares every target quantity '<X>' with its current value '<X>_current'
    at the last iteration of the highest resolution in the diagnostics.
    """
    if self.diag_dict is None:
      self.read_diagnostics()

    props = list(self.diag_dict.values())[-1]["props"]
    last  = props[list(props)[-1]]
    lines = [f"{k} error: {abs(last[k + '_current'] - last[k]) * 100 / abs(last[k]):.2f}%"
             for k in sorted(last) if k + "_current" in last and last[k] != 0]
    if not lines:
      warnings.warn("No target quantities found in the diagnostics!")
      return

    width = max(len(line) for line in lines) + 4
    print(f'  {_style("$", _DIM)} Accuracy:')
    print(f'    ╭{"─" * width}╮')
    for line in lines:
      print(f'    │  {_style(line, _BOLD)}{" " * (width - 2 - len(line))}│')
    print(f'    ╰{"─" * width}╯')


  # Write metadata to file.
  def write_metadata(self) -> None:
    """
    Writes important metadata of this initial data to a
    file.
    """
    if not self.prop_dic:
      raise SystemExit("Writing metadata requires reading properties with read_properties()!")

    # Get the timestamp of the highest-resolution
    # checkpoint.
    keys    = list(self.resolutions.keys())
    check_path = self.resolutions[keys[-1]] / "checkpoint.dat"
    if not check_path.exists():
      raise SystemExit("Could not find the 'checkpoint.dat' file in " \
                       "the highest-resolution folder. Check!")
    created = datetime.fromtimestamp(check_path.stat().st_mtime).date().isoformat()

    # Get the component masses.
    if self.system == "BHNS":
      metadata = {
        "system" : self.system,
        "simname": self.simname,
        "created": created,
        "code"   : "Elliptica",
        "properties": {
          "separation": float(self.prop_dic[f"{self.system}_separation"]),
          "omega"     : float(self.prop_dic[f"{self.system}_angular_velocity"]),
          "mass ratio": float(self.prop_dic[f"{self.system}_mass_ratio"]),
          "total ADM mass": float(self.prop_dic[f"{self.system}_ADM_mass"]),
          "total ADM angular momentum": np.sqrt(float(self.prop_dic[f"{self.system}_Jx_ADM"])**2 \
                                              + float(self.prop_dic[f"{self.system}_Jy_ADM"])**2 \
                                              + float(self.prop_dic[f"{self.system}_Jz_ADM"])**2),
          "BH irreducible mass"  : float(self.prop_dic["BH_irreducible_mass_current"]),
          "BH christodoulou mass": float(self.prop_dic["BH_Christodoulou_mass_current"]),
          "BH chi-x"             : float(self.prop_dic["BH_chi_x_current"]),
          "BH chi-y"             : float(self.prop_dic["BH_chi_y_current"]),
          "BH chi-z"             : float(self.prop_dic["BH_chi_z_current"]),
          "NS baryonic mass"     : float(self.prop_dic["NS_baryonic_mass_current"]),
          "NS TOV ADM mass"      : float(self.prop_dic["NS_TOV_ADM_mass"]),
          "NS Omega-x"           : float(self.prop_dic["NS_Omega_x"]),
          "NS Omega-y"           : float(self.prop_dic["NS_Omega_y"]),
          "NS Omega-z"           : float(self.prop_dic["NS_Omega_z"])
        },
        "eos": {
          "type" : self.prop_dic["NS_EoS_type"],
          "name" : self.prop_dic["NS_EoS_description"],
          "units": self.prop_dic["NS_EoS_unit"]
        },
        "com": {
          "com-x": float(self.prop_dic[f"{self.system}_x_CM"]),
          "com-y": float(self.prop_dic[f"{self.system}_y_CM"]),
          "com-z": float(self.prop_dic[f"{self.system}_z_CM"])
        },
        "gw": {
          "initial frequency [Hz]": get_id_gw_frequency_Hz(float(self.prop_dic[f"{self.system}_angular_velocity"])),
          "initial frequency (22) [Hz]": get_id_gw_frequency_Hz_22(float(self.prop_dic[f"{self.system}_angular_velocity"]),
                                                                   float(self.prop_dic[f"{self.system}_ADM_mass"]))
        }
      }
    else:
      raise SystemExit("Set the 'system' (currently supported: 'BHNS')!")

    # Write the file.
    json_path = self.parent_path / "metadata.json"
    with open(json_path, "w") as f:
      json.dump(metadata, f, indent=4)

    if json_path.exists():
      self.json_path = json_path
      print(f"  {_style("$", _DIM)} Created a metadata file under: {_style(self.json_path, _CYAN)}")


  # ==============================================================================
  # Parfile.
  # ==============================================================================

  # Locate a parfile.
  def locate_parfile(self) -> None:
    """
    Locate a parfile and set the path to that parfile.

    Returns:
    None; sets the 'parfile_path' class member
    """
    res = self.resolutions
    if self.path is None or not res:
      raise SystemExit("First locate ID with locate_initial_data(path) "\
                       "and then locate the resolutions with resolution_dirs()!")

    for r in res.keys():
      for file in res[r].glob('*.par'):
        self.parfile_path = file
        break

      else: # only execute if per-file loop did not break
        continue
      break # only executed if per-file loop did break

    if self.parfile_path is None:
      self.parfile_path = next(self.parent_path.glob('*.par'), None)

    if self.parfile_path is None:
      print('Could not locate a parfile with signature .par.' \
            'Check the per-resolution folders!')
    else:
      print(f'  {_style("$", _DIM)} Found parfile under: {_style(self.parfile_path, _CYAN)}')


  # Read the parfile.
  def read_parfile(self) -> None:
    """
    Read the parfile of the specified path and then
    returns a dictionary.
    """
    if not self.parfile_path:
      raise SystemExit("Locate parfile first with locate_parfile()!")

    # Read the parfile first.
    par_dic = {}
    with open (self.parfile_path) as f:
      for line in f:
        if line.startswith("#") or len(line) == 0:
          continue
        else:
          try:
            key, value = line.strip().split("=")
            if "#" in value:
              value = value.split("#")[0].strip()
            try:
              float(value)
            except ValueError:
              value = value.split("->")[-1]
            key = key.strip()
            par_dic[key] = value
          except:
            pass
    self.par_dic = par_dic

    if par_dic:
      print(f"  {_style("$", _DIM)} Read parfile and stored the data...")


  # ==============================================================================
  # Locate log files.
  # ==============================================================================

  # Locate slurm logfiles.
  def locate_logfiles(self, path: str = None) -> None:
    """
    Locate logfiles under the specified path.

    Parameters:
    path (str): path to the slurm log files.

    Returns:
    None; sets the 'parfile_path' class member
    """
    if path is None:
      path = self.path.parent
    path = Path(path)

    mtime = lambda p: p.stat().st_mtime
    logs = sorted(path.glob('slurm-*.out'), key=mtime) or sorted(path.glob('*.out'), key=mtime)
    if not logs:
      raise SystemExit('Could not locate log-files with signature '\
                       'slurm-*.out or *.out. Check the path!')
    self.logs_path = logs
    print(f'  {_style("$", _DIM)} Found logs under: {_style(self.logs_path, _CYAN)}')


  # ==============================================================================
  # Resolution directories.
  # ==============================================================================

  # Locate the initial data.
  def resolution_dirs(self) -> None:
    """
    Locate the per-resolution directories in the root path.

    Returns:
    None; sets the 'resolution' dict class member
    """
    root = self.path
    if root is None:
      raise ValueError("Root path is not set. Run locate_initial_data(path) first!")

    # Collect the per-resolution arrays.
    out = {}
    for full_path in root.iterdir():
      if not full_path.is_dir():
        continue

      m = _RES_DIR.search(full_path.name)
      if not m:
        continue

      out[tuple(map(int, m.groups()))] = full_path

    if not out:
      raise SystemExit("Located per-resolution directories earlier," \
                       "but they don't match the format. Check!")
    else:
      self.resolutions = {"x".join(map(str, k)): out[k] for k in sorted(out)}
      print(f"  {_style("$", _DIM)} Found resolutions: {list(self.resolutions.keys())}")


  # Locate the initial data.
  def convergence(self, plot: bool = False, save: bool = False) -> None:
    """
    Analyze the convergence per resolution, i.e. the worst constraint
    violation over all patches at the last iteration of each resolution.
    Results can be plotted.

    Parameters:
    plot (bool): Whether to plot the results.
    save (bool): Whether to store the plot.

    Returns:
    None; stores a dict in the 'conv_max' class member which holds the
    worst-over-patches constraint violation at the last iteration per resolution.
    """
    if self.diag_dict is None:
      self.read_diagnostics()

    self.conv_max = {
      res: {col: max(p[col][-1] for p in data["patches"].values()) for col in _CONSTRAINTS}
      for res, data in self.diag_dict.items() if data["patches"]
    }
    if not self.conv_max:
      raise SystemExit("Could not find any 0D diagnostics files with *_0d.txt signature!")

    if not plot:
      return

    rc = {
      "xtick.labelsize": 16,
      "ytick.labelsize": 16,
      "axes.labelsize":  16,
      "text.usetex":     True,
      "font.family":     "Computer Modern Serif",
      "lines.linewidth": 1.5,
      "legend.fontsize": 10.5,
    }
    with mpl.rc_context(rc):
      res = [_resolution_n(r) for r in self.conv_max]
      fig, ax = plt.subplots(1, 1, figsize=(6, 4))
      for (col, label), color in zip(_CONSTRAINTS.items(), _PALETTE):
        ax.plot(res, [v[col] for v in self.conv_max.values()], color=color, marker="^", label=label)
      ax.set_xlabel("Resolution N")
      ax.set_ylabel(r"$||L||_2$")
      ax.set_yscale("log")
      ax.legend(frameon=False)
      ax.grid()
      ax.set_title(self.simname)
      fig.tight_layout()
      if not save:
        plt.show()
      else:
        out_path = self.parent_path / "convergence.png"
        plt.savefig(out_path, dpi=150)
        print(f"  {_style("$", _DIM)} Saved convergence plot to: {_style(out_path, _CYAN)}")
      plt.close(fig)


  # ==============================================================================
  # Information.
  # ==============================================================================

  # Print useful information.
  def print_information(self) -> None:
    """Small function printing useful information about the Elliptica solver."""
    # Available solvers.
    solvers = (
      ('Black hole-neutron star', 'https://github.com/rashti-alireza/BH_NS_Binary_Initial_Data'),
      ('Binary neutron star', 'https://github.com/rashti-alireza/NS_NS_Binary_Initial_Data'),
      ('Binary black hole', 'https://github.com/rashti-alireza/BH_BH_Binary_Initial_Data'),
      ('Isolated neutron star', 'https://github.com/rashti-alireza/Single_NS_Initial_Data'),
      ('Isolated black hole', 'https://github.com/rashti-alireza/Single_BH_Initial_Data'),
    )
    name_width = max(len(name) for name, _ in solvers)

    # Compilation commands.
    commands = (
      'git clone --depth=1 git@github.com:rashti-alireza/Elliptica.git',
      'cd Elliptica',
      'make MyConfig',
      'make git_clone',
      'make -j4',
    )

    # Section and URL beautifier.
    body = []

    def section(title: str, entries: list) -> None:
      if body:
        body.append(('', ''))
      body.append((f'  {title}', f'  {_style(title, _BOLD)}'))
      for plain, decorated in entries:
        body.append((f'    {plain}', f'    {decorated}'))

    def url(link: str) -> tuple:
      return (link, _style(link, _CYAN))

    section('REPOSITORY', [url('https://github.com/rashti-alireza/Elliptica')])
    section('REQUIREMENTS',
            [('GNU make · gcc compiler · OpenMP · UMFPACK (suitesparse library)',) * 2])
    section('COMPILATION',
            [(f'$ {cmd}', f'{_style("$", _DIM)} {cmd}') for cmd in commands])
    section('INITIAL DATA SOLVERS (available)',
            [(f'{name.ljust(name_width)}  {link}',
              f'{name.ljust(name_width)}  {_style(link, _CYAN)}')
             for name, link in solvers])
    section('INITIAL DATA READER (generic)', [url('https://github.com/rashti-alireza/Elliptica_ID_Reader')])
    section('CITATION', [('{Rashti:2021ihv,Rashti:2024drr}\n',) * 2])

    # Title box.
    title = 'ELLIPTICA · initial data solver'
    width = max([len(plain) for plain, _ in body] + [len(title) + 2]) + 2
    print(f'╭{"─" * width}╮')
    print(f'│  {_style(title, _BOLD)}{" " * (width - 2 - len(title))}│')
    print(f'╰{"─" * width}╯')
    for _, decorated in body:
      print(decorated)


  # ==============================================================================
  # Schedule.
  # ==============================================================================

  # Get the scheduled iterations.
  def parse_iteration_schedule(self) -> tuple:
    """
    Parse the iteration schedule from the parfile.

    Returns:
    Returns tuple holding (it-start, it-end, resolution) and the total
    iterations of the initial data run.
    """
    parfile = self.parfile_path
    if parfile is None:
      raise SystemExit('Locate parfile first with locate_parfile()!')

    line = None
    with open(parfile) as fh:
      for row in fh:
        s = row.split("#", 1)[0].strip()
        if s.replace(" ", "").startswith("n_a="):
          line = s.split("=", 1)[1].strip()
          break
    if not line:
      sys.exit("Could not find an 'n_a = ...' line in %s" % parfile)
    boundaries, cur, total = [], 0, 0
    for chunk in line.split("->"):
      m = re.match(r"([0-9]+)\s*(?:\(x([0-9]+)\))?", chunk.strip())
      if not m:
        continue
      N = int(m.group(1))
      cnt = int(m.group(2)) if m.group(2) else 1
      boundaries.append((cur, cur + cnt, N))
      cur += cnt
      total += cnt

    return boundaries, total


  # ==============================================================================
  # Logs.
  # ==============================================================================
  # Parse the logs for the iteration speed and resolution.
  def parse_logs(self) -> list:
    """
    Parse a log file for the current resolution of the initial
    data iteration, the cumulative seconds this process took, and the
    segment ID. Combines the records from multiple log-files if the initial
    data creation was restarted from a checkpoint.

    Returns:
    List with tuples (resolution, time, log segment).
    """
    logfiles = self.logs_path
    if logfiles is None:
      raise SystemExit('Locate the logfiles first with locate_logfiles()!')

    STAMP = re.compile(r"\[(\d+)d:(\d+)h:(\d+)m:(\d+)s\]")
    DONE  = re.compile(r"\}\s*Outermost iteration (\d+) ==> Done")
    NA    = re.compile(r"N_a\s*=\s*([0-9]+)")

    recs = []
    seg = -1
    for lf in logfiles:
      seg += 1
      last_t, prev_iter, cur_res = None, None, None
      with open(lf, errors="ignore") as fh:
        for row in fh:
          m = STAMP.search(row)
          if m:
            d, h, mi, s = map(int, m.groups())
            t = ((d * 24 + h) * 60 + mi) * 60 + s
            if last_t is not None and t < last_t:
              seg += 1
            last_t = t
          mr = NA.search(row)
          if mr:
            cur_res = int(mr.group(1))
          md = DONE.search(row)
          if md and last_t is not None:
            it = int(md.group(1))
            if prev_iter is not None and it < prev_iter:
              seg += 1
            prev_iter = it
            recs.append((cur_res, last_t, seg))

    return recs


  # Compute per iteration time deltas from logs.
  def logs_per_iter_deltas(self, recs: list) -> list:
    """
    Computes the time deltas within one continuous segment.

    Returns:
    List in the format (resolution, dt_seconds).
    """
    if not recs:
      raise SystemExit('First parse the logs with parse_logs()!')

    out = [] # (resolution, dt_seconds)
    for (r0, t0, s0), (r1, t1, s1) in zip(recs, recs[1:]):
      if s0 == s1 and t1 >= t0 and r1 is not None:
        out.append((r1, t1 - t0))

    return out


  # Determine the iterations completed per resolution from 0d diagnostics.
  def counts_from_diagnostics(self) -> dict:
    """
    Determines how many iterations per resolution are already completed
    based on the 0d diagnostics output.

    Returns:
    Dict with the number of iterations already completed per resolution.
    """
    if self.diag_dict is None:
      self.read_diagnostics()

    return {r: len(data["props"]) for r, data in self.diag_dict.items()}


  # Print per-resolution timing and progress.
  def initial_data_progress(self, print_timing: bool = False,
                            outlier_factor: float = 4.0) -> float | None:
    """
    Print per-resolution timing and progress plus the estimated
    time for the initial data to finish.

    Parameters:
    outlier_factor (float): how many outliers from the median.

    Returns:
    ETA time.
    """
    # Parse the logs and get the deltas.
    boundaries, total = self.parse_iteration_schedule()
    recs   = self.parse_logs()
    deltas = self.logs_per_iter_deltas(recs)

    # Compute the metrics.
    res_seen = sorted({r for r, _, _ in recs if r is not None})
    by_res = {}
    for r, dt in deltas:
      by_res.setdefault(r, []).append(dt)

    # per-resolution timing.
    med = {}
    for N in sorted(by_res):
      v = by_res[N]; mv = statistics.median(v)
      good = [x for x in v if x <= outlier_factor * mv]
      med[N] = statistics.median(good) if good else mv

    # Progress + ETA.
    done = {_resolution_n(r): n for r, n in self.counts_from_diagnostics().items()}
    hi = max(med)
    remaining = 0.0
    for s, e, N in boundaries:
      sched = e - s
      d = done.get(N, 0 if N not in med else sched)
      rem = max(0, sched - d)
      if rem == 0:
        continue
      if N in med:
        t, src = med[N], "measured"
      else:
        t, src = med[hi] * (2.0 ** ((N - hi) / 2.0)), f"extrap from N={hi}"
      remaining += rem * t

    # General information.
    if print_timing:
      print("\nLOGS INFORMATION")
      print("=" * 72)
      print("Logs:   ", ", ".join(map(str, self.logs_path)))
      print("Schedule:", " -> ".join(f"N={N}(x{e-s})" for s, e, N in boundaries),
            f"| total = {total}")
      print(f"Resolutions present in log(s): {res_seen}   (restart-aware)")
      print("=" * 72)

      # per-resolution timing
      print("\nPER-RESOLUTION TIMING")
      print("=" * 72)
      print(f"{'N':>4} {'samples':>8} {'median/it':>11} {'min':>9} {'max':>10} {'stalls':>7}")

      for N in sorted(by_res):
        v = by_res[N]; mv = statistics.median(v)
        good = [x for x in v if x <= outlier_factor * mv]
        print(f"{N:>4} {len(v):>8} {_fmt(med[N]):>11} {_fmt(min(v)):>9} {_fmt(max(v)):>10} "
              f"{len(v)-len(good):>7}")

      print("  (median/min/max over clean iterations; 'stalls' = excluded outliers)")
      print("=" * 72)

      # progress + ETA
      print("\nPROGESS + ETA")
      print("=" * 72)
      print(f"Completed iterations per resolution (from diagnostics"
            + (f" in {self.path.name}" if self.path else ", not found") + "):")
      for s, e, N in boundaries:
        print(f"   N={N}: {done.get(N,'?')} / {e-s} scheduled")

      if not med:
        print("\nNo clean per-iteration samples; cannot estimate ETA.")
        return

      print("\nEstimated remaining wall time (measured medians; ~2x/level extrapolation if unmeasured):")
      remaining = 0.0
      for s, e, N in boundaries:
        sched = e - s
        d = done.get(N, 0 if N not in med else sched)
        rem = max(0, sched - d)
        if rem == 0:
          continue
        if N in med:
          t, src = med[N], "measured"
        else:
          t, src = med[hi] * (2.0 ** ((N - hi) / 2.0)), f"extrap from N={hi}"
        print(f"   N={N}: {rem:>4} iters x {_fmt(t)}/it = {_fmt(rem*t):>10}  ({src})")

      print(f"\t{_fmt(remaining)}  ({remaining/3600:.1f} h)")
      print("=" * 72)
      return
    else:
      return remaining


  # Get the status of the current initial data.
  def check_status(self) -> None:
    """
    Checks the status of the simulation based on the logfile.
    """
    if self.logs_path is None:
      raise SystemExit('Locate the logfiles first with locate_logfiles()!')

    self.status = 'Ongoing' if self.path.is_dir() else 'Not started'

    for log_file in self.logs_path:
      with open(log_file, 'r') as hrf:
        cont = hrf.read()
        done = "} construct_initial_data :))"
        if done in cont:
          self.status = 'Done'
          break

    print(f"  {_style("$", _DIM)} Status: {self.status}")


  # ==============================================================================
  # Free analysis.
  # ==============================================================================
  # Read 'Diagnostics_00' data.
  def read_diagnostics(self) -> None:
    """
    Read data from 'Diagnostics_00', i.e. the _properties.txt
    timeline, and if available 0D output.

    Returns:
    Dictionary with the merged timelines for _properties.txt
    and the 0D output.
    """
    diag_dict = defaultdict(lambda: defaultdict(dict))
    if not self.resolutions:
      raise SystemExit("Locate per-resolution directories first with resolution_dirs()!")

    for nabc, res in self.resolutions.items():
      diag_path = res / "Diagnostics_00"
      if not diag_path.is_dir():
        warnings.warn(f"'Diagnostics_00' does not exist in {res}, skipping!")
        continue

      for file in sorted(diag_path.iterdir()):
        if file.name.endswith("_properties.txt"):
          prop_dict = self.read_properties_timeline(file)
          diag_dict[nabc]["props"] = prop_dict

        if file.name.endswith("_0d.txt"):
          patch = file.name.removesuffix("_0d.txt")
          patch_dict = self.read_patches_timeline(file)

          diag_dict[nabc]["patches"][patch] = patch_dict

      if "props" not in diag_dict[nabc]:
        warnings.warn(f"_properties.txt does not exist in {diag_path}, skipping!")
        diag_dict.pop(nabc)
        continue

      if "patches" not in diag_dict[nabc]:
        warnings.warn(f"0D per-patch output does not exist in {diag_path}!")

    if not diag_dict:
      raise SystemExit("Could not find any diagnostics in the per-resolution directories!")

    self.diag_dict = diag_dict


  # Read per-patch 0D output if existing.
  @staticmethod
  def read_patches_timeline(file: Path) -> defaultdict:
    """
    Load the 0D output data from patches.

    Parameters:
    file (str): Path to the file _0D.txt file.

    Returns:
    Dict holding the column data for this patch.
    """
    patches = defaultdict(lambda: defaultdict(dict))

    # Read the header.
    f = open(file)
    header = f.readline()
    cols = header.split()[1:]
    f.close()

    # Load the data.
    dat = np.loadtxt(file, ndmin=2)

    for i in range(len(cols)):
      patches[cols[i]] = dat[:,i]

    return patches


  # Read _properties.txt timeline.
  @staticmethod
  def read_properties_timeline(file: Path) -> defaultdict:
    """Read _properties.txt from diagnostics."""
    props = defaultdict(lambda: defaultdict(dict))
    block = None
    with open(file, "r") as f:
      for line in f:
        # Each "# iteration = N" header precedes the block belonging to N.
        if line.startswith("#"):
          block = {}
          props[int(line.split("=")[-1])] = block
          continue

        if block is None or line == "\n" or line.startswith("-"):
          continue
        if line.startswith("NS_EoS"):
          continue
        name, value = line.split("=", 1)
        block[name.strip()] = float(value.strip())

    return props


  # Plot the constraints per patch-group.
  def plot_constraints_per_patch_group(self, output_dir: str = None,
                                       groups: dict = None) -> None:
    """
    Read the _0D.txt constraint data for the patches and group them, then
    make a plot showing the worst contraint in the group and a min-max band
    for the respective group. Four panels for Hamiltonian, and the Momentum constraints.

    Groups are derived from the patch names (e.g. 'left_NS_around_up_X0Y0Z0'
    -> 'left_NS_around'), so any binary type (BHNS, BNS, BBH, ...) works.

    Parameters:
    output_dir (str): Directory where the constraint plot should be stored. If not
                      given store in the specified data directory.
    groups (dict):    Optional mapping group name -> (color, label) to override the
                      automatic styling. Groups not listed get a color from a default
                      palette and a label derived from their name.

    Returns:
    None.
    """
    if self.diag_dict is None:
      raise SystemExit("Load diagnostics with 'read_diagnostics' first!")

    quantities = _CONSTRAINTS

    def patch_group(name: str) -> str:
      """Map e.g. 'left_NS_around_up_X0Y0Z0' -> 'left_NS_around'."""
      name = re.sub(r"_X0Y0Z0?$", "", name)
      return name.rsplit("_", 1)[0]

    # Collect the groups over all resolutions so colors stay consistent.
    found = sorted({patch_group(name)
                    for res in self.diag_dict.values()
                    for name in res["patches"]})
    if not found:
      warnings.warn("No 0D per-patch output found, skipping constraint plot.")
      return

    palette = list(_PALETTE)
    if len(found) > len(palette):
      palette += [mcolors.to_hex(c) for c in plt.get_cmap("tab20").colors]
    styles = {g: (palette[i % len(palette)], g.replace("_", " "))
              for i, g in enumerate(found)}
    styles.update(groups or {})

    fig, axs = plt.subplots(2, 2, figsize=(14, 8), sharex=True)
    last_iter = 0
    for res in self.diag_dict.keys():
      patches = self.diag_dict[res]["patches"]
      members = defaultdict(list)
      for name in patches:
        members[patch_group(name)].append(name)

      for ax, col in zip(axs.flat, quantities):
        for group, names in members.items():
          color, _ = styles[group]
          it  = patches[names[0]]["iteration"] + last_iter
          dat = np.array([patches[p][col] for p in names])
          ax.fill_between(it, dat.min(axis=0), dat.max(axis=0), color=color, alpha=0.25, lw=0)
          ax.plot(it, dat.max(axis=0), color=color, lw=1.2)

        ax.axvline(last_iter, color="black", ls="dashed", alpha=0.3)
        if last_iter != 0:
          trans = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
          ax.text(last_iter, 1.01, res, color="black", alpha=0.5,
                  ha="right", va="bottom", clip_on=False, transform=trans, fontsize=8)
      last_iter += list(self.diag_dict[res]["props"])[-1]

    for ax, lab in zip(axs.flat, quantities.values()):
      ax.set_yscale("log")
      ax.set_xlim(0, last_iter)
      ax.set_ylabel(lab, fontsize=16)
    for ax in axs[1]:
      ax.set_xlabel("Iterations", fontsize=16)

    handles = [Line2D([], [], color=styles[g][0], lw=4, alpha=0.6, label=styles[g][1])
               for g in found]
    fig.legend(handles=handles, loc="upper center", ncol=min(len(handles), 6),
               frameon=False, fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out_path = Path(output_dir) if output_dir is not None and \
               Path(output_dir).is_dir() else self.parent_path
    plt.savefig(out_path / "constraints_patches_per_group.png", dpi=150)
    print(f"  {_style("$", _DIM)} Saved constraint plot to: {_style(out_path / "constraints_patches_per_group.png", _CYAN)}")
    plt.close(fig)


  # Plotting function for the evolution of the system targets.
  def _target_quantities(self, quantities: list = None, targets: dict = None) -> list:
    """
    Select the target quantities to plot for the system target convergence. Take
    the keys of 'targets' if given, else 'quantities' if given, else every '<X>' with
    '_current' signature and non-zero target.

    Parameters:
    quantities (list): list with the target Elliptica keys, such as 'BHNS_irreducible_mass'.
    targets    (dict): dict with Elliptica keys and manual target values as items.

    Returns:
    List with target quantities to plot.
    """
    if targets:
      return list(targets)
    if quantities is not None:
      return list(quantities)
    first_props = next(iter(self.diag_dict.values()))["props"]
    first       = first_props[next(iter(first_props))]
    return sorted(k for k in first if k + "_current" in first and first[k] != 0)


  def _draw_target_convergence(self, axs: np.ndarray, quantities: list, targets: dict = None,
                               color: str = None, markers: bool = True) -> int:
    """
    Draw the target quantities of this run onto given (2,n) axes. If
    'color' is None, each quantity gets its own color and target is black.
    Quantities missing in this run are skipped.

    Parameters:
    axs: Which access to draw on.
    quantities (list): list with the target Elliptica keys, such as 'BHNS_irreducible_mass'.
    targets    (dict): dict with Elliptica keys and manual target values as items.
    color       (str): color to plot the targets of one run in (only for comparison).
    markers    (bool): whether to plot resolution markers.

    Returns:
    The last (cumulative) iteration.
    """
    targets   = targets or {}
    last_iter = 0
    for res, data in self.diag_dict.items():
      props = data["props"]
      first = props[next(iter(props))]
      it    = np.array(list(props)) + last_iter

      for i, key in enumerate(quantities):
        if key in targets:
          if key not in first:
            continue
          current = np.array([props[j][key] for j in props])
          target  = np.full_like(current, targets[key])
        else:
          if key + "_current" not in first:
            continue
          current = np.array([props[j][key + "_current"] for j in props])
          target  = np.array([props[j][key] for j in props])

        line_color = color or _PALETTE[i % len(_PALETTE)]
        axs[0,i].plot(it, current, color=line_color, lw=1.2)
        axs[0,i].plot(it, target, color=color or "black", ls="dotted", lw=1)
        axs[1,i].semilogy(it, np.abs(current - target) / np.abs(target), color=line_color, lw=1.2)

        if markers:
          for ax in axs[:,i]:
            ax.axvline(last_iter, color="black", ls="dashed", alpha=0.3)
          if last_iter != 0:
            trans = mtransforms.blended_transform_factory(axs[0,i].transData, axs[0,i].transAxes)
            axs[0,i].text(last_iter, 1.01, res, color="black", alpha=0.5,
                          ha="right", va="bottom", clip_on=False, transform=trans, fontsize=8)

      last_iter += list(props)[-1]

    return last_iter


  def plot_target_system_convergence(self, output_dir: str = None,
                                     quantities: list = None, targets: dict = None,
                                     output_file: str = None) -> None:
    """
    Plot the evolution of the target quantities (e.g. masses, spins) towards
    their targets. Every '<X>' in _properties.txt with a matching '<X>_current'
    is a target quantity.

    Parameters:
    output_dir (str):  Directory where the plot should be stored. If not
                       given store in the specified data directory.
    quantities (list): Target keys to plot, e.g. ['BH_irreducible_mass', 'NS_baryonic_mass'].
                       If not given plot all target quantities with a non-zero target.
    targets (dict):    Keys without a '_current' counterpart and their target value
                       supplied by hand, e.g. {'BHNS_ADM_mass': 8.6}. If given, only
                       these are plotted and 'quantities' is ignored.
    output_file (str): Name of the output file. If 'None', then 'system_target_convergence.png'.

    Returns:
    None.
    """
    if self.diag_dict is None:
      raise SystemExit("Load diagnostics with 'read_diagnostics' first!")

    quantities = self._target_quantities(quantities, targets)
    n = len(quantities)
    fig, axs = plt.subplots(2, n, figsize=(6*n, 6), sharex="col", squeeze=False,
                            gridspec_kw={'height_ratios': [2, 1]})
    last_iter = self._draw_target_convergence(axs, quantities, targets)
    _format_target_axes(axs, quantities, last_iter)

    handles = [Line2D([], [], color="black", ls="dotted", lw=1, label="target")]
    axs[0,0].legend(handles=handles, frameon=False, fontsize=14)

    fig.tight_layout()
    fig.subplots_adjust(hspace=0.05)
    out_dir = Path(output_dir) if output_dir is not None and \
              Path(output_dir).is_dir() else self.parent_path
    _save_figure(fig, out_dir, output_file or "system_target_convergence.png", "target_convergence")


# ==============================================================================
# EllipticaComparison class.
# ==============================================================================

# Define EllipticaComparison class.
class EllipticaComparison:
  """
  Small wrapper class dealing with comparing N Elliptica initial data
  runs. Uses the 'Elliptica' functionality.
  """
  def __init__(self, runs: list, labels: list = None) -> None:
    systems = {run.system for run in runs}
    if len(systems) != 1:
      raise SystemExit("Can only compare runs of the same system, found: "
                      f"{sorted(map(str, systems))}!")
    if labels is not None and len(labels) != len(runs):
      raise SystemExit(f"Got {len(labels)} labels for {len(runs)} runs!")

    self.runs   = runs
    self.system = systems.pop()
    self.labels = labels or [run.simname for run in runs]


  def __iter__(self) -> Iterator[Elliptica]:
    return iter(self.runs)


  def __getitem__(self, i: int) -> Elliptica:
    return self.runs[i]


  def plot_target_system_convergence(self, output_dir: str = None,
                                     quantities: list = None, targets: dict | list = None,
                                     output_file: str = "system_target_convergence_comparison.png") -> None:
    """
    Compare the evolution of the target quantities (e.g. masses, spins) towards
    their targets for all runs. Each run has its own color; its target is dotted.

    Parameters:
    output_dir (str):     Directory where the plot should be stored. If not given
                          store in the current working directory.
    quantities (list):    Target keys to plot. If not given plot the target quantities
                          that every run has.
    targets (dict, list): Keys without a '_current' counterpart and their target value
                          supplied by hand, either one dict for all runs or a list with
                          one dict per run. If given, only these are plotted.
    output_file (str):    Name of the output file.

    Returns:
    None.
    """
    per_run = targets if isinstance(targets, list) else [targets] * len(self.runs)
    if len(per_run) != len(self.runs):
      raise SystemExit(f"Got {len(per_run)} target dicts for {len(self.runs)} runs!")

    if per_run[0]:
      quantities = list(per_run[0])
    elif quantities is None:
      selected   = [run._target_quantities() for run in self.runs]
      quantities = [k for k in selected[0] if all(k in s for s in selected[1:])]
      dropped    = sorted(set().union(*selected) - set(quantities))
      if dropped:
        warnings.warn(f"Not every run has these target quantities, skipping: {dropped}")
    if not quantities:
      raise SystemExit("No target quantities to compare!")

    counts  = [run.counts_from_diagnostics() for run in self.runs]
    markers = all(c == counts[0] for c in counts)

    n = len(quantities)
    fig, axs = plt.subplots(2, n, figsize=(6*n, 6), sharex="col", squeeze=False,
                            gridspec_kw={'height_ratios': [2, 1]})
    last_iter = 0
    for i, (run, run_targets) in enumerate(zip(self.runs, per_run)):
      color = _PALETTE[i % len(_PALETTE)]
      last_iter = max(last_iter, run._draw_target_convergence(axs, quantities, run_targets,
                                                              color=color, markers=markers and i == 0))

    _format_target_axes(axs, quantities, last_iter)

    handles  = [Line2D([], [], color=_PALETTE[i % len(_PALETTE)], lw=1.2, label=label)
                for i, label in enumerate(self.labels)]
    handles += [Line2D([], [], color="gray", ls="dotted", lw=1, label="target")]
    fig.legend(handles=handles, loc="upper center", ncol=min(len(handles), 4),
               frameon=False, fontsize=14)

    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.subplots_adjust(hspace=0.05)
    out_dir = Path(output_dir) if output_dir is not None else Path.cwd()
    if not out_dir.is_dir():
      raise SystemExit(f"Output directory {out_dir} does not exist!")
    _save_figure(fig, out_dir, output_file, "target convergence comparison")
