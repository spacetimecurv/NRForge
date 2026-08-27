####################################################
#          Class for Elliptica ID solver           #
####################################################

# Built-in libraries.
import os
import sys
import glob
import re
from collections import defaultdict
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

# NRForge libraries.
from nrforge.utils.variables import get_id_gw_frequency_Hz, get_id_gw_frequency_Hz_22
from nrforge.utils.style import _style, _fmt, _BOLD, _DIM, _CYAN

# Get a iterated BH mass.
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

#--------------- ELLIPTICA ID SOLVER ----------------
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

  def __init__(self, path, logs_path=None, mode=None, user_params=None,
               system=None, print_info=False):
    """
    There are two modes:
    - 'analyze': locates the initial data under the given path, i.e. searches for
                 per-resolution directories in the specified directories, parses the
                 resolutions, locates a parfile, and locates logfiles. If logs_path is
                 None it will search for the log files inside of path, else it will
                 take the specified path if the logs files are stored elsewhere.
                 Furthermore, the status is checked, the ETA estimated and a convergence
                 plot created.
    - 'create': creates a initial data run directory and a parameter file based on
                some specified user parameters for the given binary. A Bash script
                can also be written and submitted.

    Parameters:
    path         (str): path where the initial data is created or existing.
    logs_path    (str): path to the logs file if they do not exist on the ID level.
    mode         (str): either 'analyze' or 'create'.
    user_params (dict): dictionary with the binary parameters (only in 'create' mode).
    system       (str): for which system to create initial data (currently only 'BHNS').
    print_info  (bool): prints useful information about Elliptica.
    """
    if print_info: self.print_information()

    # Select the mode.
    self.mode = mode
    if self.mode != 'create' and self.mode != 'analyze':
      raise SystemExit('Mode has to be either "create" or "analyze"!')

    # If in analyze mode, locate initial data,
    # per-resolution directories, parfile, and
    # log-files. Check the status, estimate the
    # ETA and plot the convergence.
    if self.mode == 'analyze':
      self.simname = path.split('/')[-1]
      if "BH" and "NS" in self.simname:
        self.system = "BHNS"
      else:
        raise SystemExit("Could not find object initials in simulation name, i.e. " \
                         "could not determine the system!")
      width = max([len(self.simname) + 2]) + 2
      print(f'╭{"─" * width}╮')
      print(f'│  {_style(self.simname, _BOLD)}{" " * (width - 2 - len(self.simname))}│')
      print(f'╰{"─" * width}╯')
      print(f"Assuming initial data exists under {_style(path, _CYAN)}...")
      print(_style("Searching for output...", _BOLD))
      self.locate_initial_data(path)  # locate ID
      self.resolution_dirs()          # find per-resolution directories
      self.locate_parfile()           # locate parfile
      self.read_parfile()             # read the parfile
      self.locate_logfiles(logs_path) # locate log-files
      self.locate_properties()        # locate <system>_properties.txt
      self.read_properties()          # read <system>_properties.txt
      self.check_status()             # check status of ID
      self.check_accuracy()           # check accuracy
      print(f"  {_style("$", _DIM)} Remaining time (estimated): {_fmt(self.initial_data_progress(print_timing=False))}")
      self.convergence(plot=True, save=True) # plot convergence
      self.write_metadata()           # write metadata about the binary
      self.move_data()                # move the data to the parent directory

    # If in create mode, create a parfile under
    # the specified path passing user parameters with the
    # binary parameters. Depending on the machine, a
    # slurm script will also be created.
    if self.mode == 'create':
      # Check if user_params are specified.
      if user_params is None:
        raise SystemExit('User parameters for the binary have to be specified in "create" mode!')
      else:
        self.user_params = user_params

      # Check if the specifid system is supported.
      if system != 'BHNS':
        raise SystemExit('System has to be "BHNS" (currently only supported with "create")!')
      else:
        self.system = system

      # Make the parfile.
      print(_style("Creating initial data setup...", _BOLD))
      self.make_parfile(path)

  # ------------ MOVER --------------
  # Move the data to the parent folder, once done.
  def move_data(self):
    """
    Moves the data from the highest resolution directory to
    the parent directory.
    """
    if self.status == 'Done':
      keys    = list(self.resolutions.keys())
      hr_path = self.resolutions[keys[-1]]
      subprocess.run(['cp', os.path.join(hr_path, "checkpoint.dat"), self.parent_path])
      subprocess.run(['cp', os.path.join(hr_path, f"{self.system}_properties.txt"), self.parent_path])

      print(f"  {_style("$", _DIM)} Copied the highest-resolution results to the parent folder...")

  # ------------ WRITER -------------
  # Create a parfile from the template.
  def make_parfile(self, path):
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
      self.path    = os.path.join(path, self.simname)
      try:
        os.mkdir(self.path)
        print(f"  {_style("$", _DIM)} Directory created under {_style(self.path, _CYAN)}")
      except FileExistsError:
        print(f"  {_style("$", _DIM)} Directory exists already under {_style(self.path, _CYAN)}")

    # Write the parfile.
    simpath = os.path.join(self.path, self.simname+".par")
    env = Environment(loader=FileSystemLoader(TEMPLATE_DIR))
    template = env.get_template(template_name)
    rendered = template.render(**params)

    # Write the parameter file.
    with open(simpath, "w") as f:
      f.write(rendered)
    print(f"  {_style("$", _DIM)} Parfile written to {_style(simpath, _CYAN)}")

  # Write a slurm script.
  def write_bashfile(self, bashname, cluster, id_exe):
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

    bash_path = os.path.join(self.path, bashname)
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
  def submit_job(self):
    """
    Submit the job using the written slurm file.
    """
    if not self.bash_path:
      raise SystemExit("Slurm file has to be written before submitting!")

    res = subprocess.run(['sbatch', self.bash_path], capture_output=True, text=True)
    print(f"  {_style("$", _DIM)} {res.stdout}")

  # ------------ ID -------------
  # Locate the initial data.
  def locate_initial_data(self, path):
    """
    Small wrapper to locate created or running initial data output.
    It needs to point to the directory holding the sub-directories
      with the respective resolutions.

    Parameters:
    path (str): Absolute path to the directory holding the resolution sub-directories.

    Returns:
    None; sets the 'path' class member
    """
    if not os.path.exists(path):
      raise ValueError(f'The following path does not exist: {path}')
    elif not os.listdir(path):
      raise ValueError(f'Directory is empty!')
    else:
      self.parent_path = path
      child_path = os.path.join(path, self.simname+'_00')
      if not os.path.exists(child_path):
        raise SystemExit(f'Could not locate {child_path} under the specified path!')

      cands = glob.glob(os.path.join(child_path, "*_[0-9]*x[0-9]*x[0-9]*_00"))
      if cands:
          self.path = child_path
      else:
        raise SystemExit("Could not locate per-resolution directories under %s" % child_path)

      print(f'  {_style("$", _DIM)} Set initial data output path to: {_style(self.path, _CYAN)}')

  # ------------ PROPERTIES ------------
  # Locate the <system>_properties.txt file.
  def locate_properties(self):
    """
    Reads an existing <system>_properties.txt file from the
    highest resolution directory existing.
    """
    if not self.resolutions:
      raise SystemExit("Locate per-resolution directories first with resolution_dirs()!")

    # Locate the properties file.
    keys = list(self.resolutions.keys())
    self.properties_path = None
    for file in os.listdir(self.resolutions[keys[-1]]):
      if file.endswith("_properties.txt"):
        self.properties_path = os.path.join(self.resolutions[keys[-1]], file)
        print(f"  {_style("$", _DIM)} Found properties under: {_style(self.properties_path, _CYAN)}")

    if self.properties_path is None:
      raise SystemExit("Properties file not found in the highest-resolution folder." \
                       "Check the folder!")

  # Read the metadata from properties.
  def read_properties(self):
    """
    Reads metadata into a dictionary from the <system>_properties.txt.
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

  # Check the accuracy from the properties file.
  def check_accuracy(self):
    """
    Compares the expected target quantities from the parfile with
    the current quantities from the properties file.
    """
    if not self.par_dic:
      raise SystemExit("Read parfile first with read_parfile()!")
    if not self.prop_dic:
      raise SystemExit("Read properties file first with read_properties()!")

    # Parse based on the system.
    pardic  = self.par_dic
    propdic = self.prop_dic
    if self.system == 'BHNS':
      # Expected values.
      bhmass_expected = float(pardic['BH_irreducible_mass'])
      bhchiz_expected = float(pardic['BH_chi_z'])
      nsmass_expected = float(pardic['NS_baryonic_mass'])

      # Current values.
      bhmass_current  = float(propdic['BH_irreducible_mass_current'])
      bhchiz_current  = float(propdic['BH_chi_z_current'])
      nsmass_current  = float(propdic['NS_baryonic_mass_current'])

      # Errors.
      bhmass_error    = np.abs(bhmass_current - bhmass_expected)
      bhchiz_error    = np.abs(bhchiz_current - bhchiz_expected)
      nsmass_error    = np.abs(nsmass_current - nsmass_expected)

      # Print.
      bhmass_error_string = f"BH mass error: {bhmass_error * 100 / bhmass_expected:.2f}%"
      bhchiz_error_string = f"BH chiz error: {bhchiz_error * 100 / bhchiz_expected:.2f}%"
      nsmass_error_string = f"NS mass error: {nsmass_error * 100 / nsmass_expected:.2f}%"
      width = max([len(bhmass_error_string) + 2,
                   len(bhchiz_error_string) + 2,
                   len(nsmass_error_string) + 2]) + 2
      print(f'  {_style("$", _DIM)} Accuracy:')
      print(f'    ╭{"─" * width}╮')
      print(f'    │  {_style(bhmass_error_string, _BOLD)}{" " * (width - 2 - len(bhmass_error_string))}│')
      print(f'    │  {_style(bhchiz_error_string, _BOLD)}{" " * (width - 2 - len(bhchiz_error_string))}│')
      print(f'    │  {_style(nsmass_error_string, _BOLD)}{" " * (width - 2 - len(nsmass_error_string))}│')
      print(f'    ╰{"─" * width}╯')

    else:
      raise SystemExit("Set the 'system' (currently supported: 'BHNS')!")

  # Write metadata to file.
  def write_metadata(self):
    """
    Writes important metadata of this initial data to a
    file.
    """
    if not self.prop_dic:
      raise SystemExit("Writing metadata requires reading properties with read_properties()!")

    # Get the timestamp of the highest-resolution
    # checkpoint.
    keys    = list(self.resolutions.keys())
    check_path = os.path.join(self.resolutions[keys[-1]], "checkpoint.dat")
    if not os.path.exists(check_path):
      raise SystemExit("Could not find the 'checkpoint.dat' file in " \
                       "the highest-resolution folder. Check!")
    created = datetime.fromtimestamp(os.path.getmtime(check_path)).date().isoformat()

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

    # Write the file.
    json_path = os.path.join(self.parent_path, "metadata.json")
    with open(json_path, "w") as f:
      json.dump(metadata, f, indent=4)

    if os.path.exists(json_path):
      self.json_path = json_path
      print(f"  {_style("$", _DIM)} Created a metadata file under: {_style(self.json_path, _CYAN)}")

  # ------------ PARFILE -------------
  # Locate a parfile.
  def locate_parfile(self):
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
      for file in os.listdir(res[r]):
        if file.endswith('.par'):
          self.parfile_path = os.path.join(res[r], file)
          break

      else: # only execute if per-file loop did not break
        continue
      break # only executed if per-file loop did break

    if self.parfile_path is None:
      print('Could not locate a parfile with signature .par.' \
            'Check the per-resolution folders!')
    else:
      print(f'  {_style("$", _DIM)} Found parfile under: {_style(self.parfile_path, _CYAN)}')

  # Read the parfile.
  def read_parfile(self):
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

    if not par_dic:
      print(f"  {_style("$", _DIM)} Read parfile and stored the data...")

  # ------------ LOG-FILES -------------
  # Locate slurm logfiles.
  def locate_logfiles(self, path=None):
    """
    Locate logfiles under the specified path.

    Parameters:
    path (str): path to the slurm log files.

    Returns:
    None; sets the 'parfile_path' class member
    """
    if path is None:
      path = os.path.dirname(self.path)

    logs = sorted(glob.glob(os.path.join(path, 'slurm-*.out')), key=os.path.getmtime)
    if not logs:
       logs = sorted(glob.glob(os.path.join(path, '*.out')), key=os.path.getmtime)
       if not logs:
         raise SystemExit('Could not locate log-files with signature '\
                          'slurm-*.out or *.out. Check the path!')
    else:
      self.logs_path = logs
      print(f'  {_style("$", _DIM)} Found logs under: {_style(self.logs_path, _CYAN)}')

  # ------------ RESOLUTION DIRS -------------
  # Locate the initial data.
  def resolution_dirs(self):
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
    for d in sorted(os.listdir(root)):
      full_path = os.path.join(root, d)
      if not os.path.isdir(full_path):
        continue

      m = re.search(r"_(\d+)x(\d+)x(\d+)_\d+$", d)
      if not m:
        continue

      out[int(m.group(1))] = full_path

    if not out:
      raise SystemExit("Located per-resolution directories earlier," \
                       "but they don't match the format. Check!")
    else:
      self.resolutions = out
      print(f"  {_style("$", _DIM)} Found resolutions: {list(self.resolutions.keys())}")

  # ------------ RESOLUTION DIRS -------------
  # Locate the initial data.
  def convergence(self, plot=False, save=False):
    """
    Analyze the convergence per resolution.
    Results can be plotted.

    Parameters:
    plot      (bool): Whether to plot the results
    save      (bool): Whether to store the plot.
    output_dir (str): If store=True, where?

    Returns:
    None; stores a dict in the 'conv' class member which holds
    the constraints at each iteration per resolution per patch; stores
    a dict in the 'conv_max' class member which holds a worst-over-patches reduction
    for each iteration per resolution per patch; can plot the constraints at the
    last iteration at a given resolution
    """
    # Convergence data over all resolutions and patches.
    self.conv = defaultdict(lambda: defaultdict(lambda: defaultdict(dict))) # Constraints etc.

    # Maximum violation at each iteration over all patches.
    self.conv_max = defaultdict(dict)

    res = self.resolutions
    if self.path is None or not res:
      raise SystemExit("First locate ID with locate_initial_data(path) "\
                       "and then locate the resolutions with resolution_dirs()!")

    # Load the diagnostic data.
    COL_ITER     = 0
    COL_HAM_L2   = 3                    # ham1
    COL_MOM_L2   = [6, 9, 12]           # mom1_U1, mom1_U0, mom1_U2 (L2)
    COL_EQRES_L2 = [15, 18, 21, 24, 27] # psi, alphaPsi, B0_U0, B0_U1, B0_U2 (L2)

    for r in res.keys():
      diag  = os.path.join(res[r], 'Diagnostics_00')
      files = sorted(glob.glob(os.path.join(diag, "*_0d.txt")))
      if not files:
        raise SystemExit("Could not find any 0D diagnostics files with *_0d.txt signature!")

      ham_stack = []
      niter = None
      for f in files:
        try:
          data = np.loadtxt(f, comments="#", ndmin=2)
        except Exception:
          continue

        if data.size == 0:
          continue

        n = data.shape[0]
        niter = n if niter is None else min(niter, n)

        ham_stack.append((f, data))

      if not ham_stack:
        raise SystemExit("Failed to load the diagnostic files. Check them!")

      ham_max  = np.zeros(niter)
      momx_max = np.zeros(niter)
      momy_max = np.zeros(niter)
      momz_max = np.zeros(niter)
      eq_max   = np.zeros(niter)
      it  = None
      for f, data in ham_stack:
        patch = f.split('/')[-1].split(".")[0].removesuffix("_0d")
        d = data[:niter]
        if it is None:
          it = d[:, COL_ITER]

        # Store the iteration data.
        self.conv[r][patch]['iter']        = it
        self.conv[r][patch]['ham_L2']      = d[:, COL_HAM_L2]
        self.conv[r][patch]['momx_L2']     = d[:, COL_MOM_L2[0]]
        self.conv[r][patch]['momy_L2']     = d[:, COL_MOM_L2[1]]
        self.conv[r][patch]['momz_L2']     = d[:, COL_MOM_L2[2]]
        self.conv[r][patch]['psi_L2']      = d[:, COL_EQRES_L2[0]]
        self.conv[r][patch]['alphaPsi_L2'] = d[:, COL_EQRES_L2[1]]
        self.conv[r][patch]['B0_U0_L2']    = d[:, COL_EQRES_L2[2]]
        self.conv[r][patch]['B0_U1_L2']    = d[:, COL_EQRES_L2[3]]
        self.conv[r][patch]['B0_U2_L2']    = d[:, COL_EQRES_L2[4]]

        # Best-over patches convergence.
        ham_max  = np.maximum(ham_max, np.abs(d[:, COL_HAM_L2]))
        momx_max = np.maximum(momx_max, np.abs(d[:, COL_MOM_L2[0]]))
        momy_max = np.maximum(momy_max, np.abs(d[:, COL_MOM_L2[1]]))
        momz_max = np.maximum(momz_max, np.abs(d[:, COL_MOM_L2[2]]))
        eq_max   = np.maximum(eq_max, np.min(np.abs(d[:, COL_EQRES_L2]), axis=1))

      self.conv_max[r]['ham_L2_max']  = ham_max
      self.conv_max[r]['momx_L2_max'] = momx_max
      self.conv_max[r]['momy_L2_max'] = momy_max
      self.conv_max[r]['momz_L2_max'] = momz_max
      self.conv_max[r]['eq_L2_max']   = eq_max

    # Plot the convergence data. Data taken at the last iteration
    # of each resolution.
    if plot:
      mpl.rcParams["xtick.labelsize"] = 16
      mpl.rcParams["ytick.labelsize"] = 16
      mpl.rcParams["axes.labelsize"] = 16
      mpl.rcParams["text.usetex"] = True
      mpl.rcParams["font.family"] = "Computer Modern Serif"
      mpl.rcParams['lines.linewidth'] = 1.5
      mpl.rcParams['legend.fontsize'] = 10.5

      # Prepare the data.
      ham, momx, momy, momz, res = [], [], [], [], []
      for r in self.conv_max.keys():
        res.append(r)
        ham.append(self.conv_max[r]['ham_L2_max'][-1])
        momx.append(self.conv_max[r]['momx_L2_max'][-1])
        momy.append(self.conv_max[r]['momy_L2_max'][-1])
        momz.append(self.conv_max[r]['momz_L2_max'][-1])

      id = self.path.split("/")[-1].removesuffix("_00_00")
      fig, ax = plt.subplots(1,1,figsize=(6,4))
      ax.plot(res, ham, color='cyan', linestyle='solid', marker='^', label=r'$\mathcal{H}$')
      ax.plot(res, momx, color='orange', linestyle='solid', marker='^', label=r'$M^x$')
      ax.plot(res, momy, color='green', linestyle='solid', marker='^', label=r'$M^y$')
      ax.plot(res, momz, color='pink', linestyle='solid', marker='^', label=r'$M^z$')
      ax.set_xlabel('Resolution N')
      ax.set_ylabel(r'$||L||_2$')
      ax.set_yscale('log')
      ax.legend(frameon=False)
      ax.grid()

      ax.set_title(id)
      fig.tight_layout()
      if not save:
        plt.show()
      else:
        out_path = os.path.join(self.path, 'convergence.png')
        plt.savefig(out_path, dpi=150)
        print(f"  {_style("$", _DIM)} Convergence plot saved under {_style(out_path, _CYAN)}")

  # ------------ INFORMATION -------------
  # Print useful information.
  def print_information(self):
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

    def section(title, entries):
      if body:
        body.append(('', ''))
      body.append((f'  {title}', f'  {_style(title, _BOLD)}'))
      for plain, decorated in entries:
        body.append((f'    {plain}', f'    {decorated}'))

    def url(link):
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

  # ------------ SCHEDULE -------------
  # Get the scheduled iterations.
  def parse_iteration_schedule(self):
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

  # ------------ LOGS -------------
  # Parse the logs for the iteration speed and resolution.
  def parse_logs(self):
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
  def logs_per_iter_deltas(self, recs):
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
  def counts_from_diagnostics(self):
    """
    Determines how many iterations per resolution are already completed
    based on the 0d diagnostics output.

    Returns:
    Dict with the number of iterations already completed per resolution.
    """
    res = self.resolutions
    if not res:
      raise SystemExit('Get the resolution directories with resolution_dirs() first!')

    # Get the iterations per resolution.
    counts = {}
    for r in res.keys():
      diag = os.path.join(res[r], 'Diagnostics_00')
      if not os.path.isdir(diag):
        raise SystemExit('Could not locate Diagnostics_00 in per-resolution dirs!')

      files = sorted(glob.glob(os.path.join(diag, '*_0d.txt')))
      if not files:
        raise SystemExit('Could not locate files with signature *_0d.txt in ' \
                         f'Diagnostics_00 in the per-resolution dir {res[r]}')

      n = 0
      with open(files[0], errors='ignore') as fh:
        for row in fh:
          if row.strip() and not row.lstrip().startswith('#'):
            n += 1
      counts[r] = n

    return counts

  # Print per-resolution timing and progress.
  def initial_data_progress(self, print_timing=False, outlier_factor=4.0):
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
    done = self.counts_from_diagnostics()
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
      print("Logs:   ", ", ".join(self.logs_path))
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
            + (f" in {os.path.basename(self.path)}" if self.path else ", not found") + "):")
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
  def check_status(self):
    """
    Checks the status of the simulation based on the logfile.
    """
    try:
      _ = os.listdir(self.path)
      self.status = 'Ongoing'
    except FileNotFoundError:
      self.status = 'Not started'

    for log_file in self.logs_path:
      with open(log_file, 'r') as hrf:
        cont = hrf.read()
        done = "} construct_initial_data :))"
        if done in cont:
          self.status = 'Done'
          break

    print(f"  {_style("$", _DIM)} Status: {self.status}")
