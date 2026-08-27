####################################################
#             Class for Evolution Codes            #
####################################################

# Built-in libraries.
import os
import re
import json
from enum import StrEnum
import subprocess
import math

# Third-party libraries.
import numpy as np
from jinja2 import Environment, FileSystemLoader
from pathlib import Path

# NRForge libraries.
from nrforge.utils.style import _style, _BOLD, _DIM, _CYAN
from nrforge.utils.eos_utils import get_athtab_bounds, load_athtab_header
from nrforge.utils.constants import *

class EvolutionCode(StrEnum):
  ATHENAK = "AthenaK"

class InitialDataSolver(StrEnum):
  ELLIPTICA = "Elliptica"

class EvolutionSystem(StrEnum):
  BHNS = "BHNS"

#--------------- EVOLUTION CLASS ----------------
class Evolution:
  """
  This class is a utility wrapper for evolution codes. Specifically, based
  on the specified evolution code and initial data solver, it can write
  parameter files for various systems.
  """

  def __init__(self, code, id_solver, system):
    """

    """
    # Evolution code.
    try:
      self.code = EvolutionCode(code)
    except ValueError:
      raise ValueError(
        f"Unknown evolution code: {code!r}. "
        f"Available codes: {[c.value for c in EvolutionCode]}"
      )

    # Initial data solver.
    try:
      self.id_solver = InitialDataSolver(id_solver)
    except ValueError:
      raise ValueError(
        f"Unknown initial data solver: {id_solver!r}. "
        f"Available solver: {[c.value for c in InitialDataSolver]}"
      )

    # Evolved system.
    try:
      self.system = EvolutionSystem(system)
    except ValueError:
      raise ValueError(
        f"Unknown evolution system: {system!r}. "
        f"Available system: {[c.value for c in EvolutionSystem]}"
      )

    # Print diagnostics.
    code_string = f"Code: {code}"
    id_solver_string = f"ID solver: {id_solver}"
    system_string = f"System: {system}"
    width = max([len(code_string) + 2,
                 len(id_solver_string) + 2,
                 len(system_string) + 2]) + 2
    print(f'╭{"─" * width}╮')
    print(f'│  {_style(code_string, _BOLD)}{" " * (width - 2 - len(code_string))}│')
    print(f'│  {_style(id_solver_string, _BOLD)}{" " * (width - 2 - len(id_solver_string))}│')
    print(f'│  {_style(system_string, _BOLD)}{" " * (width - 2 - len(system_string))}│')
    print(f'╰{"─" * width}╯')

    # Check the code and set the template path.
    if self.code == EvolutionCode.ATHENAK:
      if self.system == EvolutionSystem.BHNS:
        if self.id_solver == InitialDataSolver.ELLIPTICA:
          self.template = Path(__file__).resolve().parent / "templates" / "athenak" / "bhns" / "elliptica"

  # Write the parfile from template.
  def write_parfile(self, params: dict, ev_path: str, id_path: str):
    """
    Writes a parfile based on the templates that are available
    for the specified system, evolution code and initial data solver.

    Parameters:
    params (dict): dictionary holding the template keys and their values.
    ev_path (str): the path where the evolution directory should be created.
    id_path (str): the path to the directory holding the initial data.

    Returns:
    None; creates a simulation folder at the specified location and
    writes parameter file based on the specified parameters.
    """
    print(f"{_style("Setting up the simulation directory...", _BOLD)}")
    # Check the paths.
    if not os.path.exists(ev_path):
      raise SystemExit("Specified path for the creation of the evolution directory " \
                       "does not exist!")

    if not os.path.exists(id_path):
      raise SystemExit("Specified path to the initial data does not exist!")

    if not os.listdir(id_path):
      raise SystemExit("Specified initial data directory is empty!")

    # Read the initial data properties.
    if self.id_solver == InitialDataSolver.ELLIPTICA:
      for file in os.listdir(id_path):
        if file.endswith("_properties.txt"):
          self.id_properties_path = os.path.join(id_path, file)
        elif file == "metadata.json":
          self.id_metadata_path   = os.path.join(id_path, file)
        elif file == "checkpoint.dat":
          self.id_checkpoint_path = os.path.join(id_path, file)
        else:
          continue

      if not self.id_properties_path or not self.id_metadata_path:
        raise SystemExit("Failed to locate the properties and metadata files!")

      # Read the metadata.
      with open(self.id_metadata_path, "r") as f:
        self.id_metadata = json.load(f)

      # Read the properties file.
      dic = {}
      with open (self.id_properties_path) as file:
        for i, line in enumerate(file):
          if i < 3 or len(line.strip()) == 0:
            continue
          else:
            k, v = line.strip().split('=')
            dic[k.strip()] = v.strip()
      self.prop_dic = dic

      # Build the directory.
      self.simname = re.sub(r"_\d+x\d+x\d+_\d+$", "", self.id_metadata["simname"])
      self.ev_path = os.path.join(ev_path, self.simname)
      try:
        os.mkdir(self.ev_path)
        print(f"  {_style("$", _DIM)} Directory created under {_style(self.ev_path, _CYAN)}")
      except FileExistsError:
        print(f"  {_style("$", _DIM)} Directory exists already under {_style(self.ev_path, _CYAN)}")

      # Copy initial data into the run directory.
      evo_id_path = os.path.join(self.ev_path, 'ID')
      try:
        os.mkdir(evo_id_path)
      except FileExistsError:
        pass
      subprocess.run(['cp', self.id_metadata_path, evo_id_path])
      subprocess.run(['cp', self.id_properties_path, evo_id_path])
      subprocess.run(['cp', self.id_checkpoint_path, evo_id_path])
      print(f"  {_style("$", _DIM)} Copied initial data to {_style(evo_id_path, _CYAN)}")

      # Write the parameter file.
      env = Environment(loader=FileSystemLoader(self.template))
      if self.system == EvolutionSystem.BHNS and self.code == EvolutionCode.ATHENAK:
        # For BHNS in AthenaK that use Elliptica, we compute the refined regions
        # and the tracker properties based on the Elliptica data.
        # Shift damping parameter:
        separation = float(self.prop_dic["BHNS_separation"])
        params["z4c_eta"] = 2 / float(self.prop_dic["BHNS_ADM_mass"])

        # Refined regions (Elliptica binaries are set on the y-axis;
        # NS is on the negative y-axis, and BH on the positive y-axis):
        x_COM = float(self.prop_dic["BHNS_x_CM"])
        y_COM = float(self.prop_dic["BHNS_y_CM"])
        z_COM = float(self.prop_dic["BHNS_z_CM"])
        if float(self.prop_dic["BHNS_mass_ratio"]) > 1:
          bh_COM_y = (separation / 2) - y_COM
          ns_COM_y = -(separation / 2) - y_COM
        else:
          bh_COM_y = (separation / 2) + y_COM
          ns_COM_y = -(separation / 2) + y_COM

        # Black hole (radius inflated by 50%).
        factor               = 0.5
        tracker0_radius      = np.round(np.round(float(self.prop_dic["BH_min_radius"]), 0) * (1 + factor), 1)
        params["co0_mass"]   = float(self.prop_dic["BH_Christodoulou_mass_current"])
        params["co0_x"]      = x_COM
        params["co0_y"]      = bh_COM_y
        params["co0_z"]      = z_COM
        params["co0_radius"] = tracker0_radius
        params["spin"]       = float(self.prop_dic["BH_chi_z_current"])

        # Neutron star (radius inflated by 50%).
        tracker1_radius      = np.round(np.round(float(self.prop_dic["NS_TOV_radius"]), 0) * (1 + factor), 1)
        params["co1_mass"]   = float(self.prop_dic["NS_baryonic_mass_current"])
        params["co1_x"]      = x_COM
        params["co1_y"]      = ns_COM_y
        params["co1_z"]      = z_COM
        params["co1_radius"] = tracker1_radius

        # Refined regions.
        params["ref1_x1min"] = -tracker0_radius
        params["ref1_x1max"] = tracker0_radius
        params["ref1_x2min"] = bh_COM_y - tracker0_radius
        params["ref1_x2max"] = bh_COM_y + tracker0_radius
        params["ref1_x3min"] = -tracker0_radius
        params["ref1_x3max"] = tracker0_radius

        params["ref2_x1min"] = -tracker1_radius
        params["ref2_x1max"] = tracker1_radius
        params["ref2_x2min"] = ns_COM_y - tracker1_radius
        params["ref2_x2max"] = ns_COM_y + tracker1_radius
        params["ref2_x3min"] = -tracker1_radius
        params["ref2_x3max"] = tracker1_radius

        # EOS.
        ID_check = os.path.join(evo_id_path, "checkpoint.dat")
        if os.path.exists(ID_check):
          params["ID_path"] = ID_check
        else:
          raise SystemExit(f"checkpoint.dat does not exist under {ID_check} and cannot be written to parfile!")

        if not os.path.exists(params["ID_eos_path"]):
          raise SystemExit(f"ID eos file does not exist under {params["ID_eos_path"]}!")

        if not os.path.exists(params["ID_eos_evo_path"]):
          raise SystemExit(f"The 1D slice EOS for the ID does not exist under {params["ID_eos_evo_path"]}")

        if not os.path.exists(params["table_path_3D"]):
          raise SystemExit(f"3D EOS table does not exist under {params["table_path_3D"]}")

        _, scalars, _, _, _ = load_athtab_header(params["table_path_3D"])
        (n_min, _), (T_min, _) = get_athtab_bounds(params["table_path_3D"])
        if "mn" in scalars.keys():
          mb = scalars["mn"]
        else:
          mb = 939.56353 # MeV; most common baryon mass across tables
          print(f"  {_style("$", _DIM)} WARNING: could not find baryon mass in the 3D table, " \
                "Using default value of 939.56353MeV for the density floors in the parfile!")
        params["dfloor"] = 10 ** math.ceil(math.log10(n_min * mb * nb_to_rho_geo_per_MeV))
        params["tfloor"] = T_min

        template = env.get_template("athenak_bhns_elliptica.in.j2")
        rendered = template.render(**params)

        self.evo_parfile_path = os.path.join(self.ev_path, self.simname + ".athinput")
        with open(self.evo_parfile_path, "w") as f:
          f.write(rendered)

        print(f"  {_style("$", _DIM)} Wrote parfile to {_style(self.evo_parfile_path, _CYAN)}")

  # Select a batch script from batchtools.
  def select_batchtools_template(self, cluster: str):
    """
    Selects the batchtools template for the specified cluster and add's it
    to the run directory, where the parameterfile has been written.

    Parameters:
    cluster (str): name of the cluster (needs to match batchtools).
    """
    # Functionality currently only supported with AthenaK.
    if self.code == EvolutionCode.ATHENAK:
      subs      = Path(__file__).resolve().parents[2] / "external" / "batchtools" / "templates" / "athenak"
      signature = ".sub"
      clusters  = os.listdir(subs)
      if cluster not in clusters:
        raise SystemExit(f"Supported clusters are: {clusters}")

      self.batch_path = None
      for file in os.listdir(subs / cluster):
        if file.endswith(signature):
          self.batch_path = os.path.join(self.ev_path, file)
          subprocess.run(['cp', subs / cluster / file, self.ev_path])
          break

      if self.batch_path is None:
        raise SystemExit(f"Could not find batch file with signature .sub in {subs / cluster}!")
      else:
        print(f"  {_style("$", _DIM)} Copied batchtools template for {_style(cluster, _DIM)} to {_style(self.batch_path, _CYAN)}")
    else:
      raise SystemExit("Batchtools templates currently only available for AthenaK!")

