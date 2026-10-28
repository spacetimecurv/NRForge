####################################################
#           Test the Elliptica class               #
####################################################

# Built-in libraries.
import json
from pathlib import Path
from importlib.resources import files

# Import the Elliptica class.
from nrforge.initialdata.elliptica import Elliptica, get_iterated_bh_mass
from nrforge.utils.eos_utils import load_elliptica_table, calculate_enthalpy_bounds

# Import EOSympOSE.
import eosympose

_HERE      = Path(__file__).parent
_TEMPLATES = files("nrforge.initialdata.templates.elliptica")

# Create a equation-of-state table that Elliptica expects.
options = {"eos_name": "DD2",
           "output_dir": "/home/no96soq/athenak/runs/PhysicsComparisonBHNS/ID/BHNS_DD2_BH_m4.3-NS_m1.6_s0.75-d40_22x22x22_00",
           "elliptica": True,
           "elliptica_format": "compose",
           "eos_cold": True}
eos_path = eosympose.run(**options)


# Analyze mode: Analyze a current set of initial data being created,
#               get the initial data progress (ETA) and convergence.
elliptica = Elliptica.analyze(path='/home/no96soq/athenak/runs/PROJECT_largescale/initial_data/bhns/convergence_tuning/production_flat_vs_ks/P0_flat_DD2/BHNS_DD2_BH_m4.3-NS_m1.6_s0.75-d40_22x22x22_flat_00')

elliptica.initial_data_progress(print_timing=True)
elliptica.convergence(plot=False, save=False)


# Create mode: Create initial data from some user-defined parameters,
#              create a bashfile and submit the job.
mass = get_iterated_bh_mass(mass=4.3, sequence=[50, 50, 50, 20], gap=0.5)
rows = load_elliptica_table(path=_HERE / 'data' / 'DD2_eos.txt', geo=False)
h_floor, h_ceil = calculate_enthalpy_bounds(rows=rows, margin=5e-4, ceil_trim=0, print_info=False)

user_params = json.loads((_TEMPLATES / "bhns_params.json").read_text())

user_params['bh_irreducible_mass'] = mass
user_params['ns_h_floor']          = h_floor
user_params['ns_h_ceil']           = h_ceil
user_params['eos_table_path']      = eos_path / "elliptica" / f"{options['eos_name']}_compose.txt"
elliptica = Elliptica.create(path='/home/no96soq/athenak/runs/PhysicsComparisonBHNS/ID/test',
                             user_params=user_params, system='BHNS')
elliptica.write_bashfile(bashname="BHNS.slurm", cluster="ARA", id_exe="/home/no96soq/Elliptica/Exe/elliptica")
elliptica.submit_job()


# Free mode: point Elliptica class to a path, and use read_diagnostics
#            to analyze convergence and properties across patches in detail.
elliptica = Elliptica.free(path='/home/no96soq/athenak/runs/PROJECT_largescale/initial_data/bhns/convergence_tuning/production_flat_vs_ks/P0_flat_DD2')
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


# Comparison mode: specify a list of paths of the same system, and
#                  a matching list of labels. This runs through
#                  EllipticaComparison, which makes use of Elliptica
#                  functionality.
base = "/home/no96soq/athenak/runs/PROJECT_largescale/initial_data/bhns/convergence_tuning/production_flat_vs_ks"

cmp = Elliptica.compare(
  paths  = [f"{base}/P0_flat_DD2", f"{base}/P1_KS_DD2"],
  labels = ["Conformally flat", "Kerr-Schild"],
)

cmp.plot_target_system_convergence(quantities=[
  "BH_irreducible_mass", "BH_chi_z", "NS_baryonic_mass"
], output_dir=base)

for run in cmp:
  run.plot_constraints_per_patch_group()