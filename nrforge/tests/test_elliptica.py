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

# Testing the utility (mode 'analyze').
elliptica = Elliptica(path='/home/no96soq/athenak/runs/PhysicsComparisonBHNS/ID/BHNS_DD2_BH_m4.3-NS_m1.6_s0.75-d40_22x22x22_00',
                     mode='analyze', logs_path=None, print_info=False, user_params=None)

elliptica.initial_data_progress(print_timing=True)
elliptica.convergence(plot=False, save=False)

# # Testing the utility (mode 'create').
mass = get_iterated_bh_mass(mass=4.3, sequence=[50, 50, 50, 20], gap=0.5)
rows = load_elliptica_table(path=_HERE / 'data' / 'DD2_eos.txt', geo=False)
h_floor, h_ceil = calculate_enthalpy_bounds(rows=rows, margin=5e-4, ceil_trim=0, print_info=False)

user_params = json.loads((_TEMPLATES / "bhns_params.json").read_text())

user_params['bh_irreducible_mass'] = mass
user_params['ns_h_floor']          = h_floor
user_params['ns_h_ceil']           = h_ceil
user_params['eos_table_path']      = eos_path / "elliptica" / f"{options['eos_name']}_compose.txt"
elliptica = Elliptica(path='/home/no96soq/athenak/runs/PhysicsComparisonBHNS/ID/test',
                      mode='create', logs_path=None, print_info=False, user_params=user_params, system='BHNS')
elliptica.write_bashfile(bashname="BHNS.slurm", cluster="ARA", id_exe="/home/no96soq/Elliptica/Exe/elliptica")
elliptica.submit_job()
