####################################################
#           Test the Elliptica class               #
####################################################

# Built-in libraries.
import json

# Import the Elliptica class.
import nrforge as nrf
from nrforge.initialdata.elliptica import Elliptica, get_iterated_bh_mass
from nrforge.utils.eos_utils import load_elliptica_table, calculate_enthalpy_bounds

nrf.banner()

# Testing the utility (mode 'analyze').
elliptica = Elliptica(path='/home/no96soq/athenak/runs/PhysicsComparisonBHNS/ID/BHNS_DD2_BH_m4.3-NS_m1.6_s0.75-d40_22x22x22_00',
                     mode='analyze', logs_path=None, print_info=False, user_params=None)

elliptica.initial_data_progress(print_timing=True)
elliptica.convergence(plot=False, save=False)

# Testing the utility (mode 'create').
mass = get_iterated_bh_mass(mass=4.3, sequence=[50, 50, 50, 20], gap=0.5)
rows = load_elliptica_table(path='data/DD2_eos.txt', geo=False)
h_floor, h_ceil = calculate_enthalpy_bounds(rows=rows, margin=5e-4, ceil_trim=0, print_info=False)

with open("../initialdata/templates/elliptica/bhns_params.json", "r") as file:
  user_params = json.load(file)

user_params['bh_irreducible_mass'] = mass
user_params['ns_h_floor']         = h_floor
user_params['ns_h_ceil']          = h_ceil
elliptica = Elliptica(path='/home/no96soq/athenak/runs/PhysicsComparisonBHNS/ID/test',
                      mode='create', logs_path=None, print_info=False, user_params=user_params, system='BHNS')
elliptica.write_bashfile(bashname="BHNS.slurm", cluster="ARA", id_exe="/home/no96soq/Elliptica/Exe/elliptica")
#elliptica.submit_job()
