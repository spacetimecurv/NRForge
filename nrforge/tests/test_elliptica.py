####################################################
#           Test the Elliptica class               #
####################################################

# Import the Elliptica class.
import nrforge as nrf
from nrforge.initialdata.elliptica import Elliptica
from nrforge.utils.eos_utils import load_elliptica_table, calculate_enthalpy_bounds
from nrforge.initialdata.templates.elliptica_templates import get_iterated_bh_mass

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
user_params = {
  'binary_separation'      : 40,
  'bh_chi_x'               : 0.0,
  'bh_chi_y'               : 0.0,
  'bh_chi_z'               : 0.75,
  'bh_mass'                : mass,
  'ns_mass'                : 1.6,
  'ns_eos_name'            : 'DD2',
  'ns_units'               : 'compose',
  'ns_eos_table_path'      : 'data/DD2_eos.txt',
  'ns_eos_table_format'    : 'line,number_density,total_energy_density,pressure',
  'ns_eos_enth_floor'      : h_floor,
  'ns_eos_enth_ceil'       : h_ceil,
  'ns_omega_x'             : 0.0,
  'ns_omega_y'             : 0.0,
  'ns_omega_z'             : 0.0
}
elliptica = Elliptica(path='/home/no96soq/athenak/runs/PhysicsComparisonBHNS/ID/test',
                      mode='create', logs_path=None, print_info=False, user_params=user_params, system='BHNS')
elliptica.write_bashfile(bashname="BHNS.slurm", cluster="ARA", id_exe="/home/no96soq/Elliptica/Exe/elliptica")
elliptica.submit_job()
