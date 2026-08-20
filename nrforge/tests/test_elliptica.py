####################################################
#           Test the Elliptica class               #
####################################################

# Import the Elliptica class.
import nrforge as nrf
from nrforge import Elliptica

#nrf.banner()

# Testing the utility.
elliptica = Elliptica(print_info=False)
elliptica.locate_initial_data('/home/no96soq/athenak/runs/PhysicsComparisonBHNS/ID/Diagnostics/BHNS_DD2_BH_m4.3-NS_m1.6_s0.75-d40_22x22x22_00_00')

# Find per-resolution directories.
elliptica.resolution_dirs()

# Find a parfile in the per-resolution directories.
elliptica.locate_parfile()

# Parse the iteration schedule from the parfile.
boundaries, total = elliptica.parse_iteration_schedule()

# Locate/parse logfiles from a given path.
elliptica.locate_logfiles('/home/no96soq/athenak/runs/PhysicsComparisonBHNS/ID/Diagnostics/')
recs = elliptica.parse_logs()

elliptica.initial_data_progress()
elliptica.convergence(plot=False, save=False, output_dir=None)