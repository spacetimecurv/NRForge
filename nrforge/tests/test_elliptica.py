####################################################
#           Test the Elliptica class               #
####################################################

# Import the Elliptica class.
import nrforge as nrf
from nrforge import Elliptica

nrf.banner()

# Testing the utility.
elliptica = Elliptica(print_info=False)
elliptica.locate_initial_data('/home/no96soq/athenak/runs/PhysicsComparisonBHNS/ID/Diagnostics/BHNS_DD2_BH_m4.3-NS_m1.6_s0.75-d40_22x22x22_00_00')
elliptica.resolution_dirs()
elliptica.convergence(plot=False, save=False, output_dir=None)