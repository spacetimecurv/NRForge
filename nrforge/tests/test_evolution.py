####################################################
#           Test the Evolution class               #
####################################################

# Built-in libraries.
import json

# Import NRForge utility.
from nrforge.evolution.evolution import Evolution

with open("../evolution/templates/athenak/bhns/elliptica/configs/athenak_bhns_elliptica_vlr.json", "r") as f:
  params = json.load(f)

# TODO: interface batchtools for the batch scripts. Add cluster as an option. Set path to the initial data in the
# json file.

evolution = Evolution("AthenaK", "Elliptica", "BHNS")
evolution.write_parfile(
  params=params,
  ev_path="/home/no96soq/athenak/runs/PhysicsComparisonBHNS/ID/test/DD2_MBH-4.3_sBH-0.75--MNS-1.6_sNS-0.0--d40/evo",
  id_path="/home/no96soq/athenak/runs/PhysicsComparisonBHNS/ID/test/DD2_MBH-4.3_sBH-0.75--MNS-1.6_sNS-0.0--d40"
)