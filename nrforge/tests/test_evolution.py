####################################################
#           Test the Evolution class               #
####################################################

# Built-in libraries.
import json
from importlib.resources import files

# Import NRForge utility.
from nrforge.simulation.evolution import Evolution

# Import EOSympOSE.
import eosympose

# Create a equation-of-state table that Elliptica expects.
options = {"eos_name": "DD2",
           "output_dir": "/home/no96soq/athenak/runs/PhysicsComparisonBHNS/ID/BHNS_DD2_BH_m4.3-NS_m1.6_s0.75-d40_22x22x22_00",
           "elliptica": False,
           "hdf5": True,
           "athtab": True,
           "nqt": True,
           "eos_cold": True}
eos_path = eosympose.run(**options)

_CONFIGS = files("nrforge.simulation.templates.athenak.bhns.elliptica.configs")

params = json.loads((_CONFIGS / "athenak_bhns_elliptica_vlr.json").read_text())
params["table_path_3D"]   = eos_path / "athtab" / f"{options["eos_name"]}_NQT.athtab"
params["ID_eos_evo_path"] = eos_path / "athtab" / f"{options["eos_name"]}_T0.1_beta.athtab"
evolution = Evolution("AthenaK", "Elliptica", "BHNS")
evolution.write_parfile(
  params=params,
  ev_path="/home/no96soq/athenak/runs/PhysicsComparisonBHNS/ID/test/DD2_MBH-4.3_sBH-0.75--MNS-1.6_sNS-0.0--d40/evo",
  id_path="/home/no96soq/athenak/runs/PhysicsComparisonBHNS/ID/test/DD2_MBH-4.3_sBH-0.75--MNS-1.6_sNS-0.0--d40"
)
evolution.select_batchtools_template(cluster="lrz-gpu")