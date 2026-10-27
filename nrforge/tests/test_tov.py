####################################################
#                Test TOV solver                   #
####################################################

# Third-party libraries.
import numpy as np

# NRForge utility.
from nrforge.utils.eos_utils import load_compose_h5_cold_slice_code
from nrforge.utils.numerics import tov_solver, tov_from_baryonic_mass, \
                                   tov_from_gravitational_mass

# Get the interpolants from a cold-slice PyCompOSE table.
eos_path = "/home/no96soq/eos_tables/DD2/hdf5/DD2_T0.1_beta.h5"
lP, f_e, f_rho0 = load_compose_h5_cold_slice_code(eos_path)

# Integrate the TOV equations (standalone solver).
# Test corresponds to TOV with baryon mass of 1.6, when
# using DD2 cold-slice table.
lPc_test = -8.894432967423654
m_ns, mb_ns, R_ns = tov_solver(np.exp(lPc_test), lP, f_e, f_rho0)
print(f"TOV solve: log(Pc) = {lPc_test}, MNS = {m_ns}, RNS = {R_ns}, MNS_B = {mb_ns}")

# Iterative search for the target central pressure of
# a 1.6 solar mass (baryon mass) star with DD2 EOS.
Pc, mns_target, mbns_target, rns_target = tov_from_baryonic_mass(1.6, lP, f_e, f_rho0)
print(f"Targeted solve (Mb = 1.6 Msol): Pc = {Pc}, MNS = {mns_target}, "
      f"RNS = {rns_target}, Mb_diff = {np.abs(1.6 - mbns_target)}")

# Iterative search for the target central pressure of
# a 1.35 solar mass (gravitational mass) star with DD2 EOS.
Pc, mns_target, mbns_target, rns_target = tov_from_gravitational_mass(1.35, lP, f_e, f_rho0)
print(f"Targeted solve (MNS = 1.35 Msol): Pc = {Pc}, MNS_B = {mbns_target}, "
      f"RNS = {rns_target}, MNS_diff = {np.abs(1.35 - mns_target)}")