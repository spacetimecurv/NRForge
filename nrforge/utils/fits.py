####################################################
#     A collection of fits from the literature     #
####################################################

# Third-party libraries.
import numpy as np
from scipy.interpolate import interp1d

# Constants.
from nrforge.utils.constants import *

# ------------ Foucart et al. 2018 (https://arxiv.org/pdf/1807.00011) -------------
# The Foucart fits serve as an inference tool for estimating
# the post-merger remnant mass in BHNS mergers.
def normalized_R_ISCO(chi: float) -> float:
  """
  Normalized ISCO radius R_ISCO/M_BH Equ.(3).

  Parameters:
  chi (float): spin of the black hole.

  Returns:
  Normalized ISCO radius based on the input black hole spin.
  """
  Z1 = 1 + (1 - chi**2)**(1/3) * ((1 + chi)**(1/3) + (1 - chi)**(1/3))
  Z2 = np.sqrt(3 * chi**2 + Z1**2)

  return 3 + Z2 - np.sign(chi) * np.sqrt((3 - Z1) * (3 + Z1 + 2 * Z2))


def BHNS_remnant_mass_model_Foucart2018(chi: float, Mb: float, MNS: float,
                                        RNS: float, Q: float) -> float:
  """
  Remnant mass model for BHNS as in Foucart et al. 2018 Equ.(4).
  Multiply by the baryon mass of the neutron star to get the
  remnant mass.

  Parameters:
  chi (float): spin of the black hole.
  Mb  (float): baryon mass of the neutron star.
  MNS (float): TOV mass of the neutron star (in Msun).
  RNS (float): areal radius of the neutron star (in km).
  Q   (float): mass ratio of the binary.
  """
  # Normalized ISCO radius and NS compaction.
  R_isco_norm = normalized_R_ISCO(chi)
  CNS         = (G_SI * MNS * Msun_SI)/ (RNS * 10**3 * c_SI**2)
  eta         = Q / (1 + Q)**2

  # Fitting parameters Equ.(6).
  alpha = 0.406
  beta  = 0.139
  gamma = 0.255
  delta = 1.761

  # Remnant mass model.
  M_rem_model = (np.maximum((alpha * (1 - 2 * CNS) / eta**(1/3) -
                             beta * R_isco_norm * (CNS / eta) + gamma), 0))**delta

  print(f"Compactness of the NS is {CNS:.4f}.")
  print(f"Normalized ISCO is at {R_isco_norm:.4f} M.")
  print(f"Remant mass estimate is {Mb * M_rem_model:.4f} Msun.")

  return M_rem_model
