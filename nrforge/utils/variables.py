####################################################
#       A collection of derived variables          #
####################################################

# Third-party libraries.
import numpy as np

# ------------ Binary parameters --------------
# Here we collect a couple of useful
# binary paramters.
def lambda_tilde(q: float, lam1: float, lam2: float) -> float:
  """
  Computes the tidal deformability from the individual deformabilities.

  Parameters:
  q    (float): mass ratio.
  lam1 (float): tidal deformability of object 1.
  lam2 (float): tidal deformability of object 2.

  Returns:
  Tidal deformability.
  """
  m1    = q / (1 + q)
  m2    = 1 / (1 + q)
  comb1 = m1 + 12.0 * m2
  comb2 = m2 + 12.0 * m1

  return (16.0/13.0) * (comb1 * (m1**4) * lam1 + comb2 * (m2**4) * lam2)


def chi_effective(q: float, chi1: float, chi2: float) -> float:
  """
  Computes the effective spin parameter.

  Parameters:
  q    (float): mass ratio.
  chi1 (float): spin of the first object.
  chi2 (float): spin of the second object.

  Returns:
  Effective spin parameter.
  """
  m1 = q / (1 + q)
  m2 = 1 / (1 + q)

  return m1 * chi1 + m2 * chi2

# -------------- Ejecta parameters ---------------
# Here we collect a couple of useful parameters
# relating to ejecta.
def v_kick_ejecta_Kyutoku2013(Q: float, MNS: float, Mej: float, vej: float) -> float:
  """
  Computes the kick velocity exerted on remnant black hole in BHNS
  mergers due to anisotropic mass ejection. Based on Kyutoku+2013
  (https://journals.aps.org/prd/pdf/10.1103/PhysRevD.88.041503).

  Parameters:
  Q   (float): mass ratio of the binary.
  MNS (float): mass of the neutron star (in solar masses).
  Mej (float): ejecta mass (in solar masses).
  vej (float): ejecta velocity (in c).

  Returns:
  Kick velocity from ejecta (in km/s).
  """
  A5 = (1 + Q) / (1 + 5)
  vkick = 220 * (Mej / 0.03) * (vej / 0.2) * (MNS / 1.35) * A5
  print(f"The kick velocity due to ejecta is {vkick:.3f} km/s.")

  return vkick

def get_Mej_vej(path) -> float:
  """
  Compute the ejected mass and ejecta velocity from a file
  containing Mej per bin and vinf bins.

  Parameters:
  path (str): path pointing to the binned ejecta mass file.

  Returns:
  Ejecta mass and average velocity.
  """
  data = np.loadtxt(path, comments='#')

  # Get the bins.
  vinf = data[:,0]
  dM   = data[:,1]

  # Compute the mass and velocity.
  M_ej = np.sum(dM)
  vinf_avg = np.sum(vinf * dM) / M_ej

  print(f"M_ej = {M_ej:.6e} Msun")
  print(f"<vinf> = {vinf_avg:.6f} c")

  return M_ej, vinf_avg