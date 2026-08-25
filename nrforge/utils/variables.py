####################################################
#       A collection of derived variables          #
####################################################

# Third-party libraries.
import numpy as np

# NRForge stuff.
from nrforge.utils.constants import Msun_sec

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


def q_to_nu(q: float) -> float:
  """
  Converts the mass ratio q into the symmetric mass
  ratio nu.

  Parameters:
  q (float): mass ratio of the binary.

  Returns:
  Symmetric mass ratio nu.
  """
  return q / ((1.0 + q) * (1.0 + q))


def nu_to_q(nu: float) -> float:
  """
  Converts the symmetric mass ratio nu into the mass
  ratio q.

  Parameters:
  nu (float): symmetric mass ratio of the binary.

  Returns:
  Mass ratio q.
  """
  if nu == 0.0:
    q = 0.0
  else:
    a = nu
    b = 2 * nu - 1
    c = nu
    d = b**2 - 4 * a * c

    if d < 0:
      sol_1 = (-b + np.sqrt(d, dtype=complex)) / (2 * a)
      sol_2 = (-b - np.sqrt(d, dtype=complex)) / (2 * a)
    else:
      sol_1 = (-b + np.sqrt(d)) / (2 * a)
      sol_2 = (-b - np.sqrt(d)) / (2 * a)

    if sol_1 > 0.0:
      q = sol_1
    else:
      q = sol_2

  return q


def num_orbits_1pn(m1: float, m2: float, omega: float, mtot: float) -> float:
  """
  Computes the number of orbits of the initial data to 1PN order.

  Parameters:
  m1    (float): mass of the first compact object.
  m2    (float): mass of the second compact object.
  omega (float): orbital angular velocity of the binary.
  mtot  (float): total ADM mass of the binary.

  Returns:
  Number of orbits to 1PN order.
  """
  nu = m1 * m2 / (m1 + m2)**2
  return (mtot * omega)**(-5/3) / (32 * nu) / (2 * np.pi)


def energy_3pn(mo: float, q: float) -> float:
  """
  Computes the energy of a binary to 3PN order based on
  https://journals.aps.org/prd/pdf/10.1103/PhysRevD.65.124009 (Eq. (3)).

  Parameters:
  mo (float): angular orbital velocity M * Omega.
  q  (float): mass ratio.

  Returns:
  Energy of binary to 3PN order.
  """
  lam   = -1987.0 / 3080.0
  nu    = q_to_nu(q)
  term1 = 1 + (- 0.75 - nu / 12) * pow(mo, 2/3)
  term2 = (-27/8 + nu * 19/8 - nu * nu / 24) * pow(mo, 4/3)
  inside_term = 209323/4032 - np.pi * np.pi * 205/96 - lam * 110/9
  term3 = (-675/64 + inside_term * nu - nu * nu * 155/96 - nu * nu * nu * 35/5184) * mo * mo
  return -0.5 * nu * pow(mo, 2/3) * (term1 + term2 + term3)


def ang_momentum_3pn(mo: float, q: float) -> float:
    """
    Computes the angular momentum of a binary to 3PN order based on
    https://journals.aps.org/prd/pdf/10.1103/PhysRevD.65.124009 (Eq. (4)).

    Parameters:
    mo (float): angular orbital velocity M * Omega.
    q  (float): mass ratio.

    Returns:
    Energy of binary to 3PN order.
    """
    lam   = -1987.0 / 3080.0
    nu    = q_to_nu(q)
    term1 = 1 + (1.5 + nu / 6) * pow(mo, 2/3)
    term2 = (27/8 - nu * 19/8 + nu * nu / 24) * pow(mo, 4/3)
    inside_term = -209323/5040 + np.pi * np.pi * 41/24 + lam * 88/9
    term3 = (135/16 + inside_term * nu + nu * nu * 31/24 + nu * nu * nu * 7/1296) * mo * mo
    return nu * pow(mo, -1/3) * (term1 + term2 + term3)


def eccentricity_3pn(mo: float, q: float, s1: float = 0,
                     s2: float = 0, lam: float = 0, m1: float = 1,
                     m2: float = 1):
    """
    Computes the eccentricity of a binary to 3PN order
    based on https://arxiv.org/pdf/1807.06857 (Eq. (11)).

    Parameters:
    mo  (float): angular orbital velocity M * Omega.
    q   (float): mass ratio.
    s1  (float): spin of compact object 1.
    s2  (float): spin of compact object 2.
    lam (float): tidal deformability.
    m1  (float): mass of compact object 1.
    m2  (float): mass of compact object 2.

    Returns:
    Eccentricity of binary to 3PN order.
    """
    nu = q_to_nu(q)
    eb = energy_3pn(mo, q) / nu
    el = ang_momentum_3pn(mo, q) / nu
    xi = -eb * el * el

    # Spins.
    M      = m1 + m2
    stilde = (s1 + s2) / (M * M)
    dtilde = (s2/m2 - s1/m1) / M
    dM     = m1 - m2

    # Tidal parameters.
    kt2 = get_tidal_coupling_constant_kappa2T(lam, m1, m2)

    first   = -4 - 2 * nu + (-1 + 3 * nu) * xi
    second  = ((20 - 23 * nu) / xi) - 22 + 60 * nu + 3 * nu * nu \
              -(31 * nu + 4 * nu * nu) * xi
    third_a = (-2016 + (5644 - 123 * np.pi * np.pi) * nu - 252 * nu * nu) / (12 * xi * xi)
    third_b = (4848 + (-21128 + 369 * np.pi * np.pi) * nu + 2988 * nu * nu) / (24 * xi)
    third_c = -20 + 298 * nu - 186 * nu * nu - 4 * nu *nu * nu
    third_d = (-30 * nu + (283 / 4) * nu * nu + 5 * nu * nu * nu) * xi
    e2      = 1 - 2 * xi + first * eb + second * eb * eb \
              + (third_a + third_b + third_c + third_d) * eb * eb * eb
    spins   = 4 * ((5 / (np.sqrt(xi)) - 3 * np.sqrt(xi)) * stilde \
              + (dM / M) * (2 / np.sqrt(xi) - np.sqrt(xi)) * dtilde) * np.power(-eb, 3/2)
    tidal   = -8 * (4 / np.power(xi,4) - 10 / np.power(xi,3) + 5 / np.power(xi,2)) \
              * kt2 * np.power(eb, 5)
    e2      = e2 + spins + tidal
    return np.sqrt(e2)


def get_tidal_coupling_constant_kappa2T(lam: float, m1: float, m2: float) -> float:
  """
  Computes the tidal coupling constant kappa^T_2 as in
  https://arxiv.org/pdf/1807.06857 (below Eq. (12)).

  Parameters:
  lam (float): tidal deformability.
  m1  (float): mass of compact object 1.
  m2  (float): mass of compact object 2.

  Returns:
  Tidal coupling constant.
  """
  q        = m1 / m2
  nu       = q_to_nu(q)
  lamtilde = (8/13) * (1 + 7 * nu - 31 * nu * nu - np.sqrt(1 -4 * nu) \
                    * (1 + 9 * nu - 11 * nu * nu)) * lam
  return 3 * lamtilde / 16

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

def get_Mej_vej(path: str) -> float:
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

# -------------- Gravitational waves ---------------
# The following functions collect a couple of useful
# gravitational wave parameters.
def get_id_gw_frequency_Hz(omega: float) -> float:
  """
  Compute the initial gravitational wave frequency
  given the orbital angular velocity.

  Parameters:
  omega (float): orbital angular velocity of the binary.

  Returns:
  Initial gravitational wave frequency.
  """
  return omega / (np.pi * Msun_sec)

def get_id_gw_frequency_Hz_22(omega: float, mtot: float) -> float:
  """
  Compute the initial gravitational wave frequency
  of the 22-mode given the orbital angular velocity
  and total gravitational mass of the system.

  Parameters:
  omega (float): orbital angular velocity of the binary.
  mtot  (float): total gravitational mass of the binary.

  Returns:
  Initial gravitational wave frequency of the 22-mode.
  """
  return 2 * mtot * omega