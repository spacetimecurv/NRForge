####################################################
#       A collection of derived variables          #
####################################################

# Third-party libraries.
import numpy as np

# NRForge stuff.
from nrforge.utils.constants import *
from nrforge.utils.eos_utils import load_compose_h5_cold_slice_code
from nrforge.utils.numerics import tov_from_baryonic_mass

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


def infall_velocity_PN_estimate_bhns(mbh, mns, chi, D):
  """
  Compute the binary infall velocity from a Post-Newtonian estimate.

  Parameters:
  mbh (float): irreducible mass of BH (in solar masses).
  mns (float): TOV ADM mass of NS (in solar masses).
  chi (float): spin of the BH.
  D   (float): distance between objects.

  Returns:
  Infall velocity as PN estimate.
  """
  mbhc = christodolou_mass(mbh, chi)
  eta  = symmetric_mass_ratio(mbhc, mns)
  M    = mbhc + mns
  return -(64 / 5) * eta * M**3 / D**3


def christodolou_mass(Mirr, chi):
  """
  Calculate the Christodolous mass from the irreducible
  mass of the BH and the spin magnitude chi.

  Parameters:
  Mirr (float): irreducible mass of the BH.
  chi  (float): spin of the BH.

  Returns:
  Christodolou mass.
  """
  return Mirr * np.sqrt(2 / (1 + np.sqrt(1 - chi**2)))


def bhns_scattering_initial_parameters(eos_path: str, MNS_B: float,
                                       MBH: float, chi_BH: float,
                                       D: float, EoM_list: tuple,
                                       Pphi_list: tuple) -> None:
  """
  Takes a certain BHNS binary with specified component masses and distance,
  and from a Newtonian estimate at large distance with initial hyperbolic
  energy and angular momentum computes the initial infall velocity and
  orbital angular momentum that has to be specified in many initial data
  solvers such as Elliptica.

  Parameters:
  eos_path (str): Path to the EOS cold-slice in .h5 format (PyCompOSE).
  MNS_B  (float): Baryonic mass of the neutron star.
  MBH    (float): Irreducible mass of the black hole.
  chi_BH (float): Dimensionless spin of the black hole.
  D      (float): Distance between the objects (in solar masses).
  EoM_list (tuple): Tuple with initial binary energies E/M.
  Pphi_list (tuple): Tuple with initial dimensionless angular momenta.

  Returns:
  Nothing. Prints table with the initial binary parameters.
  """
  # Load the EOS and use the root finder to get the gravitational mass
  # of the NS for the specified EOS and at the target baryonic mass.
  lP, f_e, f_rho0 = load_compose_h5_cold_slice_code(eos_path)
  _, m_ns, mb_ns, R_ns = tov_from_baryonic_mass(MNS_B, lP, f_e, f_rho0)

  # Compute the masses.
  m_chr = christodolou_mass(MBH, chi_BH)
  S_bh  = chi_BH ** m_chr**2

  M  = m_chr + m_ns                      # total mass
  nu = symmetric_mass_ratio(m_chr, m_ns) # symmetric mass ratio
  mu = nu * M                            # reduced mass

  print(f"NS : M = {m_ns:.4f}  M_b = {mb_ns:.4f}  R = {R_ns:.2f} ({R_ns*1.4767:.1f} km)")
  print(f"BH : M_irr = {MBH}  chi = {chi_BH}  ->  M = {m_chr:.4f},  S = {S_bh:.3f}")
  print(f"M = {M:.4f}  mu = {mu:.4f}  nu = {nu:.4f}  q = {m_chr / m_ns:.3f}")
  print(f"D = {D} code units = {D/M:.2f} M   (2M/D = {2*M/D:.4f}: size of the Newtonian term)\n")

  hdr = (" E/M   p_phi   v_inf    b/M   rp/M  chi_N |  Omega        v_r      |"
        " |v_NS| |v_BH| |  E_ADM    J_ADM   t_peri")
  print(hdr)
  print("-" * len(hdr))

  # Loop over the list of energies and angular momenta.
  for E_over_M in EoM_list:
    # ------------------------------------------------------------------------
    # Step 1: E -> relative speed at infinity (exact special relativity).
    # ------------------------------------------------------------------------
    gamma_inf = 1 + (E_over_M**2 - 1) / (2 * nu)
    v_inf = np.sqrt(1 - 1 / gamma_inf**2)

    for p_phi in Pphi_list:
      L = p_phi * mu * M   # orbital angular momentum

      # --------------------------------------------------------------------
      # Step 2: L -> tangential relative speed at separation D.
      # --------------------------------------------------------------------
      v_t = L / (mu * D)

      # --------------------------------------------------------------------
      # Step 3: Newtonian energy conservation from infinity down to D.
      # --------------------------------------------------------------------
      v = np.sqrt(v_inf**2 + 2 * M / D)

      # --------------------------------------------------------------------
      # Step 4: the remaining speed is radial.
      # --------------------------------------------------------------------
      v_r = -np.sqrt(v**2 - v_t**2)

      # --------------------------------------------------------------------
      # Step 5: Elliptica parameters.
      # --------------------------------------------------------------------
      Omega = v_t / D

      # --------------------------------------------------------------------
      # Step 6: Diagnostics.
      # --------------------------------------------------------------------
      # Object velocity from CM mass fractions.
      v_ns = v * m_chr / M
      v_bh = v * m_ns / M

      # Newtonian impact parameter, eccentricity, periapsis and
      # scattering angle.
      b     = p_phi * M / v_inf
      ecc   = np.sqrt(1 + v_inf**2 * p_phi**2)
      r_p   = p_phi**2 * M / (1 + ecc)
      chi_N = np.degrees(2 * np.arcsin(1 / ecc))

      # Convergence parameters.
      E_adm = E_over_M * M
      J_adm = L + S_bh

      # Periapsis estimate.
      t_peri = D / abs(v_r)

      print(f"{E_over_M:5.3f} {p_phi:5.1f}  {v_inf:6.3f}  {b/M:6.2f} {r_p/M:5.2f} {chi_N:5.1f} |"
            f" {Omega:.5e} {v_r:+.5f} |"
            f"  {v_ns:.3f}  {v_bh:.3f} | {E_adm:7.4f}  {J_adm:6.2f}  {t_peri:6.0f}")


def symmetric_mass_ratio(m1, m2):
  """
  Computes the symmetric mass ratio given two component masses.

  Parameters:
  m1 (float): Component mass 1.
  m2 (float): Component mass 2.

  Returns:
  Symmetric mass ratio.
  """
  M = m1 + m2
  return (m1 * m2) / M**2


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

# -------------- Simulation parameters ---------------
# Here we collect a couple of parameters related to
# simulations.
def get_km_from_solar_masses(Msolar: float) -> float:
  """
  Convert a length given in solar masses to km.

  Parameters:
  Msolar (float): solar masses (code units).

  Returns:
  Length in kilometer.
  """
  M    = Msolar * Msun_SI
  M_km = G_SI * M / c_SI**2 / 1e3

  return M_km

def get_ms_from_solar_masses(Msolar: float) -> float:
  """
  Convert time given in solar masses to ms.

  Parameters:
  Msolar (float): solar masses (code units).

  Returns:
  Time in milliseconds.
  """
  M    = Msolar * Msun_SI
  M_ms = G_SI * M / c_SI**3 * 1e3

  return M_ms

if __name__ == '__main__':
  print(infall_velocity_PN_estimate_bhns(4.3, 1.45, 0.75, 40))
