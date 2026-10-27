####################################################
#      A collection of numerical utilities         #
####################################################

# Third-party libraries.
import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from scipy.optimize import brentq

# ------------ TOV solver -------------
# Define a generic TOV solver function.
def tov_solver(P_c: float, lP: np.array, f_e: callable,
               f_rho0: callable, r_max: float = 200.0) -> tuple:
  """
  Integrate the TOV equations at a certain central pressure and
  for given interpolants of the pressure, energy density, and
  rest-mass density from a CompOSE cold-slice table.

  Parameters:
  P_c (float): central pressure in code units.
  lP  (array): log of the table pressure (see eos_utils.py).
  f_e, f_rho0 (callable): log(e) and log(rho0) as functions of log(P).
  r_max (float): outer integration limit in code units.

  Returns:
  Tuple (gravitational mass, baryonic mass, R).
  """
  # Define the right-hand-side.
  def rhs(r, y):
    # Get the current state vector.
    m, lp, mb = y

    # Get the pressure, energy density, and rest-mass density (from interpolants).
    P = np.exp(lp)
    e = np.exp(f_e(lp))
    rho0 = np.exp(f_rho0(lp))

    # Define the TOV equations by their RHS.
    dP = -(e + P) * (m + 4 * np.pi * r**3 * P) / (r * (r - 2 * m))
    return [4 * np.pi * r**2 * e,
            dP / P,
            4 * np.pi * r**2 * rho0 / np.sqrt(1 - 2 * m / r)]

  # Stop near the table's low end. Start with the state vector
  # (0, ln(P_c), 0) in the star interior and integrate outwards.
  surface = lambda r, y: y[1] - (lP[0] + 0.5)
  surface.terminal = True
  sol = solve_ivp(rhs, [1e-5, r_max], [0, np.log(P_c), 0], events=surface,
                  rtol=1e-9, atol=1e-12, max_step=0.05)

  if sol.status != 1:
    raise RuntimeError(f"TOV surface not reached for P_c = {P_c:.3e} "
                       f"(r_max = {r_max}, status = {sol.status}).")

  return (sol.y[0, -1], sol.y[2, -1], sol.t[-1])

# Define a Brent root find to find the central pressure
# corresponding to a certain target baryonic mass of the NS.
# Iterates over a sequence of central pressures, and solves the
# TOV equations to find the root.
def tov_from_baryonic_mass(Mb_target: float, lP: np.array, f_e: callable,
                           f_rho0: callable, Pc_min: float = 1e-5,
                           Pc_max: float = 3e-3, n_scan: int = 30) -> tuple:
  """
  Search for the central pressure corresponding to a certain EOS and target
  baryonic mass. Uses a root finder that iteratively solves the TOV equations
  in a range of central pressures.

  Parameters:
  Mb_target (float): target baryonic mass (in solar masses).
  lP  (array): log of the table pressure (see eos_utils.py).
  f_e, f_rho0 (callable): log(e) and log(rho0) as functions of log(P).
  Pc_min (float): lower bound on the pressure range (in code units).
  Pc_max (float): Upper bound on the pressure range (in code units).
  n_scan   (int): Number of central pressures in the scan to find
                  the stable bracket.

  Returns:
  Central pressure in code units, and the corresponding TOV solve.
  """
  # Scan the central pressure until Mb(P_c) decreases in mass, and
  # then only bracket the stable branch.
  lPc  = np.linspace(np.log(Pc_min), min(np.log(Pc_max), lP[-1]), n_scan)
  Mb_s = []
  for x in lPc:
    Mb_s.append(tov_solver(np.exp(x), lP, f_e, f_rho0)[1])
    if len(Mb_s) > 1 and Mb_s[-1] < Mb_s[-2]:
      break

  imax = int(np.argmax(Mb_s))
  if not Mb_s[0] < Mb_target < Mb_s[imax]:
    raise ValueError(f"Mb = {Mb_target} outside stable branch "
                     f"[{Mb_s[0]:.4f}, {Mb_s[imax]:.4f}] of the scan.")

  i = int(np.searchsorted(Mb_s[:imax + 1], Mb_target))

  # Find the root in the adjusted bracket.
  lPc = brentq(lambda x: tov_solver(np.exp(x), lP, f_e, f_rho0)[1] - Mb_target,
               lPc[i - 1], lPc[i])
  Pc  = np.exp(lPc)

  return (Pc,) + tov_solver(Pc, lP, f_e, f_rho0)

# Define a Brent root find to find the central pressure
# corresponding to a certain target gravitational mass of the NS.
# Iterates over a sequence of central pressures, and solves the
# TOV equations to find the root.
def tov_from_gravitational_mass(Madm_target: float, lP: np.array, f_e: callable,
                           f_rho0: callable, Pc_min: float = 1e-5,
                           Pc_max: float = 3e-3, n_scan: int = 30) -> tuple:
  """
  Search for the central pressure corresponding to a certain EOS and target
  gravitational mass. Uses a root finder that iteratively solves the TOV equations
  in a range of central pressures.

  Parameters:
  Madm_target (float): target gravitational mass (in solar masses).
  lP  (array): log of the table pressure (see eos_utils.py).
  f_e, f_rho0 (callable): log(e) and log(rho0) as functions of log(P).
  Pc_min (float): lower bound on the pressure range (in code units).
  Pc_max (float): Upper bound on the pressure range (in code units).
  n_scan   (int): Number of central pressures in the scan to find
                  the stable bracket.

  Returns:
  Central pressure in code units, and the corresponding TOV solve.
  """
  # Scan the central pressure until Mb(P_c) decreases in mass, and
  # then only bracket the stable branch.
  lPc  = np.linspace(np.log(Pc_min), min(np.log(Pc_max), lP[-1]), n_scan)
  Madm_s = []
  for x in lPc:
    Madm_s.append(tov_solver(np.exp(x), lP, f_e, f_rho0)[0])
    if len(Madm_s) > 1 and Madm_s[-1] < Madm_s[-2]:
      break

  imax = int(np.argmax(Madm_s))
  if not Madm_s[0] < Madm_target < Madm_s[imax]:
    raise ValueError(f"Madm = {Madm_target} outside stable branch "
                     f"[{Madm_s[0]:.4f}, {Madm_s[imax]:.4f}] of the scan.")

  i = int(np.searchsorted(Madm_s[:imax + 1], Madm_target))

  # Find the root in the adjusted bracket.
  lPc = brentq(lambda x: tov_solver(np.exp(x), lP, f_e, f_rho0)[0] - Madm_target,
               lPc[i - 1], lPc[i])
  Pc  = np.exp(lPc)

  return (Pc,) + tov_solver(Pc, lP, f_e, f_rho0)