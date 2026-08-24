####################################################
#         A collection of EOS utilities            #
####################################################
# Built-in libraries.
import sys

# Third-party libraries.
import numpy as np
from scipy.interpolate import interp1d

# ------------ CompOSE MR-curves -------------
# If a CompOSE table as a MR-curve, i.e. a
# eos.mr file, fetch the data and interpolate
# the radius for a given mass or mass for
# a given radius.
def compose_get_M_from_MR(path: str, R: float) -> float:
  """
  Return the mass M for a given R based on the specified
  CompOSE MR curve.

  Paramters:
  path (str): path to the eos.mr file.
  R  (float): desired radius (in km).

  Returns:
  Interpolated mass in solar masses.
  """
  # Load the data.
  mr = np.loadtxt(path, comments='#')
  radius, mass = mr[:,0], mr[:,1]

  # Interpolator.
  mass_interpolator = interp1d(radius, mass, kind="linear")

  return mass_interpolator(R)


def compose_get_R_from_MR(path: str, M: float) -> float:
  """
  Return the radius R for a given M based on the specified
  CompOSE MR curve.

  Paramters:
  path (str): path to the eos.mr file.
  M  (float): desired mass (in Msun).

  Returns:
  Interpolated radius in km.
  """
  # Load the data.
  mr = np.loadtxt(path, comments='#')
  radius, mass = mr[:,0], mr[:,1]

  # Interpolator.
  radius_interpolator = interp1d(mass, radius, kind="linear")

  return radius_interpolator(M)

# ------------ ELLIPTICA EOS TABLES -------------
# The following functions are specific to the formatted
# equation-of-state tables Elliptica expects.
def load_elliptica_table(path: str, geo: bool) -> list:
  """
  Reads a Elliptica equation-of-state table either in
  CompOSE or geometric format as described in Elliptica
  and returns a list with the rows of the density and
  the specific enthalpy.

  Parameters:
  path (str): path to the equation-of-state .txt.
  geo (bool): whether the table is in geometric format.

  Returns:
  List of (row_index, n_or_rho0, h).
  """
  # Elliptica compose->geo conversion factors (eos_tabular.c, 2018 CODATA).
  RHO0_F = 0.002712069678583313
  E_F    = 1.619216164136643e-18
  P_F    = 1.80162095578956e-39

  rows = []
  with open(path) as fh:
    idx = 0
    for line in fh:
      s = line.strip()
      if not s or s[0] == "#":
        continue
      parts = s.split()
      try:
        if geo:
          # rest_mass_density, specific_internal_energy, pressure
          rho0, eps, p = (float(parts[0]), float(parts[1]), float(parts[2]))
          e = eps * rho0 + rho0
          dens = rho0
        else:
          # line, number_density, total_energy_density, pressure  (compose)
          if len(parts) < 4:
              continue
          n, e_raw, p_raw = (float(parts[1]), float(parts[2]), float(parts[3]))
          rho0 = n * RHO0_F
          e = e_raw * E_F
          p = p_raw * P_F
          dens = n
      except (ValueError, IndexError):
        continue
      if rho0 <= 0:
        idx += 1
        continue
      h = (p + e) / rho0
      rows.append((idx, dens, h))
      idx += 1
  if not rows:
    sys.exit("No usable rows parsed from %s (wrong --geo/format?)" % path)
  return rows

def calculate_enthalpy_bounds(rows: list, margin: float, ceil_trim: int, print_info: bool) -> tuple:
  """
  Calculate the enthalpy bounds for a given equation-of-state table. The
  enthalpy bounds are set in Elliptica's parameter file.

  Parameters:
  rows (list): list with the density and specific enthalpy.
  margin (float): extra margin above low-density hump for floor.
  ceil_trim (int): drop this many rows off high-density end for ceiling.

  Returns:
  The enthalpy bounds (floor, ceiling).
  """
  if not rows:
    raise SystemExit("Run load_elliptica_table() from utils first!")

  hs = [h for _, _, h in rows]
  n_rows = len(rows)
  imin = min(range(n_rows), key=lambda k: hs[k]) # index of min. enthalpy
  imax = max(range(n_rows), key=lambda k: hs[k]) # index of max. enthalpy

  pre_dip_hump = max(hs[: imin + 1])
  clean_idx = next((k for k in range(imin, n_rows) if hs[k] > pre_dip_hump), imin)

  rec_floor = max(pre_dip_hump, hs[clean_idx]) + margin

  ceil_row = max(0, n_rows - 1 - ceil_trim)
  rec_ceil = hs[ceil_row]

  # Count how many densities a candidate floor maps to.
  def crossings(val):
    c = [k for k in range(n_rows - 1) if (hs[k] - val) * (hs[k + 1] - val) < 0]
    return c

  a = {
    "n_rows": n_rows,
    "imin": imin, "imax": imax,
    "h_min": hs[imin], "h_max": hs[imax],
    "dip_dens": rows[imin][1], "max_dens": rows[imax][1],
    "pre_dip_hump": pre_dip_hump,
    "clean_idx": clean_idx, "clean_dens": rows[clean_idx][1], "clean_h": hs[clean_idx],
    "rec_floor": rec_floor,
    "ceil_row": ceil_row, "rec_ceil": rec_ceil,
    "crossings": crossings
  }

  if print_info:
    print("=" * 70)
    print("Rows parsed:", a["n_rows"])
    print("=" * 70)
    print(f"h range over full table:")
    print(f"   min h = {a['h_min']:.16g}  at row {a['imin']} (dens={a['dip_dens']:.3e})")
    print(f"   max h = {a['h_max']:.16g}  at row {a['imax']} (dens={a['max_dens']:.3e})")

    monotonic = a["imin"] == 0
    print("\nLow-density behaviour:")
    if monotonic:
      print("   h is monotonically increasing from the first row (min at row 0).")
      print("   -> inversion is single valued for any floor above h[0].")
    else:
      print(f"   NON-MONOTONIC: h dips to a minimum at row {a['imin']} "
            f"(dens={a['dip_dens']:.3e}).")
      print(f"   pre-dip hump      = {a['pre_dip_hump']:.16g}")
      print(f"   clean branch above hump starts at row {a['clean_idx']} "
            f"(dens={a['clean_dens']:.3e}, h={a['clean_h']:.16g})")
      print("   -> the floor MUST exceed the hump or it maps to two densities.")

    print("\nRECOMMENDED bounds:")
    print(f"   NS_EoS_enthalpy_floor   = {a['rec_floor']:.10f}")
    print(f"   NS_EoS_enthalpy_ceiling = {a['rec_ceil']:.10f}"
          + (f"   (row {a['ceil_row']}, {ceil_trim} trimmed)" if ceil_trim else "   (table max)"))
    nc = len(a["crossings"](a["rec_floor"]))
    print(f"   [check] recommended floor maps to {nc} density(ies) "
          + ("-> OK" if nc <= 1 else "-> STILL AMBIGUOUS, raise --margin"))

  return (a['rec_floor'], a['rec_ceil'])

