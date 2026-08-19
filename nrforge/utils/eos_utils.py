####################################################
#         A collection of EOS utilities            #
####################################################

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