####################################################
#       A collection of derived variables          #
####################################################

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