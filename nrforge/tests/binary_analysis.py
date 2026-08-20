####################################################
#       Get the disk mass from Foucart+2018        #
####################################################

# Grab the necessary tools.
from nrforge.utils.eos_utils import compose_get_R_from_MR
from nrforge.utils.fits import BHNS_remnant_mass_model_Foucart2018
from nrforge.utils.variables import v_kick_ejecta_Kyutoku2013, get_Mej_vej

# Get the radius from the EOS.
MNS = 1.4528 # NS TOV mass
MB  = 1.6    # NS baryon mass
R = compose_get_R_from_MR('/home/no96soq/eos_tables/DD2/compose/eos.mr', MNS)

# Get the disk mass.
chi_BH = 0.75  # black hole spin
q      = 3.387 # mass ratio
_ = BHNS_remnant_mass_model_Foucart2018(chi_BH, MB, MNS, R, q)

# Compute the kick velocity from ejecta.
Mej, vej = get_Mej_vej('data/Mej_vinf_geo.txt')
v_kick_ejecta_Kyutoku2013(q, MNS, Mej, vej)
