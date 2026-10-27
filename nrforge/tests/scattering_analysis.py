####################################################
#       Binary scattering initial parameters       #
###################################################

# NRForge utility.
from nrforge.utils.variables import bhns_scattering_initial_parameters

# EOS file and binary parameters.
eos_file = "/home/no96soq/eos_tables/DD2/hdf5/DD2_T0.1_beta.h5"
mns_b, mbh_irr, chi_bh, d = 1.6, 4.3, 0.0, 620.0

# Which encounters to tabulate.
eom_list  = (1.003, 1.005, 1.01)
pphi_list = (4.0, 5.0, 6.0)

# Call the scattering computation function.
bhns_scattering_initial_parameters(eos_file, mns_b, mbh_irr, chi_bh,
                                   d, eom_list, pphi_list)