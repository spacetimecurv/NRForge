####################################################
#           A collection of constants              #
####################################################

# ----------- SI ------------
G_SI     = 6.674 * 10**(-11) # m^3 kg^-1 s^-2
c_SI    = 299792458          # m / s
Msun_SI = 1.989 * 10**(30)   # kg

# ----------- CGS ------------
# Conversion of G = c = 1 into cgs.
Msun_sec = 4.925794970773135e-06
c_CGS    = 2.99792458e10    # cm / s
G_CGS    = 6.67408e-8       # cm^3 g^-1 s^-2
kb_CGS   = 1.38064852e-16   # erg / K
Msun_CGS = 1.98848e33       # g
MeV_CGS  = 1.6021766208e-6  # erg

# ----------- Conversions ------------
# AthenaK number density to mass density (from CompOSE [fm^-3]).
_cm_geo = c_CGS**2 / (G_CGS * Msun_CGS)  # 1 cm in geometric solar units
nb_to_rho_geo_per_MeV = (MeV_CGS / (Msun_CGS * c_CGS**2)) * (1.0e39 / _cm_geo**3)