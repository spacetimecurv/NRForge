"""Default Elliptica parameter templates, one per binary type."""

import copy as _copy

__all__ = ["get_elliptica_bhns_template"]


# ---------------- BH-NS ----------------
_PARDIC_ELLIPTICA_BHNS = {
  'Project'                               : 'BH_NS_binary_initial_data',
  'BHNS_separation'                       : '@@',
  'BHNS_angular_velocity'                 : 'auto',
  'BHNS_infall_velocity'                  : '0',
  'BHNS_start_off'                        : 'parameter_file',
  'BHNS_observe_ADM_P'                    : 'S_obj1+S_obj2,default',
  'BHNS_observe_ADM_J'                    : 'S_obj1+S_obj2,default',
  'BHNS_P_ADM_control_method'             : 'adjust(x_CM,y_CM)',
  'BHNS_P_ADM_control_update_weight'      : '0.(x10)->0.2',
  'BHNS_P_ADM_control_tolerance'          : '1E-8',
  'BHNS_P_ADM_control_threshold'          : '10',
  'BH_irreducible_mass'                   : '@@',
  'BH_chi_x'                              : '@@',
  'BH_chi_y'                              : '@@',
  'BH_chi_z'                              : '@@',
  'BH_boost_Vx'                           : 'off',
  'BH_Eq_inner_BC_fields'                 : 'XCTS',
  'BH_Eq_inner_BC_beta'                   : 'alpha+Omega*r',
  'BH_Eq_inner_BC_alpha'                  : 'none',
  'BH_start_off'                          : 'IsoSchild',
  'BH_surface_type'                       : 'perfect_s2',
  'BH_surface_Ylm_max_l'                  : '1',
  'BH_tune_BH_radius_criteria'            : 'fix_irreducible_mass',
  'BH_mass_tolerance'                     : '1E-8',
  'BH_radius_update_weight'               : '0.(x10)->0.2(x540)->0.',
  'BH_spin_update_weight'                 : '0.(x10)->0.2(x540)->0.',
  'BH_spin_tolerance'                     : '1E-3',
  'NS_baryonic_mass'                      : '@@',
  'NS_EoS_description'                    : '@@',
  'NS_EoS_type'                           : 'tabular',
  'NS_EoS_unit'                           : '@@',
  'NS_EoS_table_path'                     : '@@',
  'NS_EoS_table_format'                   : '@@',
  'NS_EoS_interpolation_method'           : 'Hermite1D',
  'NS_EoS_interpolation_use_log'          : 'yes',
  'NS_EoS_Hermite1D_FD_accuracy'          : '3',
  'NS_EoS_Hermite1D_num_points'           : '2',
  'NS_EoS_enthalpy_floor'                 : '@@',
  'NS_EoS_enthalpy_ceiling'               : '@@',
  'NS_Omega_x'                            : '@@',
  'NS_Omega_y'                            : '@@',
  'NS_Omega_z'                            : '@@',
  'NS_surface_type'                       : 'perfect_s2->topology_s2',
  'NS_surface_finder'                     : 'bisection',
  'NS_surface_change_threshold'           : '0.0',
  'NS_surface_Ylm_max_l'                  : '10',
  'NS_enthalpy_allowed_residual'          : '1E-8',
  'NS_enthalpy_update_weight'             : '0.5',
  'NS_Euler_const_criteria'               : 'fix_baryonic_mass',
  'NS_Euler_const_update_weight'          : '1.',
  'NS_force_balance_equation'             : 'none(x4)->adjust(d/dy:Omega)',
  'NS_force_balance_update_weight'        : '0.2',
  'NS_adjust_center_method'               : 'taylor_expansion',
  'NS_adjust_center_update_weight'        : '1.',
  'NS_extrapolate_matter_fields'          : 'inverse_r_expmr',
  'NS_Eq_phi_polish'                      : '0.1',
  'NS_start_off'                          : 'TOV',
  'SYS_initialize'                        : 'TOV+IsoSchild',
  'SYS_initialize_fields'                 : 'XCTS',
  'Free_data_conformal_metric'            : 'flat',
  'Free_data_conformal_Christoffel_symbol': 'flat',
  'Free_data_conformal_Ricci'             : 'flat',
  'Free_data_trK'                         : 'zero',
  'ADM_constraints_method'                : 'from_scratch',
  'ADM_B1I_form'                          : 'inspiral',
  'ADM_compute_adm_Kuu_method'            : 'use_AIJ',
  'Tij_NS_decomposition'                  : 'XCTS',
  'Tij_NS_gConf'                          : 'general',
  'checkpoint_every'                      : '0h',
  'Derivative_Method'                     : 'Spectral',
  'Interpolation_Method'                  : 'Spectral',
  'Fourier_Transformation_Method'         : 'RFT',
  'dF/du_for_Newton_Method'               : 'Spectral',
  'grid_kind'                             : 'SplitCubedSpherical(BH+NS)',
  'grid_set_NS'                           : 'left',
  'grid_set_BH'                           : 'right,excised',
  'grid_NS_central_box_length'            : 'auto',
  'grid_BH_central_box_length'            : 'auto',
  'grid_outermost_radius'                 : '1E5',
  'grid_verbose'                          : 'no',
  'n_a'                                   : '14(x200)->16(x100)->18(x100)->20(x100)->22(x100)',
  'n_b'                                   : '14(x200)->16(x100)->18(x100)->20(x100)->22(x100)',
  'n_c'                                   : '14(x200)->16(x100)->18(x100)->20(x100)->22(x100)',
  'grid_SplitCS_max_n_a'                  : '40',
  'grid_SplitCS_max_n_b'                  : '40',
  'grid_SplitCS_max_n_c'                  : '40',
  'Eq_type'                               : 'Elliptic',
  'Eq_elliptic_test'                      : 'no',
  'Eq_phi'                                : 'XCTS_curve_Type3_DDM, NS',
  'Eq_psi'                                : 'XCTS_curve_excision_Type1_DDM, .*',
  'Eq_alphaPsi'                           : 'XCTS_curve_excision_Type2_DDM, .*',
  'Eq_B0_U0'                              : 'XCTS_flat_excision_Type1_DDM , .*',
  'Eq_B0_U1'                              : 'XCTS_flat_excision_Type1_DDM , .*',
  'Eq_B0_U2'                              : 'XCTS_flat_excision_Type1_DDM , .*',
  'Eq_update_method'                      : 'relaxed_scheme',
  'Eq_update_weight_phi'                  : '0.2',
  'Eq_update_weight_psi'                  : '0.2',
  'Eq_update_weight_alphaPsi'             : '0.2',
  'Eq_update_weight_B0_U0'                : '0.2',
  'Eq_update_weight_B0_U1'                : '0.2',
  'Eq_update_weight_B0_U2'                : '0.2',
  'solve_Order'                           : 'psi,alphaPsi,B0_U0,B0_U1,B0_U2,phi',
  'solve_Newton_Update_Weight'            : '1.',
  'solve_residual'                        : '1E-10',
  'solve_residual_factor'                 : '1E-5',
  'solve_Max_Iteration'                   : '1',
  'solve_Max_Newton_Step'                 : '1',
  'solve_Method'                          : 'DDM_Schur_Complement',
  'solve_UMFPACK_refinement_step'         : '0',
  'solve_UMFPACK_size'                    : '1',
  'txt_output_0d'                         : 'ham,mom,eq_residual',
  'txt_output_1d'                         : '^phi,^psi,^alphaPsi,^B0,^beta,eq_residual,ham,mom',
  'txt_output_1d_line'                    : '(X,0.5,0.5),(0.5,Y,0.5),(0.5,0.5,Z)'
}

# Get a Elliptica BHNS template dict.
def get_elliptica_bhns_template():
  """
  Return the default Elliptica parameter template for a BH-NS binary.
  """
  return _copy.deepcopy(_PARDIC_ELLIPTICA_BHNS)

# Get a Elliptica BHNS user parameters example dict.
def get_elliptica_bhns_user_params_example():
  """
  Returns an example dictionary with the keys needed to
  fill an Elliptica template BHNS parameter file.
  """
  return {
    'binary_separation'      : 40,
    'bh_chi_x'               : 0.0,
    'bh_chi_y'               : 0.0,
    'bh_chi_z'               : 0.75,
    'bh_mass'                : 4.3,
    'ns_mass'                : 1.6,
    'ns_eos_name'            : 'DD2',
    'ns_units'               : 'compose',
    'ns_eos_table_path'      : 'data/DD2_eos.txt',
    'ns_eos_table_format'    : 'line,number_density,total_energy_density,pressure',
    'ns_eos_enth_floor'      : 1.003,
    'ns_eos_enth_ceil'       : 4,
    'ns_omega_x'             : 0.0,
    'ns_omega_y'             : 0.0,
    'ns_omega_z'             : 0.0
  }

# Get a iterated BH mass.
def get_iterated_bh_mass(mass: float, sequence: list, gap: float) -> str:
  """
  Computes an iterated mass string for a Elliptica parfile
  that gives more stability for the convergence.

  Parameters:
  mass    (float): target mass that will be iterated to.
  sequence (list): list with the iterations per iterated mass.
  gap     (float): gap between iterated masses.

  Returns:
  String holding the iteration sequence to the target mass.
  """
  string = ""
  length = len(sequence)

  for i, iter in enumerate(sequence):
    string += str((mass + ((length-i)*gap)))
    string += f"(x{sequence[i]})"
    string += f"->"

  string += str(mass)
  return string