# Initialdata templates

NRForge supports the creation of parfiles for various initialdata solvers based on some production templates. These templates are in the format that is used by the `jinja2` template engine and allows easy replacement of the desired parameters.

Below we list the solvers and the available templates:

| Solver | Systems | Templates |
| ------ | ------- | --------- |
| Elliptica | BHNS | elliptica_bhns.in.j2 |

## Elliptica

If we want to create initial data using Elliptica, we can use the `Elliptica` class in `elliptica.py` which interfaces the templates and `jinja2` to write a parameter file based on some parameter dictionary. As an example, and currently only supported, we consider a BHNS binary. The easiest way to create data for such a system is to take the example parameter dictionary `bhns_params.json` in `templates/elliptica`, modify the parameters, load the file with json (and adjust the parameters afterwards) and pass them to the `Elliptica` constructor in `create` mode. This will replace the placeholders in the template with the specified parameters (see `tests/test_elliptica.py`).

### Parameters

The available parameters inside the `bhns_params.json` file are:

- `project`: in this case `BH_NS_binary_initial_data`
- `separation`: binary separation
- `bh_irreducible_mass`: irreducible mass of the black hole (can be iterated with `get_bh_iterated_mass`)
- `bh_chix`: x-component of the black hole spin
- `bh_chiy`: y-component of the black hole spin
- `bh_chiz`: z-component of the black hole spin
- `ns_baryonic_mass`: baryonic mass of the neutron star
- `ns_eos_name`: name of the equation-of-state
- `eos_table_path`: path to the CompOSE table
- `ns_h_floor`: floor on the enthalpy (can be computed based on the equation-of-state with `calculate_enthalpy_bounds`)
- `ns_h_ceil`: ceiling on the enthalpy (can be computed based on the equation-of-state with `calculate_enthalpy_bounds`)
- `ns_omegax`: x-component of the neutron star spin
- `ns_omegay`: y-component of the neutron star spin
- `ns_omegaz`: z-component of the neutron star spin
