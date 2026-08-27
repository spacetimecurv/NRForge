# AthenaK templates

As for the initial data solvers, we use the `jinja2` templating engine to generate parameter files and setups in this case for `AthenaK`. Currently only supported are black hole-neutron star (BHNS) binaries with Elliptica. The pipeline for these systems is completely automated as described below.

## Elliptica

Below are described all the systems that are currently supported by NRForge to interface with `AthenaK` and `Elliptica`.

### BHNS

Simulating BHNS systems with `AthenaK` and `Elliptica` first requires initial data. This can be generated with the 'create' mode of the class `Elliptica` of NRForge. Specifically, a relatively robust setup would be

```python
import json
from nrforge.initialdata.elliptica import Elliptica, get_iterated_bh_mass
from nrforge.utils.eos_utils import load_elliptica_table, calculate_enthalpy_bounds

mass = get_iterated_bh_mass(mass=4.3, sequence=[50, 50, 50, 20], gap=0.5)
rows = load_elliptica_table(path='data/DD2_eos.txt', geo=False)
h_floor, h_ceil = calculate_enthalpy_bounds(rows=rows, margin=5e-4, ceil_trim=0, print_info=False)

with open("../initialdata/templates/elliptica/bhns_params.json", "r") as file:
  user_params = json.load(file)

user_params['bh_irreducible_mass'] = mass
user_params['ns_h_floor']          = h_floor
user_params['ns_h_ceil']           = h_ceil
elliptica = Elliptica(path='/path/to/id',
                      mode='create', logs_path=None, print_info=False, user_params=user_params, system='BHNS')
elliptica.write_bashfile(bashname="BHNS.slurm", cluster="ARA", id_exe="/path/to/elliptica_exe")
elliptica.submit_job()
```

The code above takes the setup of the `bhns_params.json` file located in this repo, which the user can adjust to their needs. From the equation-of-state (which can be generated with [PyCompOSE](https://github.com/computationalrelativity/PyCompOSE)), the enthalpy bounds are automatically computed and passed to the templating engine. From the specified choices, the class will create a directory based on the run parameters, and write the parameter file and the slurm file (if specified, and depending on if the cluster type is supported) into the directory. Success! The initial data is now running. It can be monitored in 'analyze' mode via

```python
elliptica = Elliptica(path='/path/to/id',
                     mode='analyze', logs_path=None, print_info=False, user_params=None)

elliptica.initial_data_progress(print_timing=True)
elliptica.convergence(plot=False, save=False)
```

Once the initial data is finished, the status will change to `Done` and the resulting checkpoint and properties file will be transferred to the parent level.

For writing the AthenaK parfile, we can run

```python
import json
from nrforge.evolution.evolution import Evolution

with open("../evolution/templates/athenak/bhns/configs/athenak_bhns_vlr.json", "r") as f:
  params = json.load(f)

evolution = Evolution("AthenaK", "Elliptica", "BHNS")
evolution.write_parfile(
  params=params,
  ev_path="/path/to/evo",
  id_path="/path/to/id"
)
```

This will take the path to the initial data we just created, create a run directory based on the binary parameters, copy the initial data there and write the parfile with the specified parameters in the .json file. The `AthenaK`-`Elliptica`-`BHNS` pipeline does a lot of automation with the initial data and the equation-of-state table so only a few of the parameters have to be set. Below we list all of them and mark the ones that don't need to be set:

- `nx1`, `nx2`, `nx3` - number of points on the root level in each direction
- `domain_minus`, `domain_plus` - outer boundary
- `boundary_condition` - boundary condition (usually `vacuum`, or `outflow`)
- `nx1_block`, `nx2_block`, `nx3_block` - number of points per MeshBlock in each direction
- `max_nmb_per_rank` - maximum number of MeshBlocks per rank
- `num_levels` - number of refinement levels
- `ref1_level` - finest level around the black hole
- `ref2_level` - finest level around the neutron star
- `ref1_x1min`, `ref1_x1max` etc. - size of the finest refinement region (can be set manually, otherwise it is automatically calculated from the object positions using the `Elliptica` properties data)
- `co0_x`, `co0_y`, `co0_z`, `co0_mass`, `co0_radius` etc. - properties of the black hole/neutron star tracker, i.e. the position mass of the object and radius of the tracker volume (automatically set from the `Elliptica` properties data - the tracker radius is calculated from the object radius times some buffer and this is fed into the size of the finest refinement region)
- `cfl` - CFL factor
- `tlim` - time limit in solar masses
- `ndiag` - diagnostic output printed every `ndiag` cycles
- `spin` - spin (not used but has to be set; directly set from the `Elliptica` initial data)
- `horizon_factor` - factor by which to multiply the apparent horizon to get the excision region
- `smooth_excision` - whether to enable smooth excision
- `tdamp` - damping time of smooth excision (only written if enabled)
- `table_path_3D` - path to the 3D CompOSE table
- `NQT` - whether to use non-quite-transcendental spacing in the table (if true, the table has to be written in that format)
- `reconstruction` - reconstruction scheme
- `riemann` - Riemann solver
- `dfloor` - density floor (automatically read by reading the 3D table; also sets `dexcise` inside the excision region)
- `tfloor` - temperature floor (automatically read by reading the 3D table; also sets `texcise` inside the excision region and `temp_min_mev` for the M1 temperature cut)
- `z4c_eta` - shift damping parameter (automatically set as $2/M_{\mathrm{ADM}}$ with the total mass from the Elliptica initial data)
- `KO_diss` - Kreiss-Oliger dissipation factor
- `rad_E_floor` - floor on the radiation energy density
- `rad_N_floor` - floor on the radiation number density
- `nb_rad_floor` - density cut below which opacities/emissivities are 0
- `use_inelastic` - enable inelastic scattering in BNS-nurates
- `use_equilibrium` - enable equilibrium distribution instead of reconstructed distribution in BNS-nurates
- `ID_path` - path to the initial data (automatically set; the class copies the initial data from the specified initial data directory to the run directory and sets that path)
- `ID_eos_path` - path to the equation-of-state used for the initial data
- `ID_eos_evo_path` - path to the cold 1D slice of the 3D table
- `bmax` - magnetic field strength in Gauss
- `r0_current` - radius of the current loop, if external magnetic fields are enabled, i.e. if `bfield_type` is `current_loop`, otherwise ignored
- `bfield_type` - type of the magnetic field (either `pressure_poloidal` (internal dipole) or `current_loop` (external dipole))
- `enable_tracer_output` - whether to enable tracer output
- `enable_sphere_output` - whether to enable sphere output for ejecta analysis etc.
- `enable_3D_output` - whether to enable 3D output
- `enable_2D_output` - whether to enable 2D output
- `nrad_wave_extraction` - number of spheres for the gravitational wave extraction
- `waveform_radius_spacing` - spacing between waveform surfaces
- `waveform_dt` - how frequent to write the wave output

For different resolutions, the user can find templates for the parameter file (the trimmed ones assuming one has used the Elliptica initialdata pipeline of NRForge) under `evolution/templates/athenak/bhns/elliptica/configs`. A full template with all the parameters can also be found. The supported resolutions are:

| File | Resolution |
| ---- | ---------- |
| `athenak_bhns_elliptica_lr.json` | 370m |
| `athenak_bhns_elliptica_mr.json` | 240m |
| `athenak_bhns_elliptica_sr.json` | 180m |
| `athenak_bhns_elliptica_hr.json` | 90m |
