####################################################
#         Classes for Elliptica ID solver          #
####################################################

# Built-in libraries.
import os
import sys
import glob
import re
from collections import defaultdict

# Third-party libraries.
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# ANSI styles used to decorate terminal reports.
_BOLD = '1'
_DIM = '2'
_CYAN = '36'

def _style(text, *codes):
  """Wrap text in ANSI escape codes, unless the output does not support them."""
  if not sys.stdout.isatty() or os.environ.get('NO_COLOR'):
    return text
  return f'\033[{";".join(codes)}m{text}\033[0m'


#--------------- ELLIPTICA ID SOLVER ----------------
class Elliptica:
  """
  This class groups useful utility for the analysis of initial
    data produced by the Elliptica solver. Some of the capabilities
    are:
      - Estimate the runtime per resolution for an ongoing run.
      - Check the run status.
      - Check the convergence on specified domain patches.
      - Write a parameter file from template.
      - Write slurm submission scripts for specific machines.
      - Print useful information about Elliptica (compilation etc.).
      - ...
  """

  def __init__(self, print_info=False):
    if print_info: self.print_information()

    # Paths.
    self.path        = None
    self.resolutions = {}

    # Convergence data over all resolutions and patches.
    self.conv        = defaultdict(lambda: defaultdict(lambda: defaultdict(dict))) # Constraints etc.

    # Maximum violation at each iteration over all patches.
    self.conv_max    = defaultdict(dict)

  # ------------ READER -------------
  # Locate the initial data.
  def locate_initial_data(self, path):
    """
    Small wrapper to locate created or running initial data output.
    It needs to point to the directory holding the sub-directories
      with the respective resolutions.

    Parameters:
    path (str): Absolute path to the directory holding the resolution sub-directories.

    Returns:
    None; sets the 'path' class member
    """
    if not os.path.exists(path):
      raise ValueError(f'The following path does not exist: {path}')
    elif not os.listdir(path):
      raise ValueError(f'Directory is empty!')
    else:
      cands = glob.glob(os.path.join(path, "*_[0-9]*x[0-9]*x[0-9]*_00"))
      if cands:
          self.path = path
      else:
        raise SystemExit("Could not locate per-resolution directories under %s" % path)

      print(f'Set initial data output path to: {self.path}')

  # ------------ RESOLUTION DIRS -------------
  # Locate the initial data.
  def resolution_dirs(self):
    """
    Locate the per-resolution directories in the root path.

    Returns:
    None; sets the 'resolution' dict class member
    """
    root = self.path
    if root is None:
      raise ValueError("Root path is not set. Run locate_initial_data(path) first!")

    # Collect the per-resolution arrays.
    out = {}
    for d in sorted(os.listdir(root)):
      full_path = os.path.join(root, d)
      if not os.path.isdir(full_path):
        continue

      m = re.search(r"_(\d+)x(\d+)x(\d+)_\d+$", d)
      if not m:
        continue

      out[int(m.group(1))] = full_path

    if not out:
      raise SystemExit("Located per-resolution directories earlier," \
                       "but they don't match the format. Check!")
    else:
      self.resolutions = out
      print(f"Found resolutions: {list(self.resolutions.keys())}")

  # ------------ RESOLUTION DIRS -------------
  # Locate the initial data.
  def convergence(self, plot=False, save=False, output_dir=None):
    """
    Analyze the convergence per resolution.
    Results can be plotted.

    Parameters:
    plot      (bool): Whether to plot the results
    save      (bool): Whether to store the plot.
    output_dir (str): If store=True, where?

    Returns:
    None; stores a dict in the 'conv' class member which holds
    the constraints at each iteration per resolution per patch; stores
    a dict in the 'conv_max' class member which holds a worst-over-patches reduction
    for each iteration per resolution per patch; can plot the constraints at the
    last iteration at a given resolution
    """
    res = self.resolutions
    if self.path is None or not res:
      raise SystemExit("First locate ID with locate_initial_data(path) "\
                       "and then locate the resolutions with resolution_dirs()!")

    # Load the diagnostic data.
    COL_ITER     = 0
    COL_HAM_L2   = 3                    # ham1
    COL_MOM_L2   = [6, 9, 12]           # mom1_U1, mom1_U0, mom1_U2 (L2)
    COL_EQRES_L2 = [15, 18, 21, 24, 27] # psi, alphaPsi, B0_U0, B0_U1, B0_U2 (L2)

    for r in res.keys():
      diag  = os.path.join(res[r], 'Diagnostics_00')
      files = sorted(glob.glob(os.path.join(diag, "*_0d.txt")))
      if not files:
        raise SystemExit("Could not find any 0D diagnostics files with *_0d.txt signature!")

      ham_stack = []
      niter = None
      for f in files:
        try:
          data = np.loadtxt(f, comments="#", ndmin=2)
        except Exception:
          continue

        if data.size == 0:
          continue

        n = data.shape[0]
        niter = n if niter is None else min(niter, n)

        ham_stack.append((f, data))

      if not ham_stack:
        raise SystemExit("Failed to load the diagnostic files. Check them!")

      ham_max  = np.zeros(niter)
      momx_max = np.zeros(niter)
      momy_max = np.zeros(niter)
      momz_max = np.zeros(niter)
      eq_max   = np.zeros(niter)
      it  = None
      for f, data in ham_stack:
        patch = f.split('/')[-1].split(".")[0].removesuffix("_0d")
        d = data[:niter]
        if it is None:
          it = d[:, COL_ITER]

        # Store the iteration data.
        self.conv[r][patch]['iter']        = it
        self.conv[r][patch]['ham_L2']      = d[:, COL_HAM_L2]
        self.conv[r][patch]['momx_L2']     = d[:, COL_MOM_L2[0]]
        self.conv[r][patch]['momy_L2']     = d[:, COL_MOM_L2[1]]
        self.conv[r][patch]['momz_L2']     = d[:, COL_MOM_L2[2]]
        self.conv[r][patch]['psi_L2']      = d[:, COL_EQRES_L2[0]]
        self.conv[r][patch]['alphaPsi_L2'] = d[:, COL_EQRES_L2[1]]
        self.conv[r][patch]['B0_U0_L2']    = d[:, COL_EQRES_L2[2]]
        self.conv[r][patch]['B0_U1_L2']    = d[:, COL_EQRES_L2[3]]
        self.conv[r][patch]['B0_U2_L2']    = d[:, COL_EQRES_L2[4]]

        # Best-over patches convergence.
        ham_max  = np.maximum(ham_max, np.abs(d[:, COL_HAM_L2]))
        momx_max = np.maximum(momx_max, np.abs(d[:, COL_MOM_L2[0]]))
        momy_max = np.maximum(momy_max, np.abs(d[:, COL_MOM_L2[1]]))
        momz_max = np.maximum(momz_max, np.abs(d[:, COL_MOM_L2[2]]))
        eq_max   = np.maximum(eq_max, np.min(np.abs(d[:, COL_EQRES_L2]), axis=1))

      self.conv_max[r]['ham_L2_max']  = ham_max
      self.conv_max[r]['momx_L2_max'] = momx_max
      self.conv_max[r]['momy_L2_max'] = momy_max
      self.conv_max[r]['momz_L2_max'] = momz_max
      self.conv_max[r]['eq_L2_max']   = eq_max

    # Plot the convergence data. Data taken at the last iteration
    # of each resolution.
    if plot:
      mpl.rcParams["xtick.labelsize"] = 16
      mpl.rcParams["ytick.labelsize"] = 16
      mpl.rcParams["axes.labelsize"] = 16
      mpl.rcParams["text.usetex"] = True
      mpl.rcParams["font.family"] = "Computer Modern Serif"
      mpl.rcParams['lines.linewidth'] = 1.5
      mpl.rcParams['legend.fontsize'] = 10.5

      # Prepare the data.
      ham, momx, momy, momz, res = [], [], [], [], []
      for r in self.conv_max.keys():
        res.append(r)
        ham.append(self.conv_max[r]['ham_L2_max'][-1])
        momx.append(self.conv_max[r]['momx_L2_max'][-1])
        momy.append(self.conv_max[r]['momy_L2_max'][-1])
        momz.append(self.conv_max[r]['momz_L2_max'][-1])

      id = self.path.split("/")[-1].removesuffix("_00_00")
      fig, ax = plt.subplots(1,1,figsize=(6,4))
      ax.plot(res, ham, color='cyan', linestyle='solid', marker='^', label=r'$\mathcal{H}$')
      ax.plot(res, momx, color='orange', linestyle='solid', marker='^', label=r'$M^x$')
      ax.plot(res, momy, color='green', linestyle='solid', marker='^', label=r'$M^y$')
      ax.plot(res, momz, color='pink', linestyle='solid', marker='^', label=r'$M^z$')
      ax.set_xlabel('Resolution N')
      ax.set_ylabel(r'$||L||_2$')
      ax.set_yscale('log')
      ax.legend(frameon=False)
      ax.grid()

      ax.set_title(id)
      fig.tight_layout()
      if not save:
        plt.show()
      else:
        if output_dir is None:
          plt.savefig(os.path.join(self.path, 'conv.png'), dpi=150)
        else:
          plt.savefig(os.path.join(output_dir, 'conv.png'), dpi=150)

  # ------------ INFORMATION -------------
  # Print useful information.
  def print_information(self):
    """Small function printing useful information about the Elliptica solver."""
    # Available solvers.
    solvers = (
      ('Black hole-neutron star', 'https://github.com/rashti-alireza/BH_NS_Binary_Initial_Data'),
      ('Binary neutron star', 'https://github.com/rashti-alireza/NS_NS_Binary_Initial_Data'),
      ('Binary black hole', 'https://github.com/rashti-alireza/BH_BH_Binary_Initial_Data'),
      ('Isolated neutron star', 'https://github.com/rashti-alireza/Single_NS_Initial_Data'),
      ('Isolated black hole', 'https://github.com/rashti-alireza/Single_BH_Initial_Data'),
    )
    name_width = max(len(name) for name, _ in solvers)

    # Compilation commands.
    commands = (
      'git clone --depth=1 git@github.com:rashti-alireza/Elliptica.git',
      'cd Elliptica',
      'make MyConfig',
      'make git_clone',
      'make -j4',
    )

    # Section and URL beautifier.
    body = []

    def section(title, entries):
      if body:
        body.append(('', ''))
      body.append((f'  {title}', f'  {_style(title, _BOLD)}'))
      for plain, decorated in entries:
        body.append((f'    {plain}', f'    {decorated}'))

    def url(link):
      return (link, _style(link, _CYAN))

    section('REPOSITORY', [url('https://github.com/rashti-alireza/Elliptica')])
    section('REQUIREMENTS',
            [('GNU make · gcc compiler · OpenMP · UMFPACK (suitesparse library)',) * 2])
    section('COMPILATION',
            [(f'$ {cmd}', f'{_style("$", _DIM)} {cmd}') for cmd in commands])
    section('INITIAL DATA SOLVERS (available)',
            [(f'{name.ljust(name_width)}  {link}',
              f'{name.ljust(name_width)}  {_style(link, _CYAN)}')
             for name, link in solvers])
    section('INITIAL DATA READER (generic)', [url('https://github.com/rashti-alireza/Elliptica_ID_Reader')])
    section('CITATION', [('{Rashti:2021ihv,Rashti:2024drr}\n',) * 2])

    # Title box.
    title = 'ELLIPTICA · initial data solver'
    width = max([len(plain) for plain, _ in body] + [len(title) + 2]) + 2
    print(f'╭{"─" * width}╮')
    print(f'│  {_style(title, _BOLD)}{" " * (width - 2 - len(title))}│')
    print(f'╰{"─" * width}╯')
    for _, decorated in body:
      print(decorated)

