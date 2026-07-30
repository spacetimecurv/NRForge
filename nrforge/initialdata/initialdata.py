####################################################
#                                                  #
####################################################

# Built-in libraries.
import os
import sys

# Third-party libraries.
import numpy as np

# ANSI styles used to decorate terminal reports.
_BOLD = '1'
_DIM = '2'
_CYAN = '36'

def _style(text, *codes):
  """Wrap text in ANSI escape codes, unless the output does not support them."""
  if not sys.stdout.isatty() or os.environ.get('NO_COLOR'):
    return text
  return f'\033[{";".join(codes)}m{text}\033[0m'


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
      - ...
  """

  def __init__(self):
    pass

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
    section('CITATION', [('{Rashti:2021ihv,Rashti:2024drr}',) * 2])

    # Title box.
    title = 'ELLIPTICA · initial data solver'
    width = max([len(plain) for plain, _ in body] + [len(title) + 2]) + 2
    print(f'╭{"─" * width}╮')
    print(f'│  {_style(title, _BOLD)}{" " * (width - 2 - len(title))}│')
    print(f'╰{"─" * width}╯')
    for _, decorated in body:
      print(decorated)

# Testing the utility.
elliptica = Elliptica()
elliptica.print_information()