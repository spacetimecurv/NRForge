####################################################
#                 Style utilities                  #
####################################################

# Built-in libraries.
import sys
import os

# ANSI styles used to decorate terminal reports.
_BOLD = '1'
_DIM = '2'
_CYAN = '36'

def _style(text, *codes):
  """Wrap text in ANSI escape codes, unless the output does not support them."""
  if not sys.stdout.isatty() or os.environ.get('NO_COLOR'):
    return text
  return f'\033[{";".join(codes)}m{text}\033[0m'

# Timing format.
def _fmt(sec):
  sec = int(round(sec))
  d, r = divmod(sec, 86400)
  h, r = divmod(r, 3600)
  m, s = divmod(r, 60)
  return (f"{d}d " if d else "") + f"{h}h {m}m"