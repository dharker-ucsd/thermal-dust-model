"""Dust thermal emission model.

Approach and methodology based on
  - Harker et al. 2002, ApJ 580, 579–597.
  - ...

Submodules
----------
fit : Fit models to a spectrum.
materials : Minerals, porosity, grain size distribution, etc.
model : Thermal equilibrium and emission model.
results : Instances of the model.
util : Core library and utility functions.

"""

from . import fit
from . import materials
from . import model
from . import results
from . import util

from .fit import *
from .materials import *
from .model import *
from .results import *
from .util import *

########################################################################
def _configure():
    """Default configuration: read or create config file."""
    import os
    import configparser

    config = configparser.ConfigParser()
    config_file = os.sep.join((os.environ['HOME'], '.config', 'dust', 'config.ini'))
    if os.path.exists(config_file):
        config.read(config_file)
    else:
        path = os.sep.join((os.environ['HOME'], 'Projects', 'src', 'thermal-dust-model', 'data'))
        config['fit-idl-save'] = {'Path': path}
        del path

        i = 0
        while i >= 0:
            if i != 0:  # no need to create a root directory!
                if not os.path.isdir(config_file[:i]):
                    os.mkdir(config_file[:i])
            i = config_file.find(os.sep, i + 1)

        with open(config_file, 'w') as outf:
            config.write(outf)

    return config

_config = _configure()
