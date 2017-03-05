#!/usr/bin/python3
import os
import re
import argparse
import numpy as np
from scipy.io import idl
import astropy.units as u
from astropy.io import ascii
import dust

def list_of(type):
    """Return a fuction that will split a string into a list of `type` objects.

    The string is assumed to be comma-separated.

    Parameters
    ----------
    type : object
      The fuction will return a list of `type` objects.

    Returns
    -------
    f : function
      The parsing function.

    """
    return lambda s: [type(item) for item in s.split(',')]

parser = argparse.ArgumentParser(description='Fit a comet spectrum with a model stored in IDL save file format.')
parser.add_argument('spectrum', type=ascii.read, help='Name of the comet spectrum.  Must be a readable astropy Table, with wavelength in units of μm, and spectral data in units of flux density.')
parser.add_argument('rh', type=float, help='Use the model evaulated at this heliocentric distance in units of au.')
parser.add_argument('output', help='File name prefix for best-fit results.')
parser.add_argument('-n', type=int, default=10000, help='Number of Monte Carlo simulations to run for final uncertainty estimates.  Set to 0 to skip MC fitting.')
parser.add_argument('-D', type=list_of(float), default=[3, 2.857, 2.727, 2.609, 2.5], help='Fit these specific fractal dimensions, default=3,2.857,2.727,2.609,2.5.')
parser.add_argument('--gsd', default='(pow|han).*', help='Regular expression used to determine which GSD models to fit.')
parser.add_argument('--unit', default='W/(m2 um)', type=u.Unit, help='Flux density units of the comet spectrum, default="W/(m2 um)".')
parser.add_argument('--columns', type=list_of(str), default='wave,fluxd,unc', help='Comet spectrum column names for the wavelength, spectral values, and uncertainties.  Default="wave,fluxd,unc".')
parser.add_argument('--overwrite', action='store_true', help='Overwrite previous output, if it exists.')

args = parser.parse_args()

# Set up comet spectrum, taking advantage of argparse magic:
wave = args.spectrum[args.columns[0]]
fluxd = args.spectrum[args.columns[1]]
unc = args.spectrum[args.columns[2]]

# Open IDL save files with idl.readsav, preserve this order:
files = ['fpyr50.idl', 'fol50.idl', 'fcar_e.idl', 'fsfor.idl', 'fens.idl']
models = [idl.readsav(os.sep.join((dust._config['fit-idl-save']['path'], f))) for f in files]
mfluxd = [m['flux'] for m in models]

# Pick out rh
i = np.array([np.isclose(args.rh, rh) for rh in models[0]['r_h']])
assert any(i), 'rh={} not in {}'.format(args.rh, models[0]['r_h'])
assert sum(i) == 1, 'Found more than one matching r_h for {}: {}'.format(args.rh, models[0]['r_h'])
i = np.flatnonzero(i)[0]
mfluxd = [m[:, i] for m in mfluxd]

# Pick out GSDs
gsds = models[0]['gsd'].astype(str)  # IDL bytes to python3 string
i = np.array([re.match(args.gsd, gsd) is not None for gsd in gsds])
assert any(i), 'Error: None of gsd={} in {}'.format(args.gsd, gsds)
gsds = gsds[i]
mfluxd = [m[:, i] for m in mfluxd]

# For fractal dimension, amorphous grains have 5 values, but
# crystalline only have 1.  Pick out the user's requested D values for
# the amorphous grains, then for the crystals repeat their values
# along the D axis to match the number of user requested amorphous Ds.
# For example:
#
#  >>> mfluxd[0].shape  # amorphous dust
#  (300, 1, 59, 5, 1)   # D axis has length 5
#  >>> mfluxd[3].shape  # crystalline dust
#  (300, 1, 59, 1, 6)   # D axis has length 1
#  >>> np.repeat(mfluxd[3], 5, 3).shape  # repeat cyrstalline Ds 5 times
#  (300, 1, 59, 5, 6)

# Pick out D, only for amorphous
Ds = models[0]['D']
i = np.zeros(len(Ds), bool)
for j in range(len(Ds)):
    for D in args.D:
        if np.isclose(D, Ds[j]):
            i[j] = True
            break
Ds = Ds[i]

assert any(i), 'Error: None of D={} in {}'.format(args.D, Ds)
for j in range(3):  # amorphous dust
    mfluxd[j] = mfluxd[j][:, :, i]

for j in range(3, 5):  # crystalline dust
    mfluxd[j] = np.repeat(mfluxd[j], len(Ds), 2)

# Pick out dirtiness, this is not user configurable
mfluxd = [m[..., -1] for m in mfluxd]

# Now all flux density arrays should have the same shape and may be
# combined.
mfluxd = np.array(mfluxd)

# Pass models to fit to dust.fit_all.
# mfluxd will be an array with axis order: material, wavelength, D, gsd
mwave = models[0]['wave_f']

tab = dust.fit_all(wave, fluxd, unc, mwave, mfluxd, (Ds, gsds),
                   parameter_names=('D', 'GSD'),
                   material_names=('ap', 'ao', 'ac', 'co', 'cp'))

# Save fit_all results.

# Determine best model.

# If If args.n > 0, pass to dust.fit_uncertainty.  Otherwise, set uncertainty
# to 0?  Save all mcfits.

# Save best model results.
