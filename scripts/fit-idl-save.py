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
parser.add_argumnet('--overwrite', action='store_true', help='Overwrite previous output, if it exists.')

args = parser.parse_args()

# Set up comet spectrum, taking advantage of argparse magic:
wave = args.spectrum[args.columns[0]]
fluxd = args.spectrum[args.columns[1]]
unc = args.spectrum[args.columns[2]]

# Open IDL save files with idl.readsav

# Verify requested rh and D exists in saved model

# Pick out GSDs to fit, e.g.,
#   i = [re.match(args.gsd, str(gsd)) is not None for gsd in model['gsd']]
# Need to convert gsd to str because the gsd array in the IDL save
# file is an array of byte strings, which causes issues in Python 3.

# Pass models to fit to dust.fit_all.

# Save fit_all results.

# Determine best model.

# If If args.n > 0, pass to dust.fit_uncertainty.  Otherwise, set uncertainty
# to 0?  Save all mcfits.

# Save best model results.
