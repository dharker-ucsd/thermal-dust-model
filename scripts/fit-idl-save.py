#!/usr/bin/python3
import os
import re
import sys
import argparse
from collections import OrderedDict
import numpy as np
from scipy.io import idl
import astropy.units as u
from astropy.io import ascii
from astropy.table import Table
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
parser.add_argument('spectrum', help='Name of the comet spectrum.  Must be a readable astropy Table, with wavelength in units of μm, and spectral data in units of flux density.')
parser.add_argument('rh', type=float, help='Use the model evaulated at this heliocentric distance in units of au.')
parser.add_argument('out_prefix', help='File name prefix for best-fit results.')
parser.add_argument('-n', type=int, default=10000, help='Number of Monte Carlo simulations to run for final uncertainty estimates.  Set to 0 to skip MC fitting.')
parser.add_argument('-D', type=list_of(float), default=[3, 2.857, 2.727, 2.609, 2.5], help='Fit these specific fractal dimensions, default=3,2.857,2.727,2.609,2.5.')
parser.add_argument('--gsd', default='(pow|han).*', help='Regular expression used to determine which GSD models to fit.')
parser.add_argument('--unit', default='W/(m2 um)', type=u.Unit, help='Flux density units of the comet spectrum, default="W/(m2 um)".')
parser.add_argument('--columns', type=list_of(str), default='wave,fluxd,unc', help='Comet spectrum column names for the wavelength, spectral values, and uncertainties.  Default="wave,fluxd,unc".')
parser.add_argument('--overwrite', action='store_true', help='Overwrite previous output, if it exists.')

args = parser.parse_args()

# output file check
filenames = {
    'all': '{}.ALL.txt'.format(args.out_prefix),
    'best': '{}.BEST.txt'.format(args.out_prefix),
    'bestmodel': '{}.BESTMODEL.txt'.format(args.out_prefix),
    'mcall': '{}.MCALL.fits'.format(args.out_prefix),
    'mcbest': '{}.MCBEST.txt'.format(args.out_prefix)
}
for f in filenames.values():
    if os.path.exists(f):
        if args.overwrite:
            os.unlink(f)
        else:
            raise AssertionError('{} exists, remove or use --overwrite')

assert args.unit.is_equivalent('W/(cm2 um)', u.spectral_density(1 * u.um)), 'Comet spectrum must be in units of flux density.'

# Set up comet spectrum
spectrum = ascii.read(args.spectrum)
wave = spectrum[args.columns[0]]
fluxd = spectrum[args.columns[1]]
unc = spectrum[args.columns[2]]

# Open IDL save files with idl.readsav, preserve this order:
files = ['fpyr50.idl', 'fol50.idl', 'fcar_e.idl', 'fsfor.idl', 'fens.idl']
models = [idl.readsav(os.sep.join((dust._config['fit-idl-save']['path'], f))) for f in files]
mwave = models[0]['wave_f']
conv = u.Unit('W/(cm2 um)').to(args.unit, 1.0, u.spectral_density(mwave * u.um))
mfluxd = [m['flux'] * conv for m in models]  # now in units of args.unit

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
material_names = ('ap', 'ao', 'ac', 'co', 'cp')
material_classes = (
    dust.AmorphousPyroxene50,
    dust.AmorphousOlivine50,
    dust.AmorphousCarbon,
    dust.HotOrthoEnstatite,
    dust.HotForsterite95
)
tab = dust.fit_all(wave, fluxd, unc, mwave, mfluxd, (gsds, Ds),
                   parameter_names=('GSD', 'D'), material_names=material_names)

meta = OrderedDict()
meta['fit-idl-save.py parameters'] = ' '.join(sys.argv[1:])
meta['comet spectrum'] = args.spectrum
meta['wavelength unit'] = 'um'
meta['flux density unit'] = str(args.unit)
tab.meta['comments'] = [' = '.join((k, str(v))) for k, v in meta.items()]

# Save fit_all results.
tab.write(filenames['all'], format='ascii.fixed_width_two_line')

# Determine best model, save it.
i = tab['rchisq'].argmin()
Np = np.array([tab[i][m] for m in material_names])
D = tab[i]['D']
gsd_name = tab[i]['GSD']
rchisq = tab[i]['rchisq']
dof = len(wave) - len(material_names) - 1

j, k = np.unravel_index(i, mfluxd.shape[2:]) # indices for best D and GSD
mfluxd_best = mfluxd[..., j, k]  # pick out best model fluxes
f = Np[:, np.newaxis] * mfluxd_best  # scale model
best_model = Table(names=('wave', ) + material_names, data=np.vstack((mwave[np.newaxis], f)).T)

meta['rchisq'] = rchisq
meta['dof'] = dof
meta['Np'] = dict()
Np = np.empty(len(material_names))
for j, m in enumerate(material_names):
    meta['Np'][m] = tab[i][m]
    Np[j] = tab[i][m]
best_model.meta['comments'] = [' = '.join((k, str(v))) for k, v in meta.items()]

best_model.write(filenames['bestmodel'], format='ascii.fixed_width_two_line')

# Save direct and derived parameters.
materials = []
porosity = dust.FractallyPorous(0.1, D)
if gsd_name.startswith('han'):
    N, M = [float(x) for x in gsd_name.split()[1:]]
    gsd = dust.HannerGSD(0.1, N, M)
elif gsd_name.startswith('pow'):
    N = float(gsd_name.split()[1])
    gsd = dust.PowerLaw(0.1, N)

for i in range(len(material_names)):
    if material_names[i] in ['ap', 'ao', 'ac']:
        # use fractal porosity
        materials.append(material_classes[i](porosity=porosity, gsd=gsd))
    else:
        # crystals are solidn and do not accept porosity models
        materials.append(material_classes[i](gsd=gsd))

# Save best model results.
best_results = dust.ModelResults(materials, Np, rchisq, dof)
best_results.table().write(filenames['best'],
                           format='ascii.fixed_width_two_line')

# If args.n > 0, pass to dust.fit_uncertainties.  Save all mcfits.
if args.n > 0:
    mcall, mcbest = dust.fit_uncertainties(wave, fluxd, unc, mwave,
                                           mfluxd_best, best_results)
    mcall.table().write(filenames['mcall'],
                        format='ascii.fixed_width_two_line')
    
    mcbest.write(filenames['mcbest'],
                 format='ascii.fixed_width_two_line')
