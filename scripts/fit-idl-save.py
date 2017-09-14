#!/usr/bin/python3
import os
import re
import sys
import time
import argparse
from collections import OrderedDict
import numpy as np
from scipy.io import idl
import astropy.units as u
from astropy.io import ascii
from astropy.table import Table
from astropy import constants as const
import dust
from dust.materials import MaterialType

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
parser.add_argument('--delta', default=1., type=float, help='The geocentric distance of the comet in AU.')
parser.add_argument('--columns', type=list_of(str), default='wave,fluxd,unc', help='Comet spectrum column names for the wavelength, spectral values, and uncertainties.  Default="wave,fluxd,unc".')
parser.add_argument('--overwrite', action='store_true', help='Overwrite previous output, if it exists.')
parser.add_argument('--materials', type=list_of(str), default=['ap','ao','ac','co','cp'], help='Names of the materials to include in model fits, default="ap","ao","ac","co","cp"')

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
            raise AssertionError('{} exists, remove or use --overwrite'.format(f))

assert args.unit.is_equivalent('W/(cm2 um)', u.spectral_density(1 * u.um)), 'Comet spectrum must be in units of flux density.'

# Set up comet spectrum
spectrum = ascii.read(args.spectrum)
wave = spectrum[args.columns[0]]
fluxd = spectrum[args.columns[1]]
unc = spectrum[args.columns[2]]

# Open IDL save files with idl.readsav
files = []
materials = []
for mat in args.materials:
    if mat == 'ap':
        files += ['fpyr50.idl']
        materials += [dust.amorphous_pyroxene50]
    elif mat == 'ao':
        files += ['fol50.idl']
        materials += [dust.amorphous_olivine50]
    elif mat == 'ac':
        files += ['fcar_e.idl']
        materials += [dust.amorphous_carbon]
    elif mat == 'co':
        files += ['fsfor.idl']
        materials += [dust.hot_forsterite95]
    elif mat == 'cp':
        files += ['fens.idl']
        materials += [dust.hot_ortho_enstatite]
    else:
        raise ValueError('Requested material is not recognized: {}'.format(mat))

materials_abbrev = [m.abbrev for m in materials]

models = [idl.readsav(os.sep.join((dust._config['fit-idl-save']['path'], f))) for f in files]
mwave = models[0]['wave_f']
delta = args.delta * const.au.to('cm').value
conv = u.Unit('W/(cm2 um)').to(args.unit, 1.0, u.spectral_density(mwave * u.um))
mfluxd = [m['flux'] * conv for m in models]  # unitless but scaled to units of args.unit

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
Ds = np.around(models[0]['D'].astype('float64'), 3)
i = np.zeros(len(Ds), bool)
for j in range(len(Ds)):
    for D in args.D:
        if np.isclose(D, Ds[j]):
            i[j] = True
            break
Ds = Ds[i]
assert any(i), 'Error: None of D={} in {}'.format(args.D, Ds)

# Given the nature of the IDL save files, amorphous and crystalline materials
# are treated differently.
for j, mat in enumerate(materials):
    if MaterialType.AMORPHOUS in mat.mtype:
        mfluxd[j] = mfluxd[j][:, :, i]  # amorphous dust
    else:
        mfluxd[j] = np.repeat(mfluxd[j], len(Ds), 2) # crystalline dust

# Pick out dirtiness, this is not user configurable
mfluxd = [m[..., -1] for m in mfluxd]

# Now all flux density arrays should have the same shape and may be
# combined.
mfluxd = np.array(mfluxd)

# Pass models to fit to dust.fit_all.
# mfluxd will be an array with axis order: material, wavelength, D, gsd

# meta data for all tables
meta = OrderedDict()
meta['fit-idl-save.py parameters'] = ' '.join(sys.argv[1:])
meta['run on'] = time.strftime("%a %b %d %Y %I:%M:%S")
meta['comet spectrum'] = args.spectrum
meta['materials included'] = materials_abbrev
meta['r_h (AU)'] = args.rh
meta['Delta (AU)'] = args.delta

# fit_all results.
tab = dust.fit_all(wave, fluxd, unc, mwave, mfluxd, (gsds, Ds),
                   parameter_names=('GSD', 'D'), materials=materials)

scale_names = ['s({})'.format(m.abbrev) for m in materials]

# factor in delta scale factor for saved tables, but not for in-memory Np
Np = np.array([tab[m] for m in scale_names])
for col in scale_names:
    tab[col] *= delta**2

# Save fit_all results.
tab.meta = meta
tab['D'].format = '{:.3f}'
tab.write(filenames['all'], format='ascii.ecsv')

# Determine best model, save it.
i = tab['chisq'].argmin()
Np = np.array([Np[m][i] for m in range(len(Np))])
D = tab[i]['D']
gsd_name = tab[i]['GSD']
chisq = tab[i]['chisq']
dof = len(wave) - len(args.materials) - 1

j, k = np.unravel_index(i, mfluxd.shape[2:]) # indices for best D and GSD
mfluxd_best = mfluxd[..., j, k]  # pick out best model fluxes
f = Np[:, np.newaxis] * mfluxd_best  # scale model

meta['rchisq'] = float(chisq / dof)
meta['dof'] = dof
meta['GSD'] = str(gsd_name)
if gsd_name.startswith('han'):
    N, M = [float(x) for x in gsd_name.split()[1:]]
    gsd = dust.HannerGSD(0.1, N, M)
    meta['a_p'] = '{} um'.format(round(gsd.ap, 1))
meta['D'] = float(D)
meta['Np'] = OrderedDict()
for j, m in enumerate(scale_names):
    meta['Np'][m] = float(tab[i][m]) * delta**2

fluxd_names = ['F({})'.format(m.abbrev) for m in materials]
best_model = Table(names=['wave', 'F(total)'] + fluxd_names,
                   data=np.vstack((mwave[np.newaxis], np.sum(f, axis=0), f)).T,
                   meta=meta)

# add units and material details to meta data
best_model['wave'].unit = 'um'
for i, col in enumerate(best_model.colnames[1:]):
    best_model[col].unit = str(args.unit)
    if i == 0:  # first column is total
        best_model[col].description = 'Total model spectrum'
    else:  # remaining are materials
        best_model[col].description = materials[i - 1].name
    
best_model.write(filenames['bestmodel'], format='ascii.ecsv')

# Save direct and derived parameters.
porosity = dust.FractallyPorous(0.1, D)
if gsd_name.startswith('han'):
    N, M = [float(x) for x in gsd_name.split()[1:]]
    gsd = dust.HannerGSD(0.1, N, M)
elif gsd_name.startswith('pow'):
    N = float(gsd_name.split()[1])
    gsd = dust.PowerLaw(0.1, N)

grains = []
for m in materials:
    if MaterialType.AMORPHOUS in m.mtype:
        # use fractal porosity
        grains.append(dust.Grains(m, porosity=porosity, gsd=gsd))
    else:
        # crystals are solid
        grains.append(dust.Grains(m, porosity=dust.Solid(), gsd=gsd))

# Save best model results.
ratios = OrderedDict()
ratios['AS'] = ([MaterialType.AMORPHOUS, MaterialType.SILICATE],
                [MaterialType.DUST])
ratios['CS'] = ([MaterialType.CRYSTALLINE, MaterialType.SILICATE],
                    [MaterialType.DUST])
ratios['S/C'] = ([MaterialType.SILICATE],
                 [MaterialType.CARBONACEOUS])
ratios['fcryst'] = ([MaterialType.CRYSTALLINE, MaterialType.SILICATE],
                [MaterialType.SILICATE])
best_results = dust.ModelResults(grains, Np * delta**2, chisq=chisq, dof=dof)
best_results.table(ratios=ratios).write(filenames['best'], format='ascii.ecsv')

# If args.n > 0, pass to dust.fit_uncertainties.  Save all mcfits.
if args.n > 0:
    # Need unscaled Nps here
    best_results_unscaled = dust.ModelResults(grains, Np, chisq=chisq, dof=dof)
    mcall, mcbest = dust.fit_uncertainties(
        wave, fluxd, unc, mwave, mfluxd_best, best_results_unscaled)

    # Incorporate delta scaling factor
    mcall.scales *= delta**2
    
    for pfx in ['', '+', '-']:
        for s in scale_names + ['Mtot']:
            mcbest[pfx + s] *= delta**2

    # save all fits to FITS
    mcall.table().write(filenames['mcall'], format='fits')

    # summarize MCFITS
    meta['s(#), +s(#), -s(#)'] = 'Nps - number of grains at the peak grain size and range for each material'
    meta['Mtot, +Mtot, -Mtot'] = 'total mass of the submicron sized grains in grams'
    meta['f(#), +f(#), -f(#)'] = 'relative mass of the submicron sized grains and range for each material'
    meta['AS, +AS, -AS'] = 'Amorphous silicate dust fraction'
    meta['CS, +CS, -CS'] = 'Crystalline silicate dust fraction'
    meta['S/C, +S/C, -S/C'] = 'Silicate to carbon ratio'
    meta['fcryst, +fcryst, -fcryst'] = 'Ratio of crystalline silicate mass to total silicate mass.'

    mcbest.meta = meta
    mcbest.write(filenames['mcbest'], format='ascii.ecsv')
