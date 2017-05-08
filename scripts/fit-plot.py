#!/usr/bin/python3
import argparse
import matplotlib
import numpy as np
import astropy.units as u
from astropy.io import ascii
import matplotlib.pyplot as plt

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


parser = argparse.ArgumentParser(description='Plot a comet spectrum with the model fit.  Input files do not need to have the same units, except wavelength must be μm, and the spectra must be have units of flux density.')
parser.add_argument('spectrum', help='Name of the comet spectrum.')
parser.add_argument('model', help='Name of the file which has the model spectra.')
parser.add_argument('--lfl', action='store_true', help='Plot lambda*F_lambda')
parser.add_argument('--xlog', action='store_true', help='Set log x-axis')
parser.add_argument('--ylog', action='store_true', help='Set log y-axis')
parser.add_argument('--xlim', type=list_of(float), default='3.0, 25.0', help='Limits of the x-axis.  Default = "3.0 - 25.0"')
parser.add_argument('--ylim', type=list_of(float), default='1e-19, 1e-15', help='Limits of the y-axis.  Default = "1e-19 - 1e-15"')
parser.add_argument('--unit', default='', type=u.Unit, help='Flux density units on which to plot the comet spectrum and the model.  The default are from the spectrum.')
parser.add_argument('--dash', action='store_true', help='Plot the unconstrained materials with a dashed line.  Need the relevant MCBEST file with same prefix as the BESTMODEL file in the same directory.')
parser.add_argument('--colspec', type=list_of(str), default='wave,fluxd,unc', help='Comet spectrum column names for the wavelength, spectral values, and uncertainties.  Default="wave,fluxd,unc".')

args = parser.parse_args()

#------------------------------------------------
# Set up comet spectrum
#------------------------------------------------
spectrum = ascii.read(args.spectrum)
wave = spectrum[args.colspec[0]]
fluxd = spectrum[args.colspec[1]] #* u.Unit(unit_in)
unc = spectrum[args.colspec[2]] #* u.Unit(unit_in)
header = ascii.read(spectrum.meta['comments'], delimiter='=', format='no_header', names=['key', 'val'])
sunit_in = str(header[header['key'] == 'flux density unit']['val']).split('\n')[2] # units in the spectrum file

if args.unit != '':
    if args.unit == 'W/(cm2 um)':
        if args.lfl:
            fluxd = fluxd * u.Unit(sunit_in).to('W/(cm2)', 1.0, u.spectral_density(wave * u.um))
            unc = unc * u.Unit(sunit_in).to('W/(cm2)', 1.0, u.spectral_density(wave * u.um))
        else:
            fluxd = fluxd * u.Unit(sunit_in).to('W/(cm2 um)', 1.0, u.spectral_density(wave * u.um))
            unc = unc * u.Unit(sunit_in).to('W/(cm2 um)', 1.0, u.spectral_density(wave * u.um))
        units = 'W/(cm2 um)'
    if args.unit == 'W/(m2 um)':
        if args.lfl:
            fluxd = fluxd * u.Unit(sunit_in).to('W/(m2)', 1.0, u.spectral_density(wave * u.um))
            unc = unc * u.Unit(sunit_in).to('W/(m2)', 1.0, u.spectral_density(wave * u.um))
        else:
            fluxd = fluxd * u.Unit(sunit_in).to('W/(m2 um)', 1.0, u.spectral_density(wave * u.um))
            unc = unc * u.Unit(sunit_in).to('W/(m2 um)', 1.0, u.spectral_density(wave * u.um))
        units = 'W/(m2 um)'
    if args.unit == 'Jy':
        fluxd = fluxd * u.Unit(sunit_in).to('Jy', 1.0, u.spectral_density(wave * u.um))
        unc = unc * u.Unit(sunit_in).to('Jy', 1.0, u.spectral_density(wave * u.um))
        units = 'Jy'
else:
    if sunit_in == 'W / (cm2 um)':
        if args.lfl:
            fluxd = fluxd * u.Unit(sunit_in).to('W/(cm2)', 1.0, u.spectral_density(wave * u.um))
            unc = unc * u.Unit(sunit_in).to('W/(cm2)', 1.0, u.spectral_density(wave * u.um))
        else:
            fluxd = fluxd * u.Unit(sunit_in).to('W/(cm2 um)', 1.0, u.spectral_density(wave * u.um))
            unc = unc * u.Unit(sunit_in).to('W/(cm2 um)', 1.0, u.spectral_density(wave * u.um))
        units = 'W/(cm2 um)'
    if sunit_in == 'W / (m2 um)':
        if args.lfl:
            fluxd = fluxd * u.Unit(sunit_in).to('W/(m2)', 1.0, u.spectral_density(wave * u.um))
            unc = unc * u.Unit(sunit_in).to('W/(m2)', 1.0, u.spectral_density(wave * u.um))
        else:
            fluxd = fluxd * u.Unit(sunit_in).to('W/(m2 um)', 1.0, u.spectral_density(wave * u.um))
            unc = unc * u.Unit(sunit_in).to('W/(m2 um)', 1.0, u.spectral_density(wave * u.um))
        units = 'W/(m2 um)'
    if sunit_in == 'Jy':
        fluxd = fluxd * u.Unit(sunit_in).to('Jy', 1.0, u.spectral_density(wave * u.um))
        unc = unc * u.Unit(sunit_in).to('Jy', 1.0, u.spectral_density(wave * u.um))
        units = 'Jy'

#------------------------------------------------
# Set up model spectra
#
# Required column names based on fit-idl-save.py: wave, F(total), F(*)
# where * is a material abbreviation.
#
#------------------------------------------------
model = ascii.read(args.model) # read the model file
wmodel = u.Quantity(model['wave'])  # pull out the wavelength column
tmodel = u.Quantity(model['F(total)'])  # pull out the total model
materials = model.meta['materials included']
mcols = u.Quantity(np.zeros((len(materials), len(model))), tmodel.unit) # set up array for the individual materials
for i, m in enumerate(model.meta['materials included']):
    mcols[i, :] = u.Quantity(model['F({})'.format(m)]) # pull out the individual materials

# Convert model flux density units to plot units
if args.lfl:
    # plot is lfl, but model is flux density
    tmodel = wmodel * tmodel.to(u.Unit(units) / u.um, u.spectral_density(wmodel))
    mcols = wmodel * mcols.to(u.Unit(units) / u.um, u.spectral_density(wmodel))
else:
    # both plot and model are flux density
    tmodel = tmodel.to(units, u.spectral_density(wmodel))
    mcols = mcols.to(units, u.spectral_density(wmodel))

#------------------------------------------------
# Are the materials constrained?  Draw a dashed line if it is not.
#------------------------------------------------

if args.dash:
    try:
        np_table = ascii.read('{}.MCBEST.txt'.format(args.model.split('.BESTMODEL.txt')[0]))
    except FileNotFoundError:
        print('Sorry, MCBEST file not found. Drawing all lines as solid.')
        line_dash = ["solid" for x in range(len(materials))]
    else:
        line_dash = []
        for i in range(len(materials)):
            num = float(np_table['s{}'.format(i)]) 
            mnum = float(-np_table['-s{}'.format(i)]) 
            if num + mnum == 0:
                line_dash += ['dashed']
            else:
                line_dash += ['solid']
else:
    line_dash = ["solid" for x in range(len(materials))]

#------------------------------------------------
# We have all the data, so now start the plotting
#------------------------------------------------
fig = plt.figure(num=1, figsize=[7,7]) # initialize frame and size
ax = fig.add_subplot(111) # full single frame

#hfont = {'fontname':'Helvetica'} # set font to Helvetica

# thicken the border
ax.spines['top'].set_linewidth(2)
ax.spines['right'].set_linewidth(2)
ax.spines['bottom'].set_linewidth(2)
ax.spines['left'].set_linewidth(2)

plt.minorticks_on() # turn on minor ticks
plt.rc('font', weight='bold') # bold the tick labels
plt.rc('xtick', labelsize=14) # set the x-axis tick label size
plt.rc('ytick', labelsize=14) # set the y-axis tick label size
plt.tick_params(length=10) # set the length of the major ticks
plt.tick_params(direction='in',which='minor',length=5) # set the direction and length of the minor ticks
plt.tick_params(direction='in',which='both',width=2) # set the width of all ticks
plt.ticklabel_format(axis='both', fontweight='bold') # bold the tock labels

plt.xlim(args.xlim) # set x-axis limits
plt.ylim(args.ylim) # set y-axis limits

# Define the axis labels
plt.xlabel('Wavelength ($\mu$m)', fontsize=14, fontweight='bold') #, **hfont) # set x-axis label
if (units == 'W/(cm2 um)') and (args.lfl):
    ylabel_name = '$\lambda$F$_{\lambda}$ (W cm$^{-2}$)'
if (units == 'W/(cm2 um)') and (not args.lfl):
    ylabel_name = 'F$_{\lambda}$ (W cm$^{-2}$ $\mu$m$^{-1}$)'
if (units == 'W/(m2 um)') and (args.lfl):
    ylabel_name = '$\lambda$F$_{\lambda}$ (W m$^{-2}$)'
if (units == 'W/(m2 um)') and (not args.lfl):
    ylabel_name = 'F$_{\lambda}$ (W m$^{-2}$ $\mu$m$^{-1}$)'
if units == 'Jy':
    ylabel_name = 'F(Jy)'

plt.ylabel(ylabel_name, fontsize=14, fontweight='bold') #, **hfont)  # set y-axis label 

# Set axis to log if flagged.
if args.xlog:
    ax.set_xscale("log", nonposx='clip')
if args.ylog:
    ax.set_yscale("log", nonposx='clip')

# Plot the data.
ax.plot(wave,fluxd,'ko',markersize=4) # plot data
ax.plot(wave,fluxd,'w.',markersize=2) # plot data
ax.errorbar(wave,fluxd,yerr=unc,ecolor='k',fmt='none', capsize=2) # plot error bars

# Set up the colors for the materials in a dictionary
colors = {'ap': 'blue', 'ao': 'cyan', 'ac': 'darkorange', 'co': 'green', 'cp': 'magenta', 'other': 'black'}

# Plot the materials
for i, mats in enumerate(materials):
    try:
        colors[mats]
    except KeyError:
        ax.plot(wmodel, mcols[i,:], color=colors['other'], linestyle=line_dash[i])
    else:
        ax.plot(wmodel, mcols[i,:], color=colors[mats], linestyle=line_dash[i]) 

# Plot the total model in red
ax.plot(wmodel.value, tmodel.value, color='red')

plt.show()


