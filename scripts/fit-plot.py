#!/usr/bin/python3
import sys
import time
import argparse
import matplotlib
import numpy as np
import astropy.units as u
from astropy.io import ascii
import matplotlib.pyplot as plt
import matplotlib

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


parser = argparse.ArgumentParser(description='Plot a comet spectrum with the model fit.  Input files do not need to have the same units, except wavelength must be μm, and both input files must be have units of flux density.')
parser.add_argument('spectrum', help='Name of the comet spectrum.')
parser.add_argument('model', help='Name of the file which has the model spectra.')
parser.add_argument('--xlog', action='store_true', help='Set log x-axis')
parser.add_argument('--ylog', action='store_true', help='Set log y-axis')
parser.add_argument('--xlim', type=list_of(float), default=[3.0, 25.0], help='Limits of the x-axis.  Default = "3.0 - 25.0"')
parser.add_argument('--ylim', type=list_of(float), default=[1e-19, 1e-15], help='Limits of the y-axis.  Default = "1e-19 - 1e-15"')
parser.add_argument('--yunit', type=u.Unit, default='W/(cm2 um)', help='Use this for the y-axis.  May be units of F_λ, F_ν, λF_λ, or νF_ν.  For the latter, either --lfl or --nfn must be set.')
parser.add_argument('--lfl', action='store_true', help='--yunit is λF_λ.')
parser.add_argument('--nfn', action='store_true', help='--yunit is νF_ν.')
parser.add_argument('--unit', type=u.Unit, help='Comet spectrum flux desnity units.  Default is to divine units from the spectrum header.')
parser.add_argument('--dash', action='store_true', help='Plot the unconstrained materials with a dashed line.  Need the relevant MCBEST file with same prefix as the BESTMODEL file in the same directory.')
parser.add_argument('--colspec', type=list_of(str), default='wave,fluxd,unc', help='Comet spectrum column names for the wavelength, spectral values, and uncertainties.  Default="wave,fluxd,unc".')
parser.add_argument('--fscale', type=int, default=7, help='Size of the frame scale factor.  Default=7')
parser.add_argument('--savefigure', action='store_true', help='Set to write out a PDF file of the figure named "fit-plot-figure-YYYMMDD-HHMMSS.pdf"')


args = parser.parse_args()

assert (args.lfl + args.nfn) != 2, 'Only one of --lfl or --nfn can be specified at one time.'

if args.yunit.is_equivalent('W/m2'):
    assert args.lfl or args.nfn, 'If --yunit is λF_λ, or νF_ν, then --lfl or --nfn must be set'

#------------------------------------------------
# Set up comet spectrum
#------------------------------------------------
spectrum = ascii.read(args.spectrum)
if args.unit is None:
    try:
        unit = u.Unit(spectrum.meta['flux density unit'])
        pass
    except KeyError:
        print('--unit not specified and "flux density unit" not found in spectrum table header.')
        sys.exit()
else:
    unit = args.unit

if spectrum['fluxd'].unit:
    wave = u.Quantity(spectrum[args.colspec[0]])
    spec = u.Quantity(spectrum[args.colspec[1]])
    unc = u.Quantity(spectrum[args.colspec[2]])
else:
    wave = spectrum[args.colspec[0]] * u.um
    spec = spectrum[args.colspec[1]] * unit
    unc = spectrum[args.colspec[2]] * unit

spec = spec.to(args.yunit, u.spectral_density(wave))
unc = unc.to(args.yunit, u.spectral_density(wave))

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

tmodel = tmodel.to(args.yunit, u.spectral_density(wmodel))
mcols = mcols.to(args.yunit, u.spectral_density(wmodel))

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
        for m in materials:
            num = float(np_table['s({})'.format(m)]) 
            mnum = float(-np_table['-s({})'.format(m)]) 
            if num + mnum == 0:
                line_dash += ['dashed']
            else:
                line_dash += ['solid']
else:
    line_dash = ["solid" for x in range(len(materials))]


#------------------------------------------------
# We have all the data, so now start the plotting
#------------------------------------------------
fig = plt.figure(num=1, figsize=[args.fscale,args.fscale]) # initialize frame and size
fig.clear()

plt.rc('font', weight='bold') # bold the tick labels NEEDS TO GO BEFORE add_subplot command
plt.rc('xtick', labelsize=args.fscale*2) # set the x-axis tick label size
plt.rc('ytick', labelsize=args.fscale*2) # set the y-axis tick label size

ax = fig.add_subplot(111) # full single frame

#hfont = {'fontname':'Helvetica'} # set font to Helvetica

# thicken the border
ax.spines['top'].set_linewidth(2)
ax.spines['right'].set_linewidth(2)
ax.spines['bottom'].set_linewidth(2)
ax.spines['left'].set_linewidth(2)

plt.minorticks_on() # turn on minor ticks

ax.tick_params(top='on') # turn on top major ticks
ax.tick_params(right='on') # turn on right major ticks
ax.tick_params(axis='x', which='minor', top='on') # turn on top minor ticks
ax.tick_params(axis='y', which='minor', right='on') # turn on right minor ticks
plt.tick_params(length=10) # set the length of the major ticks
plt.tick_params(direction='in',which='minor',length=5) # set the direction and length of the minor ticks
plt.tick_params(direction='in',which='both',width=2) # set the width of all ticks

plt.xlim(args.xlim) # set x-axis limits
plt.ylim(args.ylim) # set y-axis limits

# Define the axis labels
plt.xlabel('Wavelength ($\mu$m)', fontsize=args.fscale*2, fontweight='bold') #, **hfont) # set x-axis label

if args.lfl:
    ylabel = r'$\lambda F_\lambda$ ({})'
elif args.nfn:
    ylabel = r'$\nu F_\nu$ ({})'
elif args.yunit.is_equivalent('W/(m2 um)'):
    ylabel = r'$F_\lambda$ ({})'
else:
    # must be per frequency
    ylabel = r'$F_\nu$ ({})'

ylabel = ylabel.format(args.yunit.to_string('latex_inline'))
# ugly hack to fix unit order  :(  Need to fix in astropy.units.format.latex
ylabel = ylabel.replace('\\mu m^{-1}\\,cm^{-2}', 'cm^{-2}\\,\\mu m^{-1}')
ylabel = ylabel.replace('\\mu m^{-1}\\,m^{-2}', 'm^{-2}\\,\\mu m^{-1}')
plt.ylabel(ylabel, fontsize=args.fscale*2, fontweight='bold') #, **hfont)  # set y-axis label 

# Set axis to log if flagged.
if args.xlog:
    ax.set_xscale("log", nonposx='clip')
    f = matplotlib.ticker.FuncFormatter(lambda x, pos: '{:.0f}'.format(x))
    ax.xaxis.set_major_formatter(f)
    ax.xaxis.set_minor_formatter(f)
if args.ylog:
    ax.set_yscale("log", nonposx='clip')

# Plot the data
ax.plot(wave.value, spec.value, 'ko', markersize=4) # plot data
ax.plot(wave.value, spec.value, 'w.', markersize=2) # plot data
ax.errorbar(wave.value, spec.value, yerr=unc.value, ecolor='k', fmt='none', capsize=2) # plot error bars

# Set up the colors for the materials in a dictionary
colors = {'ap': 'blue', 'ap50': 'blue', 'ao': 'cyan', 'ao50': 'cyan', 'ac': 'darkorange', 'co': 'green', 'cp': 'magenta', 'other': 'black'}

# Plot the materials
for i, mats in enumerate(materials):
    try:
        colors[mats]
    except KeyError:
        ax.plot(wmodel.value, mcols[i, :].value, color=colors['other'], linestyle=line_dash[i])
    else:
        ax.plot(wmodel.value, mcols[i,:].value, color=colors[mats], linestyle=line_dash[i]) 


# Plot the total model in red
ax.plot(wmodel.value, tmodel.value, color='red')

plt.tight_layout()
plt.draw()

# If chosen, write out PDF file
if args.savefigure:
    fig.savefig("fit-plot-figure-{}.pdf".format(time.strftime('%Y%m%d-%I%M%S')), dpi=300, bbox_inches='tight')
else:
    plt.show()
