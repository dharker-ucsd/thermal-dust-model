#!/usr/bin/python3
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table
from astropy.io import ascii, fits

def plot_columns(fignum, tab, colnames, bins=None, best=None, **kwargs):
    nx = len(colnames) - 1
    ny = len(colnames) - 1
    plt.close(fignum)
    fig, axes = plt.subplots(ny, nx, sharex='col', sharey='row', num=fignum,
                             figsize=(10, 10))

    for x in range(nx):
        for y in range(ny):
            if x > y:
                fig.delaxes(axes[y, x])
                continue

            coly = colnames[y + 1]
            colx = colnames[x]
            param_distribution(axes[y, x], tab[coly], tab[colx], bins, **kwargs)
            
            if best is not None:
                axes[y, x].errorbar(best[colx], best[coly],
                                    (best['+' + coly], best['-' + coly]),
                                    (best['+' + colx], best['-' + colx]),
                                    marker='o', color='w', mfc='none',
                                    mew=1, ecolor='w', elinewidth=1)

            if x == 0:
                axes[y, x].set_ylabel(coly)
            if y == ny - 1:
                axes[y, x].set_xlabel(colx)

    if best is not None:
        s = """{comet spectrum}
Fit date: {run on}
Materials: {materials included}
$r_\\mathrm{{h}}$: {r_h (AU)} au
$\\Delta$: {Delta (AU)} au
$\\chi^2_\\nu$: {rchisq:.2f}
GSD: {GSD}
$a_p$: {a_p}
$D$: {D}""".format(**best.meta)
        fig.text(0.95, 0.98, s, va='top', ha='right', linespacing=1.5)
        
    return fig

def param_distribution(ax, py, px, bins, range=None):
    h, edges = np.histogramdd(np.c_[py, px], bins=bins, range=range)
    extent = (edges[1][0], edges[1][-1], edges[0][0], edges[0][-1])
    ax.imshow(h, extent=extent)

parser = argparse.ArgumentParser(description='Plot parameter correlations based on Monte Carlo fitting.')
parser.add_argument('mcall', help='Name of the file with the Monte Carlo simulations.')
parser.add_argument('prefix', help='File name prefix for plots.')
parser.add_argument('--bins', default=30, help='Number of bins for histograms (default: 30).')
parser.add_argument('--format', default='png', help='File name extension for plot output.')
parser.add_argument('--no-best', action='store_false', dest='best', help='Do not mark the best-fit locations.')
parser.add_argument('--best-fit', help='Use this file for the best-fit values.Do not mark the best-fit locations.  Default is take the MC fit file name and replace "MCALL.fits" with "MCBEST.txt".')

args = parser.parse_args()

tab = Table(fits.getdata(args.mcall))

# get best-fit values, if requested
best = None
if args.best:
    if args.best_fit is None:
        if 'MCALL' in args.mcall:
            best_file = args.mcall.replace('MCALL.fits', 'MCBEST.txt')
        else:
            best_file = ''
    else:
        best_file = args.best_fit

    if os.path.exists(best_file):
        best = ascii.read(best_file)
    else:
        print('Warning: Best-fit file not found: {}'.format(best_file))

# mass fractions
colnames = [c for c in tab.colnames if c.startswith('f(')]
fig = plot_columns(1, tab, colnames, bins=args.bins, best=best)
fig.savefig('{}-mass-v-mass.{}'.format(args.prefix, args.format))

# composition classes, only plot if they are available
colnames = []
for col in ['f(ac)', 'AS', 'CS', 'fcryst']:
    if col in tab.colnames:
        colnames.append(col)

if len(colnames) > 1:
    fig = plot_columns(2, tab, colnames, bins=args.bins, best=best)
    fig.savefig('{}-comp-v-comp.{}'.format(args.prefix, args.format))

