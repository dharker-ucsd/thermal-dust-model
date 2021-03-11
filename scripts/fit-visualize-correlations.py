#!/usr/bin/env python3
import os
from copy import copy
import argparse
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from astropy.table import Table
from astropy.io import ascii, fits

AXIS_LABELS = {
    'f(ac)': 'Amor. Carbon',
    'f(ao50)': 'Amor. Olivine',
    'f(ap50)': 'Amor. Pyroxene',
    'f(co)': 'Cryst. Olivine',
    'f(cp)': 'Cryst. Pyroxene',
    'AS': 'Amor. Silicate',
    'CS': 'Cryst. Silicate',
    'fcryst': '$f_{\mathrm{cryst}}$'
}


def plot_columns(fignum, tab, colnames, bins=None, best=None, image=True,
                 contours=False, title=None, **kwargs):
    n = len(colnames) - 1
    plt.close(fignum)
    fig, axes = plt.subplots(n, n, sharex='col', sharey='row', num=fignum,
                             figsize=(11, 8), gridspec_kw={'right': 8 / 11})

    if best is None:
        d = [0, 1]
    else:
        # get 95% CL ranges
        ll = [best[col] - best['-' + col] for col in colnames]
        ul = [best[col] + best['-' + col] for col in colnames]
        d = np.floor(np.min(ll) * 10) / 10, np.ceil(np.max(ul) * 10) / 10

    if kwargs['range']:
        d = kwargs['range']
    del kwargs['range']

    # xcolnames = colnames[:n]
    # ycolnames = colnames[:n - 1][::-1] + colnames[-1:]

    for x in range(n):
        for y in range(n):
            if x > y:
                if (y, x) == (0, 1):
                    # save this position
                    pos = axes[y, x].get_position()
                fig.delaxes(axes[y, x])
                continue

            coly = colnames[y + 1]
            colx = colnames[x]

            param_distribution(axes[y, x], tab[coly],
                               tab[colx], bins, range=[d, d], **kwargs)

            if best is not None:
                axes[y, x].errorbar(best[colx], best[coly],
                                    (best['-' + coly], best['+' + coly]),
                                    (best['-' + colx], best['+' + colx]),
                                    marker='o', color='k', mfc='none',
                                    mew=1, ecolor='k', elinewidth=0.5)

            if axes[y, x].is_first_col():
                axes[y, x].set_ylabel(AXIS_LABELS[coly])
            if axes[y, x].is_last_row():
                axes[y, x].set_xlabel(AXIS_LABELS[colx])

            plt.setp(axes[y, x], xlim=d, ylim=d)

    if best is not None:
        if title is None:
            title = "{comet spectrum}\nFit date: {run on}".format(
                **best.meta)

        materials = ', '.join([m.upper()
                               for m in best.meta['materials included']])

        s = ("""{title}
Materials: {materials}
Modules
$r_\\mathrm{{h}}$ = {r_h (AU)} au
$\\Delta$ = {Delta (AU):.1f} au
$\\chi^2_\\nu$ = {rchisq:.2f}
$a_p$ = {a_p}
$N$ = {N}
$D$ = {D}""".format(title=title, N=best.meta['GSD'].split()[1],
                    materials=materials, **best.meta)
             .replace(' um', ' μm'))
        fig.text(0.95, 0.96, s, va='top', ha='right', linespacing=1.5, size=16)

    for ax in axes.ravel():
        ax.minorticks_on()
        ax.tick_params(axis='both', which='major', labelsize=12)
        ax.tick_params(axis='both', which='minor', labelsize=9)
        plt.setp((ax.xaxis.get_label(), ax.yaxis.get_label()),
                 fontsize=14)

    # fig.tight_layout()
    return fig, pos


def plot_spectrum(fig, spec, model):
    last_ax = fig.axes[-1]
    bbox = fig.transFigure.inverted().transform(last_ax.bbox)
    ax = fig.add_axes(
        (bbox[1, 0] + 0.65 / 11, bbox[0, 1], 2.2 / 11, bbox[1, 1] - bbox[0, 1]))

    ac = np.interp(spec['wave'], model['wave'], model['F(ac)'])
    ax.scatter(spec['wave'], spec['fluxd'] / ac, marker='.')

    colors = {'ap': 'blue', 'ap50': 'blue', 'ao': 'cyan', 'ao50': 'cyan',
              'ac': 'darkorange', 'co': 'green', 'cp': 'magenta', 'other': 'black'}

    # Plot the materials
    for mat in model.meta['materials included'] + ['total']:
        offset = 0 if mat in ['total', 'ac'] else 0
        ax.plot(
            model['wave'],
            model[f'F({mat})'] / model['F(ac)'] + offset,
            color=colors.get(mat, colors['other']))

    xlim = (np.floor(spec['wave'].min() * 0.8),
            np.ceil(spec['wave'].max() * 1.2))

    ylim = (
        # spec['fluxd'][spec['fluxd'] > 0].min() / 30,
        0.9,
        np.ceil((np.median(spec['fluxd'] / ac - 1) * 3 + 1) * 10) / 10
    )
    ax.minorticks_on()
    plt.setp(ax, xlim=xlim, yscale='linear', ylim=ylim)


def param_distribution(ax, py, px, bins, range=None, image=True, contours=False):
    h, edges = np.histogramdd(np.c_[py, px], bins=bins, range=range)
    extent = (edges[1][0], edges[1][-1], edges[0][0], edges[0][-1])
    cm = copy(plt.cm.viridis)
    cm.set_bad('w')
    if image:
        ax.imshow(h, extent=extent, origin='lower',
                  cmap=cm, norm=mpl.colors.LogNorm())
    if contours:
        ax.contour(h, extent=extent, origin='lower')


parser = argparse.ArgumentParser(
    description='plot parameter correlations based on Monte Carlo fitting')
parser.add_argument(
    'mcall', help='name of the file with the Monte Carlo simulations')
parser.add_argument('prefix', help='file name prefix for plots')
parser.add_argument('--bins', type=int, default=30,
                    help='number of bins for histograms (default: 30)')
parser.add_argument('--format', default='png',
                    help='file name extension for plot output')
parser.add_argument('--no-best', action='store_false',
                    dest='best', help='do not mark the best-fit locations')
parser.add_argument(
    '--best-fit', help='use this file for the best-fit values; default is take the MC fit file name and replace "MCALL.fits" with "MCBEST.txt"')
parser.add_argument(
    '--best-model', help='use this file for the best model spectrum; default is to replace "MCALL.fits" with "BESTMODEL.txt"'


)
parser.add_argument(
    '--spectrum', help='use this file for the comet spectrum; default is to replace "MCALL.fits" with "TOFIT.txt"'
)
parser.add_argument('--contours', action='store_true',
                    help='plot contours')
parser.add_argument('--no-image', action='store_false',
                    dest='image', help='do not plot the 2D histogram as an image')
parser.add_argument(
    '--title', help="use this title instead of file name, date, and materials")
parser.add_argument('--range', help='use this range for all axes')
parser.add_argument('--parameters', default='f(ao50),f(ap50),AS,CS,f(ac)',
                    help='plot these parameters, comma-separated')

args = parser.parse_args()

if not (args.image + args.contours):
    raise ValueError('No image and no contours means nothing to plot.')

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

spec = None
if args.spectrum:
    spec = ascii.read(args.spectrum)
else:
    spec = ascii.read(args.mcall.replace("MCALL.fits", "TOFIT.txt"))

model = None
if args.best_model:
    model = ascii.read(args.best_model)
else:
    model = ascii.read(args.mcall.replace("MCALL.fits", "BESTMODEL.txt"))

opts = dict(bins=args.bins, best=best, title=args.title,
            range=None, image=args.image, contours=args.contours)
if args.range:
    opts['range'] = [float(x) for x in args.range.split(',')]

# only plot if they are available
colnames = []
for col in args.parameters.split(','):
    if col in tab.colnames:
        colnames.append(col)
    else:
        print(col, 'not in file')

if len(colnames) > 1:
    fig, axes = plot_columns(1, tab, colnames, **opts)
    plot_spectrum(fig, spec, model)
    fig.savefig('{}-comp-v-comp.{}'.format(args.prefix, args.format))
