import numpy as np

class ModelResults:
    """Model results and calculated parameters.

    Use `ModelResults` for fits of a particular model instance to a
    single spectrum, possibly including a Monte Carlo uncertainty
    analysis.

    Parameters
    ----------
    grains : list of Grains
      List of Grains used in the fit.
    scales : array-like
      Grain size distribution scale factors for `grains`.  May be an
      array with the same length as `grains`, or, to hold multiple
      sets of scale factors, a 2-dimensional array with size `N x
      len(grains)` (same as `fit.mcfit` results).
    chisq : array-like, optional
      The chi-squared statistic for each set of scales.
    dof : int, optional
      The number of degrees of freedom.

    """

    def __init__(self, grains, scales, chisq=None, dof=None):
        from .materials import Grains

        assert all([isinstance(g, Grains) for g in grains])
        self.grains = grains

        self.scales = np.array(scales)
        assert self.scales.ndim in [1, 2], '`scales` must have 1 or 2 dimensions.'
        if self.scales.ndim == 1:
            assert len(self.scales) == len(grains)
        else:
            assert self.scales.shape[1] == len(grains)

        self.chisq = self._dimension_check(chisq)
        self.dof = dof

    def _dimension_check(self, a):
        if a is None:
            return a

        a = np.array(a)
        assert a.ndim in [0, 1, 2]

        while a.ndim < self.scales.ndim:
            # add an axis to the end
            a = a[..., np.newaxis]

        if a.ndim != self.scales.ndim:
            raise ValueError("Parameter has too many dimensions.")

        return a

    def table(self, arange=(0.1, 1), ratios=None):
        """Results summarized as a table.

        Parameters
        ----------
        arange : two-element array
          The grain size range in μm.
        ratios : dictionary of lists
          A dictionary of ratios to compute, or `None to use the
          defaults.  The dictionary keys are used as column headings.
          Each value is a two-element list of material types to use as
          the numerator and denominator.  For example, to compute the
          silicate to carbon ratio, use:

            {'S/C': ([MaterialType.SILICATE], [MaterialType.CARBONACEOUS])}

          For the crystalline silicate to total silicate ratio:

            {'fcryst': ([MaterialType.CRYSTALLINE, MaterialType.SILICATE], [MaterialType.SILICATE])}

          To compute the total amorphous silicate dust mass fraction:

            {'AS': ([MaterialType.AMORPHOUS, MaterialType.SILICATE], [MaterialType.DUST])}

          Use `MaterialType` to match any material:

            {'fice': ([MaterialType.ICE, MaterialType])}

        """

        from collections import OrderedDict
        from astropy.table import Table
        from .materials import MaterialType

        if ratios is None:
            ratios = OrderedDict()
            ratios['AS'] = ([MaterialType.AMORPHOUS, MaterialType.SILICATE], [MaterialType.DUST])
            ratios['CS'] = ([MaterialType.CRYSTALLINE, MaterialType.SILICATE], [MaterialType.DUST])
            ratios['S/C'] = ([MaterialType.SILICATE], [MaterialType.CARBONACEOUS])
            ratios['fcryst'] = ([MaterialType.CRYSTALLINE, MaterialType.SILICATE], [MaterialType.SILICATE])

        for r in ratios.values():
            for types in r:
                assert all([isinstance(t, MaterialType) for t in types]), '`ratios` must be a list of two-element arrays.  The elements are lists of the types to use in the numerator and denominator.'

        m = self.total_mass(arange)  # grams
        if m.ndim == 1:
            m = m.sum()
        else:
            m = m.sum(-1)

        # Compute the mass fraction of each grain collection
        f = self.mass_fraction(arange)

        # Name of the scale factors (Nps)
        names = ['s({})'.format(g.material.abbrev) for g in self.grains]
        # Total mass
        names += ['Mtot']
        # Name of the mass fractions
        names += ['f({})'.format(g.material.abbrev) for g in self.grains]

        # Nps, total mass, and mass fractions
        data = [self.scales, self._dimension_check(m), f]

        # Assmble all columns from the data sources
        data = np.hstack(data)

        meta = OrderedDict()
        for g in self.grains:
            meta[g.material.abbrev] = g.material.name

        if self.dof is not None:
            meta['dof'] = self.dof

        meta['Radius range for masses'] = arange

        # Define inital table content with headers and data
        tab = Table(names=names, data=data, meta=meta)

        # add column meta data
        for g in self.grains:
            m = '{}'.format(g.material.abbrev)
            tab['s({})'.format(m)].description = 'Grain size distribution scale factor for component {}'.format(m)
            tab['f({})'.format(m)].description = 'Mass fraction for component {} over requested grain size range'.format(m)
            
        tab['Mtot'].unit = 'g'
        tab['Mtot'].description = 'Total grain mass over requested grain size range'

        # Calculate and include ratios, if requested
        if len(ratios) > 0:
            for i, (name, equation) in enumerate(ratios.items()):
                numerator = 0
                denominator = 0
                for j, g in enumerate(self.grains):
                    if all([m in g.material.mtype for m in equation[0]]):
                        numerator = numerator + f[..., j]
                    if all([m in g.material.mtype for m in equation[1]]):
                        denominator = denominator + f[..., j]

                tab[name] = numerator / denominator

                numerator_names = [x.value for x in equation[0]]
                denominator_names = [x.value for x in equation[1]]
                tab[name].description = '{} / {}'.format(
                    ' '.join(numerator_names), ' '.join(denominator_names))
            
        # Last column is chisq, if available
        if self.chisq is not None:
            if len(tab) == 1:
                tab['chisq'] = self.chisq[0]
            else:
                tab['chisq'] = self.chisq
            tab['chisq'].description = 'chi-squared statistic'        
        
        return tab

    def total_mass(self, arange):
        """Total mass of Grains for the given size range.

        Parameters
        ----------
        arange : array, optional
          Consider grain radii from `arange[0]` to `arange[1]`.  Unit: μm.

        """

        from .util import avint

        assert np.iterable(arange)
        assert len(arange) == 2
        assert arange[0] <= arange[1]

        if arange[0] == arange[1]:
            return m

        m = np.zeros(len(self.grains))
        for i in range(len(m)):
            m[i] = self.grains[i].total_mass(arange)

        return self.scales * m

    def mass_fraction(self, ar):
        """Mass fraction of Grains for the given size range.

        Parameters
        ----------
        ar : array, optional
          Consider grain radii from `ar[0]` to `ar[1]`.  Unit: μm.

        """

        m = self.total_mass(ar)
        if m.ndim == 1:
            f = m / m.sum()
        else:
            f = (m.T / m.sum(-1)).T

        return f

class ModelSpectra:
    """Model spectra.

    Use `ModelSpectra` to hold the model spectra of a set of
    materials.

    Parameters
    ----------
    wave : Quantity
      The spectral wavelengths.
    fluxd : array of (key, value) pairs
      Key-value pairs providing the material name (key) and the flux
      density at each `wave` (value : Quantity).
    meta : dictionary-like, optional
      Meta data for the spectrum.

    """

    default_colors = {
        'ap': 'blue',
        'ap50': 'blue',
        'ao': 'cyan',
        'ao50': 'cyan',
        'ac': 'darkorange',
        'co': 'green',
        'cp': 'magenta',
        'total': 'red',
        'other': 'black'
    }

    def __init__(self, wave, fluxd, meta=None):
        from collections import OrderedDict
        from astropy.table import Table
        import astropy.units as u

        self.meta = meta if meta is not None else OrderedDict()

        self.table = Table(names=['wave'] + [k for k, v in fluxd],
                           data=[wave] + [v for k, v in fluxd])
        self.table.meta = meta if meta is not None else OrderedDict()

    def __getitem__(self, k):
        import astropy.units as u
        if k == 'total' and 'total' not in self.table:
            t = 0
            for col in self.table.colnames:
                if col == 'wave':
                    continue
                t += u.Quantity(self.table[col].data, self.table[col].unit)
            return t
        else:
            return u.Quantity(self.table[k].data, self.table[k].unit)

    @property
    def materials(self):
        return [k for k in self.table.colnames if k not in ['wave', 'total']]

    @classmethod
    def from_fitidlsave(cls, filename):
        """Read model spectra from a fit-idl-save file."""

        from astropy.io import ascii
        import astropy.units as u

        tab = ascii.read(filename)
        wave = u.Quantity(tab['wave'], tab['wave'].unit)
        fluxd = []
        for col in tab.colnames:
            if col == 'wave':
                continue
            f = u.Quantity(tab[col], tab[col].unit)
            fluxd.append((col[2:-1], f))

        return cls(wave, fluxd, meta=tab.meta)

    def plot(self, ax=None, materials=None, unconstrained=None, colors=None,
             wunit='um', unit=None, total=True):
        """Plot spectra.

        Parameters
        ----------
        ax : matplotlib Axis, optional
          Plot to this matplotlib axis.
        materials : list, optional
          Only plot these materials.
        unconstrained : list, optional
          These materials are not detected, but should be plot with a
          dashed line.
        colors : dict, optional
          `(material, color)` pairs.
        wunit : str or astropy Unit
          Convert wavelength this unit.
        unit : str or astropy Unit, optional
          Convert flux density to this unit.
        total : bool, optional
          Set to `True` to also plot the total.

        Returns
        -------
        lines : list
          The results from `ax.plot`.

        """

        import matplotlib.pyplot as plt
        import astropy.units as u
        
        ax = ax if ax is not None else plt.gca()
        colors = {} if colors is None else colors
        materials = self.materials if materials is None else materials
        unconstrained = [] if unconstrained is None else unconstrained

        w = self['wave'].to(wunit)
        lines = []
        for k in materials:
            color = colors.get(k, self.default_colors.get(k, self.default_colors['other']))
            ls = 'dashed' if k in unconstrained else 'solid'
            if unit is None:
                f = self[k]
            else:
                f = self[k].to(unit, u.spectral_density(w))
            lines.append(ax.plot(w, f, color=color, ls=ls))

        if total:
            color = colors.get('total', self.default_colors['total'])
            if unit is None:
                f = self['total']
            else:
                f = self['total'].to(unit, u.spectral_density(w))
            lines.append(ax.plot(w, f, color=color))
            
        return lines
