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
            tab['chisq'] = self.chisq[0]
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
