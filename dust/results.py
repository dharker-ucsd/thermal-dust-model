import numpy as np

class ModelResults:
    """Model results and calculated parameters.

    Use `ModelResults` for fits of a particular model instance to a
    single spectrum, possibly including a Monte Carlo uncertainty
    analysis.

    Parameters
    ----------
    dust : list of Dust
      List of dust collections used in the fit.
    scales : array-like
      Grain size distribution scale factors for each dust collection.
      May be an array with the same length as `dust`, or, to hold
      multiple sets of scale factors, a 2-dimensional array with size
      `N x len(dust)` (same as `fit.mcfit` results).
    rchisq : array-like, optional
      The reduced chi-squared statistic for each set of scales.
    dof : int, optional
      The number of degrees of freedom.

    """

    def __init__(self, dust, scales, rchisq=None, dof=None):
        from .materials import Dust

        assert all([isinstance(d, Dust) for d in dust])
        self.dust = dust

        self.scales = np.array(scales)
        assert self.scales.ndim in [1, 2], '`scales` must have 1 or 2 dimensions.'
        if self.scales.ndim == 1:
            assert len(self.scales) == len(dust)
        else:
            assert self.scales.shape[1] == len(dust)

        self.rchisq = self._dimension_check(rchisq)
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

    def table(self, arange=(0.1, 1), ratios={}):
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
        from materials import MaterialType

        if ratios is None:
            ratios = OrderedDict()
            ratios['AS'] = ([MaterialType.AMORPHOUS, MaterialType.SILICATE], [MaterialType.DUST])
            ratios['CS'] = ([MaterialType.CRYSTALLINE, MaterialType.SILICATE], [MaterialType.SILICATE])
            ratios['fcryst'] = ([MaterialType.CRYSTALLINE, MaterialType.SILICATE], [MaterialType.DUST])
            ratios['S/C'] = ([MaterialType.SILICATE], [MaterialType.CARBONACEOUS])

        for r in ratios:
            for types in r:
                assert all([isinstance(t, MaterialType) for t in types]), '`ratios` must be a list of two-element arrays.  The elements are lists of the types to use in the numerator and denominator.'

        Ndust = len(self.dust)
        Nsca = len(self.scales)
        m = self.total_mass(arange)
        if m.ndim == 1:
            m = m.sum()
        else:
            m = m.sum(-1)

        # Compute the mass fraction of each dust collection
        f = self.mass_fraction(arange)

        # Name of the scale factors (Nps)
        names = ['s({})'.format(d.material.abbrev) for d in self.dust]
        # Total mass
        names += ['M_total']
        # Name of the mass fractions
        names += ['f({})'.format(d.material.abbrev) for d in self.dust]

        # Nps, total mass, and mass fractions
        data = [self.scales, self._dimension_check(m), f]

        meta = OrderedDict()
        for d in enumerate(self.dust):
            meta[d.material.abbrev] = d.material.name

        if self.dof is not None:
            meta['dof'] = self.dof

        meta['Radius range for masses'] = arange

        # Calculate and include ratios, if requested
        if len(ratios) > 0:
            ratio_names = []
            if f.ndim == 1:
                mass_ratios = np.zeros(len(ratios))
            else:
                mass_ratios = np.zeros((len(ratios), f.shape[0]))

            for i, (name, equation) in enumerate(ratios.items()):
                meta[name] = '{} / {}'.format(' '.join(equation[0]),
                                              ' '.join(equation[1])).lower()
                numerator = 0
                denominator = 0
                for j, d in enumerate(self.dust):
                    if all([m in d.material.mtype for m in equation[0]]):
                        numerator = numerator + f[j]
                    if all([m in d.material.mtype for m in equation[1]]):
                        denominator = denominator + f[j]

                names.append(name)
                mass_ratios[i] = numerator / denominator
            
            data.append(mass_ratios)
        
        # Last column is rchisq, if available
        if self.rchisq is not None:
            names.append('rchisq')
            data.append(self.rchisq)

        # Assmble all columns from the data sources
        data = np.hstack(data)

        # Define the table with headers and data
        tab = Table(names=names, data=data, meta=meta)
        
        return tab

    def total_mass(self, arange):
        """Total mass of each dust collection for the given size range.

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

        m = np.zeros(len(self.dust))
        for i in range(len(m)):
            m[i] = self.dust[i].total_mass(arange)

        return self.scales * m

    def mass_fraction(self, ar):
        """Mass fraction of each dust collection for the given size range.

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
