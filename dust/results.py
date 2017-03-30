import numpy as np

class ModelResults:
    """Model results and calculated parameters.

    Use `ModelResults` for fits of a particular model instance to a
    single spectrum, possibly including a Monte Carlo uncertainty
    analysis.

    Parameters
    ----------
    materials : list of Material
      List of materials used in the fit.
    scales : array-like
      Grain size distribution scale factors for each material.  May be
      an array with the same length as `materials`, or, to hold
      multiple sets of scale factors, a 2-dimensional array with size
      `N x len(materials)` (same as `fit.mcfit` results).
    rchisq : array-like, optional
      The reduced chi-squared statistic for each set of scales.
    dof : int, optional
      The number of degrees of freedom.

    """

    def __init__(self, materials, scales, rchisq=None, dof=None):
        from .materials import Material

        assert all([isinstance(m, Material) for m in materials])
        self.materials = materials

        self.scales = np.array(scales)
        assert self.scales.ndim in [1, 2], '`scales` must have 1 or 2 dimensions.'
        if self.scales.ndim == 1:
            assert len(self.scales) == len(materials)
        else:
            assert self.scales.shape[1] == len(materials)

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

    def table(self, ar=(0.1, 1)):
        """Results summarized as a table."""

        from astropy.table import Table

        Nmat = len(self.materials)
        Nsca = len(self.scales)
        m = self.total_mass(ar)
        if m.ndim == 1:
            m = m.sum()
        else:
            m = m.sum(-1)

        f = self.mass_fraction(ar)

        names = ['s{}'.format(i) for i in range(Nmat)]
        names += ['Mtot']
        names += ['f{}'.format(i) for i in range(Nmat)]
        d = [self.scales, self._dimension_check(m), f]

        if self.rchisq is not None:
            names.append('rchisq')
            d.append(self.rchisq)

        data = np.hstack(d)

        tab = Table(names=names, data=data)
        for i, m in enumerate(self.materials):
            tab.meta['material {}'.format(i)] = m.name

        if self.dof is not None:
            tab.meta['dof'] = self.dof

        tab.meta['Radius range for masses'] = ar

        return tab

    def total_mass(self, ar):
        """Total mass of each material for the given size range.

        Parameters
        ----------
        ar : array, optional
          Consider grain radii from `ar[0]` to `ar[1]`.  Unit: μm.

        """

        from .util import avint

        assert np.iterable(ar)
        assert len(ar) == 2
        assert ar[0] <= ar[1]

        if ar[0] == ar[1]:
            return m

        m = np.zeros(len(self.materials))
        for i in range(len(m)):
            m[i] = self.materials[i].total_mass(ar)

        return self.scales * m

    def mass_fraction(self, ar):
        """Mass fraction of each material for the given size range.

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
