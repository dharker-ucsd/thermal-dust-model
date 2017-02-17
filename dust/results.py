import numpy as np

class ModelResults:
    """Model results and calculated parameters.

    Parameters
    ----------
    materials : list of Material
      List of materials used in the fit.
    scales : array-like
      Grain size distribution scale factors for each material.

    """

    def __init__(self, materials, scales, gsd=None):
        from .materials import Material

        assert all([isinstance(m, Material) for m in materials])
        assert all([isinstance(s, (int, float)) for s in scales])
        assert len(materials) == len(scales)

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

        m = np.zeros(len(self.materials))

        if ar[0] == ar[1]:
            return m
        
        for i in range(len(m)):
            m[i] = self.scale[i] * self.material[i].total_mass(ar)

        return m

    def mass_fraction(self, ar):
        """Mass fraction of each material for the given size range.

        Parameters
        ----------
        ar : array, optional
          Consider grain radii from `ar[0]` to `ar[1]`.  Unit: μm.

        """

        m = self.mass(ar)
        return m / m.sum()
