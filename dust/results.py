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

    def mass(self, alim):
        """Total mass of each material for the given size range.

        Parameters
        ----------
        alim : array, optional
          Consider grain radii from `alim[0]` to `alim[1]`.  Unit: μm.

        """

        from .util import avint

        assert alim[0] <= alim[1]

        m = np.zeros(len(self.materials))

        if alim[0] == alim[1]:
            return m
        
        for i in range(len(m)):
            m[i] = self.scale[i] * self.material[i].mass(alim)

        return m

    def mass_fraction(self, alim):
        """Mass fraction of each material for the given size range.

        Parameters
        ----------
        alim : array, optional
          Consider grain radii from `alim[0]` to `alim[1]`.  Unit: μm.

        """

        m = self.mass(alim)
        return m / m.sum()
