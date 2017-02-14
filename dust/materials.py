from abc import ABC, abstractmethod
import numpy as np

__all__ = [
    'Solid',
    'FractallyPorous',
    'ConstantPorosity',
    'Material',
    'AmorphousOlivine50',
    'AmorphousPyroxene50',
    'AmorphousCarbon',
    'HotForsterite95',
    'HotOrthoEnstatite',
]

class PorosityModel(ABC):
    """Abstract base class for porosity models.

    Porosity models define how porosity varies with size.  Derived
    objects must define the `__call__` method.

    """

    @abstractmethod
    def __call__(self, a):
        """Porosity, i.e., vacuum fraction, for grain size `a`."""
        pass

class Solid(PorosityModel):
    """Solid dust."""
    def __init__(self):
        pass
    
    def __call__(self, a):
        """Porosity, i.e., vacuum fraction, for grain size `a`.

        Parameters
        ----------
        a : float or array
          Grain radius in μm.  Porosity is 0 for all `a`.

        """
        return np.zeros_like(a)

class FractallyPorous(PorosityModel):
    """Fractally porous dust.

    Parameters
    ----------
    a0 : Quantity
      The radius of the smallest grain unit in μm.  Porosity for `a <= a0`
      will be 0.
    D : float
      Fractal dimension.  Must be `0 <= D <= 3.0`.

    """
    def __init__(self, a0, D):
        self.a0 = a0
        self.D = D
    
    def __call__(self, a):
        """Porosity, i.e., vacuum fraction, for grain size `a`.

        Parameters
        ----------
        a : float or array
          Grain radius in μm.

        """
        return 1 - (a / self.a0)**(self.D - 3)

class ConstantPorosity(PorosityModel):
    """Uniform porosity for all grain sizes.

    Parameters
    ----------
    p : float
      The porosity to use for all grain sizes.

    """
    def __init__(self, p):
        self.p = p
    
    def __call__(self, a):
        """Porosity, i.e., vacuum fraction, for grain size `a`.

        Parameters
        ----------
        a : float or array
          Grain radius in μm.

        """
        return self.p * np.ones_like(a)

class Material:
    """A single instance of a material.

    Parameters
    ----------
    name : string
      The name of the material.
    rho0 : float
      The bulk material density in g/cm3.
    porosity : PorosityModel, optional
      A description of the porosity as a function of size.
    
    """

    def __init__(self, name, rho0, porosity=Solid()):
        self.name = name
        self.rho0 = rho0
        self.porosity = porosity

    def mass(self, a):
        """Grain mass for radius `a`.

        Parameters
        ----------
        a : float or array
          Grain radius in μm.

        Returns
        -------
        m : float or ndarray
          Grain mass in g.

        """
        from numpy import pi
        return 4e-12 / 3 * pi * a**3 * self.rho0 * (1 - self.porosity(a))

class AmorphousOlivine50(Material):
    """Amorphous olivine, Mg/Fe = 50/50.

    Parameters
    ----------
    porosity : PorosityModel, optional
      A description of the porosity as a function of size.

    """

    def __init__(self, porosity=Solid()):
        Material.__init__(self, 'Amorphous olivine 50/50', 3.3,
                          porosity=porosity)

class AmorphousPyroxene50(Material):
    """Amorphous pyroxene, Mg/Fe = 50/50.

    Parameters
    ----------
    porosity : PorosityModel, optional
      A description of the porosity as a function of size.

    """

    def __init__(self, porosity=Solid()):
        Material.__init__(self, 'Amorphous pyroxene 50/50', 3.3,
                          porosity=porosity)

class AmorphousCarbon(Material):
    """Amorphous carbon.

    Parameters
    ----------
    porosity : PorosityModel, optional
      A description of the porosity as a function of size.

    """

    def __init__(self, porosity=Solid()):
        Material.__init__(self, 'Amorphous carbon', 2.5,
                          porosity=porosity)

class HotForsterite95(Material):
    """Mg-rich olivine (Fo95), hot crystal model."""

    def __init__(self):
        Material.__init__(self, 'Hot forsterite 95', 3.3, porosity=Solid())

class HotOrthoEnstatite(Material):
    """Mg-rich ortho enstantite, hot crystal model."""

    def __init__(self):
        Material.__init__(self, 'Hot ortho-enstatite', 3.3, porosity=Solid())
