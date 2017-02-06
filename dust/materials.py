from abc import ABC, abstractmethod

__all__ = [
    'Solid',
    'FractallyPorous',
    'ConstantPorosity',
    'Material'
]

class PorosityModel(ABC):
    """Abstract base class for porosity models.

    Porosity models define how porosity varies with size.  Derived
    objects must define the `__call__` method.

    """

    @abstractmethod
    def __call___(self, a):
        """Porosity, i.e., vacuum fraction, for grain size `a`."""
        pass

class Solid(PorosityModel):
    """Solid dust."""
    def __init__(self):
        pass
    
    @abstractmethod
    def __call___(self, a):
        """Porosity, i.e., vacuum fraction, for grain size `a`.

        Parameters
        ----------
        a : astropy Quanitity
          Grain radius.  Porosity is 0 for all `a`.

        """

        import astropy.unit as u

        assert isinstance(a, u.Quantity)
        assert a.unit.is_equivalent(u.um)

        # astropy magic returns a Quantity for zeros_like(a).  Use
        # .value to get the magnitude.
        p = np.zeros_like(a).value
        return p

class FractallyPorous(PorosityModel):
    """Fractally porous dust.

    Parameters
    ----------
    a0 : Quantity
      The radius of the smallest grain unit.  Porosity for `a <= a0`
      will be 0.
    D : float
      Fractal dimension.  Must be `0 <= D <= 3.0`.

    """
    def __init__(self, a0, D):
        import astropy.units as u
        
        assert isinstance(a0, u.Quantity)
        assert a0.unit.is_equivalent(u.um)
        assert isinstance(D, (float, int))
        
        self.a0 = a0
        self.D = D
    
    def __call___(self, a):
        """Porosity, i.e., vacuum fraction, for grain size `a`.

        Parameters
        ----------
        a : astropy Quanitity
          Grain radius.

        """

        import astropy.unit as u

        assert isinstance(a, u.Quantity)
        assert a.unit.is_equivalent(u.um)

        p = 1 - (a / self.a0).value**(self.D - 3)
        
        return p

class Material:
    """A single instance of a material.

    Parameters
    ----------
    name : string
      The name of the material.
    rho0 : astropy Quantity
      The bulk material density.
    porosity : PorosityModel, optional
      A description of the porosity as a function of size.
    
    """

    def __init__(self, name, rho0, porosity=Solid()):
        
