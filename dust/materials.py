from abc import ABC, abstractmethod
from enum import Enum
import numpy as np

__all__ = [
    'Solid',
    'FractallyPorous',
    'ConstantPorosity',
    'HannerGSD',
    'PowerLaw',
    'MaterialType',
    'Material',
    'amorphous_olivine50',
    'amorphous_pyroxene50',
    'amorphous_carbon',
    'hot_forsterite95',
    'hot_ortho_enstatite',
    'Grains',
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

class GrainSizeDistribution(ABC):
    """Abstract base class for differential grain size distributions.

    GSD models define the differential GSD for particular materials.  
    Derived objects must define the `__call__` method.

    """
    def __init__(self, alim=[0, np.inf]):
        self.alim = alim

    def __call__(self, a):
        """Differential size distribution evaluated at radius `a`.
        ...
        """

        dnda = self._n(a)
        if np.iterable(a):
            if any(a < self.alim[0]):
                dnda[a < self.alim[0]] = 0.
            if any(a > self.alim[1]):
                dnda[a > self.alim[1]] = 0.
            return dnda
        else:
            if a < self.alim[0] or a > self.alim[1]:
                dnda = 0
            return dnda

    @abstractmethod
    def _n(self, a):
        """Differential grain size distribution relative fraction for grain size `a`."""
        pass
        
class HannerGSD(GrainSizeDistribution):
    """Hanner (modified power law) differential grain size distribution.

    The HannerGSD is normalized by the peak of the GSD curve.
    
      `dn/da = (1 - a0 / a)**M * (a0 / a)**N`

    Parameters
    ----------
    a0 : float
      The radius of the smallest grain unit in μm.  GSD for `a <= a0`
      will be < 0.
    N : float
      Large grain slope
    M : float
      Small grain slope
    alim : Two element array
      Upper and lower grain radii in microns

    """
    def __init__(self, a0, N, M, alim=[0.1,100]):
        self.a0 = a0
        self.N = N
        self.M = M
        self.alim = alim

    @property
    def ap(self):
        """Peak of the GSD."""
        from .util import hanner_ap
        return hanner_ap(self.a0, self.N, self.M)

    def _n(self, a):
        """GSD, i.e., relative amount, for grain size `a`.

        Parameters
        ----------
        a : float or array
          Grain radius in μm.

        """
        from .util import hanner_gsd
        return hanner_gsd(a, self.a0, self.N, self.M)
        
class PowerLaw(GrainSizeDistribution):
    """Power Law differential grain size distribution.

    The GSD is normalized by the smallest grain size.

      `dn/da = (a0 / a)**powlaw`
    
    Parameters
    ----------
    a0 : float
      The radius of the smallest grain unit in μm.  GSD for `a <= a0`
      will be < 0.
    powlaw : float
      The power for the GSD.  It is a POSITIVE number when the slope
      is negative, i.e., `dn/da` is proportional to `a**-powlaw`.
    alim : Two element array
      Upper and lower grain radii in microns

    """
    def __init__(self, a0, powlaw, alim=[0.1,100]):
        self.a0 = a0
        self.powlaw = powlaw
        self.alim = alim
    
    def _n(self, a):
        """GSD, i.e., relative amount, for grain size `a`.

        Parameters
        ----------
        a : float or array
          Grain radius in μm.

        """
        from .util import power_law
        return power_law(a, self.a0, self.powlaw)

class MaterialType(Enum):
    AMORPHOUS = 'amorphous'
    CRYSTALLINE = 'crystalline'
    SILICATE = 'silicate'
    CARBONACEOUS = 'carbonaceous'
    DUST = 'dust'
    ICE = 'ice'
    
class Material:
    """Bulk material properties.

    Parameters
    ----------
    name : string
      The name of the material.
    abbrev : string
      An unqiue abbreviation for the material.
    rho0 : float
      The bulk material density in g/cm3.
    mtype : tuple of MaterialTypes, optional
      Type of material.
    
    """

    def __init__(self, name, abbrev, rho0, mtype=None):
        self.name = name
        self.abbrev = abbrev
        self.rho0 = rho0
        assert isinstance(mtype, (list, tuple))
        self.mtype = tuple() if mtype is None else tuple(mtype)

amorphous_olivine50 = Material(
    'Amorphous olivine Mg/Fe 50/50', 'ao50', 3.3,
    (MaterialType.AMORPHOUS, MaterialType.SILICATE, MaterialType.DUST))

amorphous_pyroxene50 = Material(
    'Amorphous pyroxene Mg/Fe 50/50', 'ap50', 3.3,
    (MaterialType.AMORPHOUS, MaterialType.SILICATE, MaterialType.DUST))

amorphous_carbon = Material(
    'Amorphous carbon', 'ac', 2.5,
    (MaterialType.AMORPHOUS, MaterialType.CARBONACEOUS, MaterialType.DUST))

hot_forsterite95 = Material(
    'Hot forsterite 95', 'co', 3.3,
    (MaterialType.CRYSTALLINE, MaterialType.SILICATE, MaterialType.DUST))

hot_ortho_enstatite = Material(
    'Hot ortho-enstatite', 'cp', 3.3,
    (MaterialType.CRYSTALLINE, MaterialType.SILICATE, MaterialType.DUST))

class Grains:
    """A grain, or collection thereof.

    Parameters
    ----------
    material : Material
      The bulk material of which the dust is composed.
    porosity : PorosityModel, optional
      A description of the porosity as a function of size.
    gsd : GrainSizeDistribution, optional
      The differential grain size distribution as a function of size.
    
    """

    def __init__(self, material, porosity=Solid(), gsd=PowerLaw(1.,0)):
        assert isinstance(material, Material)
        assert isinstance(porosity, PorosityModel)
        assert isinstance(gsd, GrainSizeDistribution)
        self.material = material
        self.porosity = porosity
        self.gsd = gsd

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
        from .util import mass

        rho = self.material.rho0 * (1 - self.porosity(a))
        
        return mass(a, rho)

    def total_mass(self, arange):
        """Total mass over a range of radii weighted by the GSD.
        
        Parameters
        ----------
        arange : two element array
          Lower and upper grain radii range over which to compute mass.
          
        """
        
        from numpy import pi
        from .util import avint

        assert np.iterable(arange)
        assert len(arange) == 2
        
        log_arange = np.log10(arange)
        n = max(log_arange.ptp(), 1) * 10000
        a = np.logspace(log_arange[0], log_arange[1], n)
        dmda = self.gsd(a) * self.mass(a)
        
        return avint(a, dmda, arange)

