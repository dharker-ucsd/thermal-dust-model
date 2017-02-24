import numpy as np
from .. import materials as mat
from ..util import avint

class TestPorosity:
    def test_solid(self):
        a = np.logspace(-1, 2)
        porosity = mat.Solid()
        assert np.all(porosity(a) == 0)

    def test_fractally_porous(self):
        a = np.array([0.1, 1, 10, 100])
        porosity = mat.FractallyPorous(0.1, 3.0)
        assert np.all(porosity(a) == 0)

        porosity = mat.FractallyPorous(0.1, 2.727)
        p = np.array([0,
                      0.4666651045123792,
                      0.7155538892552087,
                      0.8482949632540665])
        assert np.allclose(porosity(a), p)

    def test_constant_porosity(self):
        a = np.logspace(-1, 2)
        porosity = mat.ConstantPorosity(0.5)
        assert np.all(porosity(a) == 0.5)

class TestMaterial:
    def test_mass(self):
        ac = mat.Material('amorphous carbon', 2.5)
        assert np.isclose(ac.mass(1.0), 1.0471975511965977e-11)

        ac.porosity = mat.ConstantPorosity(0.5)
        assert np.isclose(ac.mass(1.0), 5.2359877559829886e-12)

        # gsd should not affect mass
        ac.gsd = mat.HannerGSD(0.1, 3.7, 37)
        assert np.isclose(ac.mass(1.0), 5.2359877559829886e-12)
        assert np.isclose(ac.mass(5.0), 7.776724279537361e-11)
        
    def test_total_mass(self):
        ac = mat.AmorphousCarbon(mat.FractallyPorous(0.1, 3.0),
                                 mat.HannerGSD(0.1, 3.7, 37, alim=[0.1, 100]))
        assert np.isclose(ac.total_mass((0.1, 100)), 3.5740712092113057e-09)

        # dn/da = a^-3 has equal mass per decade
        ac = mat.AmorphousCarbon(mat.Solid(), mat.PowerLaw(0.1, 3))
        assert np.isclose(ac.total_mass((0.1, 1)), ac.total_mass((1, 10)))
