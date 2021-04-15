import numpy as np
import astropy.units as u
from astropy.modeling.models import BlackBody
from .. import util


class TestBbody:
    def test_bbody(self):
        # Test the BBODY function just using wave and temp.

        wave = np.arange(50) + 1.
        temp = 300.
        planck = util.bbody(wave, temp)

        wave = wave * u.um
        bb = BlackBody(temp * u.K, scale=1 * u.Unit('W/(cm2 um sr)'))
        # Remove surface brightness units to make spectral density conversion
        astropy_planck = (bb(wave) * u.sr).to_value('W/(cm2 um)')

        assert np.allclose(planck, astropy_planck)


class TestInterp:
    # Test the interpolation function
    def test_interp(self):
        w1 = np.linspace(1., 2 * np.pi, 50)
        w2 = np.linspace(1., 2 * np.pi, 100)
        f2 = np.sin(w2)

        interp = util.interp_model2comet(w1, w2, f2)
        assert interp.shape[0] == 50
        assert np.allclose(interp, np.sin(w1))


class TestAVInt:
    def test_x2(self):
        x = np.linspace(0, 100)
        integ = util.avint(x, x**2, x[[0, -1]])
        assert np.isclose(integ, (x[-1]**3 - x[0]**3) / 3)

    def test_x3(self):
        x = np.linspace(20, 36)
        integ = util.avint(x, x**3, x[[0, -1]])
        assert np.isclose(integ, (x[-1]**4 - x[0]**4) / 4)

    def test_x_ln(self):
        x = np.linspace(1, 100)
        integ = util.avint(x, x - np.log(x), (5, 50))
        def f(x): return 0.5 * x * (x - 2 * np.log(x) + 2)
        assert np.isclose(integ, f(50) - f(5))

    def test_trapz(self):
        # avint uses trapezoidal rule when there are only two points
        x = np.array([1, 2])
        integ = util.avint(x, x**2, x[[0, -1]])
        assert np.isclose(integ, 2.5)


class TestHannerGSD:
    def test_value(self):
        gsd = util.hanner_gsd(5.0, 0.1, 3.7, 37)
        assert np.isclose(gsd, 0.05940979728725421)

    def test_peak(self):
        a0, N, M = 0.1, 3.7, 37
        ap = util.hanner_ap(a0, N, M)
        gsd = util.hanner_gsd(ap, a0, N, M)
        assert np.isclose(gsd, 1.0)

    def test_M(self):
        a0, N, ap = 0.1, 3.7, 1.1
        M = util.hanner_M(a0, N, ap)
        assert np.isclose(M, 37)
        gsd = util.hanner_gsd(ap, a0, N, M)
        assert np.isclose(gsd, 1.0)


class TestPowerLaw:
    def test_value(self):
        gsd = util.power_law((5, 50), 0.1, 3.5)
        assert np.isclose(gsd[0], 1.1313708498984761e-06)
        assert np.isclose(gsd[0] / gsd[1], 10**3.5)

    def test_peak(self):
        assert np.isclose(util.power_law(0.1, 0.1, 3.5), 1.0)
