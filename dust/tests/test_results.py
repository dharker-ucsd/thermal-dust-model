import numpy as np
from scipy.integrate import quad
from ..results import ModelResults
from .. import materials as mat
from .. import util


class TestModelResults:
    def test_table(self):
        materials = [mat.amorphous_carbon, mat.amorphous_olivine50]
        grains = [mat.Grains(m) for m in materials]

        scales = np.arange(len(grains))
        chisq = 1100
        dof = 100
        results = ModelResults(grains, scales, chisq=chisq, dof=dof)
        assert len(results.table()) == 1

        N = 10
        scales = np.arange(len(grains) * N).reshape((N, len(materials)))
        chisq = np.arange(N)
        dof = 100
        results = ModelResults(grains, scales, chisq=chisq, dof=dof)
        assert len(results.table()) == N

    def test_total_mass(self):
        gsd = mat.PowerLaw(0.1, 0)
        materials = [mat.amorphous_carbon, mat.amorphous_olivine50]
        grains = [mat.Grains(m, gsd=gsd) for m in materials]

        ar = (0.1, 1)
        scales = np.arange(len(grains))
        results = ModelResults(grains, scales)
        M = results.total_mass(ar)
        M0 = scales * np.array([g.total_mass(ar) for g in grains])
        assert np.allclose(M, M0)

        scales = np.arange(len(grains) * 10).reshape((10, len(grains)))
        results = ModelResults(grains, scales)
        M = results.total_mass(ar)
        assert np.allclose(M[0], M0)

    def test_total_mass_large_ap(self):
        """Check sub-micron mass results for Hanner GSD and large ap."""

        gsd = mat.HannerGSD.from_ap(0.1, 3.5, 2.0)
        g = mat.Grains(mat.amorphous_olivine50, porosity=mat.Solid(), gsd=gsd)
        m_solid = g.total_mass([0.1, 1])

        # repeat with a higher porosity: 50%
        g = mat.Grains(mat.amorphous_olivine50,
                       porosity=mat.ConstantPorosity(0.5),
                       gsd=gsd)
        m_c50 = g.total_mass([0.1, 1])
        assert np.isclose(m_c50 / m_solid, 0.5)

        # repeat with fractal porosity
        # expected value is ratio of the GSD weighted densities
        def f(a, gsd, a0=0.1, D=2.5):
            return (a / a0)**(D - 3) * gsd(a)
        y = quad(f, 0.1, 1.0, args=(gsd,))[0]
        rho_f25 = y / quad(lambda a: gsd(a), 0.1, 1.0)[0]

        g = mat.Grains(mat.amorphous_olivine50,
                       porosity=mat.FractallyPorous(0.1, 2.5),
                       gsd=gsd)
        m_f25 = g.total_mass([0.1, 1])
        assert np.isclose(m_f25 / m_solid, rho_f25, rtol=0.02)
