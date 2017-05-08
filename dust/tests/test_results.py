import numpy as np

class TestModelResults:
    def test_table(self):
        from ..results import ModelResults
        from .. import materials as mat

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
        from ..results import ModelResults
        from .. import materials as mat
        
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
