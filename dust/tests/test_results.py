import numpy as np

class TestModelResults:
    def test_table(self):
        from ..results import ModelResults
        from .. import materials as mat

        materials = [mat.AmorphousCarbon(), mat.AmorphousOlivine50()]

        scales = np.arange(len(materials))
        results = ModelResults(materials, scales)
        assert len(results.table()) == 1

        scales = np.arange(len(materials) * 10).reshape((10, len(materials)))
        results = ModelResults(materials, scales)
        assert len(results.table()) == 10

    def test_total_mass(self):
        from ..results import ModelResults
        from .. import materials as mat
        
        gsd = mat.PowerLaw(0.1, 0)
        materials = [mat.AmorphousCarbon(gsd=gsd),
                     mat.AmorphousOlivine50(gsd=gsd)]

        ar = (0.1, 1)
        scales = np.arange(len(materials))
        results = ModelResults(materials, scales)
        M = results.total_mass(ar)
        M0 = scales * np.array([m.total_mass(ar) for m in materials])
        assert np.allclose(M, M0)

        scales = np.arange(len(materials) * 10).reshape((10, len(materials)))
        results = ModelResults(materials, scales)
        M = results.total_mass(ar)
        assert np.allclose(M[0], M0)
