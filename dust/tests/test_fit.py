import numpy as np

class TestFitOne:
    def setup(self):
        N = 100
        self.x = np.arange(N) + 1
        self.A = np.c_[np.ones(len(self.x)), self.x].T
        self.m = 0.5
        self.b = 5.0
        y0 = self.m * self.x + self.b
        self.u = np.sqrt(y0) / 10000
        self.y = y0 + self.u * np.random.randn(N)
        self.u_n = self.u * 100
        self.y_n = y0 + self.u * np.random.randn(N)

    def fit(self, y, u, tol, method):
        from ..fit import fit_one
        best, chi2 = fit_one(y, u, self.A, guess=[1.0, 0.6], method=method)
        return (np.isclose(best[0], self.b, tol)
                and np.isclose(best[1], self.m, tol))

    def test_nnls_tight(self):
        # weighted fit, low noise
        self.setup()
        assert self.fit(self.y, self.u, 1e-4, 'nnls')

    def test_nnls_loose(self):
        # weighted fit, high noise
        self.setup()
        assert self.fit(self.y_n, self.u_n, 1e-2, 'nnls')

    #def test_slsqp_tight(self):
    #    self.setup()
    #    assert self.fit(self.y, self.u, 1e-3, 'slsqp')

    #def test_slsqp_loose(self):
    #    self.setup()
    #    assert self.fit(self.y_n, self.u_n, 0.01, 'slsqp')

class TestMCFit:
    def setup(self):
        TestFitOne.setup(self)

    def fit(self, y, u, tol, method):
        from ..fit import fit_one, mcfit
        best, chisq = fit_one(y, u, self.A, method=method)
        fits, chisqs = mcfit(y, u, self.A, best, nmc=1000, method=method)
        r = np.corrcoef(fits.T)
        best = fits.mean(0)
        unc = fits.std(0)
        return best, unc, r

    def test_nnls_tight(self):
        self.setup()
        best, unc, r = self.fit(self.y, self.u, 1e-4, 'nnls')
        assert (best[0] - self.b) / unc[0] < 3
        assert (best[1] - self.m) / unc[1] < 3

    def test_nnls_loose(self):
        self.setup()
        best, unc, r = self.fit(self.y_n, self.u_n, 1e-2, 'nnls')
        assert (best[0] - self.b) / unc[0] < 3
        assert (best[1] - self.m) / unc[1] < 3

class TestSummarizeMCFit:
    def test_limits(self):
        from scipy.special import erf
        from ..fit import summarize_mcfit
        from ..results import ModelResults
        from .. import materials as mat

        best_scales = 1000, 100
        unc = 100, 10
        scales = np.random.randn(20000).reshape((10000, 2))
        scales[:, 0] = scales[:, 0] * unc[0] + best_scales[0]
        scales[:, 1] = scales[:, 1] * unc[1] + best_scales[1]

        gsd = mat.PowerLaw(0.1, 0)
        materials = [mat.AmorphousCarbon(gsd=gsd),
                     mat.AmorphousOlivine50(gsd=gsd)]
        
        best = ModelResults(materials, best_scales)
        results = ModelResults(materials, scales)

        s = 2  # get 2-sigma results
        cl = 0.5 * (erf(s / np.sqrt(2.0)) - erf(-s / np.sqrt(2.0))) * 100
        summary = summarize_mcfit(results, cl=cl)

        limits = best_scales[0] + s * unc[0] * np.array((-1, 1))
        assert np.allclose(summary['s0'][1:], limits, rtol=0.05)

        limits = best_scales[1] + s * unc[1] * np.array((-1, 1))
        assert np.allclose(summary['s1'][1:], limits, rtol=0.05)

    def test_best_fit_finding(self):
        from ..fit import summarize_mcfit
        from ..results import ModelResults
        from .. import materials as mat

        best_scales = 1000, 100
        unc = 100, 10
        scales = np.random.randn(20000).reshape((10000, 2))
        scales[:, 0] = scales[:, 0] * unc[0] + best_scales[0]
        scales[:, 1] = scales[:, 1] * unc[1] + best_scales[1]

        gsd = mat.PowerLaw(0.1, 0)
        materials = [mat.AmorphousCarbon(gsd=gsd),
                     mat.AmorphousOlivine50(gsd=gsd)]
        
        results = ModelResults(materials, scales)

        summary = summarize_mcfit(results)

        assert np.isclose(summary['s0'][0], best_scales[0], rtol=0.05)
        assert np.isclose(summary['s1'][0], best_scales[1], rtol=0.05)
