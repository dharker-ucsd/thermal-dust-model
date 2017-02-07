import numpy as np


class TestBbody:
    # Test the BBODY function just using wave and temp.
    def setup(self):
        N = 50
        self.wave = np.arange(N) + 1
        self.temp = 300.

    def util(self, wave, temp):
        from ..util import bbody
        planck = bbody(wave, temp)
        return planck.any()

    def test_bbody(self):
        self.setup()
        assert self.util(self.wave,self.temp)

class TestBbodyE:
    # Test the BBODY function with wave, temp, scaling factor and errors
    # in temp and scaling factor.
    def setup(self):
        N = 50
        self.wave = np.arange(N) + 1
        self.temp = 300.
        self.sigmaT = 10.
        self.S = 1e-5
        self.sigmaS = 1e-8

    def util(self, wave, temp):
        from ..util import bbody
        planck, error = bbody(wave, temp, sigmaT = self.sigmaT, S = self.S, sigmaS = self.sigmaS)
        return planck.any() and error.any()

    def test_bbodye(self):
        self.setup()
        assert self.util(self.wave,self.temp)


class TestInterp:
    # Test the interpolation function
    def setup(self):
        self.w1 = np.linspace(1., 50., 50)
        self.w2 = np.linspace(1., 50., 100)
        self.f2 = np.sin(self.w2)

    def util(self, w1, w2, f2):
        from ..util import interp_model2comet
        interp = interp_model2comet(w1, w2, f2)
        if 50 == interp.shape[0]:
            return True
        else:
            return False

    def test_interp(self):
        self.setup()
        assert self.util(self.w1, self.w2, self.f2)

