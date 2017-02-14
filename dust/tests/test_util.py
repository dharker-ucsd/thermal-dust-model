import numpy as np

class TestBbody:
	def test_bbody(self):
		# Test the BBODY function just using wave and temp.
		from astropy.analytic_functions import blackbody_lambda
		import astropy.units as u
		from ..util import bbody

		wave = np.arange(50) + 1.
		temp = 300.
		planck = bbody(wave, temp)

		wave = wave * u.um
		astropy_planck = blackbody_lambda(wave, temp)
		# Remove surface brightness units to make spectral density conversion
		conv = (astropy_planck.unit * u.sr).to(
			'W/(cm2 um)', 1.0, u.spectral_density(wave))
		astropy_planck = astropy_planck.value * conv
        
		assert np.allclose(planck, astropy_planck, rtol=1e-25)


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

