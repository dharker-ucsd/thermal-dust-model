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
        
		assert np.allclose(planck, astropy_planck, rtol=1e-5)

	def test_bbody_error(self):
		# test bbody (Planck function) error analysis
		from ..util import bbody
		wave = np.arange(50) + 1
		temp = 300.
		sigmaT = 10.
		S = 1e-5
		sigmaS = 1e-8
		planck, error = bbody(wave, temp, sigmaT=sigmaT, S=S, sigmaS=sigmaS)

		# Hard coded test for now:
		# mskpy.planck(wave, 300, unit='W/(cm2 um sr)')
		# mskpy.planck(wave, 300, unit='W/(cm2 um sr)', deriv='t')
		mskpy_planck = np.array([
			1.76806236e-17,  1.43404410e-08,  5.59127987e-06,
			7.21975863e-05,  2.60268174e-04,  5.17503009e-04,
			7.50597255e-04,  9.07835350e-04,  9.83006211e-04,
			9.92402971e-04,  9.57317694e-04,  8.96136943e-04,
			8.22272665e-04,  7.44566932e-04,  6.68394792e-04,
			5.96743556e-04,  5.31055650e-04,  4.71823226e-04,
			4.18983886e-04,  3.72173742e-04,  3.30883993e-04,
			2.94554611e-04,  2.62628073e-04,  2.34578273e-04,
			2.09924363e-04,  1.88235674e-04,  1.69131568e-04,
			1.52278573e-04,  1.37386235e-04,  1.24202502e-04,
			1.12509143e-04,  1.02117425e-04,  9.28641837e-05,
			8.46083325e-05,  7.72277892e-05,  7.06168121e-05,
			6.46836979e-05,  5.93488036e-05,  5.45428514e-05,
			5.02054792e-05,  4.62840012e-05,  4.27323494e-05,
			3.95101691e-05,  3.65820457e-05,  3.39168428e-05,
			3.14871351e-05,  2.92687224e-05,  2.72402119e-05,
			2.53826590e-05,  2.36792574e-05])
		mskpy_dB_dT = np.array([
			2.82649710e-18,  1.14626090e-09,  2.97948354e-07,
			2.88546857e-06,  8.32207607e-06,  1.37930180e-05,
			1.71600935e-05,  1.81865963e-05,  1.75459187e-05,
			1.59971518e-05,  1.40928914e-05,  1.21618545e-05,
			1.03708797e-05,  8.78796240e-06,  7.42705620e-06,
			6.27560849e-06,  5.31008018e-06,  4.50409159e-06,
			3.83235529e-06,  3.27232555e-06,  2.80467316e-06,
			2.41319690e-06,  2.08449624e-06,  1.80757094e-06,
			1.57342660e-06,  1.37472011e-06,  1.20545584e-06,
			1.06073229e-06,  9.36534376e-07,  8.29564720e-07,
			7.37107474e-07,  6.56918699e-07,  5.87138235e-07,
			5.26218835e-07,  4.72869115e-07,  4.26007541e-07,
			3.84725244e-07,  3.48255889e-07,  3.15951179e-07,
			2.87260900e-07,  2.61716603e-07,  2.38918220e-07,
			2.18523054e-07,  2.00236691e-07,  1.83805482e-07,
			1.69010289e-07,  1.55661287e-07,  1.43593608e-07,
			1.32663704e-07,  1.22746281e-07])
		mskpy_error = S * np.sqrt(mskpy_dB_dT**2 * sigmaT**2
								+ mskpy_planck**2 * sigmaS**2)

		assert np.allclose(error, mskpy_error, rtol=1e-5, atol=1e-16)


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

