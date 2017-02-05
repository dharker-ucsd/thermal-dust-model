import numpy as np
import logging

__all__ = [
    'fit_one',
    'mcfit',
]

logger = logging.getLogger('thermal-dust-model')

def fit_one(fluxd, unc, mfluxd, method='nnls', guess=None, **kwargs):
    """Fit model dust scale factors to a spectrum.

    Parameters
    ----------
    fluxd : ndarray
      The observed flux density.
    unc : ndarray
      `fluxd` measurement uncertainties.
    mfluxd : ndarray
      NxM array of model flux densities, where `N` is the number of
      dust species, and `M` is the spectral dimension.  Each element
      along axis `M` corresponds to the same element in the `fluxd`
      and `unc` arrays.
    guess : ndarray, optional
      Initial model guess for `scipy.optimize.minimize` solvers.
    method : string, optional
      The method to find the least-squares solution, either 'nnls' or
      one of the `scipy.optimize.minimize` solvers.
    **kwargs
      Keyword arguments for `scipy.optimize.minimize`.

    Returns
    -------
    best : ndarray
      Best-fit dust model scale factors.
    chi2 : float
      Chi-squared statistic.

    """

    from scipy.optimize import nnls, minimize

    if method == 'nnls':
        # minimize ||A x - b||^2, for x >= 0 with non-negative
        # least-squares matrix factorization

        # weight data and model by uncertainties as you would for
        # computing the chi^2 statistic
        A = (mfluxd / unc).T
        b = fluxd / unc
        
        best, r = nnls(A, b)
        chi2 = np.sum(r)
    else:
        kwargs['tol'] = kwargs.get('tol', 1e-5)  # default tolerance
        
        if guess is None:
            guess = np.ones(mfluxd.shape[0]) * fluxd.mean() / mfluxd.mean()
            
        r = minimize(_fit_one_chi2, guess, args=(fluxd, unc, mfluxd), **kwargs)
        assert r.success, 'fit_one minimization failure: {}'.format(r)
        best = r.x
        chi2 = r.fun
        
    return best, chi2

def _fit_one_chi2(scales, fluxd, unc, mfluxd):
    """Function for `scipy.optimize.minimize` to optimize."""
    m = (mfluxd.T * scales).sum(1)
    chi2 = np.sum(((fluxd - m) / unc)**2)
    logger.debug(scales, chi2)
    return chi2

def mcfit(fluxd, unc, mfluxd, best, nmc=10000, method='nnls', **kwargs):
    """Derive fit uncertainties using a Monte Carlo approach.

    For each run:

      1. Generates a new spectrum using the spectral uncertainties and
         a normally distributed variate:

           ```
           dfluxd = np.random.rand(n_wave) * unc
           fluxd_new = fluxd + dfluxd
           ```

      2. Fit that spectrum, and store the result.


    Parameters
    ----------
    fluxd : ndarray
      Spectrum to be fit.
    unc : ndarray
      `fluxd` uncertainties.
    mfluxd : ndarray
      NxM array of model flux densities, where `N` is the number of
      dust species, and `M` is the spectral dimension.  Each element
      along axis `M` corresponds to the same element in the `fluxd`
      and `unc` arrays.
    best : ndarray
      The best fit scales for `mfluxd`, used as an initial guess for
      `scipy.optimize.minimize`.
    nmc : int
      Number of Monte Carlo simulations to run.
    method : string, optional
      The method to find the least-squares solutions, either 'nnls' or
      one of the `scipy.optimize.minimize` solvers.
    **kwargs
      Keyword arguments for `scipy.optimize.minimize`.

    Returns
    -------
    scales : ndarray
      Dust model scale factors, `nmc` x `len(best)`, one for each
      Monte Carlo simulation.
    chi2 : ndarray
      Chi-squared statistic, one for each Monte Carlo simulation.

    """

    from scipy.optimize import nnls, minimize

    scales = np.empty((nmc, len(best)))
    chi2 = np.empty(nmc)
    n_wave = len(fluxd)

    for i in range(nmc):
        dfluxd = np.random.randn(n_wave) * unc
        fluxd_i = fluxd + dfluxd

        scales[i], chi2[i] = fit_one(fluxd, unc, mfluxd, method=method,
                                     guess=best, **kwargs)

    return scales, chi2
