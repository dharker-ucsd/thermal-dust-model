import numpy as np
import logging

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
    scales : ndarray
      `N` dust model scale factors.
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
