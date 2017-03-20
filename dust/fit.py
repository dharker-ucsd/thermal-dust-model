import numpy as np
import logging

__all__ = [
    'fit_all',
    'fit_one',
    'mcfit',
    'summarize_mcfit'
]

logger = logging.getLogger('thermal-dust-model')

def fit_all(wave, fluxd, unc, mwave, mfluxd, parameters, parameter_names=None,
            material_names=None):
    """Run through a range of models with various parameters to fit
        to a spectrum

    Parameters
    ----------
    wave : ndarray
      Wavelengths of spectrum to fit, units of μm.
    fluxd : ndarray
      The observed flux density.
    unc : ndarray
      `fluxd` measurement uncertainties.
    mwave : ndarray
      Wavelengths of models, units of μm.
    mfluxd : ndarray
      NxMx#parameters array of model flux densities, where `N` is the number of
      dust species, `M` is the spectral dimension, and each subsequent dimension
      corresponds to a separate parameter (for example 'D' or 'GSD'.  Each element
      along axis `M` corresponds to the same element in the `fluxd`
      and `unc` arrays.
    parameters : ?
      An input array with the parameters corresponding to the parameters in mfluxd
    parameter_name : ?
      A list(?) with the parameter names corresponding to the input parameters
    material_names = ?
      A list(?) with the material names corresponding to the number of dust species 
      in the N dimension of mfluxd

    Returns
    -------
    tab : table
      Table with the best-fit dust model scale factors over each parameter.

"""


    from astropy.table import Table
    from itertools import product
    from dust import util

    n = len(material_names)
    if material_names is None:
        material_names = ['mat{}'.format(i) for i in range(mfluxd.shape[0])]

    if parameter_names is None:
        parameter_names = ['p{}'.format(i) for i in range(len(parameters))]

    names = parameter_names + material_names + ('rchisq',)
    dtype = []
    for p in parameters:
        if type(p) == str:
            dtype.append((str, 128))
        elif type(p) == np.ndarray:
            dtype.append(p.dtype)
        else:
            dtype.append(type(p))
    dtype += [float] * (n + 1)

    tab = Table(names=names, dtype=dtype)

    # Loop over each parameter combination
    for indices in product(*[range(x) for x in mfluxd.shape[2:]]):
        i = (slice(None), slice(None)) + indices

        # Interpolate over each material onto the wavelength grid of the observed spectrum
        mfluxd_i = np.zeros((n, len(wave)))
        for j in np.arange(0, n):
            mfluxd_i[j,:] = util.interp_model2comet(wave, mwave, mfluxd[i][j,:])

        # Calculate the best fit.
        best, chi2 = fit_one(fluxd, unc, mfluxd_i)

        # To get the parameters for the table
        row = []
        for k, p in zip(indices, parameters):  # pairs up each index with each parameter
          row.append(p[k])

        # Add the best fit and chi2
        row.extend(best)
        row.append(chi2)  # or rchisq

        # Add the whole row to the table
        tab.add_row(row)

    return tab

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

        scales[i], chi2[i] = fit_one(fluxd_i, unc, mfluxd, method=method,
                                     guess=best, **kwargs)

    return scales, chi2

def summarize_mcfit(results, best=None, cl=95, ar=(0.1, 1), bins=31):
    """Summarize the results of a Monte Carlo fit.

    Parameters
    ----------
    results : ModelResults
      The ModelResults.
    best : ModelResults
      Use these scale factors for the best fit, otherwise use the mode
      of each parameter for the best fit, estimated from a histogram.
    cl : float, optional
      Use this confidence limit to define uncertainties. [percentile]
    ar : array-like, optional
      Use this radius range for computing dust masses.
    bins : int, optional
      Number of bins to use for best-fit estimation.

    Returns
    -------
    summary : dict
      The summary.  For each parameter the values are the best fit,
      lower limit, and upper limit.

    """

    from .results import ModelResults

    assert isinstance(results, ModelResults)
    assert isinstance(best, (ModelResults, type(None)))
    assert len(ar) == 2
    
    tab = results.table(ar=ar)
    if best is not None:
        best_tab = best.table(ar=ar)

    summary = dict()
    for col in tab.colnames:
        # find the upper and lower limits
        ll = np.percentile(tab[col], (100 - cl) / 2)
        ul = np.percentile(tab[col], (100 - cl) / 2 + cl)
        
        # find the best fit
        if best is None:
            h = np.histogram(tab[col], range=(ll, ul), bins=31)
            c = h[1][h[0].argmax()]
        else:
            c = best_tab[col]

        summary[col] = c, ll, ul
        logger.info('{} = {:.4g} +{:.4g} -{:.4g}'.format(
            col, c, ul - c, ll - c))

    return summary

def fit_uncertainties(wave, fluxd, mwave, mfluxd, best):
    """Derive uncertainties on direct and derived model parameters.

    Uses the Monte Carlo method to explore parameter space.

    Parameters
    ----------
    wave : ndarray
      Wavelengths of spectrum to fit, units of μm.
    fluxd : ndarray
      Flux density of spectrum to fit, same units as `mfluxd`.
    mwave : ndarray
      Spectral wavelengths for the model, units of μm.
    mfluxd : ndarray
      Model flux densities in an `NxM` array, where `N` is the number
      of materials, and `M` is the number of wavelengths.
    best : ModelResults
      Nominal best-fit for given spectrum and model.
    
    Results
    -------
    mcfits : ModelResults
      All the Monte Carlo results.

    summary : dict

      A summary of the Monte Carlo analysis for all direct and derived
      model parameters.

    """
    return None, None
