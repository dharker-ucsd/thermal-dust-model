# Licensed under a 3-clause BSD style license - see LICENSE.rst
#

"""
dust_model --- Radiative Equilibrium Model Calculations
================================================================================

.. autosummary::
   :toctree: generated/

   q_solar
   bbody

"""

__all__ = [
    'q_solar'
    'bbody'
]

import numpy as np
from scipy.integrate import simps

def q_solar(a, wave, q):
    """Calculate the absorption efficiency averaged over the solar spectrum
       for particles of arbitrary size and constituent.  Similar to QBAR.
       FOR USE WHEN Q_ABS IS ALREADY KNOWN.
       
       REVISION HISTORY:
            Ported from IDL (q_solar.pro) on August 31, 2016    D. Harker (UCSD)

    Parameters
    ----------
    a : float
      grain radius in microns
    wave : array
      A 1 element array of wavelength values in micron
    q : array
      A 1 element array (same size as wave) of q values

    Returns
    -------
    ans : float
      Planck mean average over solar spectrum.
    ierr: int
      Error code

    """

    # SOLAR DATA - wavelength in microns, flux in erg/cm^2/ang/s
    w = np.array([.20,.22,.24,.26,.28,.30,.32,.34,.36,.37,.38,.39,.40,.41,.42, \
    .43,.44,.45,.46,.48,.50,.55,.60,.65,.70,.75,.80,.90,1.0,1.1,1.2,1.4,1.6, \
    1.8,2.0,2.5,3.0,4.,5.,6.,8.,10.,12.])
    # 'f' is the solar flux at 1 AU, or PI*B(lambda,T)*(Rs/AU)^2
    f = np.array([0.65,4.5,5.2,13.,23.,56.,76.,91.,97.,113.,107.,103.,148., \
    170.,173.,159.,184.,200.,205.,203.,192.,188.,177.,159.,141.,127.,114., \
    94.,75.,61.,52.,35.,25.5,16.9,11.6,5.2,2.6,.9,.4,.21,.063,.023,.012])
    
    ncount = 0
    i = 0
    ierr = 0
    num = w.size()
    
    return ierr, ans

def bbody(wave,temp,sigmaT=None,S=1.,sigmaS=None):
    """Compute the Planck funcion in W cm^-2 micron^-1 st^-1

    Parameters
    ----------
    wave : array or quantity
      A linear array of wavelength values in microns
    temp : float or array
      A temperature in Kelvin
    sigmaT : boolean(?)
      If set to True, the error of the resulting curve will be computed.
    S : float
      A multiplier to scale the resulting curve.
    sigmaS : boolean(?)
      If set to True, the error of the resulting curve will be computed.

    Return
    ------
    bbflux : ndarray
      Planck function values in W cm^-2 micron^-1 st^-1
    error : ndarray
      The error of the resulting curve.  Will only return if either sigmaT or sigmaS are set to True.

    """
    bbflux = wave*0.

    c1 = 1.1926926e8                        # W m^-2 micron^-1 st^-1
    c2 = 14388.3
    val = c2/wave/temp
    good = np.where(val < 88)               # avoid floating underflow
    bbflux[good] = c1/(wave[good]**5 * (np.exp(val[good])-1.))
    bbflux = bbflux * 1e-4                  # W cm^-2 micron^-1 st^-1

    if sigmaT is not None:
        bberr = wave*0.
        # Partial derivative of Planck Function w.r.t. T.  Scaled and in units of W cm^-2 micron^-1
        bberr[good] = (S * 1e-4) * c1 * c2/(wave[good]**6 * temp**2 * (np.exp(val[good])-1.)**2) * np.exp(val[good])
        # First quadrature term
        err1 = bberr * sigmaT
    else:
        err1 = 0.

    if sigmaS is not None:
        # Partial derivative of Planck Function w.r.t. S is "bbflux" in units of W cm^-2 micron^-1
        # Second quadrature term
        err2 = bbflux*sigmaS
    else:
        err2 = 0.

    # Compute final error by adding in quadrature.
    error = np.sqrt(err1**2 + err2**2)

    if (sigmaT is None) and (sigmaS is None):
         return bbflux * S
    else:
         return bbflux * S, error


