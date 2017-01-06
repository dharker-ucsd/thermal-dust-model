# Licensed under a 3-clause BSD style license - see LICENSE.rst
#

"""
dust_model --- Radiative Equilibrium Model Calculations
================================================================================

.. autosummary::
   :toctree: generated/

   q_solar

"""

__all__ = [
    'q_solar'
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

