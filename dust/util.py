import numpy as np

# by default, only import functions a user might be interested in
__all__ [
    'bbody',
    'hanner_gsd',
    'power_law',
]

def interp_model2comet(wave_comet, wave_model, fluxd_model):
    """Interpolate the model spectrum to the same wavelength grid as the comet
    spectrum.

    Parameters
    ----------
    wave_comet : array
            Wavelength grid of the comet spectrum.
    wave_model : array
            Wavelength grid of the model spectrum.
    fluxd_model : array
            Model spectrum

    Returns
    -------
    fluxd_model_interp : array
            The model spectrum interpolated on the wavelength grid of the comet
            spectrum.

    """

    from scipy import interpolate
    
    tck = interpolate.splrep(wave_model, fluxd_model, s=0)
    
    fluxd_model_interp = interpolate.splev(wave_comet, tck, der=0)
    
    return fluxd_model_interp
        

def bbody(wave,temp):
    """Compute the Planck funcion in W cm^-2 micron^-1 st^-1

    Parameters
    ----------
    wave : array or quantity
      A linear array of wavelength values in microns
    temp : float or array
      A temperature in Kelvin

    Return
    ------
    bbflux : ndarray
      Planck function values in W cm^-2 micron^-1 st^-1

    """
    bbflux = wave*0.

    c1 = 1.1910428681415875e8	            # W m^-2 micron^-1 st^-1
    c2 = 14387.7695998
    val = c2/wave/temp
    good = np.where(val < 88)               # avoid floating underflow
    bbflux[good] = c1/(wave[good]**5 * (np.exp(val[good])-1.))
    bbflux = bbflux * 1e-4                  # W cm^-2 micron^-1 st^-1

    return bbflux 

def avint(x, y, xlim):
    """Integrate tabulated function with arbitrarily-spaced abscissas.

    Trapezoial rule is used when there are only two function values,
    otherwise, the method requires at least three values within the
    range `xlim`.

    Paramters
    ---------
    x : array
      The abscissas, must be in increasing order, and must have length
      2 or more.
    y : array
      The function values at each `x`.
    xlim : float
      The integration limits, must be within the closed interval
      `x[0]` to `x[1]`.

    Returns
    -------
    z : float
      The estimated integral.

    Notes
    -----
    From the SLATEC Common Mathematical Library:

    Original program from *Numerical Integration* by Davis & Rabinowitz
    Adaptation and modifications by Rondall E Jones.

    References: R. E. Jones, Approximate integrator of functions C
    tabulated at arbitrarily spaced abscissas, C Report SC-M-69-335,
    Sandia Laboratories, 1969.

    """

    x = np.array(x)
    y = np.array(y)
    xlim = np.array(xlim)

    assert len(x) >= 2
    assert all(np.diff(x) > 0), "Abscissas must be strictly increasing."
    assert len(x) == len(y)
    assert (x[0] <= xlim[0]) and (xlim[0] <= x[-1]), "xlim must be within x"
    assert (x[0] <= xlim[1]) and (xlim[1] <= x[-1]), "xlim must be within x"

    z = 0  # the result    

    if xlim[0] == xlim[1]:
        return z

    if len(x) == 2:
        # trapezoidal rule
        slope = (y[1] - y[0]) / (x[1] - x[0])
        fl = y[0] + slope * (min(xlim) - x[0])
        fr = y[1] + slope * (max(xlim) - x[1])
        z = 0.5 * (fl + fr) * (max(xlim) - min(xlim))
    else:
        # overlapping parabolas
        assert min(xlim) < x[-2], "Less than three function values between the limits of integration."
        assert max(xlim) > x[2], "Less than three function values between the limits of integration."

        left = np.flatnonzero(x >= min(xlim))[0]
        right = np.flatnonzero(x <= max(xlim))[-1]
        assert (right - left) >= 2, "Less than three function values between the limits of integration."

        istart = left
        if left == 0:
            istart += 1

        istop = right
        if right == len(x) - 1:
            istop -= 1

        syl = min(xlim)
        syl2 = syl**2
        syl3 = syl**3

        for i in range(istart, istop + 1):
            x1 = x[i - 1]
            x2 = x[i]
            x3 = x[i + 1]

            x12 = x1 - x2
            x13 = x1 - x3
            x23 = x2 - x3

            term1 = y[i - 1] / (x12 * x13)
            term2 = -y[i] / (x12 * x23)
            term3 = y[i + 1] / (x13 * x23)

            A = term1 + term2 + term3
            B = -(x2 + x3) * term1 - (x1 + x3) * term2 - (x1 + x2) * term3
            C = x2 * x3 * term1 + x1 * x3 * term2 + x1 * x2 * term3

            if i == istart:
                CA = A
                CB = B
                CC = C

            CA = 0.5 * (A + CA)
            CB = 0.5 * (B + CB)
            CC = 0.5 * (C + CC)

            syu = x2
            syu2 = x2**2
            syu3 = x2**3
            z += (CA * (syu3 - syl3) / 3
                  + CB * 0.5 * (syu2 - syl2)
                  + CC * (syu - syl))

            CA = A
            CB = B
            CC = C
            
            syl = syu
            syl2 = syu2
            syl3 = syu3

        syu = max(xlim)
        z += (CA * (syu**3 - syl3) / 3
              + CB * 0.5 * (syu**2 - syl2)
              + CC * (syu - syl))
        
    if xlim[1] < xlim[0]:
        # the integration is backwards, which is OK
        return -z
    else:
        return z
