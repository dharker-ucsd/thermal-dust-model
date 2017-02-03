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
      The error of the resulting curve.  Will only return if either sigmaT or 
      sigmaS are set to True.

    """
    bbflux = wave*0.

    c1 = 1.1926926e8                        # W m^-2 micron^-1 st^-1
    c2 = 14388.3
    val = c2/wave/temp
    good = np.where(val < 88)               # avoid floating underflow
    bbflux[good] = c1/(wave[good]**5 * (np.exp(val[good])-1.))
    bbflux = bbflux * 1e-4                  # W cm^-2 micron^-1 st^-1

    if sigmaT:
        bberr = wave*0.
        # Partial derivative of Planck Function w.r.t. T.  Scaled and in units of W cm^-2 micron^-1
        bberr[good] = (S * 1e-4) * c1 * c2/(wave[good]**6 * temp**2 * (np.exp(val[good])-1.)**2) * np.exp(val[good])
        # First quadrature term
        err1 = bberr * sigmaT
    else:
        err1 = 0.

    if sigmaS:
        # Partial derivative of Planck Function w.r.t. S is "bbflux" in units of W cm^-2 micron^-1
        # Second quadrature term
        err2 = bbflux*sigmaS
    else:
        err2 = 0.

    # Compute final error by adding in quadrature.
    error = np.sqrt(err1**2 + err2**2)

    if sigmaT or sigmaS:
         return bbflux * S, error
    else:
         return bbflux * S