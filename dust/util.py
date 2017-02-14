import numpy as np

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
