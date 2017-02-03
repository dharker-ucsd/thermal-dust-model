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
