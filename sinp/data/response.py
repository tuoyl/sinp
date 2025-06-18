"""
JAX-optimized response class
"""

import warnings
import numpy as np
import jax.numpy as jnp
from astropy.io import fits
from typing import Dict, List, Tuple, Optional, Union, Any, NamedTuple

__all__ = ['Response']


class Response:
    """
    JAX-optimized response class for spectral analysis
    """

    def __init__(self,
                 energy_low: Optional[np.ndarray] = None, 
                 energy_high: Optional[np.ndarray] = None,
                 rmf: Optional[np.ndarray] = None, 
                 arf: Optional[np.ndarray] = None,
                 channel: Optional[np.ndarray] = None, 
                 ebounds_min: Optional[np.ndarray] = None, 
                 ebounds_max: Optional[np.ndarray] = None):
        """
        Initialize Response object
        
        Parameters
        ----------
        energy_low : array-like
            Lower energy boundaries of incident spectrum in keV
        energy_high : array-like
            Upper energy boundaries of incident spectrum in keV
        rmf : array-like
            Redistribution matrix
        arf : array-like
            Ancillary response (effective area)
        channel : array-like
            Channel numbers
        ebounds_min : array-like
            Lower energy boundaries of detected spectrum
        ebounds_max : array-like
            Upper energy boundaries of detected spectrum
        """
        self.energy_low = energy_low
        self.energy_high = energy_high
        self._rmf = rmf
        self._arf = arf
        self.channel = channel
        self.ebounds_min = ebounds_min
        self.ebounds_max = ebounds_max
        
        # JAX arrays for efficient computation
        self._energy_low_jax = None
        self._energy_high_jax = None
        self._rmf_jax = None
        self._arf_jax = None
        self._drm_jax = None
        self._ebounds_min_jax = None
        self._ebounds_max_jax = None
        
        # Update JAX arrays if data is provided
        if energy_low is not None:
            self._update_jax_arrays()

    def _update_jax_arrays(self):
        """Update JAX arrays when data changes"""
        if self.energy_low is not None:
            self._energy_low_jax = jnp.array(self.energy_low)
            self._energy_high_jax = jnp.array(self.energy_high)
        
        if self._rmf is not None:
            self._rmf_jax = jnp.array(self._rmf)
        
        if self._arf is not None:
            self._arf_jax = jnp.array(self._arf)
            
        if self.ebounds_min is not None:
            self._ebounds_min_jax = jnp.array(self.ebounds_min)
            self._ebounds_max_jax = jnp.array(self.ebounds_max)
        
        # Update DRM if possible
        if self._rmf_jax is not None:
            self._update_drm()

    def _update_drm(self):
        """Update the detector response matrix (DRM)"""
        if self._arf_jax is not None:
            self._drm_jax = jnp.diag(self._arf_jax) @ self._rmf_jax
        else:
            self._drm_jax = self._rmf_jax

    @property
    def energy_centroid(self) -> np.ndarray:
        """Central energy of incident spectrum bins"""
        return (self.energy_low + self.energy_high) / 2

    @property
    def energy_centroid_jax(self) -> jnp.ndarray:
        """JAX array of central energies"""
        if self._energy_low_jax is None:
            self._update_jax_arrays()
        return (self._energy_low_jax + self._energy_high_jax) / 2

    @property
    def ebounds_centroid(self) -> np.ndarray:
        """Central energy of detected spectrum bins"""
        return (self.ebounds_min + self.ebounds_max) / 2
    
    @property
    def ebounds_centroid_jax(self) -> jnp.ndarray:
        """JAX array of detected spectrum central energies"""
        if self._ebounds_min_jax is None:
            self._update_jax_arrays()
        return (self._ebounds_min_jax + self._ebounds_max_jax) / 2

    @property
    def ebounds_low(self) -> np.ndarray:
        """Lower energy boundaries (alias for ebounds_min)"""
        return self.ebounds_min
    
    @property
    def ebounds_high(self) -> np.ndarray:
        """Upper energy boundaries (alias for ebounds_max)"""
        return self.ebounds_max

    @property
    def drm(self) -> np.ndarray:
        """
        Combined RMF and ARF matrix (detector response matrix)
        If ARF is None, returns RMF
        """
        if self._arf is not None:
            return np.dot(np.diag(self._arf), self._rmf)
        else:
            return self._rmf

    @property
    def drm_jax(self) -> jnp.ndarray:
        """JAX array of detector response matrix"""
        if self._drm_jax is None:
            self._update_drm()
        return self._drm_jax

    @property
    def rmf(self) -> np.ndarray:
        return self._rmf

    @rmf.setter
    def rmf(self, rmf: np.ndarray):
        self._rmf = rmf
        if rmf is not None:
            self._rmf_jax = jnp.array(rmf)
            self._update_drm()

    @property
    def arf(self) -> np.ndarray:
        return self._arf

    @arf.setter
    def arf(self, arf: np.ndarray):
        self._arf = arf
        if arf is not None:
            self._arf_jax = jnp.array(arf)
            self._update_drm()

    @property
    def n_channels(self) -> int:
        """Number of detector channels"""
        return len(self.channel) if self.channel is not None else 0

    @property
    def n_energy_bins(self) -> int:
        """Number of incident energy bins"""
        return len(self.energy_low) if self.energy_low is not None else 0

    def load_rmf_file(self, filename: str):
        """
        Load RMF (Redistribution Matrix File)
        
        Parameters
        ----------
        filename : str
            Path to RMF FITS file
        """
        with fits.open(filename) as hdulist:
            # Find MATRIX extension
            matrix_idx = self._find_extension(hdulist, 'MATRIX')
            self.energy_low = hdulist[matrix_idx].data.field("ENERG_LO")
            self.energy_high = hdulist[matrix_idx].data.field("ENERG_HI")
            
            # Find EBOUNDS extension
            ebounds_idx = self._find_extension(hdulist, 'EBOUNDS')
            self.channel = hdulist[ebounds_idx].data.field("CHANNEL")
            self.ebounds_min = hdulist[ebounds_idx].data.field("E_MIN")
            self.ebounds_max = hdulist[ebounds_idx].data.field("E_MAX")

            # Handle different RMF formats
            matrix_data = hdulist[matrix_idx].data.field("MATRIX")
            if isinstance(matrix_data[0], np.ndarray):
                # Variable-length array format
                self._rmf = self._expand_variable_rmf(hdulist[matrix_idx].data)
            else:
                # Fixed format
                self._rmf = matrix_data

            
        # Update JAX arrays
        self._update_jax_arrays()

    def _expand_variable_rmf(self, rmf_data) -> np.ndarray:
        """
        Expand variable-length RMF format to full matrix
        
        Parameters
        ----------
        rmf_data : FITS data
            RMF data in variable-length format
            
        Returns
        -------
        np.ndarray
            Full RMF matrix
        """
        n_energy = len(rmf_data)
        n_channels = len(self.channel)
        
        # Initialize full matrix
        rmf_full = np.zeros((n_energy, n_channels))
        
        # Fill matrix
        for i, row in enumerate(rmf_data):
            n_grp = row['N_GRP']
            if n_grp == 0:
                continue
                
            f_chan = row['F_CHAN']
            n_chan = row['N_CHAN']
            matrix = row['MATRIX']
            
            # Handle multiple groups
            if isinstance(f_chan, np.ndarray):
                idx = 0
                for j in range(n_grp):
                    start = int(f_chan[j]) - 1  # FITS is 1-indexed
                    end = start + int(n_chan[j])
                    rmf_full[i, start:end] = matrix[idx:idx+n_chan[j]]
                    idx += n_chan[j]
            else:
                # Single group
                start = int(f_chan) - 1
                end = start + int(n_chan)
                rmf_full[i] = matrix

        return rmf_full

    def load_arf_file(self, filename: str):
        """
        Load ARF (Ancillary Response File)

        Parameters
        ----------
        filename : str
            Path to ARF FITS file
        """
        with fits.open(filename) as hdulist:
            specresp_idx = self._find_extension(hdulist, 'SPECRESP')
            self._arf = hdulist[specresp_idx].data.field('SPECRESP')
            
            # Verify energy bins match RMF if loaded
            if self.energy_low is not None:
                arf_elo = hdulist[specresp_idx].data.field('ENERG_LO')
                arf_ehi = hdulist[specresp_idx].data.field('ENERG_HI')
                
                if not (np.allclose(arf_elo, self.energy_low) and 
                        np.allclose(arf_ehi, self.energy_high)):
                    warnings.warn("ARF energy bins don't match RMF energy bins")
        
        # Update JAX arrays
        self._update_jax_arrays()

    @staticmethod
    def _find_extension(hdulist, ext_name: str) -> int:
        """Find extension by name in FITS file"""
        for i in range(len(hdulist)):
            if 'EXTNAME' in hdulist[i].header:
                ext = hdulist[i].header['EXTNAME']
                if ext.lower() == ext_name.lower():
                    return i
        raise FitsFileError(f"Extension '{ext_name}' not found in FITS file")

    def fold_model(self, model_flux: Union[np.ndarray, jnp.ndarray]) -> jnp.ndarray:
        """
        Fold model through response matrix
        
        Parameters
        ----------
        model_flux : array-like
            Model photon flux at incident energies
            
        Returns
        -------
        jnp.ndarray
            Predicted counts in detector channels
        """
        if isinstance(model_flux, np.ndarray):
            model_flux = jnp.array(model_flux)
        
        return model_flux @ self.drm_jax

    def effective_area(self, energy: Optional[Union[float, np.ndarray]] = None) -> Union[float, np.ndarray]:
        """
        Get effective area at given energy/energies
        
        Parameters
        ----------
        energy : float or array-like, optional
            Energy/energies in keV. If None, returns ARF array
            
        Returns
        -------
        float or array
            Effective area in cm^2
        """
        if self._arf is None:
            raise ValueError("No ARF loaded")
            
        if energy is None:
            return self._arf
        
        # Simple linear interpolation
        energy = np.atleast_1d(energy)
        return np.interp(energy, self.energy_centroid, self._arf)

    def energy_resolution(self, energy: float, channel: Optional[int] = None) -> float:
        """
        Estimate energy resolution (FWHM) at given energy
        
        Parameters
        ----------
        energy : float
            Energy in keV
        channel : int, optional
            Specific channel. If None, uses channel closest to energy
            
        Returns
        -------
        float
            FWHM energy resolution in keV
        """
        if channel is None:
            # Find channel closest to energy
            channel = np.argmin(np.abs(self.ebounds_centroid - energy))
        
        # Find energy bin containing this energy
        energy_bin = np.argmin(np.abs(self.energy_centroid - energy))
        
        # Get response for this energy bin
        response = self._rmf[energy_bin, :]
        
        # Calculate weighted standard deviation
        if np.sum(response) > 0:
            mean = np.sum(response * self.ebounds_centroid) / np.sum(response)
            variance = np.sum(response * (self.ebounds_centroid - mean)**2) / np.sum(response)
            fwhm = 2.355 * np.sqrt(variance)  # Convert std to FWHM
            return fwhm
        else:
            return 0.0

    def plot_response(self, energy_bins: Optional[List[int]] = None, 
                     ax: Optional['matplotlib.axes.Axes'] = None):
        """
        Plot response matrix
        
        Parameters
        ----------
        energy_bins : list of int, optional
            Specific energy bins to plot. If None, plots full matrix
        ax : matplotlib axes, optional
            Axes to plot on
        """
        import matplotlib.pyplot as plt
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 8))
        
        if energy_bins is None:
            # Plot full matrix
            im = ax.imshow(self._rmf.T, aspect='auto', origin='lower',
                          extent=[self.energy_low[0], self.energy_high[-1],
                                  self.ebounds_min[0], self.ebounds_max[-1]])
            ax.set_xlabel('Incident Energy (keV)')
            ax.set_ylabel('Detected Energy (keV)')
            ax.set_title('Response Matrix')
            plt.colorbar(im, ax=ax, label='Response')
        else:
            # Plot specific energy bins
            for ebin in energy_bins:
                ax.plot(self.ebounds_centroid, self._rmf[ebin, :],
                       label=f'E = {self.energy_centroid[ebin]:.1f} keV')
            ax.set_xlabel('Detected Energy (keV)')
            ax.set_ylabel('Response')
            ax.set_title('Energy Redistribution')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        return ax

    def __repr__(self) -> str:
        return (f"Response(n_energy_bins={self.n_energy_bins}, "
                f"n_channels={self.n_channels}, "
                f"has_arf={self._arf is not None})")


class FitsFileError(Exception):
    """FITS file error"""
    pass
