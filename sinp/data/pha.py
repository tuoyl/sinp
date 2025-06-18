"""
JAX-optimized spectrum class for PHA (Pulse Height Analyzer) data
"""
import numpy as np
import jax.numpy as jnp
from astropy.io import fits
from typing import Optional, Union, Tuple, Dict
import warnings

__all__ = ['Spectrum', 'BackgroundSpectrum']


class Spectrum:
    """
    JAX-optimized spectrum class for X-ray spectral data
    """
    
    def __init__(self):
        """Initialize empty spectrum"""
        self.nchannel = 0
        self.exposure = 0.0
        self.channel = np.array([])
        self.counts = np.array([])
        self.rate = np.array([])
        self.error = np.array([])
        self.grouping = np.array([])
        self.quality = np.array([])
        self.is_poisson = True
        self.backscale = 1.0
        self.areascale = 1.0
        self.respfile = None
        self.backfile = None
        self.ancrfile = None
        
        # Metadata
        self.telescope = None
        self.instrument = None
        self.filter = None
        self.object = None
        self.ra = None
        self.dec = None
        self.obsid = None
        
        # Private attributes
        self._background = None
        self._response_obj = None
        
        # JAX arrays for efficient computation
        self._counts_jax = None
        self._error_jax = None
        self._grouping_jax = None
        self._quality_jax = None

    def _update_jax_arrays(self):
        """Update JAX arrays when data changes"""
        if self.counts.size > 0:
            self._counts_jax = jnp.array(self.counts)
            self._error_jax = jnp.array(self.error)
            if self.grouping.size > 0:
                self._grouping_jax = jnp.array(self.grouping)
            if self.quality.size > 0:
                self._quality_jax = jnp.array(self.quality)

    def load_pha_file(self, filename: str, extension: int = 1, 
                      rowid: Optional[int] = None):
        """
        Load spectrum from PHA FITS file
        
        Parameters
        ----------
        filename : str
            Path to PHA file
        extension : int
            FITS extension number (default: 1)
        rowid : int, optional
            Row number for Type II PHA files
        """
        with fits.open(filename) as hdulist:
            hdu = hdulist[extension]
            header = hdu.header
            
            # Check if it's counts or rate
            if 'COUNTS' in hdu.data.names:
                self.load_cspec_file(filename, extension, rowid)
            elif 'RATE' in hdu.data.names:
                self.load_rate_file(filename, extension, rowid)
            else:
                raise ValueError("No COUNTS or RATE column found in PHA file")
            
            # Load additional metadata
            self._load_metadata(header)
            
            # Load scaling factors
            if 'BACKSCAL' in header:
                self.backscale = header['BACKSCAL']
            elif 'BACKSCAL' in hdu.data.names:
                if rowid is not None:
                    self.backscale = hdu.data['BACKSCAL'][rowid]
                else:
                    self.backscale = hdu.data['BACKSCAL']
                    
            if 'AREASCAL' in header:
                self.areascale = header['AREASCAL']
            elif 'AREASCAL' in hdu.data.names:
                if rowid is not None:
                    self.areascale = hdu.data['AREASCAL'][rowid]
                else:
                    self.areascale = hdu.data['AREASCAL']

    def load_cspec_file(self, filename: str, extension: int = 1, 
                        rowid: Optional[int] = None):
        """
        Load counts spectrum from FITS file
        
        Parameters
        ----------
        filename : str
            Path to spectrum file
        extension : int
            FITS extension number
        rowid : int, optional
            Row number for Type II PHA files
        """
        with fits.open(filename) as hdulist:
            hdu = hdulist[extension]
            
            if rowid is None:
                # Type I PHA file
                self.channel = hdu.data.field("CHANNEL")
                self.nchannel = self.channel.size
                self.exposure = hdu.header['EXPOSURE']
                self.counts = hdu.data.field("COUNTS").astype(np.float64)
                self.rate = self.counts / self.exposure
                self.is_poisson = hdu.header.get('POISSERR', True)
                
                if self.is_poisson:
                    self.error = np.sqrt(self.counts)
                else:
                    self.error = hdu.data.field("STAT_ERR")
                
                # Optional columns
                if 'GROUPING' in hdu.data.names:
                    self.grouping = hdu.data.field("GROUPING")
                else:
                    self.grouping = np.ones_like(self.channel, dtype=np.int16)
                    
                if 'QUALITY' in hdu.data.names:
                    self.quality = hdu.data.field("QUALITY")
                else:
                    self.quality = np.zeros_like(self.channel, dtype=np.int16)
                    
            else:
                # Type II PHA file
                self.channel = np.arange(len(hdu.data.field("COUNTS")[rowid])) + 1
                self.nchannel = self.channel.size
                self.exposure = hdu.data.field('EXPOSURE')[rowid]
                self.counts = hdu.data.field("COUNTS")[rowid].astype(np.float64)
                self.rate = self.counts / self.exposure
                self.is_poisson = hdu.header.get('POISSERR', True)
                
                if self.is_poisson:
                    self.error = np.sqrt(self.counts)
                else:
                    self.error = hdu.data.field("STAT_ERR")[rowid]
                    
                if 'GROUPING' in hdu.data.names:
                    self.grouping = hdu.data.field("GROUPING")[rowid]
                else:
                    self.grouping = np.ones_like(self.channel, dtype=np.int16)
                    
                if 'QUALITY' in hdu.data.names:
                    self.quality = hdu.data.field("QUALITY")[rowid]
                else:
                    self.quality = np.zeros_like(self.channel, dtype=np.int16)
        
        # Update JAX arrays
        self._update_jax_arrays()

    def load_rate_file(self, filename: str, extension: int = 1, 
                       rowid: Optional[int] = None):
        """
        Load rate spectrum from FITS file
        
        Parameters
        ----------
        filename : str
            Path to spectrum file
        extension : int
            FITS extension number
        rowid : int, optional
            Row number for Type II PHA files
        """
        with fits.open(filename) as hdulist:
            hdu = hdulist[extension]
            
            if rowid is None:
                # Type I PHA file
                self.channel = hdu.data.field("CHANNEL")
                self.nchannel = self.channel.size
                self.exposure = hdu.header['EXPOSURE']
                self.rate = hdu.data.field("RATE").astype(np.float64)
                self.counts = self.rate * self.exposure
                self.is_poisson = hdu.header.get('POISSERR', True)
                
                if self.is_poisson:
                    self.error = np.sqrt(self.counts)
                else:
                    # Convert rate error to counts error
                    self.error = hdu.data.field("STAT_ERR") * self.exposure
                    
                if 'GROUPING' in hdu.data.names:
                    self.grouping = hdu.data.field("GROUPING")
                else:
                    self.grouping = np.ones_like(self.channel, dtype=np.int16)
                    
                if 'QUALITY' in hdu.data.names:
                    self.quality = hdu.data.field("QUALITY")
                else:
                    self.quality = np.zeros_like(self.channel, dtype=np.int16)
                    
            else:
                # Type II PHA file
                self.channel = np.arange(len(hdu.data.field("RATE")[rowid])) + 1
                self.nchannel = self.channel.size
                self.exposure = hdu.data.field('EXPOSURE')[rowid]
                self.rate = hdu.data.field("RATE")[rowid].astype(np.float64)
                self.counts = self.rate * self.exposure
                self.is_poisson = hdu.header.get('POISSERR', True)
                
                if self.is_poisson:
                    self.error = np.sqrt(self.counts)
                else:
                    self.error = hdu.data.field("STAT_ERR")[rowid] * self.exposure
                    
                if 'GROUPING' in hdu.data.names:
                    self.grouping = hdu.data.field("GROUPING")[rowid]
                else:
                    self.grouping = np.ones_like(self.channel, dtype=np.int16)
                    
                if 'QUALITY' in hdu.data.names:
                    self.quality = hdu.data.field("QUALITY")[rowid]
                else:
                    self.quality = np.zeros_like(self.channel, dtype=np.int16)
        
        # Update JAX arrays
        self._update_jax_arrays()

    def _load_metadata(self, header: fits.Header):
        """Load metadata from FITS header"""
        self.telescope = header.get('TELESCOP', None)
        self.instrument = header.get('INSTRUME', None)
        self.filter = header.get('FILTER', None)
        self.object = header.get('OBJECT', None)
        self.ra = header.get('RA', None)
        self.dec = header.get('DEC', None)
        self.obsid = header.get('OBS_ID', None)
        self.respfile = header.get('RESPFILE', None)
        self.backfile = header.get('BACKFILE', None)
        self.ancrfile = header.get('ANCRFILE', None)

    @property
    def net_counts(self) -> np.ndarray:
        """Net counts after background subtraction"""
        if self._background is None:
            return self.counts
        
        # Scale background to source extraction region
        bkg_scaled = self._background.counts * (self.backscale / self._background.backscale)
        return self.counts - bkg_scaled

    @property
    def net_counts_jax(self) -> jnp.ndarray:
        """JAX array of net counts"""
        if self._background is None:
            return self._counts_jax
        
        # Scale background
        scale_factor = self.backscale / self._background.backscale
        return self._counts_jax - self._background._counts_jax * scale_factor

    @property
    def net_cspec(self) -> np.ndarray:
        """Net counts spectrum (alias for net_counts)"""
        return self.net_counts

    @property
    def net_rate(self) -> np.ndarray:
        """Net count rate after background subtraction"""
        return self.net_counts / self.exposure

    @property
    def net_error(self) -> np.ndarray:
        """Error on net counts"""
        if self._background is None:
            return self.error
        
        # Propagate errors including scaling
        src_var = self.error**2
        bkg_scale = self.backscale / self._background.backscale
        bkg_var = (self._background.error * bkg_scale)**2
        
        return np.sqrt(src_var + bkg_var)

    @property
    def net_error_jax(self) -> jnp.ndarray:
        """JAX array of net count errors"""
        if self._background is None:
            return self._error_jax
        
        src_var = self._error_jax**2
        bkg_scale = self.backscale / self._background.backscale
        bkg_var = (self._background._error_jax * bkg_scale)**2
        
        return jnp.sqrt(src_var + bkg_var)

    @property
    def background(self) -> Optional['BackgroundSpectrum']:
        """Background spectrum object"""
        return self._background

    @background.setter
    def background(self, background_file: Union[str, 'BackgroundSpectrum']):
        """Set background spectrum from file or object"""
        if isinstance(background_file, str):
            back_obj = BackgroundSpectrum()
            back_obj.load_pha_file(background_file)
            self._background = back_obj
        elif isinstance(background_file, BackgroundSpectrum):
            self._background = background_file
        else:
            raise TypeError("Background must be filename or BackgroundSpectrum object")

    @property
    def response_obj(self) -> Optional['Response']:
        """Response object"""
        return self._response_obj

    @response_obj.setter
    def response_obj(self, obj: 'Response'):
        """Set response object"""
        self._response_obj = obj

    @property
    def energy_centroid(self) -> Optional[np.ndarray]:
        """Energy centroids from response"""
        if self._response_obj is None:
            return None
        return self._response_obj.ebounds_centroid

    @property
    def energy_low(self) -> Optional[np.ndarray]:
        """Lower energy boundaries from response"""
        if self._response_obj is None:
            return None
        return self._response_obj.ebounds_min

    @property
    def energy_high(self) -> Optional[np.ndarray]:
        """Upper energy boundaries from response"""
        if self._response_obj is None:
            return None
        return self._response_obj.ebounds_max

    def group_channels(self, min_counts: Optional[int] = None, 
                      min_sigma: Optional[float] = None,
                      max_bins: Optional[int] = None):
        """
        Group channels to improve statistics
        
        Parameters
        ----------
        min_counts : int, optional
            Minimum counts per grouped bin
        min_sigma : float, optional
            Minimum significance per grouped bin
        max_bins : int, optional
            Maximum number of bins after grouping
        """
        if min_counts is not None:
            self._group_min_counts(min_counts)
        elif min_sigma is not None:
            self._group_min_sigma(min_sigma)
        elif max_bins is not None:
            self._group_max_bins(max_bins)
        else:
            raise ValueError("Must specify grouping criterion")

    def _group_min_counts(self, min_counts: int):
        """Group channels to have minimum counts per bin"""
        self.grouping = np.ones(self.nchannel, dtype=np.int16)
        self.quality = np.zeros(self.nchannel, dtype=np.int16)
        
        current_counts = 0
        for i in range(self.nchannel):
            if current_counts == 0:
                self.grouping[i] = 1
            else:
                self.grouping[i] = -1
                
            current_counts += self.net_counts[i]
            
            if current_counts >= min_counts:
                current_counts = 0

    def _group_min_sigma(self, min_sigma: float):
        """Group channels to have minimum significance per bin"""
        self.grouping = np.ones(self.nchannel, dtype=np.int16)
        self.quality = np.zeros(self.nchannel, dtype=np.int16)
        
        current_counts = 0
        current_error_sq = 0
        
        for i in range(self.nchannel):
            if current_counts == 0:
                self.grouping[i] = 1
            else:
                self.grouping[i] = -1
                
            current_counts += self.net_counts[i]
            current_error_sq += self.net_error[i]**2
            
            if current_counts > 0:
                significance = current_counts / np.sqrt(current_error_sq)
                if significance >= min_sigma:
                    current_counts = 0
                    current_error_sq = 0

    def _group_max_bins(self, max_bins: int):
        """Group channels to have maximum number of bins"""
        # Simple linear grouping for now
        channels_per_bin = self.nchannel // max_bins
        remainder = self.nchannel % max_bins
        
        self.grouping = np.ones(self.nchannel, dtype=np.int16)
        
        idx = 0
        for i in range(max_bins):
            # First bin of group
            self.grouping[idx] = 1
            idx += 1
            
            # Rest of channels in group
            n_in_group = channels_per_bin - 1
            if i < remainder:
                n_in_group += 1
                
            for j in range(n_in_group):
                if idx < self.nchannel:
                    self.grouping[idx] = -1
                    idx += 1

    def apply_grouping(self) -> Dict[str, np.ndarray]:
        """
        Apply grouping to get grouped spectrum
        
        Returns
        -------
        dict
            Dictionary with grouped data:
            - 'counts': grouped counts
            - 'error': grouped errors
            - 'channel': grouped channel numbers
            - 'exposure': exposure time
        """
        if self.grouping.size == 0:
            raise ValueError("No grouping defined")
            
        # Find group starts
        group_starts = np.where(self.grouping == 1)[0]
        n_groups = len(group_starts)
        
        # Initialize grouped arrays
        grouped_counts = np.zeros(n_groups)
        grouped_error_sq = np.zeros(n_groups)
        grouped_channels = np.zeros(n_groups)
        
        # Apply grouping
        for i, start in enumerate(group_starts):
            # Find end of group
            if i < n_groups - 1:
                end = group_starts[i + 1]
            else:
                end = self.nchannel
                
            # Sum counts and propagate errors
            mask = (self.quality[start:end] == 0)  # Good quality channels
            grouped_counts[i] = np.sum(self.net_counts[start:end][mask])
            grouped_error_sq[i] = np.sum(self.net_error[start:end][mask]**2)
            grouped_channels[i] = np.mean(self.channel[start:end][mask])
        
        return {
            'counts': grouped_counts,
            'error': np.sqrt(grouped_error_sq),
            'channel': grouped_channels,
            'exposure': self.exposure,
            'n_channels': n_groups
        }

    def ignore_channels(self, channel_ranges: list):
        """
        Ignore channels by setting quality flag
        
        Parameters
        ----------
        channel_ranges : list of tuples
            List of (low, high) channel ranges to ignore
        """
        for low, high in channel_ranges:
            mask = (self.channel >= low) & (self.channel <= high)
            self.quality[mask] = 2  # Bad quality flag

    def notice_channels(self, channel_ranges: list):
        """
        Notice (use) channels by clearing quality flag
        
        Parameters
        ----------
        channel_ranges : list of tuples
            List of (low, high) channel ranges to use
        """
        # First ignore all channels
        self.quality[:] = 2
        
        # Then notice specified ranges
        for low, high in channel_ranges:
            mask = (self.channel >= low) & (self.channel <= high)
            self.quality[mask] = 0  # Good quality flag

    def ignore_energy(self, energy_ranges: list):
        """
        Ignore energy ranges by setting quality flag
        
        Parameters
        ----------
        energy_ranges : list of tuples
            List of (low, high) energy ranges in keV to ignore
        """
        if self._response_obj is None:
            raise ValueError("No response loaded - cannot convert energy to channels")
            
        for low, high in energy_ranges:
            mask = (self.energy_centroid >= low) & (self.energy_centroid <= high)
            self.quality[mask] = 2

    def notice_energy(self, energy_ranges: list):
        """
        Notice (use) energy ranges by clearing quality flag
        
        Parameters
        ----------
        energy_ranges : list of tuples
            List of (low, high) energy ranges in keV to use
        """
        if self._response_obj is None:
            raise ValueError("No response loaded - cannot convert energy to channels")
            
        # First ignore all channels
        self.quality[:] = 2
        
        # Then notice specified ranges
        for low, high in energy_ranges:
            mask = (self.energy_centroid >= low) & (self.energy_centroid <= high)
            self.quality[mask] = 0

    def get_good_channels(self) -> np.ndarray:
        """Get mask of good quality channels"""
        if self.quality.size == 0:
            return np.ones(self.nchannel, dtype=bool)
        return self.quality == 0

    def plot(self, ax=None, plot_background=False, plot_errors=True,
             energy_units=True, **kwargs):
        """
        Plot spectrum
        
        Parameters
        ----------
        ax : matplotlib axes, optional
            Axes to plot on
        plot_background : bool
            Whether to plot background spectrum
        plot_errors : bool
            Whether to show error bars
        energy_units : bool
            Whether to use energy units (requires response)
        **kwargs
            Additional keyword arguments for plotting
        """
        import matplotlib.pyplot as plt
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
            
        # Determine x-axis
        if energy_units and self._response_obj is not None:
            x = self.energy_centroid
            x_low = self.energy_low
            x_high = self.energy_high
            xlabel = 'Energy (keV)'
        else:
            x = self.channel
            x_low = self.channel - 0.5
            x_high = self.channel + 0.5
            xlabel = 'Channel'
            
        # Get good channels
        good = self.get_good_channels()
        
        # Plot data
        if plot_errors:
            ax.errorbar(x[good], self.rate[good], yerr=self.error[good]/self.exposure,
                       xerr=[x[good] - x_low[good], x_high[good] - x[good]],
                       fmt='o', capsize=0, label='Source', **kwargs)
        else:
            ax.scatter(x[good], self.rate[good], label='Source', **kwargs)
            
        # Plot background if requested
        if plot_background and self._background is not None:
            bkg_rate = self._background.rate * (self.backscale / self._background.backscale)
            if plot_errors:
                bkg_error = self._background.error * (self.backscale / self._background.backscale)
                ax.errorbar(x[good], bkg_rate[good], 
                           yerr=bkg_error[good]/self.exposure,
                           xerr=[x[good] - x_low[good], x_high[good] - x[good]],
                           fmt='s', capsize=0, label='Background', alpha=0.5)
            else:
                ax.scatter(x[good], bkg_rate[good], label='Background', alpha=0.5)
        
        ax.set_xlabel(xlabel)
        ax.set_ylabel('Count Rate (counts/s)')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add title with metadata
        title_parts = []
        if self.object:
            title_parts.append(f"Object: {self.object}")
        if self.obsid:
            title_parts.append(f"ObsID: {self.obsid}")
        if self.exposure:
            title_parts.append(f"Exposure: {self.exposure:.1f}s")
        if title_parts:
            ax.set_title(', '.join(title_parts))
        
        return ax

    def write(self, filename: str, overwrite: bool = False):
        """
        Write spectrum to FITS file
        
        Parameters
        ----------
        filename : str
            Output filename
        overwrite : bool
            Whether to overwrite existing file
        """
        # Create columns
        cols = []
        cols.append(fits.Column(name='CHANNEL', format='I', array=self.channel))
        cols.append(fits.Column(name='COUNTS', format='E', array=self.counts))
        
        if not self.is_poisson:
            cols.append(fits.Column(name='STAT_ERR', format='E', array=self.error))
            
        if self.grouping.size > 0:
            cols.append(fits.Column(name='GROUPING', format='I', array=self.grouping))
            
        if self.quality.size > 0:
            cols.append(fits.Column(name='QUALITY', format='I', array=self.quality))
        
        # Create HDU
        hdu = fits.BinTableHDU.from_columns(cols)
        
        # Add header keywords
        hdu.header['EXTNAME'] = 'SPECTRUM'
        hdu.header['TELESCOP'] = self.telescope or 'UNKNOWN'
        hdu.header['INSTRUME'] = self.instrument or 'UNKNOWN'
        hdu.header['FILTER'] = self.filter or 'NONE'
        hdu.header['EXPOSURE'] = self.exposure
        hdu.header['BACKSCAL'] = self.backscale
        hdu.header['AREASCAL'] = self.areascale
        hdu.header['POISSERR'] = self.is_poisson
        
        if self.respfile:
            hdu.header['RESPFILE'] = self.respfile
        if self.backfile:
            hdu.header['BACKFILE'] = self.backfile
        if self.ancrfile:
            hdu.header['ANCRFILE'] = self.ancrfile
            
        # Create primary HDU
        primary = fits.PrimaryHDU()
        
        # Write file
        hdul = fits.HDUList([primary, hdu])
        hdul.writeto(filename, overwrite=overwrite)

    def __repr__(self) -> str:
        return (f"Spectrum(nchannel={self.nchannel}, "
                f"exposure={self.exposure:.1f}s, "
                f"telescope={self.telescope}, "
                f"has_background={self._background is not None})")


class BackgroundSpectrum(Spectrum):
    """
    Background spectrum class
    
    Inherits all functionality from Spectrum class
    """
    
    def __init__(self):
        super().__init__()
        
    def __repr__(self) -> str:
        return (f"BackgroundSpectrum(nchannel={self.nchannel}, "
                f"exposure={self.exposure:.1f}s, "
                f"telescope={self.telescope})")