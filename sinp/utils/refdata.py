import importlib.resources as pkg_resources
from astropy.io import fits

# Load cross-section data into memory
def load_tbabs_cross_section():

    with pkg_resources.files("sinp.utils.absorption").joinpath("xsect_tbabs_wilm.fits").open("rb") as f:
        with fits.open(f) as hdulist:
            energy = hdulist[1].data["ENERGY"]  # keV
            sigma = hdulist[1].data["SIGMA"]  # cm^2
    return energy, sigma

def load_phabs_cross_section():

    with pkg_resources.files("sinp.utils.absorption").joinpath("xsect_phabs_aspl.fits").open("rb") as f:
        with fits.open(f) as hdulist:
            energy = hdulist[1].data["ENERGY"]  # keV
            sigma = hdulist[1].data["SIGMA"]  # cm^2
    return energy, sigma
