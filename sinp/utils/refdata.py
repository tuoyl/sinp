import pooch
from astropy.io import fits
import numpy as np

# Configure the file manager
table_manager = pooch.create(
    path=pooch.os_cache("jaxspec"),
    base_url="https://github.com/renecotyfanboy/jaxspec-database/raw/main/",
    registry={
        "xsect_tbabs_wilm.fits": "sha256:3cf45e45c9d671c4c4fc128314b7c3a68b30f096eede6b3eb08bf55224a44935",
    }
)

# Load cross-section data into memory
def load_tbabs_cross_section():
    file_path = table_manager.fetch("xsect_tbabs_wilm.fits")
    with fits.open(file_path) as hdul:
        energy = hdul[1].data["ENERGY"]  # keV
        sigma = hdul[1].data["SIGMA"]  # cm^2
    return energy, sigma
