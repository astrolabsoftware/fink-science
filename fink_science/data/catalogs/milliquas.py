from astropy.io import fits
import pandas as pd
import numpy as np

data = fits.open("milliquas.fits")

ra = data[1].data["RA"]
dec = data[1].data["DEC"]
type_ = data[1].data["TYPE"]
redshift = data[1].data["Z"]

pdf = pd.DataFrame(
    {
        "ra": np.array(ra, dtype=float),
        "dec": np.array(dec, dtype=float),
        "type": [i.strip() for i in np.array(type_, dtype=str)],
        "redshift": np.array(redshift, dtype=float),
    }
)

pdf.to_parquet("milliquas.parquet")
