# Manual catalog creation

## VSX

```bash
wget https://cdsarc.cds.unistra.fr/viz-bin/nph-Cat/fits?B/vsx/vsx.dat -O B_vsx_vsx.dat.fits
```

Then:

```python
from astropy.io import fits
import pandas as pd

data = fits.open("B_vsx_vsx.dat.fits")
table = data[1].data
pdf = pd.DataFrame(table)
pdf = pdf[["RAdeg", "DEdeg",  "Type"]]
pdf["Type"] = pdf["Type"].apply(lambda x: x.strip())
size = len(pdf)
!mkdir vsx
for index, i in enumerate(range(0, size, int(size/10)+1)):
    pdf.loc[i:i+int(size/10)].to_parquet(f"vsx/vsx_{index}.parquet")
```

## SPICY

```
wget https://cdsarc.cds.unistra.fr/viz-bin/nph-Cat/fits?J/ApJS/254/33/table1.dat.gz -O spicy_20260216.fits
```

```python
from astropy.io import fits
import pandas as pd

data = fits.open("spicy")
table = data[1].data
pdf = pd.DataFrame(table)
pdf[["RAdeg", "DEdeg", "SPICY", "class"]].to_parquet("spicy.parquet")
```

## REGALADE (`regalade_minimal_ZTF.fits`)

Used by `ztf.superluminous.slsn_classifier.get_regalade_photoz` for host
galaxy crossmatch and photo-z. Source: REGALADE v2
(https://github.com/htranin/regalade), reduced to the 7 columns needed
(`gal_ra`, `gal_dec`, `R1`, `R2`, `PA`, `z`, `z_err`) and to the
ZTF-visible sky (`dec > -30`).

**Not committed to git or bundled in the pip package** -- unlike the other
catalogs here, it is ~1.5GB, far above GitHub's 100MB push limit and
impractical to ship in a wheel/sdist. It must be placed manually at
`kernel.regalade_path` (`fink_science/data/catalogs/regalade_minimal_ZTF.fits`)
before running anything that imports `slsn_classifier` in production
(`get_regalade_photoz` reads it lazily on first use, so import itself
does not require the file to be present).

TODO: distribution story not finalized yet (external host + download
link, git-lfs, or something else) -- see the training pipeline's
`create_photoz_table.py` for how the file itself is built from the full
REGALADE catalog.

