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
siz = len(pdf)
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
pdf[["RAdeg", "DEdeg",  "SPICY", "class"]].to_parquet("spicy.parquet")
```

