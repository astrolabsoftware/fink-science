import pandas as pd
from fink_tns.utils import download_catalog

with open("../../tns/tns_marker.txt") as f:
    tns_marker = f.read().replace("\n", "")

pdf_tns = download_catalog("../../tns/tns_api.key", tns_marker)
pdf = pd.read_parquet("../data/alerts/datatest")

# keep only confirmed
f1 = ~pdf_tns["type"].isna()
pdf_tns_filt = pdf_tns[f1]

# stupid xmatch by name
pdf_tns_filt["objectId"] = pdf_tns_filt["internal_names"]

pdf_x = pd.merge(pdf, pdf_tns_filt, on="objectId", how="left")
pdf_x = pdf_x[~pdf_x["type"].isnull()]
pdf_x = pdf_x[pdf_tns.columns]
pdf_x.to_parquet("../data/catalogs/tns.parquet")
