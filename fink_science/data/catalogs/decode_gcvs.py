import io
import numpy as np
import pandas as pd

from astropy.coordinates import SkyCoord
from astropy import units as u

def format_gcvs(pdf):
    ra = pdf['RAh'] + ':' + pdf['RAm'] + ':' + pdf['RAs']
    dec = pdf['DE-'] + pdf['DEd'] + ':' + pdf['DEm'] + ':' + pdf['DEs']
    mask = ra == '  :  :     '
    catalog = SkyCoord(ra[~mask], dec[~mask], unit=[u.hourangle, u.deg])

    pdf = pdf[~mask]
    pdf['ra'] = np.array(catalog.ra.deg, dtype=np.double)
    pdf['dec'] = np.array(catalog.dec.deg, dtype=np.double)
    return pdf

with open('gcvs5.txt', 'rb') as f:
    data = f.read()

buf = io.BytesIO(data)

header = [
    'Constell', 'Number', 'Component', 'GCVS', 'NoteFlag',
    'RAh', 'RAm', 'RAs', 'DE-', 'DEd', 'DEm', 'DEs', 'u_DEs',
    'VarType', 'l_magMax', 'magMax', 'u_magMax', 'l_magMinI',
    'magMinI', 'u_magMinI', 'n_magMinI', 'f_magMinI', 'l_magMinII',
    'magMinII', 'u_magMinII', 'n_magMinII', 'f_magMinII', 'magCode',
    'Epoch', 'q_Epoch', 'YearNova', 'q_Year', 'l_Period', 'Period',
    'u_Period', 'n_Period', 'M-m/D', 'u_M-m/D', 'n_M-m/D', 'SpType',
    '*Ref1', '*Ref2', '*Exists', 'PMa', 'PMd', 'Epoch coor', 'u_Ident',
    'Ident', 'VarTypeII', 'GCVSII'
]

bytes_step = [
    [0,2], [2,6], [6,7], [8,18], [18,19],
    [20,22], [22,24], [24,29], [30,31], [31,33], [33,35], [35,39],
    [39,40], [41,51], [52,53], [53,59], [59,60], [62,64], [64,70],
    [70,71], [71,73], [73,74], [75,77], [77,83], [83,84], [84,86],
    [86,87], [88,90], [91,102], [102,103], [104,108], [108,109],
    [110,111], [111,127], [127,128], [128,130], [131,134], [134,135],
    [135,136], [137,154], [155,160], [161,166], [167,178], [179,185],
    [186,192], [193, 201], [202,203], [204,213], [214,224], [225,235]
]

arr = []
data_ = buf.readline()

for line in buf.readlines():
    arr.append([line[i[0]:i[1]].decode('utf8') for i in bytes_step])

pdf = pd.DataFrame(arr, columns=header)
pdf = format_gcvs(pdf)

# Keep only useful columns for the crossmatch
cols = ['ra', 'dec', 'VarType']

pdf = pdf[cols]
pdf.to_parquet('gcvs.parquet', index=False)

print(pdf)

vsx_tmp = pd.read_csv('vsx_tmp.tsv', sep=';', header=None, names=['OID','Name','VType','Period','RAJ2000','DEJ2000'])

cols = ['RAJ2000', 'DEJ2000', 'VType']
vsx = vsx_tmp[cols]

vsx.to_parquet('vsx.parquet', index=False)
