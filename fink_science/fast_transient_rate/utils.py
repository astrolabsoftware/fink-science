import numpy as np


def to_flux(mag):
    # from Serguey Karpov
    return 10**(0.4*(27.5 - mag)) # FLUXCAL, mag = 27.5 - 2.5*np.log10(flux)

def to_fluxerr(magerr, flux):
    return magerr * flux * np.log(10)/2.5 # magerr = 2.5/np.log(10) * fluxerr / flux

def to_mag(flux):
    return 27.5 - 2.5*np.log10(flux)

def to_magerr(fluxerr, flux):
    return 2.5/np.log(10) * fluxerr / flux


def stack_columns(df, *cols):
    return list(np.dstack([df[c] for c in cols])[0])


def stack_column(col, N):
    return np.stack([col for _ in range(N)]).T