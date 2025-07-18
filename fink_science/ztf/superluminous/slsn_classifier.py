import pandas as pd
import numpy as np
from light_curve.light_curve_py import RainbowFit
from light_curve.light_curve_py.features.rainbow._scaler import MultiBandScaler, Scaler
import warnings
from light_curve.light_curve_py import warnings as rainbow_warnings
warnings.filterwarnings("ignore", category=rainbow_warnings.ExperimentalWarning)
import sncosmo
from astropy.table import QTable, Table, Column
import light_curve as lcpckg
import fink_science.ztf.superluminous.kernel as kern
import joblib


def fit_rainbow(lc, rainbow_model):

    # Shift time
    lc['cjd'] = lc['cjd'] - lc['cjd'][np.argmax(lc['cflux'])]

    #Sort values 
    zipped = zip(lc['cjd'], lc['cflux'], lc['csigflux'], lc['cfid'])
    lc['cjd'], lc['cflux'], lc['csigflux'], lc['cfid'] = zip(*sorted(zipped, key=lambda x: x[0]))
    lc['cjd'], lc['cflux'], lc['csigflux'], lc['cfid'] = np.array(lc['cjd']), np.array(lc['cflux']), np.array(lc['csigflux']), np.array(lc['cfid'])

    t_scaler = Scaler.from_time(lc['cjd'])
    m_scaler = MultiBandScaler.from_flux(lc['cflux'], lc['cfid'], with_baseline=False)

    try:
        result, errors = rainbow_model._eval_and_get_errors(t=lc['cjd'],
                                     m=lc['cflux'],
                                     sigma=lc['csigflux'],
                                     band=lc['cfid'], debug=True)
        
        return list(result[:-1]) + list(result[:-1]/errors) + [result[-1]]
    
    except RuntimeError:
        return [np.nan] * (2 * len(rainbow_model.names) + 1)

def fit_salt(lc, salt_model):

    int_to_filter = {1: 'ztfg', 2: 'ztfr', 3: 'ztfi'}
    lc_table = Table(data = {'time': lc['cjd']-lc['cjd'][np.argmax(lc['cflux'])],
                         'band': [int_to_filter[k] for k in lc['cfid']],
                         'flux':lc['cflux'],
                         'fluxerr':lc['csigflux'],
                         'zp':[25.0] * len(lc['cjd']),
                         'zpsys':['ab'] * len(lc['cjd'])})
    
    try:
        # run the fit
        result, fitted_model = sncosmo.fit_lc(
            lc_table, salt_model,
            ['z', 't0', 'x0', 'x1', 'c'],  # parameters of model to vary
            bounds={'z':(0, 0.5)})  # bounds on parameters (if any)
    
        return list(result.parameters) + [result.chisq]

    except RuntimeError:
        return [np.nan] * 6


def statistical_features(lc):

    amplitude = lcpckg.Amplitude()
    kurtosis = lcpckg.Kurtosis()
    max_slope = lcpckg.MaximumSlope()
    skew = lcpckg.Skew()
    
    # Feature extractor, it will evaluate all features in more efficient way
    extractor = lcpckg.Extractor(amplitude, kurtosis, max_slope, skew)
    
    # Array with all 5 extracted features
    result = extractor(lc['cjd'], lc['cflux'], lc['csigflux'].astype(np.float64), sorted=True, check=True)
    return list(result)

def extract_features(data):  

    rainbow_model = RainbowFit.from_angstrom(kern.band_wave_aa, with_baseline=False,
                                    temperature=kern.temperature,
                                    bolometric=kern.bolometric)
    salt_model = sncosmo.Model(source='salt2')

    rainbow_pnames = rainbow_model.names
    salt_pnames = salt_model.param_names
    
    pdf = pd.DataFrame(columns = ['distnr', 'duration', 'flux_amplitude', 'kurtosis', 'max_slope', 'skew'] + rainbow_pnames + ['snr_' + k for k in rainbow_pnames] + ['chi2_rainbow'] + salt_pnames + ['chi2_salt'])
    
    for pdf_idx in range(len(data)):
    
        lc = data.iloc[pdf_idx].copy()
        
        all_valid_bands = all(kern.min_points_perband <= np.array([sum(lc['cfid']==band) for band in np.unique(lc['cfid'])]))
        enough_total_points = len(lc['cjd']) > kern.min_points_total
        duration = np.ptp(lc['cjd'])
        distnr = lc['distnr']
        
        if all_valid_bands & enough_total_points:
                rainbow_features = fit_rainbow(lc, rainbow_model)
                salt_features = fit_salt(lc, salt_model)
                stat_features = statistical_features(lc)

                row = [distnr, duration] + stat_features + rainbow_features + salt_features
                pdf.loc[pdf_idx] = row

        else:
            pdf.loc[pdf_idx] = [distnr, duration] + [np.nan] * (np.shape(pdf)[1] - 2)

    return pdf

