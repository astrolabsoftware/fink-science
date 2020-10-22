import numpy as np
from astropy.table import Table


class LightCurve:

    def __init__(self, data, object_id):

        self.df = Table.from_pandas(data)
        self.object_id = object_id

        self.brightness_col_name = 'FLUXCAL'
        self.brightness_err_col_name = 'FLUXCALERR'
        self.band_col_name = 'FLT'
        self.time_col_name = 'MJD'

    def calc_priodic_penalty(self):
        """ Calculate the period penalty for each event

            Returns
            ----------
                penalty: np.float64
                    calculated periodic penalty for a given event
        """
        flux_and_error_diff = np.abs(self.df[self.brightness_col_name]) - np.abs(self.df[self.brightness_err_col_name])
        flux_err_ratio = np.abs(self.df[self.brightness_col_name]) > 2 * self.df[self.brightness_err_col_name]
        index = (np.abs(flux_and_error_diff) > 10) & flux_err_ratio
        object_df = self.df[index]

        if len(object_df) == 0:
            penalty = 0

        else:

            max_flux_pos = np.argmax(self.df[self.brightness_col_name])

            # length =len(object_df['mjd'])
            max_flux_date = object_df[self.time_col_name][max_flux_pos]
            max_flux_val = self.df[self.brightness_col_name][max_flux_pos]
            # print(object_df['flux_err'])

            time_from_max = np.abs(object_df[self.time_col_name] - max_flux_date)
            time_from_max[np.where(time_from_max < 7)] = 0
            # filtered_flux = object_df['flux']
            # normalization = np.sum(index_greater_than_half)
            penalty = np.sum(np.abs(self.df[self.brightness_col_name]) * np.abs(time_from_max)) / max_flux_val
            print(penalty)
            penalty = np.log(np.abs(penalty) + 1)
        return penalty
