# Copyright 2020-2023 AstroLab Software
# Author: Roman Le Montagner
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import numpy as np


def to_flux(mag):
    # from Sergey Karpov
    return 10 ** (0.4 * (27.5 - mag))  # FLUXCAL, mag = 27.5 - 2.5*np.log10(flux)


def to_fluxerr(magerr, flux):
    return magerr * flux * np.log(10) / 2.5  # magerr = 2.5/np.log(10) * fluxerr / flux


def to_mag(flux):
    return 27.5 - 2.5 * np.log10(flux)


def to_magerr(fluxerr, flux):
    return 2.5 / np.log(10) * fluxerr / flux


def stack_columns(df, *cols):
    return list(np.dstack([df[c] for c in cols])[0])


def stack_column(col, N):
    return np.stack([col for _ in range(N)]).T
