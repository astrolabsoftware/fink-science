# Classifying microlensing events with Fink

The idea of this module is to use the [LIA](https://github.com/dgodinez77/LIA) classifier to perform early detection of microlensing events (before the peak of the event $t_0$).

## Offline studies

With Tristan B., Etienne B. and Marc M., we investigated:
Step 0: Create a simulated dataset from ZTF DR2
Step 1: Integrate classification algorithm (LIA)
Step 2: Perform Early Detection of microlensing events (full light-curve).
Step 3: Perform Early Detection of microlensing events (alerts).
Step 4: Use on real microlensing data from ZTF

Results can be found in this [presentation](https://docs.google.com/presentation/d/1Te6aFJvHTysyEqJ8I7yQ5o1SEek864ddaFrrGYVDkww/edit?usp=sharing).

## Fink microlensing science module

### Training

The science module is for the moment very simple. It consists in loading alert data, and applying a pre-trained classifier.

The training has been done offline using simulated data based on ZTF DR2 (for the time sampling), and internal LIA simulation (for the amplitudes and shapes). We currently compute the models with:

```python
from LIA import training_set

# `times` is a representative set of ZTF DR2 time samplings
training_set.create(times[::1], min_mag=15, max_mag=20, noise=None, n_class=500)
```

We achieved ~72% accuracy using full light-curve information. If now we chunk full ligh-curves into alerts of 30 days (moving window), the accuracy goes down to 20%, and only half of it happens before the peak of the event (i.e. only less than 10% of total events are flagged before the peak by their alert data). The current training procedure, and input data set can be found at: https://github.com/Azhom/ZTF_mulens_simulator ([PR1](https://github.com/Azhom/ZTF_mulens_simulator/pull/1) for the exact version of this initial run).

### Installing LIA on the cluster

Unfortunately LIA is not distributed using pip or conda, hence

## Limitations and future upgrades

* LIA is currently adapted for single lens detection only.
* We use the default noise generator in LIA, that might not represent the noise in our data. We can do better!
* ZTF has 30 days of history attached with each alert. This is rather short for detecting microlensing events. LSST will have 1 year of historical data attached to each alert, which will enable better triggering capabilities.
* The classification is very sensitive to the time samplings. We should try training and testing on interpolated data to minimise the impact.
* We currently use the classifier as-is. But we could add additional information prior the classification, such as filtering events whose last 3 measurements are 3 sigma above the baseline, or cross-match information.
