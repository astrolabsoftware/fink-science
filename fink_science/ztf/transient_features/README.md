# Features for static transients

This module is based on the work of 
1. https://ui.adsabs.harvard.edu/abs/2020ApJ...904...35P/abstract
2. https://ui.adsabs.harvard.edu/abs/2020ApJ...895...32F/abstract
3. https://zenodo.org/records/4054129

we identified 6 new features to add:
1. `faint`: `magpsf` is currently fainter than 19.8, or the source had a very recent detection fainter than 19.
2. `pointunderneath`:  is likely sitting on top of or blended with a star in Pan-STARRS.
3. `positivesubtraction`: is brighter than the template image.
4. `real`: is likely a genuine astrophysical transient and not an artifact.
5. `stationary`: is not a moving source.
6. `brightstar`: is likely contaminated by a nearby bright star.
7. `variablesource`: is likely a variable star

Note that what is called `rock` in the initial filter is not necessary here as we already provide a column `roid` to assess the asteroid nature of an object.
