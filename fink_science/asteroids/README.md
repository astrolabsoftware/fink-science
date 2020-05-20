# Asteroid catcher

This module determines if an alert is an asteroid using two criteria:
1. The alert has been flagged as an asteroid by ZTF (MPC) within 5"
2. First time the alert is seen, and |mag_r - mag_g| ~ 0.5

The alerts are labeled using:
- [3] if the asteroid has been flagged by ZTF
- [2] if the asteroid has been flagged by Fink
- [1] if is the first time ZTF sees this object
- [0] if it is likely not an asteroid

## Todo

- Use measurement times to compute the magnitude difference (we want close measurements between the two bands).
- Use timestamps to flag lines that correspond to asteroids.
- Introduce new Fink criteria to flag asteroids. Ideally, this would be a student project to develop a real asteroid catcher (linked to ephemerids, and archives).
