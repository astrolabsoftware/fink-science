# Asteroid catcher

This module determines if an alert is a potential Solar System object (SSO) using two criteria:

1. The alert has been flagged as an SSO by ZTF (MPC) within 5"
2. The alert satisfies Fink criteria for a SSO
    1. No stellar counterpart from PS1, sgscore1 < 0.76 (Tachibana & Miller 2018)
    2. Number of detection is 1 or 2
    3. No Panstarrs counterpart within 1"
    4. If 2 detections, observations must be done within 30 min.

The alerts are labeled using:

	[3] if the alert has been flagged by ZTF as SSO candidate
	[2] if the alert has been flagged by Fink as SSO candidate
	[1] if is the first time ZTF sees this object
	[0] if it is likely not a SSO


## Todo

- Use postage stamps to flag lines that correspond to asteroids.
- Introduce new Fink criteria to flag future SSO. Ideally, this would be a student project to develop a real asteroid catcher (linked to ephemerids, and archives).
