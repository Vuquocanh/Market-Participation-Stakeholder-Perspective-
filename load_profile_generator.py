import numpy as np


MIN_LOAD  = 220.0   # kW
MAX_LOAD  = 600.0   # kW
MAX_DELTA = 35.0    # kW per minute


def generate_load_profiles(n_profiles=300, n_minutes=60, seed=42):
    """
    Generate random load profiles for the FCR-D UP flexible load.

    Each profile is a 60-minute time series where:
      - consumption stays within [220, 600] kW
      - minute-to-minute change does not exceed 35 kW

    Returns
    -------
    profiles : ndarray, shape (n_profiles, n_minutes)
    """
    rng = np.random.default_rng(seed)
    profiles = np.empty((n_profiles, n_minutes))

    for i in range(n_profiles):
        profile = np.empty(n_minutes)
        profile[0] = rng.uniform(MIN_LOAD, MAX_LOAD)

        for t in range(1, n_minutes):
            prev = profile[t - 1]
            lo = max(MIN_LOAD, prev - MAX_DELTA)
            hi = min(MAX_LOAD, prev + MAX_DELTA)
            profile[t] = rng.uniform(lo, hi)

        profiles[i] = profile

    return profiles


def split_profiles(profiles, n_in_sample=100):
    """
    Split profiles into in-sample and out-of-sample sets.

    Returns
    -------
    in_sample     : ndarray, shape (n_in_sample, n_minutes)
    out_of_sample : ndarray, shape (n_profiles - n_in_sample, n_minutes)
    """
    return profiles[:n_in_sample], profiles[n_in_sample:]
