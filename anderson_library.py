import numpy as np
from scipy.stats import anderson, norm, uniform, expon, gamma, beta, lognorm, weibull_min, weibull_max


# Custom Anderson-Darling test for various distributions
def anderson_darling_test(x, dist='gamma'):
    """
    Anderson-Darling test for data coming from a specific distribution.

    Parameters
    ----------
    x : array_like
        Array of sample data.
    dist : str
        The type of distribution to test against. One of 'norm', 'uniform', 'expon', 'gamma', 'beta', 'lognorm', 'weibull_min', 'weibull_max'.

    Returns
    -------
    statistic : float
        The Anderson-Darling test statistic.
    critical_values : array
        The critical values for the test.
    significance_level : array
        The significance levels for the test.
    """
    # Fit the chosen distribution to the data
    if dist == 'norm':
        params = norm.fit(x)
        cdf_func = lambda x: norm.cdf(x, *params)
        critical_values = [0.576, 0.656, 0.787, 0.918, 1.092]
        significance_level = [15, 10, 5, 2.5, 1]
    elif dist == 'uniform':
        params = uniform.fit(x)
        cdf_func = lambda x: uniform.cdf(x, *params)
        critical_values = [0.675, 0.785, 0.918, 1.092, 1.340]
        significance_level = [15, 10, 5, 2.5, 1]
    elif dist == 'expon':
        return anderson(x, dist='expon')
        critical_values = [0.922, 1.078, 1.341, 1.606, 1.957]
        significance_level = [15, 10, 5, 2.5, 1]
    elif dist == 'gamma':
        params = gamma.fit(x)
        cdf_func = lambda x: gamma.cdf(x, *params)
        # These are approximate values; for accurate testing, refer to specific statistical tables
        critical_values = [0.576, 0.656, 0.787, 0.918, 1.092]
        significance_level = [15, 10, 5, 2.5, 1]
    elif dist == 'beta':
        params = beta.fit(x)
        cdf_func = lambda x: beta.cdf(x, *params)
        # Approximate values
        critical_values = [0.576, 0.656, 0.787, 0.918, 1.092]
        significance_level = [15, 10, 5, 2.5, 1]
    elif dist == 'lognorm':
        params = lognorm.fit(x)
        cdf_func = lambda x: lognorm.cdf(x, *params)
        # Approximate values
        critical_values = [0.576, 0.656, 0.787, 0.918, 1.092]
        significance_level = [15, 10, 5, 2.5, 1]
    elif dist == 'weibull_min':
        params = weibull_min.fit(x)
        cdf_func = lambda x: weibull_min.cdf(x, *params)
        # Approximate values
        critical_values = [0.576, 0.656, 0.787, 0.918, 1.092]
        significance_level = [15, 10, 5, 2.5, 1]
    elif dist == 'weibull_max':
        params = weibull_max.fit(x)
        cdf_func = lambda x: weibull_max.cdf(x, *params)
        # Approximate values
        critical_values = [0.576, 0.656, 0.787, 0.918, 1.092]
        significance_level = [15, 10, 5, 2.5, 1]
    else:
        raise ValueError(
            "Invalid distribution type. Must be one of 'norm', 'uniform', 'expon', 'gamma', 'beta', 'lognorm', 'weibull_min', 'weibull_max'.")

    # Calculate the Anderson-Darling test statistic
    n = len(x)
    sorted_x = np.sort(x)
    cdf_vals = cdf_func(sorted_x)

    S = np.sum((2 * np.arange(1, n + 1) - 1) * (np.log(cdf_vals) + np.log(1 - cdf_vals[::-1])))
    A2 = -n - S / n

    return A2, critical_values, significance_level