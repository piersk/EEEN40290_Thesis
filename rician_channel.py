from scipy.special import iv  # Modified Bessel function of the first kind
from scipy.stats import rice

def rician_fading_params(v: float, sigma: float):
    """
    Compute Rician K-factor and Omega from dominant signal and scattering.
    """
    K = (v ** 2) / (2 * sigma ** 2)
    Omega = v ** 2 + 2 * sigma ** 2
    return K, Omega


def rician_pdf(x: float, K: float, Omega: float):
    """
    Compute PDF value of the Rician distribution at x.
    """
    bessel_term = iv(0, 2 * x * np.sqrt(K * (K + 1) / Omega))
    return ((2 * (K + 1) * x) / Omega) * np.exp(-K - ((K + 1) * x ** 2) / Omega) * bessel_term


def rician_channel_gain(K: float, Omega: float):
    """
    Simulate a single fading gain sample using the Rician distribution.
    """
    # `rice.rvs` in scipy uses non-centrality parameter b = sqrt(K / (K + 1))
    b = np.sqrt(K / (K + 1))
    return rice.rvs(b=b, scale=np.sqrt(Omega / (2 * (K + 1))))
