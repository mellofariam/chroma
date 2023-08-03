import numpy as np


def auto_corr(data):
    """Fourier transform implementation"""

    # Nearest size with power of 2
    size = 2 ** np.ceil(np.log2(2 * len(data) - 1)).astype("int")

    # Variance
    var = np.var(data)

    # Normalized data
    ndata = data - np.mean(data)

    # Compute the FFT
    fft = np.fft.fft(ndata, size)

    # Get the power spectrum
    pwr = np.abs(fft) ** 2

    # Calculate the autocorrelation from inverse FFT of the power spectrum
    acorr = np.fft.ifft(pwr).real / var / len(data)

    return acorr[: len(data)]


def calc_reconfig_time(acorr_data, lag_times=None, thresh=0):
    if lag_times is None:
        lag_times = np.array(range(len(acorr_data)))

    assert len(acorr_data) == len(
        lag_times
    ), "Correlation data and Lag Times should have the same size"

    i = 0
    while acorr_data[i] >= 0 + thresh:
        i += 1

    x = lag_times[:i]
    log_data = np.log(acorr_data[:i])
    curve_fit = np.polyfit(x, log_data, 1)

    reconfig_time = 1 / np.negative(curve_fit[0])

    y = [np.exp(-t / reconfig_time) for t in x]

    return reconfig_time, x, y
