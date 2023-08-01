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