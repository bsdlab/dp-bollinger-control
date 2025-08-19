import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.signal import butter, decimate, hilbert, sosfiltfilt

d = loadmat(
    r"D:\data\dareplane\<path_to_ao_matfile>.mat",
    simplify_cells=True,
)
x = d["CECOG_HF_2___04___Array_3___04"]
sfreq = 22_000
tsfreq = 100

# first simplified callibration attempt, use causal filtering for actual calibration
sos = butter(8, (5, 8), "bandpass", fs=tsfreq, output="sos")
x = decimate(x, int(sfreq // tsfreq))
sfreq = tsfreq
xf = sosfiltfilt(sos, x)
xh = np.abs(hilbert(xf))

decoder_signal = pd.Series(xh)

plt.plot(np.arange(len(xh)) / sfreq, xh)
plt.show()


def calibrate_to_switching_rate(
    decoder_signal: np.ndarray, sw_rate: float, sfreq: float
) -> dict:
    """Given a target switching rate, find the bollinger band parameters
    so that during online use, this switching rate is realized

    Just do a brute force search for now

    Parameters
    ----------
    decoder_signal : np.ndarray
        1D array of the decoder signal on which the bollinger bands will be
        evaluated

    sw_rate : float
        the number of switches per second that should be achieved

    sfreq : float
        the sampling frequency of the decoder signal


    Returns
    -------
    dict
        time_horizon_s : float
            the time horizon in seconds
        n_std : float
            the distance of the upper and lower band to the moving average
            in number of standard deviation

    """

    res = []

    rec_time_s = len(decoder_signal) / sfreq

    for time_horizon_n in [
        10,
        25,
        100,
        1_000,
        5_000,
        10_000,
    ]:  # , 20_000, 50_000]:
        mean = decoder_signal.rolling(time_horizon_n).mean().to_numpy()
        std = decoder_signal.rolling(time_horizon_n).std().to_numpy()

        for n_std in [0.1, 0.2, 0.5, 0.8, 1, 2, 2.5, 3]:
            lower = mean - std * n_std
            upper = mean + std * n_std

            n_switchings = find_switchings_n_switchings(decoder_signal, upper, lower)

            res.append(
                {
                    "time_horizon_n": time_horizon_n,
                    "n_std": n_std,
                    "n_switchings": n_switchings,
                    "switching_rate": n_switchings / rec_time_s,
                }
            )

            print(f"{res[-1]}")

    plot_bollinger_bands(decoder_signal, upper, lower, sfreq)


def find_switchings_n_switchings(
    xf: np.ndarray, upper: np.ndarray, lower: np.ndarray
) -> int:
    """Find number of switched between were x is above upper and below lower"""

    x_h = xf > upper
    x_l = xf < lower

    if (~x_h.any()) or (~x_l.any()):
        print("No switchings found")
        return 0

    stim = np.asarray([np.nan] * len(xf))
    stim[x_h] = 1
    stim[x_l] = 0

    # forward fill - using pandas for better readability
    # https://stackoverflow.com/questions/41190852/most-efficient-way-to-forward-fill-nan-values-in-numpy-array
    sstim = pd.Series(stim)
    sstim = sstim.ffill()
    sstim = sstim.fillna(0)  # the start until the first stim

    # start from first turning on
    diff = sstim.diff()[1:]

    n_switchings = len(diff[(diff != 0)])

    return n_switchings


def plot_bollinger_bands(
    decoder_signal: np.ndarray,
    upper: np.ndarray,
    lower: np.ndarray,
    sfreq: float,
):
    """Plot the decoder signal with the bollinger bands"""
    t = np.arange(len(decoder_signal)) / sfreq
    fig, ax = plt.subplots()
    ax.plot(t, decoder_signal, label="signal")
    ax.plot(t, upper, label="upper", linestyle="dashed", color="#333")
    ax.plot(t, lower, label="lower", linestyle="dashed", color="#333")

    stim = np.asarray([np.nan] * len(decoder_signal))
    stim[decoder_signal > upper] = 1
    stim[decoder_signal < lower] = 0
    sstim = pd.Series(stim)
    sstim = sstim.ffill().fillna(0)
    diff = sstim.diff()[1:]

    segments = np.split(sstim.to_numpy(), diff[diff != 0].index)

    i = 0
    for sgm in segments:
        j = min(len(t) - 1, i + len(sgm))
        if sgm[0] == 1:
            ax.axvspan(t[i], t[j], color="#f33", alpha=0.2)
        else:
            ax.axvspan(t[i], t[j], color="#33f", alpha=0.2)

        i = j
    ax.set_xlabel("Time [s]")
    ax.legend()
    plt.show()
