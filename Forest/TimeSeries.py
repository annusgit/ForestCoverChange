
"""
    Parse the values to test the data we have obtained
"""

from __future__ import print_function
from __future__ import division
import matplotlib.pyplot as pl
import numpy as np
import scipy
import json
from pprint import pprint

all_values = ["value_min",
              "value_max",
              # "value_sum",
              "value_range",
              "value_mean",
              "value_variance",
              "value_stddev"]
all_vals = {}
for i in all_values:
    all_vals[i] = []


def smooth(x, window_len=11, window='hanning'):
    """smooth the data using a window with requested size.

    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.

    input:
        x: the input signal
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal

    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)

    see also:

    np.hanning, np.hamming, np.bartlett, np.blackman, np.convolve
    scipy.signal.lfilter

    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """

    if x.ndim != 1:
        raise ValueError, "smooth only accepts 1 dimension arrays."

    if x.size < window_len:
        raise ValueError, "Input vector needs to be bigger than window size."

    if window_len < 3:
        return x

    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError, "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'"

    s = np.r_[x[window_len - 1:0:-1], x, x[-2:-window_len - 1:-1]]
    # print(len(s))
    if window == 'flat':  # moving average
        w = np.ones(window_len, 'd')
    else:
        w = eval('np.' + window + '(window_len)')

    y = np.convolve(w / w.sum(), s, mode='valid')
    return y


def main():
    # this_file = '/home/amna/Downloads/statistics_250m_16_days_NDVI.json' #local
    this_file = '/home2/amna_gillani/PyCharmProject/statistics_250m_16_days_NDVI.json' #remote
    with open(this_file) as this:
        read = json.load(this)
    # pprint(read)
    statistics = read['statistics']
    for t in statistics:
        for v in all_values:
            val = float(t[v])
            all_vals[v].append(val)

    for v in all_vals.keys():
        this_signal = smooth(x=np.asarray(all_vals[v]), window_len=5)
        pl.plot(this_signal, label=v)
    pl.legend(loc='upper right')
    pl.show()

# if __name__ == '__main__':
    # main()

main()