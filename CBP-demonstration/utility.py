from __future__ import division
import numpy as np
import pandas
from matplotlib import pyplot as plt
from neurokit2.signal import signal_smooth
import scipy


def random_index_generator(num_index:int=10):
    device_array = np.array(pandas.read_csv('./data/subject-info.csv')['Device'])
    pos_device = np.argwhere(device_array==1)+1
    forbidden_array = np.array([20,24,33,65,69,79,85,88,97,138,178,180,242,244,247,252,272,274,288,308,326,334,335,377,397,410,430,433,440,442,450,451,457,475,496,528,547,580,585,596,611,621,628,630,649,670,692,693,726,727,752,754,774,784,803,812,813,814,834,849,856,861,895,898,906,909,918,921,925,937,959,970,977,983,984,998,1029,1044,1063,1069,1078,1095,1098,1101,1120])
    random_index = np.zeros(num_index)
    for i in range(num_index):
        index_pos_device = np.random.randint(low=0,high=int(pos_device.size),size=1)
        while(np.sum(np.isin(pos_device[index_pos_device],forbidden_array)) == 1 or np.sum(np.isin(pos_device[index_pos_device],random_index)) == 1):
            index_pos_device = np.random.randint(low=0,high=int(pos_device.size),size=1)
        random_index[i] = pos_device[index_pos_device]
    random_index = random_index.astype(int)
    return random_index

def ppg_findpeaks_elgendi(
    signal,
    sampling_rate=1000,
    peakwindow=0.261,
    beatwindow=0.767,
    beatoffset=0.02,
    mindelay=0.3,
    show=False,
):
    """
    We copy the original implementation of _ppg_findpeaks_elgendi from neurokit2.ppg.ppg_findpeaks. The default values
    of peakwindow and beatwindow are changed to 0.261 and 0.767, respectively.
    """

    """Implementation of Elgendi M, Norton I, Brearley M, Abbott D, Schuurmans D (2013) Systolic Peak Detection in
    Acceleration Photoplethysmograms Measured from Emergency Responders in Tropical Conditions. PLoS ONE 8(10): e76585.
    doi:10.1371/journal.pone.0076585.

    All tune-able parameters are specified as keyword arguments. `signal` must be the bandpass-filtered raw PPG
    with a lowcut of .5 Hz, a highcut of 8 Hz.
    
    """
    if show:
        _, (ax0, ax1) = plt.subplots(nrows=2, ncols=1, sharex=True)
        ax0.plot(signal, label="filtered")

    # Ignore the samples with negative amplitudes and square the samples with
    # values larger than zero.
    signal_abs = signal.copy()
    signal_abs[signal_abs < 0] = 0
    sqrd = signal_abs**2

    # Compute the thresholds for peak detection. Call with show=True in order
    # to visualize thresholds.
    ma_peak_kernel = int(np.rint(peakwindow * sampling_rate))
    ma_peak = signal_smooth(sqrd, kernel="boxcar", size=ma_peak_kernel)

    ma_beat_kernel = int(np.rint(beatwindow * sampling_rate))
    ma_beat = signal_smooth(sqrd, kernel="boxcar", size=ma_beat_kernel)

    thr1 = ma_beat + beatoffset * np.mean(sqrd)  # threshold 1

    if show:
        ax1.plot(sqrd, label="squared")
        ax1.plot(thr1, label="threshold")
        ax1.legend(loc="upper right")

    # Identify start and end of PPG waves.
    waves = ma_peak > thr1
    beg_waves = np.where(np.logical_and(np.logical_not(waves[0:-1]), waves[1:]))[0]
    end_waves = np.where(np.logical_and(waves[0:-1], np.logical_not(waves[1:])))[0]
    # Throw out wave-ends that precede first wave-start.
    end_waves = end_waves[end_waves > beg_waves[0]]

    # Identify systolic peaks within waves (ignore waves that are too short).
    num_waves = min(beg_waves.size, end_waves.size)
    min_len = int(np.rint(peakwindow * sampling_rate))  # this is threshold 2 in the paper
    min_delay = int(np.rint(mindelay * sampling_rate))
    peaks = [0]

    for i in range(num_waves):

        beg = beg_waves[i]
        end = end_waves[i]
        len_wave = end - beg

        if len_wave < min_len:
            continue

        # Visualize wave span.
        if show:
            ax1.axvspan(beg, end, facecolor="m", alpha=0.5)

        # Find local maxima and their prominence within wave span.
        data = signal[beg:end]
        locmax, props = scipy.signal.find_peaks(data, prominence=(None, None))

        if locmax.size > 0:
            # Identify most prominent local maximum.
            peak = beg + locmax[np.argmax(props["prominences"])]
            # Enforce minimum delay between peaks.
            if peak - peaks[-1] > min_delay:
                peaks.append(peak)

    peaks.pop(0)

    if show:
        ax0.scatter(peaks, signal_abs[peaks], c="r")
        ax0.legend(loc="upper right")
        ax0.set_title("PPG Peaks (Method by Elgendi et al., 2013)")

    peaks = np.asarray(peaks).astype(int)
    return peaks

def decision_function(decision_value, window, num_series):
    offset = int (decision_value.shape[0] / num_series)
    pred_y = []
    for i in range(num_series):
        temp_decision_value = np.sum(decision_value[i*offset: i*offset+window],0)
        pred_y.append(np.argmax(temp_decision_value))
    return np.array(pred_y)