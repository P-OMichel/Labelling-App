import numpy as np
from scipy.signal import butter, sosfiltfilt

def get_filtered_signal(y,fs,list_freq_int):

    N=y.size
    N_freq_int=len(list_freq_int)
    filtered_signals=np.zeros(shape=(N_freq_int,N))
    for k in range(N_freq_int):
        filtered_signals[k,:]=filter_butterworth(y,fs,list_freq_int[k])
    return filtered_signals

def filter_butterworth(y,fs,f_int,order=4):

    nyq = 0.5 * fs
    low = f_int[0] / nyq
    high = f_int[-1] / nyq
    sos = butter(order, [low, high], btype='band', output='sos')
    filtered_signal = sosfiltfilt(sos, y)

    return filtered_signal
