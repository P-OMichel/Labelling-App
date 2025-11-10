import numpy as np
import scipy as sc
from Functions.filter import filter_butterworth
from Functions.utils import detect_pos_1, diff_envelops, envelope_maxima


def detect_supp_threshold(y, fs, T):

    signal = filter_butterworth(y, fs,[4,25])**2
    n = int(fs/4)
    h = np.ones(n)/n
    P_signal = np.convolve(signal, h, mode = 'same')

    mask_supp =P_signal <= T

    mask_supp = erode_dilate(mask_supp, 0.5, 0.3, fs)

    return mask_supp


#-----------------------------------------------------------------------------------#
#----------------------------- Morphological routines ------------------------------#
#-----------------------------------------------------------------------------------#

def erosion_dilation(mask,min_band,max_gap,fs):
    '''
    Function that return a mask after erosion/dilatation/erosion

    Inputs:
    - mask      <-- mask of 1 at a position of a value of interest and 0 elsewhere
    - min_band  <-- min lenght of correct values (minimal lenght of segment of 1)
    - max_gap   <-- max lenght of incorrect values (maximal lenght of segment of 0)
    
    Outputs:
    - new_mask  <-- mask created after a process of erosion/dilatation/erosion (explain each phase)
    '''

    # translate the min_band and max_gap from time duration to interval length
    min_band = int(fs*min_band) 
    max_gap = int(fs*max_gap) - 1   # minus 1 to have the correct effect on erosion (erosion with int 3 for instance takes away 2 by construction of the function)

    # routine to erode, dilate and erode the mask
    new_mask = sc.ndimage.binary_erosion(mask,np.ones(min_band),iterations=1)
    new_mask = sc.ndimage.binary_dilation(new_mask,np.ones(min_band+max_gap),iterations=1)
    new_mask = sc.ndimage.binary_erosion(new_mask,np.ones(max_gap),iterations=1)

    return new_mask

def closing_opening(mask, min_band, max_gap, fs):
    '''
    Function that return a mask after erosion/dilatation/erosion

    Inputs:
    - mask      <-- mask of 1 at a position of a value of interest and 0 elsewhere
    - min_band  <-- min lenght of correct values (minimal lenght of segment of 1)
    - max_gap   <-- max lenght of incorrect values (maximal lenght of segment of 0)
    
    Outputs:
    - new_mask  <-- mask created after a process of erosion/dilatation/erosion (explain each phase)
    '''

    # translate the min_band and max_gap from time duration to interval length
    min_band = int(fs*min_band) 
    max_gap = int(fs*max_gap)-1   # minus 1 to have the correct effect on erosion (erosion with int 3 for instance takes away 2 by construction of the function)

    # routine to erode, dilate and erode the mask
    new_mask = sc.ndimage.binary_closing(mask, np.ones(max_gap), iterations=1) # fills holes smaller than max_gap
    new_mask = sc.ndimage.binary_opening(mask, np.ones(min_band), iterations=1) # remove segments smaller than min_band

    return new_mask



#-----------------------------------------------------------------------------------#
#-------- Detection of suppressions using power of signal cf. Chris Sun ------------#
#-----------------------------------------------------------------------------------#

from scipy.ndimage import binary_erosion, binary_dilation

def erode_dilate(mask, min_time, max_gap, Fs):
    """
    Remove short detections and merge nearby detections.

    Parameters:
        mask (array): Binary mask (1D numpy array).
        min_time (float): Minimum duration to keep (in seconds).
        max_gap (float): Maximum gap to bridge (in seconds).
        Fs (float): Sampling frequency.

    Returns:
        mask_out (array): Processed binary mask.
    """
    min_samples = int(np.floor(min_time * Fs))
    gap_samples = int(np.floor(max_gap * Fs))
    structure1 = np.ones(min_samples)
    structure2 = np.ones(min_samples + gap_samples)
    structure3 = np.ones(gap_samples)

    mask = mask.astype(bool)  # Ensure mask is boolean
    mask_out = binary_erosion(mask, structure=structure1)
    mask_out = binary_dilation(mask_out, structure=structure2)
    mask_out = binary_erosion(mask_out, structure=structure3)

    return mask_out.astype(int)

from scipy.signal import find_peaks

def get_envelope(t, xt):
    """
    Compute the difference between the upper and lower envelopes of a signal.

    Parameters:
        t (array): Time vector.
        xt (array): Signal vector.

    Returns:
        dx (array): Envelope difference (upper - lower).
    """
    # Find local maxima
    peaks_max, _ = find_peaks(xt)
    # Find local minima by inverting the signal
    peaks_min, _ = find_peaks(-xt)

    # Interpolate to get upper and lower envelopes
    x_max = np.interp(t, t[peaks_max], xt[peaks_max], left=np.nan, right=np.nan)
    x_min = np.interp(t, t[peaks_min], xt[peaks_min], left=np.nan, right=np.nan)

    # Compute the envelope difference
    dx = x_max - x_min
    return dx

def detect_sup_amplitude(t, y):
    Fs = 1 / (t[1] - t[0])
    min_sup_time = 0.9
    min_IES_time = 1.1
    max_sup_gap = 0.5
    max_IES_gap = 0.5
    rms = np.sqrt(np.mean(y ** 2))

    # Amplitude threshold
    diff_y = get_envelope(t, y)
    diff_y = diff_y[~np.isnan(diff_y)]

    thres_IES = np.quantile(diff_y, 0.5)
    thres_IES = min(8, thres_IES)

    # Alpha threshold
    y_norm = y / rms
    y_high = filter_butterworth(y_norm, Fs, [6, 17])
    diff_y_high = get_envelope(t, y_high)
    diff_y_high = diff_y_high[~np.isnan(diff_y_high)]

    thres_a = np.quantile(diff_y_high, 0.5)
    thres_a = min(0.25, thres_a)

    mask_a = ((-thres_a < y_high) & (y_high < thres_a))

    # Delta threshold
    y_low = filter_butterworth(y_norm, Fs, [0.5, 4])
    diff_y_low = get_envelope(t, y_low)
    diff_y_low = diff_y_low[~np.isnan(diff_y_low)]

    thres_d = np.quantile(diff_y_low, 0.5)
    thres_d = min(0.25, thres_d)

    mask_d = ((-thres_d < y_low) & (y_low < thres_d))

    # Detect IES
    mask_IES = mask_d & mask_a & ((-thres_IES < y) & (y < thres_IES))
    mask_IES = erode_dilate(mask_IES, min_IES_time, max_IES_gap, Fs)

    # Detect aS
    mask_aS = (mask_a & ~mask_IES)
    mask_aS = erode_dilate(mask_aS, min_sup_time, max_sup_gap, Fs)

    # Detect dS
    mask_dS = (mask_d & ~mask_IES)
    mask_dS = erode_dilate(mask_dS, min_sup_time, max_sup_gap, Fs)

    return mask_IES, mask_aS, mask_dS

#-----------------------------------------------------------------------------------#
#---------------- Detection of suppressions using power of signal ------------------#
#-----------------------------------------------------------------------------------#

def env_suppression(signal, fs, T, q, val_min = 0, val_max = None):
    '''
    y: signal in which to look for suppressions

    T: threshold to multiply to the quantile value
    q: quantile
    val_min: minimum value to crop for quantile estimation
    val_max: maximum value to crop for quantile estimation
    '''

    #--- get maximum envelope
    env = diff_envelops(signal)
    env_maxima = envelope_maxima(env)

    #--- get mask of values within val_min and val_max, in case val_max is specified
    if val_max != None:
        mask_val = (val_min <= env_maxima) & (env_maxima <= val_max)

    #--- get mask of values above val_min, in case val_max is not specified
    else: 
        mask_val = (val_min <= env_maxima)

    #--- get threshold for suppressions values
    threshold = T * np.quantile(env_maxima[mask_val], q)

    #--- get mask of suppressions
    mask_supp = (env_maxima <= threshold)

    length = 0.5
    gap = 0.4
    mask_supp = erosion_dilation(mask_supp, length, gap, fs) * 1

    return mask_supp, env, env_maxima, threshold

def env_P_suppression(signal, fs, T, q, val_min = 0, val_max = None):
    '''
    y: signal in which to look for suppressions

    T: threshold to multiply to the quantile value
    q: quantile
    val_min: minimum value to crop for quantile estimation
    val_max: maximum value to crop for quantile estimation
    '''

    #--- get power of signal
    y2 = signal ** 2

    #--- get maximum envelope
    L_y2= []
    L_y2.append(y2)
    env_maximum = envelope_maxima(y2)
    L_y2.append(env_maximum)

    #--- get mask of values within val_min and val_max, in case val_max is specified
    if val_max != None:
        mask_val = (val_min <= env_maximum) & (env_maximum <= val_max)

    #--- get mask of values above val_min, in case val_max is not specified
    else: 
        mask_val = (val_min <= env_maximum)

    #--- get threshold for suppressions values
    threshold = T * np.quantile(env_maximum[mask_val], q)

    #--- get mask of suppressions
    mask_supp = (env_maximum <= threshold)

    length = 0.5
    gap = 0.4
    mask_supp = erosion_dilation(mask_supp, length, gap, fs) * 1

    return mask_supp, L_y2, threshold

def P_suppression(signal, fs, t_smoothing, T, q, val_min = 0, val_max = None):
    '''
    y: signal in which to look for suppressions
    fs: sampling frequency
    N_smoothing: list of time (s) by which to smooth. If more than one element the signal is smooth iteratively
    T: threshold to multiply to the quantile value
    q: quantile
    val_min: minimum value to crop for quantile estimation
    val_max: maximum value to crop for quantile estimation
    '''

    #--- get power of signal
    y2 = signal ** 2

    #--- smooth iteratively
    L_y2 = []
    L_y2.append(y2)
    for i in range(len(t_smoothing)):
        n = int(t_smoothing[i] * fs)
        h = np.ones(n) / n
        y2 = np.convolve(y2, h, mode = 'same')
        L_y2.append(y2)

    #--- get mask of values within val_min and val_max, in case val_max is specified
    if val_max != None:
        mask_val = (val_min <= y2) & (y2 <= val_max)

    #--- get mask of values above val_min, in case val_max is not specified
    else: 
        mask_val = (val_min <= y2)

    #--- get threshold for suppressions values
    threshold = T * np.quantile(y2[mask_val], q)

    #--- get mask of suppressions
    mask_supp = (y2 <= threshold)

    length = 0.5
    gap = 0.4
    mask_supp = erosion_dilation(mask_supp, length, gap, fs) * 1

    return mask_supp, L_y2, threshold

def detect_suppressions_power(y,fs,T_IES_max,T_alpha_max):
    ''''
    Function to detect the alpha-suppressions and IES
    '''
    
    N_points = int(fs/4)
    h = np.ones(N_points) / N_points #  0.25 s 

    # smoothed power of signal between [1.5,30]] Hz
    y2 = np.convolve(filter_butterworth(y,fs,[1.5,30])**2,h,mode='same')
    N = len(y2)
    
    # smoothed power of signal between [7,14]] Hz
    y2_alpha = np.convolve(filter_butterworth(y,fs,[7,14])**2,h,mode='same')

    # smoothed power of signal between [15,20] Hz
    y2_beta = np.convolve(filter_butterworth(y,fs,[15,20])**2,h,mode='same')

    # smoothed power of signal between [30,45]] Hz
    y2_gamma = np.convolve(filter_butterworth(y,fs,[40,45])**2,h,mode='same')

    #--- IES threshold
    # find if zone is ok or mostly suppression
    q = np.quantile(y2,0.75)
    
    if q <= 8: # condition indicating it is mostly suppresions
        T_IES = min(10,q*3)

    else:
        # take only values that are lower than very high values from high power region and artefact, ground checks
        # what happens if power in Burst is same as high values signal ?
        try:
            indices=np.where(y2<T_IES_max*12)[0]
            T_IES=np.quantile(y2[indices],0.9)*0.12
        except: # indices is empty
            T_IES=np.quantile(y2,0.9)*0.12

        # Threshold for IES
        T_IES = min(T_IES,T_IES_max)

    # if T_IES<=8: # can be if just a burst or artefact that make the quantile 75 high and 0.12*quantile(0.9) low #8
    #     print('T_IES', T_IES)
    #     T_IES = 8

    #--- alpha-suppressions threshold
    try:
        indices=np.where(y2_alpha<T_alpha_max*15)[0]
        T_alpha=np.quantile(y2_alpha[indices],0.9)*0.15
    except: # indices is empty
        T_alpha=np.quantile(y2_alpha,0.9)*0.15

        # threshold for alpha supp
        T_alpha = min(T_alpha, T_IES_max)

    # if T_alpha<=1: # can be if just a burst or artefact that make the quantile 75 high and 0.12*quantile(0.9) low #8
    #     print('T_alpha', T_alpha)
    #     T_IES = 1

    #--- beta threshold
    T_beta = T_alpha * 0.75

    #--- get shallow signals mask
    r_gamma_delta = np.convolve(filter_butterworth(y,fs,[30,45])**2,h,mode='same') / np.convolve(filter_butterworth(y,fs,[1,4])**2,np.ones(fs)/fs,mode='same')
    P_y = np.convolve(filter_butterworth(y,fs,[0.1,45])**2,h,mode='same')
    mask_shallow_signal = np.zeros(N)
    mask_shallow_signal[np.where((r_gamma_delta >= 0.05) & (P_y <= 100))[0]] = 1
    mask_shallow_signal = erosion_dilation(mask_shallow_signal,0.5,0.5,fs)*1

    #--- ground check mask
    mask_ground_check = get_mask_ground_check(y,fs)

    #--- mask of suppressions
    Om_alpha = np.where((y2_alpha < T_alpha) & (y2_beta < T_beta) & (mask_shallow_signal != 1) & (y2_gamma < 0.25) & (mask_ground_check != 1))[0]    # list of indices where the condition is satisfied for alpha-suppressions
    mask_alpha = np.zeros(N)
    mask_alpha[Om_alpha] = 1           #  set 1 where there is an alpha suppression, 0 elsewhere

    Om_IES = np.where((mask_alpha == 1) & (y2 < T_IES))[0]     # list of indices where the condition is satisfied for IES
    Om_IES = np.where((y2 < T_IES))
    mask_IES = np.zeros(N)
    mask_IES[Om_IES] = 1           #  set 1 where there is an IES suppression, 0 elsewhere

    #--- Erosion and dilatation routine
    mask_alpha = erosion_dilation(mask_alpha,0.6,0.5,fs)*1
    mask_IES = erosion_dilation(mask_IES,1.1,0.9,fs)*1 

    # remove alpha_supp where there is an IES
    mask_alpha[np.where((mask_alpha-mask_IES) != 1)[0]] = 0
    #mask_alpha = erosion_dilation(mask_alpha,0.5,0.5,fs)

    # get proportion of shallow signals
    shallow_signal_proportion = np.sum(mask_shallow_signal)/N

    # get position of suppressions
    pos_IES = detect_pos_1(mask_IES)
    pos_alpha = detect_pos_1(mask_alpha)

    # get proportion of IES in window
    IES_proportion = np.sum(mask_IES)/N
    alpha_suppression_proportion = np.sum(mask_alpha)/N

    return y2,y2_alpha,pos_IES,pos_alpha,shallow_signal_proportion, mask_IES, mask_alpha, IES_proportion,alpha_suppression_proportion

def detect_shallow_signal(y,fs,T_ratio = 0.05,T_P = 100):
    ''''
    Function to detect the alpha-suppressions and IES
    '''
    
    N_points = int(fs/4)
    h = np.ones(N_points) / N_points #  0.25 s 

    N = len(y)

    # mask supp
    y2_supp = np.convolve(filter_butterworth(y,fs,[1,30])**2,h,mode='same')

    #--- get shallow signals mask
    r_gamma_delta = np.convolve(filter_butterworth(y,fs,[30,45])**2,h,mode='same') / np.convolve(filter_butterworth(y,fs,[1,2])**2,np.ones(fs)/fs,mode='same')
    P_y = np.convolve(filter_butterworth(y,fs,[0.1,45])**2,h,mode='same')
    mask_shallow_signal = np.zeros(N)
    mask_shallow_signal[np.where((r_gamma_delta >= T_ratio) & (P_y <= T_P) & (y2_supp >= 20))[0]] = 1
    mask_shallow_signal = erosion_dilation(mask_shallow_signal,0.5,0.5,fs)*1
    #mask_shallow_signal = transform_mask(mask_shallow_signal)*1

    # get proportion of shallow signals
    shallow_signal_proportion = np.sum(mask_shallow_signal)/N

    return shallow_signal_proportion

def get_mask_ground_check_l(signal,fs, dilatation_segment = 3, min_segment_length_l=1, T_h = 10000, T_l = 10):

    # NOTE:  high power signal mask
    y2 = np.convolve(signal**2, np.ones(32)/32, mode='same')
    mask_h = np.zeros_like(y2)
    mask_h[np.where(y2 >= T_h)[0]] = 1
    
    # dilation to make sure emnough signal is removed
    mask_h =sc.ndimage.binary_dilation(mask_h,np.ones(dilatation_segment*fs),iterations=1)*1

    # NOTE: low amplitude signal mask
    y = filter_butterworth(signal,fs,[0.1,4])

    diff_env_lf = diff_envelops(y)
    mask_l = np.zeros_like(diff_env_lf)
    mask_l[np.where(diff_env_lf<=T_l)[0]] = 1

    # Create a structuring element for erosion and dilation (a 1D array of ones)
    structuring_element = np.ones(int(min_segment_length_l * fs))

    # Erode the mask: small segments will become zero
    #mask_l = sc.ndimage.binary_erosion(mask_l, structure=structuring_element)

    # Dilate the eroded mask to restore large enough segments
    mask_l = sc.ndimage.binary_dilation(mask_l, structure=structuring_element)*1

    return mask_l, mask_h

def join_close_segments(mask, max_gap):
    """Joins segments of 1s if they are separated by less than max_gap zeros."""
    mask = mask.copy()
    ones_indices = np.where(mask == 1)[0]
    
    if len(ones_indices) == 0:
        return mask
    
    # Identify gaps between consecutive 1s
    gaps = np.diff(ones_indices)
    
    # Find small gaps and fill them
    for i, gap in enumerate(gaps):
        if 0 < gap <= max_gap:
            mask[ones_indices[i] + 1 : ones_indices[i + 1]] = 1
            
    return mask

def remove_small_segments(mask, min_size):
    """Keeps only segments with more than min_size ones."""
    mask = mask.copy()
    labeled = np.split(mask, np.where(np.diff(mask) != 0)[0] + 1)  # Split into segments
    new_mask = np.zeros_like(mask)

    pos = 0
    for segment in labeled:
        if np.all(segment == 1) and len(segment) > min_size:
            new_mask[pos:pos+len(segment)] = 1
        pos += len(segment)
    
    return new_mask

def transform_mask(binary_mask, max_gap = 64, min_size = 64):
    """Applies both transformations: joining close segments and removing small ones."""
    binary_mask = join_close_segments(binary_mask, max_gap)
    binary_mask = remove_small_segments(binary_mask, min_size)
    return binary_mask

#-----------------------------------------------------------------------------------------------------------------------#
#---------------------------------- Detection of suppressions with TF representation -----------------------------------#
#-----------------------------------------------------------------------------------------------------------------------#

def compute_fragmentation_spectro(spectro,t_spectro, f_spectro, N_max = 2):
    '''
    Inputs:
    spectro: clipped spectrogram  
    N_max: maximum number of point per colums higher than the threshold

    Ouput:
    '''
    # create matrix where in a column 0,0--> 0 | 0,1 --> 0 | 1,0 --> 0 | 1,1 --> 1 
    M = spectro[:-1, :] * spectro[1:, :]

    # cumulative of each colums
    reversed_spectro = np.flipud(spectro)
    reversed_cumulative_spectro = np.cumsum(reversed_spectro, axis=0)
    cumulative_spectro = np.flipud(reversed_cumulative_spectro)

    # find first frequency where M is 1
    list_frequencies = []
    for j in range(len(t_spectro)):
        try:
            i = max(np.where(cumulative_spectro[:, j] == N_max)[0])
            k = max(np.where(M[:, j] == 1)[0]) + 1
            index = max(i,k)
            list_frequencies.append(f_spectro[index]) # check if it cannot be higher than index in f_spectro
        except:
            list_frequencies.append(0)
            #print(t_spectro[j])

    return np.array(list_frequencies)

def get_mask_ground_check(y,fs, delta_f = 1, n_overlap = 16):

    # compute the spectrogram
    nfft = int(fs / delta_f)
    overlap = nfft - n_overlap
    f_spectro, t_spectro, spectro = sc.signal.spectrogram(y, fs, nperseg=nfft, noverlap=overlap)
    delta_f = f_spectro[1] - f_spectro[0]
    j = int(45 / delta_f)
    f_spectro = f_spectro[:j]
    spectro = spectro[:j, :]  

    # get sum of values above T_h in a spectro per colum
    T_h = 10
    T_h_spectro = np.zeros_like(spectro)
    T_h_spectro[np.where(spectro >= T_h)] = 1

    sum_h = np.sum(T_h_spectro, axis = 0)

    # get sum of values bellow T_l
    T_l = 0.005

    T_l_spectro = np.zeros_like(spectro)
    T_l_spectro[np.where(spectro <= T_l)] = 1

    sum_l = np.sum(T_l_spectro, axis = 0) 

    # mask of values higher or lower than threhsolds
    mask_h = np.zeros_like(sum_h)
    mask_h[np.where(sum_h >= 30)] = 1

    mask_l = np.zeros_like(sum_l)
    mask_l[np.where(sum_l >= 20)] = 1

    delta_t = t_spectro[1] - t_spectro[0]

    return delta_t, mask_l, mask_h   


#-----------------------------------------------------------------------------------------------------------------------#
#--------------------------  detection based on Spectro power within a frequency range  --------------------------------#
#-----------------------------------------------------------------------------------------------------------------------#

def detect_supp_P_spectro(edge_frequency, spectro, f_spectro, f_int = [5,30], T = 0.2, q = 0.8):

    delta_f = f_spectro[1] - f_spectro[0]
    i = int((f_int[0] - f_spectro[0]) / delta_f)
    j = int((f_int[-1] - f_spectro[0]) / delta_f)
    P_spectro = np.sum(spectro[i:j , :], axis = 0)

    threshold = T * np.quantile(P_spectro, q)

    mask = ((P_spectro <= threshold) & (edge_frequency <= 7)).astype(int)

    return mask, P_spectro, threshold

#-----------------------------------------------------------------------------------------------------------------------#
#-------------------------------------  Suppressions detection based on edge frequency  --------------------------------#
#-----------------------------------------------------------------------------------------------------------------------#

def detect_supp_2_edge_frequency(edge_frequency_hf, edge_frequency, Fs, f_hf = 15, f_a=7, f_IES=3, list_morph=[[1.1,0.9],[0.6,0.5]]): 
    '''
    obtain mask and proportion of suppressions using the edge frequency

    Fs: sampling frequency for the edge_frequency (not for the eeg signal, here it is the inverse of the time-frequency transform time resolution)

    '''

    N = len(edge_frequency)

    # get mask of alpha-suppressions and IES
    #mask_alpha = ((edge_frequency_hf <= f_hf) & (edge_frequency <= f_a)).astype(int)
    #mask_IES = ((edge_frequency_hf <= f_hf) & (edge_frequency <= f_IES)).astype(int)

    mask_alpha = (edge_frequency <= f_a).astype(int)
    mask_IES = (edge_frequency <= f_IES).astype(int)

    # erosion/dilation routine to join close suppressions and delete suppressions like segments that are too shorts
    mask_alpha = erosion_dilation(mask_alpha,list_morph[1][0],list_morph[1][1],Fs)*1
    mask_IES = erosion_dilation(mask_IES,list_morph[0][0],list_morph[0][1],Fs)*1 

    # remove IES part overlapping alpha-suppressions for the compuation of alpha-suppression proportion
    mask_alpha_minus_IES = np.where(mask_IES == 1, 0, mask_alpha)

    # get proportion of IES and alpha-suppression in window
    IES_proportion = np.sum(mask_IES)/N
    alpha_suppression_proportion = np.sum(mask_alpha_minus_IES)/N

    return mask_IES, mask_alpha, mask_alpha_minus_IES, IES_proportion, alpha_suppression_proportion

def detect_supp_edge_frequency(edge_frequency, Fs, f_a=7, f_IES=3, list_morph=[[0.7,0.5],[0.3,0.2]]):#list_morph=[[1.1,0.9],[0.6,0.5]]): 
    '''
    obtain mask and proportion of suppressions using the edge frequency

    Fs: sampling frequency for the edge_frequency (not for the eeg signal, here it is the inverse of the time-frequency transform time resolution)
    '''

    N = len(edge_frequency)

    # get mask of alpha-suppressions and IES
    mask_alpha = np.zeros(N)
    mask_IES = np.zeros(N)

    mask_alpha[np.where(edge_frequency <= f_a)[0]] = 1
    mask_IES[np.where(edge_frequency <= f_IES)[0]] = 1

    # erosion/dilation routine to join close suppressions and delete suppressions like segments that are too shorts
    #mask_alpha = erosion_dilation(mask_alpha,list_morph[1][0],list_morph[1][1],Fs)*1
    mask_IES = erosion_dilation(mask_IES,list_morph[0][0],list_morph[0][1],Fs)*1 
    #mask_alpha = closing_opening(mask_alpha,list_morph[1][0],list_morph[1][1],Fs)*1
    #mask_IES = erosion_dilation(mask_IES,list_morph[0][0],list_morph[0][1],Fs)*1 

    # remove IES part overlapping alpha-suppressions for the compuation of alpha-suppression proportion
    mask_alpha_minus_IES = np.where(mask_IES == 1, 0, mask_alpha)

    # get proportion of IES and alpha-suppression in window
    IES_proportion = np.sum(mask_IES)/N
    alpha_suppression_proportion = np.sum(mask_alpha_minus_IES)/N

    return mask_IES, mask_alpha, mask_alpha_minus_IES, IES_proportion, alpha_suppression_proportion

def detection_supp_high_prominence(edge_frequencies, delta_t, f_upper = 20, f_lower= 8, max_duration= 0.8, delta_value = 5):
    '''
    Detect supp of small duration but in between some very distinctive burst of high frequency.
    The duration can be smaller than the usual cut off for suppressions. The cut off frequency for supp can be higher
    '''

    # detect parts above f_upper
    mask_upper = np.zeros_like(edge_frequencies)
    mask_upper[np.where(edge_frequencies >= f_upper)[0]] = 1

    # get positions mask edges
    pos_mask_upper_edges = detect_pos_1(mask_upper)

    # get minimum between two peaks above f_upper
    pos_minimum = []
    N = len(pos_mask_upper_edges)

    # mask for suppressions
    mask_supp = np.zeros_like(mask_upper)

    # check there is at least 2 peaks above f_upper so we can check for a minimum in between
    # also check we are and not into some high freq area such as awake period using the median of the edge frequencies
    if N > 1 and np.median(edge_frequencies) <= 20:

        for i in range(N - 1):
            start = pos_mask_upper_edges[i][-1]
            end = pos_mask_upper_edges[i+1][0]

            # check that distance between bursts is not too long
            if end  -  start <= 2 / delta_t:
                # get signal part between the two peaks
                signal = edge_frequencies[start : end + 1]
                # get pos of minimum
                pos_current_minimum = np.argmin(signal)

                # check the current minimum is bellow f_lower
                if signal[pos_current_minimum] <= f_lower:
                    # check that is it a narrow down peak
                    left_index, right_index = find_closest_indices_numpy(signal, pos_current_minimum, delta_value)
                    if right_index - left_index <= max_duration / delta_t:
                        pos_minimum.append(pos_current_minimum + pos_mask_upper_edges[i][-1])
                        mask_supp[pos_mask_upper_edges[i][-1] + left_index : pos_mask_upper_edges[i][-1] + right_index] = 1

    #--- get proportion of supp
    supp_prop = np.sum(mask_supp) / len(mask_supp)

    return pos_minimum, mask_supp, supp_prop 

def find_closest_indices_numpy(arr, pos, delta_value):

    target = arr[pos] + delta_value 

    # Indices to the left of `pos`
    left_indices = np.where(arr[:pos] >= target)[0]
    left_index = left_indices[-1] # if left_indices.size > 0 else None

    # Indices to the right of `pos`
    right_indices = np.where(arr[pos+1:] >= target)[0]
    right_index = right_indices[0] + pos + 1 # if right_indices.size > 0 else None

    return left_index, right_index

def supp_edge_frequency_prominence(edge_frequencies, fs):

    peaks, properties = sc.signal.find_peaks(-edge_frequencies, 
                                            prominence = 5,
                                            width = [int(fs / 16), int(fs)],
                                            rel_height = 0.5,
                                            wlen = int(fs) 
                                          )
    widths = properties['widths']

    # mask of artefacts
    N = len(edge_frequencies)
    mask = np.zeros(N)
    for i in range(len(peaks)):
        if edge_frequencies[peaks[i]] <= 7:
            start = int(peaks[i] - widths[i])
            end = int(peaks[i] + widths[i])

            if start < 0:
                start = 0
            if end >= N:
                end = N - 1
        
            mask[start:end] = 1

    return mask

#-----------------------------------------------------------------------------------------------#
#----------------------------------- mask suppressions envelopes -------------------------------#
#-----------------------------------------------------------------------------------------------#

from Functions.filter import get_filtered_signal
from scipy.interpolate import interp1d
from scipy.ndimage import uniform_filter1d

def get_supp_filter(y, fs, list_freq_int = [[1,4],[7,14],[15,30],[30,45]], lower_T = [0.15]*4):

    #--- filter signals
    signals = get_filtered_signal(y, fs, list_freq_int)
    #signals = signals **2
    N_signal = len(signals[0])
    t_signal = np.linspace(0, N_signal / fs, N_signal)

    #--- get masks
    mask_delta = mask_env_power(t_signal, signals[0], lower_T[0])
    mask_alpha = mask_env_power(t_signal, signals[1], lower_T[1])
    mask_beta = mask_env_power(t_signal, signals[2], lower_T[2])
    mask_gamma = mask_env_power(t_signal, signals[-1], lower_T[-1])

    # mask_delta = mask_env(t_signal, signals[0], lower_T[0])
    # mask_alpha = mask_env(t_signal, signals[1], lower_T[1])
    # mask_beta = mask_env(t_signal, signals[2], lower_T[2])
    # mask_gamma = mask_env(t_signal, signals[-1], lower_T[-1])

    #--- erosion dilation
    length = 0.5
    gap = 0.4
    mask_delta = erosion_dilation(mask_delta, length, gap, fs)*1
    mask_alpha = erosion_dilation(mask_alpha, length, gap, fs)*1
    mask_beta = erosion_dilation(mask_beta, length, gap, fs)*1
    mask_gamma = erosion_dilation(mask_gamma, length, gap, fs)*1

    #--- combine masks
    mask_alpha_supp = ((mask_alpha == 1) & (mask_beta == 1) & (mask_gamma == 1)).astype(int)


    # signal = get_filtered_signal(y, fs, [[7,45]])[0]
    # mask_signal = mask_env_power(t_signal, signal, lower_T[0])
    # mask_alpha_supp = erosion_dilation(mask_signal, length, gap, fs)*1

    #--- suppressions proportion
    supp = np.sum(mask_alpha_supp) / N_signal
    

    return mask_alpha_supp, supp

def smooth(data, window_size):
    """Smooth data using a uniform filter."""
    return uniform_filter1d(data, size=window_size, mode='nearest')

def mask_env(t, signal, lower_T):

    std = np.std(signal)

    span = 1

    # Local maxima and minima
    TF_max = (signal[1:-1] > signal[:-2]) & (signal[1:-1] > signal[2:])  # Local maxima
    TF_min = (signal[1:-1] < signal[:-2]) & (signal[1:-1] < signal[2:])  # Local minima

    # Interpolation for upper and lower envelopes
    max_indices = np.where(TF_max)[0] + 1  # Adjust index
    min_indices = np.where(TF_min)[0] + 1  # Adjust index

    x_up = interp1d(t[max_indices], smooth(signal[max_indices], span), bounds_error=False, fill_value="extrapolate")(t)
    x_low = interp1d(t[min_indices], smooth(signal[min_indices], span), bounds_error=False, fill_value="extrapolate")(t)

    # Find spindles cut
    d = np.abs(x_up - x_low)

    mask = (d <= lower_T * std).astype(int)

    return mask

def mask_env_power(t, signal, lower_T):
    
    h = np.ones(16)/16
    signal = np.convolve(signal**2,h,mode='same')
    signal = smooth(signal, 32)

    std = np.quantile(signal, 0.9)

    mask = (signal <= lower_T * std).astype(int)

    return mask

