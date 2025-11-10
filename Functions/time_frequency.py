import numpy as np
import scipy as sc
from scipy.ndimage import median_filter, gaussian_filter, uniform_filter, minimum_filter, maximum_filter
#-----------------------------------------------------------------------------------------------------------------------#
#--------------------------------------------- time-frequency representation -------------------------------------------#
#-----------------------------------------------------------------------------------------------------------------------#

def spectrogram_low_res(y, fs, delta_f=1, n_overlap=16, f_cut=45):

    # compute spectrogram
    nfft = int(fs / delta_f)
    overlap = nfft - n_overlap
    f_spectro, t_spectro, spectro = sc.signal.spectrogram(y, fs, nperseg=nfft, noverlap=overlap)
    delta_f = f_spectro[1] - f_spectro[0]
    j = int(f_cut / delta_f)
    f_spectro = f_spectro[:j]
    spectro = spectro[:j, :]  

    return t_spectro, f_spectro, spectro


def spectrogram(y, fs, nperseg_factor=1, noverlap_factor=0.9, nfft_factor=4, detrend = False, scaling = 'density', f_cut = 45):
    
    # parameters
    nperseg = int(nperseg_factor*fs)
    noverlap = int(noverlap_factor * nperseg)
    nfft = int(nfft_factor * nperseg)
    window = sc.signal.windows.hamming(nperseg, sym = True)

    # compute spectrogram
    f_spectro, t_spectro, spectro = sc.signal.spectrogram(y, fs = fs, window = window,
                                                        nperseg = nperseg, noverlap = noverlap,
                                                        nfft = nfft, detrend = False, scaling = 'density')
    
    # cut at limit frequency
    delta_f = f_spectro[1] - f_spectro[0]
    j = int(f_cut / delta_f)
    f_spectro = f_spectro[:j]
    spectro = spectro[:j, :]

    return t_spectro, f_spectro, spectro

#-----------------------------------------------------------------------------------------------------------------------#

#-----------------------------------------------------------------------------------------------------------------------#
#----------------------------------------------- Spectrogram normalisation ---------------------------------------------#
#-----------------------------------------------------------------------------------------------------------------------#

def quantile_normalisation(M,  q0, q1):
    '''
    Normalize spectrogram with values:
    - 0 if m_ij <= quantile(M, q0)
    - 255 if m_ij >= quanyile(M, q1)
    and linear scaling in between
    '''

    #--- scale spectrogram in log2
    M = np.log2(M + 0.00001)

    #--- get quantile values
    q0_val = np.quantile(M, q0)
    q1_val = np.quantile(M, q1)

    #--- clip spectrogram in between quantiles
    M = np.clip(M, q0_val, q1_val)

    #--- set into [0, 255] range
    M = ((M - q0_val) / (q1_val - q0_val) * 255).astype(np.uint8)

    return M

#-----------------------------------------------------------------------------------------------------------------------#
#-------------------------------------------------- Power in spectrogram -----------------------------------------------#
#-----------------------------------------------------------------------------------------------------------------------#

def extract_power(M,  f, list_f_int, method = 'quantile_0.85'):

    delta_f = f[1] - f[0]

    N = len(M[0,:])

    # initialize array to contain the different powers
    P = np.zeros(len(list_f_int))

    if method == 'mean_psd': # first project onto y axis
        for index, (f0, f1) in enumerate(list_f_int):
            # get indices corresponding to current frequency range
            i = int((f0 - f[0]) / delta_f)
            j = int((f1 - f[0]) / delta_f)

            # extract power
            P[index] = np.mean(np.sum(M[i:j+1,:], axis = 1) / N)
        
    elif method == 'median_psd':
        for index, (f0, f1) in enumerate(list_f_int):
            # get indices corresponding to current frequency range
            i = int((f0 - f[0]) / delta_f)
            j = int((f1 - f[0]) / delta_f)

            # extract power
            P[index] = np.median(np.sum(M[i:j+1,:], axis = 1) / N)

    elif method == 'mean': # directly take mean of bins values
        for index, (f0, f1) in enumerate(list_f_int):
            # get indices corresponding to current frequency range
            i = int((f0 - f[0]) / delta_f)
            j = int((f1 - f[0]) / delta_f)

            # extract power
            P[index] = np.mean(M[i:j+1,:])

    elif method == 'median': # directly take median of bins values
        for index, (f0, f1) in enumerate(list_f_int):
            # get indices corresponding to current frequency range
            i = int((f0 - f[0]) / delta_f)
            j = int((f1 - f[0]) / delta_f)

            # extract power
            P[index] = np.median(M[i:j+1,:])

    elif method == 'quantile_0.85': # directly take median of bins values
        for index, (f0, f1) in enumerate(list_f_int):
            # get indices corresponding to current frequency range
            i = int((f0 - f[0]) / delta_f)
            j = int((f1 - f[0]) / delta_f)

            # extract power
            P[index] = np.quantile(M[i:j+1,:], 0.85)      

    return P


#-----------------------------------------------------------------------------------------------------------------------#

#-----------------------------------------------------------------------------------------------------------------------#
#------------------------- Edge frequency using the limit value on a time-frequency representation ---------------------#
#-----------------------------------------------------------------------------------------------------------------------#


def edge_frequencies_double_quantiles(M, f, T0 = 1, q0 = 0.5, T1 = 1, q1 = 0.95, v_max_outliers = 1000):

    #--- get mask of  values higher than first quantile based threshold
    #    and large outlier removal
    mask = (M >= T0 * np.quantile(M, q0)) & (M < v_max_outliers)
    
    #--- determine second threshold on data present in the mask
    threshold = T1 * np.quantile(M[mask], q1)
    print('double qunatile threshold', threshold)

    # get list of frequencies:
    edge_frequencies = get_edge_limit_value(M, f, threshold)

    return edge_frequencies

def edge_frequencies_double_threshold(M, f, T0, T1):
    '''
    M: time_frequency matrix
    f: associated frequencies
    T0: first threshold, all values lower than T0 are set to zeros
    T1: standard threshold for ef estimation
    '''

    M[M <= T0] = 0

    # get list of frequencies:
    edge_frequencies = get_edge_limit_value(M, f, T1)

    return edge_frequencies


def edge_frequencies_limit_value(M, f, min_val = 0.001, max_val=20, threshold=None, T=5, q=0.75, threshold_min=0.01, threshold_max=20, factor_hf=5, smooth = False, second_check = True):
    '''
    M: time_frequency matrix
    f: associated frequencies
    min_val: minimum value to clip M
    max_val: maximum value to clip M 
    T: Value by which to multiply to get the threshold
    q: quantile value to get the on the M elements
    threshold_min: minimum value for the threshold
    threshold_max: maximum value for the threshold
    smooth: when True the edge_frequency is smoothed with a (3,1) Savitsky-Golay filter
    '''

    # mask of high frequency high power parts
    T_M = (M >= max_val/10).astype(int)
    count = np.sum(T_M[35:,:], axis=0)
    mask_hf = 1 - (count >= 1).astype(int)
    

    # thresholding   
    if threshold == None:
        if np.sum(mask_hf) != 0 and np.prod(mask_hf) !=1:
            threshold = T * np.quantile(np.clip(M[:, mask_hf == 1], min_val, max_val), q)
        else:
            threshold = T * np.quantile(np.clip(M, min_val, max_val), q)
        # min max values 
        print('threshold', threshold)
        threshold = min(threshold_max, threshold)
        threshold = max(threshold_min, threshold)

    # get list of frequencies:
    edge_frequencies = get_edge_limit_value(M, f, threshold)

    # smooth
    if smooth:
        edge_frequencies = sc.signal.savgol_filter(edge_frequencies,3,1)

    # estimation of the hf edge frequency
    threshold_hf = threshold / factor_hf

    # get list of frequencies:
    edge_frequencies_hf = get_edge_limit_value(M, f, threshold_hf)

    # smooth
    if smooth:
        edge_frequencies_hf = sc.signal.savgol_filter(edge_frequencies_hf,3,1)
    

    return edge_frequencies, edge_frequencies_hf, threshold


def get_edge_limit_value(spectro, f_spectro, threshold):
    '''
    Inputs:
    spectro: clipped spectrogram  
    N_max: maximum number of point per colums higher than the threshold

    Ouput:
    '''

    # cumulative of each colums
    reversed_spectro = np.flipud(spectro)
    reversed_cumulative_spectro = np.cumsum(reversed_spectro, axis=0)
    cumulative_spectro = np.flipud(reversed_cumulative_spectro)

    cond = cumulative_spectro <= threshold
    indices = np.argmax(cond, axis = 0)
    # check where the condition cannot be reached as it return 0 and change to -1 to get highest frequency later
    valid = cond.any(axis=0)
    indices[~valid] = -1

    edge_frequencies = f_spectro[indices]

    return edge_frequencies

#-----------------------------------------------------------------------------------------------------------------------#


#-----------------------------------------------------------------------------------------------------------------------#
#--------------------- Edge frequency using the significant values on a time-frequency representation ------------------#
#-----------------------------------------------------------------------------------------------------------------------#

def edge_frequencies_significant_value(M, f, min_val=0.001, max_val=20, threshold=None, T=5, q=0.5, threshold_min=0.05, threshold_max=2, N_max=2, smooth = False):
    
    M = sc.ndimage.maximum_filter(M, 3)

    # thresholding
    if threshold == None:
        threshold = T * np.quantile(np.clip(M, min_val, max_val), q)
        # min max values 
        threshold = min(threshold_max, threshold)
        threshold = max(threshold_min, threshold)

    # thresholding on the TF matrix
    thresholded_M = np.zeros_like(M)
    thresholded_M[np.where(M >= threshold)] = 1

    # get list of frequencies:
    edge_frequencies = get_edge_significant_value(thresholded_M, f, N_max)

    # smooth
    if smooth:
        edge_frequencies = sc.signal.savgol_filter(edge_frequencies,3,1)

    return edge_frequencies, threshold

def edge_frequencies_significant_value_hf(M, f, min_val=0.001, max_val=20, threshold=None, T=5, q=0.5, threshold_min=0.05, threshold_max=2, N_max=2, factor_hf = 10, smooth = False, second_check = True):
    
    # thresholding
    if threshold == None:
        threshold = T * np.quantile(np.clip(M, min_val, max_val), q)
        # min max values 
        threshold = min(threshold_max, threshold)
        threshold = max(threshold_min, threshold)

    # thresholding on the TF matrix
    thresholded_M = np.zeros_like(M)
    thresholded_M[np.where(M >= threshold)] = 1

    # get list of frequencies:
    edge_frequencies = get_edge_significant_value(thresholded_M, f, N_max)

    # smooth
    if smooth:
        edge_frequencies = sc.signal.savgol_filter(edge_frequencies,3,1)

    # call function again after removing IES potential parts

    if threshold == None and second_check:
        mask = (edge_frequencies >= 4).astype(int)
        threshold = T * np.quantile(np.clip(M[:, mask == 1], min_val, max_val), q)
        # min max values 
        threshold = min(threshold_max, threshold)
        threshold = max(threshold_min, threshold)

        thresholded_M = np.zeros_like(M)
        thresholded_M[np.where(M >= threshold)] = 1

        # get list of frequencies:
        edge_frequencies = get_edge_significant_value(thresholded_M, f, N_max)

        # smooth
        if smooth:
            edge_frequencies = sc.signal.savgol_filter(edge_frequencies,3,1)

    # estimation of the hf edge frequency
    threshold_hf = threshold / factor_hf

    thresholded_M_hf = np.zeros_like(M)
    thresholded_M_hf[np.where(M >= threshold_hf)] = 1

    # get list of frequencies:
    edge_frequencies_hf = get_edge_significant_value(thresholded_M, f, N_max)

    # smooth
    if smooth:
        edge_frequencies_hf = sc.signal.savgol_filter(edge_frequencies_hf,3,1)
    

    return edge_frequencies, edge_frequencies_hf, threshold

def get_edge_significant_value(thresholded_M, f, N_max):
    '''
    Inputs:
    spectro: clipped spectrogram  
    N_max: maximum number of point per colums higher than the threshold

    Ouput:
    '''

    # create matrix where in a column 0,0--> 0 | 0,1 --> 0 | 1,0 --> 0 | 1,1 --> 1 
    M_01 = thresholded_M[:-1, :] * thresholded_M[1:, :]

    # cumulative of each columns of the thresholded matrix
    reversed_spectro = np.flipud(thresholded_M)
    reversed_cumulative_spectro = np.cumsum(reversed_spectro, axis=0)
    cumulative_spectro = np.flipud(reversed_cumulative_spectro)

    # find first frequency where M_01 is 1 or when the cumulative reaches N_max
    N = len(thresholded_M[0,:])
    edge_frequencies = np.zeros(N)
    for j in range(N):
        try:
            i = max(np.where(cumulative_spectro[:, j] == N_max)[0])
            k = max(np.where(M_01[:, j] == 1)[0]) + 1
            index = max(i,k)
            edge_frequencies[j] = f[index] # check if it cannot be higher than index in f_spectro
        except:
            edge_frequencies[j] = 0 # already 0 check if can be changed

    return edge_frequencies #, cumulative_spectro


#-----------------------------------------------------------------------------------------------------------------------#

#-----------------------------------------------------------------------------------------------------------------------#
#------------------------------------ Spectrogram centroid -----------------------------------------#
#-----------------------------------------------------------------------------------------------------------------------#

def compute_centroids(S, freqs, times):
    """
    Computes spectral and temporal centroids from a TFR.
    
    Parameters
    ----------
    S : np.ndarray
        2D array of shape (n_freqs, n_times), the power or magnitude.
    freqs : np.ndarray
        1D array of frequencies (length = n_freqs).
    times : np.ndarray
        1D array of times (length = n_times).
    
    Returns
    -------
    spectral_centroid : np.ndarray
        1D array of length n_times — the spectral centroid at each time bin.
    temporal_centroid : np.ndarray
        1D array of length n_freqs — the temporal centroid at each frequency bin.
    """

    # Avoid division by zero
    eps = 1e-10

    # Spectral Centroid (per time slice): weighted average of frequencies
    numerator_spec = np.dot(freqs, S)
    denominator_spec = np.sum(S, axis=0) + eps
    spectral_centroid = numerator_spec / denominator_spec

    # Temporal Centroid (per frequency slice): weighted average of times
    numerator_temp = np.dot(S, times)
    denominator_temp = np.sum(S, axis=1) + eps
    temporal_centroid = numerator_temp / denominator_temp

    return spectral_centroid, temporal_centroid


#-----------------------------------------------------------------------------------------------------------------------#
#------------------------------------ Threshold spectrogram per frequency band -----------------------------------------#
#-----------------------------------------------------------------------------------------------------------------------#

def combined_FTTFM(M, f, 
                   list_freq = [1, 4, 7, 14, 20, 30], 
                   T = [0.2]*7,#[1,    1,   1,   1,   1,   1,   1], 
                   q = [0.9]*7):#[0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]):

    delta_f = f[1] - f[0]

    #--- first frequency
    j = int((list_freq[0] - f[0]) / delta_f)
    M[:j, :] = threshold_M(M[:j, :], T[0], q[0])

    #--- last_frequency
    i = int((list_freq[-1] - f[0]) / delta_f)
    M[i:, :] = threshold_M(M[i:, :], T[-1], q[-1])

    #--- subdivisions in between
    for k in range(len(list_freq) - 1):
        i = int((list_freq[k] - f[0]) / delta_f) 
        j = int((list_freq[k + 1] - f[0]) / delta_f) 
        M[i:j,:] = threshold_M(M[i:j,:], T[k + 1], q[k + 1])

    return f, M

def FTTFM(M, f, f_int, T, q):
    '''
    Frequency Thresholded Time-Frequency Matrix
    '''

    delta_f = f[1] - f[0]
    i = int((f_int[0] - f[0]) / delta_f)
    j = int((f_int[-1] - f[0]) / delta_f)
    M = M[i : j, :]

    M = threshold_M(M, T, q)

    return M
    
def threshold_M(M, T, q):

    M = (M >= T * np.quantile(M, q)).astype(int)
    
    return M