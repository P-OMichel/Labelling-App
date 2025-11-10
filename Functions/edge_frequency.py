import numpy as np
import scipy as sc

#-----------------------------------------------------------------------------------------------------------------------#
#--------------------- Edge frequency using the significant values on a time-frequency representation ------------------#
#-----------------------------------------------------------------------------------------------------------------------#

def edge_frequencies_significant_value(M, f, min_val=0.001, max_val=20, threshold=None, T=5, q=0.5, threshold_min=0.05, threshold_max=2, N_max=2, smoothing = None):
    
    #--- smoothing of the image
    if smoothing == 'maximum' or smoothing == 'Maximum':
        M = sc.ndimage.maximum_filter(M, 3)

    elif smoothing == 'median' or smoothing == 'Median':
        M = sc.ndimage.median_filter(M, 3)

    elif smoothing == 'gaussian' or smoothing == 'Gaussian':
        M = sc.ndimage.gaussian_filter(M, 3)

    elif smoothing == 'minimum' or smoothing == 'Minimum':
        M = sc.ndimage.minimum_filter(M, 3)

    #--- adaptive thresholding if a fixed threshold value is not specified
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

    return edge_frequencies, threshold


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
            edge_frequencies[j] = f[0] # already 0 check if can be changed

    return edge_frequencies #, cumulative_spectro