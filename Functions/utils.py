import numpy as np
import scipy as sc

'''
Function that detects the position of a 0 to 1 edge in a mask (the position is the one of the 1 in the mask)
as well as the position of a 1 to 0 edge (position is the one of the 1 in the mask)
'''
def detect_mask_edge(mask):
    '''
    Input:
    - mask   <--- 1D numpy array of 0 and 1
    Outputs:
    - edge_0_1   <--- position of the 1 in a 0 to 1 edge
    - edge_1_0   <--- position of the 1 in a 1 to 0 edge
    '''
    # in case array start with 1 to detect it as 0 to 1 edge
    mask=np.insert(mask,0,0)
    # in case array finishes with 1 to detect it as 1 to 0 edge
    mask=np.append(mask,0)

    # transformation that makes:
    # 0,0 <-- 0
    # 0,1 <-- 2
    # 1,0 <-- 1
    # 1,1 <-- 3
    mask=mask[:-1]+2*mask[1:]

    edge_0_1=np.where(mask==2)[0]
    edge_1_0=np.where(mask==1)[0]-1

    return edge_0_1,edge_1_0

'''
Function that returns a list of starting and ending position of 1 in a mask
ex:
Input=[0,1,1,0,1]
Output=[[1,2],[4,4]]
'''
def detect_pos_1(mask):

    edge_0_1,edge_1_0=detect_mask_edge(mask)
    N=edge_0_1.size
    pos=[]
    for i in range(N):
        pos.append([edge_0_1[i],edge_1_0[i]])
    
    return pos

def pos_to_mask(mask_template, pos_list):
    '''
    mask_template: mask of zeros of length the size of the original signal from which pos_list is extracted
    '''

    for pos in pos_list:
        mask_template[pos[0]:pos[-1]] = 1

    return mask_template

def diff_envelops_signals(signals):

    N_line,N_column=np.shape(signals)
    env_signals=np.zeros(shape=(N_line,N_column))

    for i in range(N_line):
        env_signals[i,:]=diff_envelops(signals[i,:])

    return env_signals



def envelope_maxima(y):
    '''
    Inputs:
    - y        <-- signal for which the difference between the upper and lower envelops is computed (signal or filtered signal in a band f_int=[f_0,f_1] from the filter function and normalized by its RMS)

    Outputs:
    - diff_env  <-- difference of the interpolated upper and lower envelops
    '''
    N=y.size

    # creates upper and lower envelops of a signal
    list_pos_maxima = find_extremum(y)[0]

    # set origin as a max and min to start with a difference envelope of 0
    list_pos_maxima=np.insert(list_pos_maxima,0,0)

    # get list of maxima and minima
    list_maxima = y[list_pos_maxima]

    # create an interpolation of the envelope at every time of the signal (otherwise we only have a list at the position of the extremum)
    index = [i for i in range(N)]
    ######### maybe index with np.arange is faster
    upper_env=np.interp(index,list_pos_maxima,list_maxima)  # upper envelop interpolation

    return upper_env

'''
return the list of the interpolated upper and lower envelopes at every index of the list y
''' 
def diff_envelops(y):
    '''
    Inputs:
    - y        <-- signal for which the difference between the upper and lower envelops is computed (signal or filtered signal in a band f_int=[f_0,f_1] from the filter function and normalized by its RMS)

    Outputs:
    - diff_env  <-- difference of the interpolated upper and lower envelops
    '''
    N=y.size

    # creates upper and lower envelops of a signal
    list_pos_maxima,list_pos_minima=find_extremum(y)

    # set origin as a max and min to start with a difference envelope of 0
    list_pos_maxima=np.insert(list_pos_maxima,0,0)
    list_pos_minima=np.insert(list_pos_minima,0,0)

    # set last point as a max and min to end with a difference envelope of 0
    list_pos_maxima=np.append(list_pos_maxima,N-1)
    list_pos_minima=np.append(list_pos_minima,N-1)  

    # get list of maxima and minima
    list_maxima,list_minima=y[list_pos_maxima],y[list_pos_minima]

    # create an interpolation of the envelope at every time of the signal (otherwise we only have a list at the position of the extremum)
    index=[i for i in range(N)]
    ######### maybe index with np.arange is faster
    upper_env=np.interp(index,list_pos_maxima,list_maxima)  # upper envelop interpolation
    lower_env=np.interp(index,list_pos_minima,list_minima)  # lower envelop interpolation

    # difference of the envelops
    diff_env=upper_env-lower_env

    return diff_env, upper_env, lower_env

def find_extremum(y):
    '''
    Input:
    - y    <-- the signal where we want to find the local minimum and local maxima
    '''

    # compute the list of differences with right neighbour
    L_right=y[:-1]-y[1:]

    # it misses last point so we add it to keep same indexing as y (pos i is the difference of y_i with y_i+1)
    L_right=np.append(L_right,0)

    # compute the list of differences with left neighbour
    L_left=y[1:]-y[:-1]
    # it misses first point so we add it to keep same indexing as y (pos i is the difference of y_i with y_i+1)
    L_left=np.insert(L_left,0,0)

    # list that discriminates if max, min or not extremum (+3 for max, -1 if min, -3 or 1 if not extremum)
    L=np.sign(L_right)*np.sign(L_left)+2*np.sign(L_right)

    # get list of maximum
    list_pos_maxima=np.where(L==3)[0]
    
    # get list of minimum
    list_pos_minima=np.where(L==-1)[0]

    return list_pos_maxima,list_pos_minima

def find_maximum(y):
    '''
    Input:
    - y    <-- the signal where we want to find the local minimum and local maxima
    '''

    # compute the list of differences with right neighbour
    L_right=y[:-1]-y[1:]

    # it misses last point so we add it to keep same indexing as y (pos i is the difference of y_i with y_i+1)
    L_right=np.append(L_right,0)

    # compute the list of differences with left neighbour
    L_left=y[1:]-y[:-1]
    # it misses first point so we add it to keep same indexing as y (pos i is the difference of y_i with y_i+1)
    L_left=np.insert(L_left,0,0)

    # list that discriminates if max, min or not extremum (+3 for max, -1 if min, -3 or 1 if not extremum)
    L=np.sign(L_right)*np.sign(L_left)+2*np.sign(L_right)

    # get list of maximum
    list_pos_maxima=np.where(L==3)[0]

    return list_pos_maxima

def find_minimum(y):
    '''
    Input:
    - y    <-- the signal where we want to find the local minimum and local maxima
    '''

    # compute the list of differences with right neighbour
    L_right=y[:-1]-y[1:]

    # it misses last point so we add it to keep same indexing as y (pos i is the difference of y_i with y_i+1)
    L_right=np.append(L_right,0)

    # compute the list of differences with left neighbour
    L_left=y[1:]-y[:-1]
    # it misses first point so we add it to keep same indexing as y (pos i is the difference of y_i with y_i+1)
    L_left=np.insert(L_left,0,0)

    # list that discriminates if max, min or not extremum (+3 for max, -1 if min, -3 or 1 if not extremum)
    L=np.sign(L_right)*np.sign(L_left)+2*np.sign(L_right)

    # get list of minimum
    list_pos_minima=np.where(L==-1)[0]

    return list_pos_minima

def zero_crossing(signal):
    '''
    Input:
    - signal <-- 1D numpy array
    Output
    - N_zero_crossing <-- number of time the signal crosses 0
    '''

    # remove all zero values to avoid not counting crossing ( 3 0 -3 becomes 3 -3 and there is a direct crossing that is seen)
    non_zeros_indices = np.where(signal != 0)[0]
    non_zero_signal = signal[non_zeros_indices]

    # get the sign list
    sign_list = np.sign(non_zero_signal)

    # multiplication of the sign_list with itself shifted
    # it gives:
    #          -1 when there is a crossing of zero
    #          +1 when there is no crossing
    #           0 when at least one of the two multiplied value is zero
    mult_list = sign_list[:-1]*sign_list[1:]

    # count the number of time there is -1
    N_zero_crossing=len(np.where(mult_list == -1)[0])

    return N_zero_crossing

def log_reg(v,w):
    '''
    compute the logistic regression value of the values v and the weigths w (ordered)
    '''
    sum_value=np.sum(np.array([w[i+1]*v[i] for i in range(len(v))]))
    exp_value=np.exp(-(w[0]+sum_value))
    return 1 / (1 + exp_value)

def barycenter(x,y):
    
    return np.sum(x*y)/np.sum(y)

def get_PSD_barycenter(y,fs,height):

    f, Pxx_den = sc.signal.periodogram(y, fs)
    Pxx_den=Pxx_den/np.max(Pxx_den)
    indices=sc.signal.find_peaks(Pxx_den,height=height)[0]
    x_b=barycenter(f[indices],Pxx_den[indices])

    return f,Pxx_den,x_b

def tot_var(y):

    grad = np.gradient(y)

    return np.sum(np.abs(grad))
     
def get_re(y):
    '''
    returns the reltative error between the last and first point of y
    '''
    
    return (y[-1] - y[0]) / np.abs(y[0])

from scipy.interpolate import interp1d

def resize_binary_mask(mask, new_length):
    """
    Resample a 1D binary mask to a new length using interpolation.
    
    Parameters:
    - mask (np.ndarray): Original binary mask (1D array of 0s and 1s).
    - new_length (int): Desired length of the output mask.

    Returns:
    - np.ndarray: Resized binary mask (1D array of 0s and 1s).
    """
    
    original_length = len(mask)
    
    # Create interpolation function
    x_old = np.linspace(0, 1, original_length)
    x_new = np.linspace(0, 1, new_length)
    
    f = interp1d(x_old, mask, kind='linear')
    resized = f(x_new)
    
    # Threshold to get binary mask again
    binary_resized = (resized >= 0.5).astype(int)
    
    return binary_resized


def filter_binary_mask(mask, min_length):
    """
    Keeps only sequences of consecutive 1s longer than or equal to min_length in a 1D binary mask.
    
    Parameters:
        mask (np.ndarray): 1D numpy array of binary values (0 and 1).
        min_length (int): Minimum number of consecutive 1s to keep (default is 6).
    
    Returns:
        np.ndarray: Filtered binary mask.
    """
    mask = np.asarray(mask, dtype=np.uint8)
    result = np.zeros_like(mask)
    
    start = None
    for i, val in enumerate(mask):
        if val == 1:
            if start is None:
                start = i
        else:
            if start is not None and i - start >= min_length:
                result[start:i] = 1
            start = None
    # Handle case where the mask ends with a segment of 1s
    if start is not None and len(mask) - start >= min_length:
        result[start:] = 1
    
    return result


def remove_short_segments(mask, min_length):
    """
    Sets to 0 all segments of consecutive 1s shorter than min_length in a 1D binary mask.
    
    Parameters:
        mask (np.ndarray): 1D numpy array of 0s and 1s.
        min_length (int): Minimum length of consecutive 1s to keep.
    
    Returns:
        np.ndarray: Filtered binary mask.
    """
    mask = np.asarray(mask, dtype=np.uint8)
    result = np.zeros_like(mask)

    start = None
    for i, val in enumerate(mask):
        if val == 1:
            if start is None:
                start = i
        else:
            if start is not None and i - start >= min_length:
                result[start:i] = 1
            start = None

    # Handle case where the mask ends with a segment of 1s
    if start is not None and len(mask) - start >= min_length:
        result[start:] = 1

    return result

def get_quantile(y, x, q):

    cumulative = np.cumsum(y)
    cumulative = cumulative / cumulative[-1]
    # Find the index where cumulative reaches the quantile value
    q_index = np.where(cumulative >= q)[0][0]
    
    return x[q_index]
      
        

