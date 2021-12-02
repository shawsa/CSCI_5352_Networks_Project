'''
Some functions implementing the Kuramoto measure on time series data.
'''

import numpy as np

def refine_max(guess_index, series, window=10, max_iter=100):
    '''
    Refines an approximate times series spike index to be more accurate.
    '''
    old_index = guess_index
    try:
        for _ in range(max_iter):
            index = np.argmax(series[old_index-window:old_index+window]) + (old_index-window)
            if index == old_index:
                break
            old_index = index
        return index
    except:
        return None

# Some threshold functions

def series_midpoint(series):
    my_max, my_min = np.max(series), np.min(series)
    return (my_max - my_min)/2 + my_min

def quantile_90(series):
    return np.quantile(series, 0.9)

def get_spike_indices_approx(ts, series, threshold_function):
    '''
    Find approximate spike indices for the given time series. Applies a Heaviside filter
    with a threshold decided by the provided threshold_function. Then Does a 2 point finite
    difference to identify approximate rising times.
    '''
    threshold = threshold_function(series)
    spiking = np.heaviside(series-threshold, 0)
    kernel = np.array([1, -1]) #detect rises
    rising_signal = np.convolve(spiking, kernel, mode='same')
    spike_mask = rising_signal == 1
    return np.arange(len(series))[spike_mask]

def get_spike_indices(ts, series, threshold_function, window):
    spike_indices_approx = get_spike_indices_approx(ts, series, threshold_function)
    spike_indices = [refine_max(guess, series, window) for guess in spike_indices_approx]
    spike_indices = [index for index in spike_indices if index is not None]
    spike_indices = list(set(spike_indices)) # remove duplicates
    spike_indices.sort()
    return spike_indices

def get_kuramoto_phase(t, ts, spike_indices):
    next_spike_num = np.argmax(np.heaviside(ts[spike_indices] - t, 0))
    next_spike_num = np.argmax(np.heaviside(ts[spike_indices] - t, 0))
    if next_spike_num == 0:
        return np.nan #returns nan if before first spike or after last spike
    tf = ts[spike_indices[next_spike_num]]
    t0 = ts[spike_indices[next_spike_num-1]]
    phase = (t-t0)/(tf-t0) * 2*np.pi
    return phase

def series_to_phase(ts, series, threshold_function, window):
    spike_indices = get_spike_indices(ts, series, threshold_function, window=window)
    phases = np.empty(len(ts))
    phases[:spike_indices[0]-1] = np.nan
    for upper, lower in zip(spike_indices[1:], spike_indices[:-1]):
        phases[lower:upper] = np.linspace(0, 2*np.pi, upper-lower)
    phases[upper:] = np.nan
    return phases

def kuramoto_measure(ts, time_series, threshold_function=quantile_90, window=10):
    phases = np.empty(time_series.shape)
    for series_index, series in enumerate(time_series):
        phases[series_index] = series_to_phase(ts, series, threshold_function, window=window)
    
    kuramoto = np.abs(np.sum(np.exp(1j*phases), axis=0)/phases.shape[0])
    return kuramoto

def _nan_filter(series):
    return series[np.logical_not(np.isnan(series))]

def average_kuramoto(ts, time_series, time_span=None, threshold_function=quantile_90, kuramoto=None):
    if time_span is None:
        t_start_index = 0
        t_stop_index = len(ts) - 1 
    else:
        assert len(time_span) == 2
        t_start, t_stop = time_span
        t_start_index = np.argmin(np.abs(ts - t_start))
        t_stop_index = np.argmin(np.abs(ts - t_stop))
    if kuramoto is None:
        kuramoto = kuramoto_measure(ts, time_series, threshold_function=threshold_function)
    kuramoto = kuramoto[t_start_index: t_stop_index]
    return np.mean(_nan_filter(kuramoto))


def pearson_mean(ts, series, time_span=None):
    '''
    The average of the pair-wise Pearson corrolation coefficients between each
    time series.
    '''
    if time_span is None:
        t_start_index = 0
        t_stop_index = len(ts) - 1 
    else:
        assert len(time_span) == 2
        t_start, t_stop = time_span
        t_start_index = np.argmin(np.abs(ts - t_start))
        t_stop_index = np.argmin(np.abs(ts - t_stop))
    cor_mat = np.corrcoef(series[t_start_index: t_stop_index], rowvar=False)
    N = cor_mat.shape[0]
    num_pairs = N*(N-1)/2
    return np.sum(np.triu(cor_mat, k=1))/num_pairs


