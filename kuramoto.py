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

threshold_function = quantile_90

def get_spike_indices_approx(ts, series, threshold_function=threshold_function):
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

def get_spike_indices(ts, series, threshold_function=threshold_function, window=10):
    spike_indices_approx = get_spike_indices_approx(ts, series, threshold_function=threshold_function)
    spike_indices = [refine_max(guess, series) for guess in spike_indices_approx]
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

def series_to_phase(ts, series, threshold_function=threshold_function):
    spike_indices = get_spike_indices(ts, series, threshold_function=threshold_function)
    phases = [get_kuramoto_phase(t, ts, spike_indices) for t in ts]
    return phases

def kuramoto_measure(ts, time_series, threshold_function=threshold_function):
    phases = np.empty(time_series.shape)
    for series_index, series in enumerate(time_series):
        phases[series_index] = series_to_phase(ts, series, threshold_function=threshold_function)
    
    kuramoto = np.abs(np.sum(np.exp(1j*phases), axis=0)/phases.shape[0])
    return kuramoto

def average_kuramoto(kuramoto):
    return np.mean(kuramoto[np.logical_not(np.isnan(kuramoto))])
