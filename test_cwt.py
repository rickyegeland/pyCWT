import cwt

import pytest
import numpy as np

def test_class():
    t = np.arange(10.)
    data = t
    wavelet = cwt.Morlet()
    result = wavelet.cwt(t, data)
    assert result # exists
    assert isinstance(result, cwt.WaveletResult) # right type
    assert isinstance(result.mother, cwt.Morlet) # right type
    assert isinstance(result.coefs, np.ndarray) # right type

def test_dimensions():
    N_samples = 11
    N_scales = 22
    t = np.linspace(0., 10., N_samples)
    data = t
    scales = np.linspace(1., 10., N_scales)
    wavelet = cwt.Morlet()
    result = wavelet.cwt(t, data, scales=scales)
    assert result.coefs.ndim == 2 # 2D array
    assert result.coefs.shape[0] == N_scales # scale axis
    assert result.coefs.shape[1] == N_samples # time axis
    periods = result.periods() # get period axis
    assert periods.size == N_scales

def test_sampf():
    D = 100   # duration
    N = 10 * D # number of samples
    t = np.linspace(0., D, N)
    data = t
    wavelet = cwt.Morlet()
    result = wavelet.cwt(t, data)
    assert result.mother.sampf == N/D

def test_peak():
    P = 13.0 # period
    D = 100   # duration
    N = 10 * D # number of samples
    t = np.linspace(0, D, N) # time axis
    data = np.sin(2*np.pi/P*t) # sine wave

    wavelet = cwt.Morlet()
    result = wavelet.cwt(t, data)
    periods = result.periods()

    # calculate the period of peak power, and make sure that it agrees
    # with the input period to within a small tolerance
    power = np.abs(result.coefs)**2
    i = N/2 # time index in the middle of series
    slice = power[:,i] # periodogram slice at time i
    j = np.argmax(slice) # index of peak power
    peak_period = periods[j]
    tol = 0.05 # tolerance of 5%
    P_hi = P + P * tol
    P_lo = P - P * tol
    assert peak_period >= P_lo and peak_period <= P_hi # correct period
