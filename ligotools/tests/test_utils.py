import numpy as np
import os
from scipy.io import wavfile
from ligotools import utils

# Test #1: Testing whiten function
def test_whiten_basic():
    # Here is a a simple sine wave 
    fs = 1024
    t = np.arange(0, 1, 1/fs)
    # I will set to 10 hz
    data = np.sin(2 * np.pi * 10 * t)  
    psd = lambda f: np.ones_like(f)
    whitened = utils.whiten(data, psd, dt=1/fs)
    # testing the output length and I will also test the type
    assert len(whitened) == len(data)
    assert isinstance(whitened, np.ndarray)
    # test that values are roughly normalized
    assert np.max(np.abs(whitened)) <= 2.0 

#Test #2: Testing the reqshift function
def test_reqshift_basic():
    fs = 1024
    t = np.arange(0, 1, 1/fs)
    # I will assign set to 10hz
    data = np.sin(2 * np.pi * 10 * t)  
    shifted = utils.reqshift(data, fshift=10, sample_rate=fs)
    # Here I will test the output length and type similar to before 
    # but for reqshift
    assert len(shifted) == len(data)
    assert isinstance(shifted, np.ndarray)
