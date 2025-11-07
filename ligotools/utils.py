# ligotools/utils.py

import numpy as np
from scipy.signal import butter, filtfilt
from scipy.io import wavfile

def whiten(strain, interp_psd, dt, fband=None, fs=None):
    Nt = len(strain)
    freqs = np.fft.rfftfreq(Nt, dt)
    hf = np.fft.rfft(strain)
    norm = 1./np.sqrt(1./(dt*2))
    white_hf = hf / np.sqrt(interp_psd(freqs)) * norm
    white_ht = np.fft.irfft(white_hf, n=Nt)
    if fband is not None and fs is not None:
        bb, ab = butter(4, [fband[0]*2./fs, fband[1]*2./fs], btype='band')
        normalization = np.sqrt((fband[1]-fband[0])/(fs/2))
        white_ht = filtfilt(bb, ab, white_ht) / normalization
    return white_ht

def write_wavfile(filename, fs, data, scale=0.9):
    d = np.int16(data / np.max(np.abs(data)) * 32767 * scale)
    wavfile.write(filename, int(fs), d)

def reqshift(data, fshift=100, sample_rate=4096):
    """Frequency shift the signal by constant"""
    x = np.fft.rfft(data)
    T = len(data)/float(sample_rate)
    df = 1.0/T
    nbins = int(fshift/df)
    y = np.roll(x.real, nbins) + 1j*np.roll(x.imag, nbins)
    y[0:nbins] = 0.
    z = np.fft.irfft(y)
    return z
