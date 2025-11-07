# ligotools/utils.py

import numpy as np
from scipy.signal import butter, filtfilt
from scipy.io import wavfile
import matplotlib.pyplot as plt

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

# Plotting code
def plotting(pcolor, strain_whitenbp, time, timemax, tevent, template_match, template_fft,
             datafreq, freqs, data_psd, fs, SNR, det, d_eff, eventname, plottype):

    plt.figure(figsize=(10,8))
    plt.subplot(2,1,1)
    plt.plot(time-timemax, SNR, pcolor,label=det+' SNR(t)')
    #plt.ylim([0,25.])
    plt.grid('on')
    plt.ylabel('SNR')
    plt.xlabel('Time since {0:.4f}'.format(timemax))
    plt.legend(loc='upper left')
    plt.title(det+' matched filter SNR around event')
    # zoom in
    plt.subplot(2,1,2)
    plt.plot(time-timemax, SNR, pcolor,label=det+' SNR(t)')
    plt.grid('on')
    plt.ylabel('SNR')
    plt.xlim([-0.15,0.05])
    #plt.xlim([-0.3,+0.3])
    plt.grid('on')
    plt.xlabel('Time since {0:.4f}'.format(timemax))
    plt.legend(loc='upper left')
    plt.savefig('figures/' + eventname+"_"+det+"_SNR."+plottype)

    plt.figure(figsize=(10,8))
    plt.subplot(2,1,1)
    plt.plot(time-tevent,strain_whitenbp,pcolor,label=det+' whitened h(t)')
    plt.plot(time-tevent,template_match,'k',label='Template(t)')
    plt.ylim([-10,10])
    plt.xlim([-0.15,0.05])
    plt.grid('on')
    plt.xlabel('Time since {0:.4f}'.format(timemax))
    plt.ylabel('whitened strain (units of noise stdev)')
    plt.legend(loc='upper left')
    plt.title(det+' whitened data around event')
    
    plt.subplot(2,1,2)
    plt.plot(time-tevent,strain_whitenbp-template_match,pcolor,label=det+' resid')
    plt.ylim([-10,10])
    plt.xlim([-0.15,0.05])
    plt.grid('on')
    plt.xlabel('Time since {0:.4f}'.format(timemax))
    plt.ylabel('whitened strain (units of noise stdev)')
    plt.legend(loc='upper left')
    plt.title(det+' Residual whitened data after subtracting template around event')
    plt.savefig('figures/' + eventname+"_"+det+"_matchtime."+plottype)
                 
    # -- Display PSD and template
    # must multiply by sqrt(f) to plot template fft on top of ASD:
    plt.figure(figsize=(10,6))
    template_f = np.absolute(template_fft)*np.sqrt(np.abs(datafreq)) / d_eff
    plt.loglog(datafreq, template_f, 'k', label='template(f)*sqrt(f)')
    plt.loglog(freqs, np.sqrt(data_psd),pcolor, label=det+' ASD')
    plt.xlim(20, fs/2)
    plt.ylim(1e-24, 1e-20)
    plt.grid()
    plt.xlabel('frequency (Hz)')
    plt.ylabel('strain noise ASD (strain/rtHz), template h(f)*rt(f)')
    plt.legend(loc='upper left')
    plt.title(det+' ASD and template around event')
    plt.savefig('figures/' + eventname+"_"+det+"_matchfreq."+plottype)