'''
By Paweł A. Pierzchlewicz
=========================
A library that has all the required functions for signal analysis
=================================================================
'''



'''
==========================
Imports
==========================
'''

import numpy as np
import matplotlib.pyplot as plt
import warnings
import scipy.stats as st
from scipy import signal


'''
==================
Misc functions
==================
'''
def time(T=1, Fs=128):
	return np.arange(0, T, 1/Fs)

def drawArray(A, columns=1, method=plt.plot, show=True):
	N = len(A)
	for i, s in enumerate(A):
		plt.subplot(N,columns, i+1)
		method(s[0], s[1])
	if show: 
		plt.show()

'''
==========================
Signal Definition Functons
==========================
'''

# Sin function
def sin(T = 1, f = 10, Fs = 128, phi = 0):
	'''
		T - Time length
		f - sin frequency
		Fs - probing frequency
		phi - phase shift
	'''
	#Test for niquist frequency
	if f >= Fs/2:
		raise Exception("[Aliasing imannet] {} Hz must be smaller than {} Hz".format(f, Fs/2) )
	if Fs%128 != 0:
		warnings.warn("{} Hz is not the most optimal frequency for FFT".format(Fs), Warning)
	t = time(T, Fs)
	s = np.sin(2*np.pi*f*t + phi)
	return (t, s)

def white_noise(mu=0, std=1, T = 1, Fs = 128):
	'''
		White noise, a gaussian noise function
		mu - mean of gauss
		std - standard deviation of gauss
		T - Time length
		Fs - probing frequency
	'''
	t = time(T, Fs)
	s = st.norm.rvs(loc=mu, scale=std, size=T*Fs)
	return (t, s)

def delta(t0, T = 1, Fs = 128, fit=True):
	'''
		A test function with one point at which it is equal 1
		t0 - point at which equal 1
		T - Time length
		Fs - probing frequency
	'''
	t = time(T, Fs)
	s = np.zeros(T*Fs)
	if t0 > T:
		raise Exception("t0 is {}s, but it cannot be larger than {}s".format(t0, T))
	if not (np.where(t == t0)[0]) and not fit:
		warnings.warn("{} is not found in the time given, set fit=True to find the closest value to t0".format(Fs), Warning)
	if not fit:
		s[s == t0] = 1
	else:
		subtract = abs(t - t0)
		closest = np.where(subtract == min(subtract))[0]
		s[closest] = 1
	return (t, s)

'''
==================
Window functions
==================
'''

# Module Constants
square = 'square'
hamming = 'hamming'
hanning = 'hanning'
blackman = 'blackman'
bartlett = 'bartlett'
hann = 'hann'

def window_signal(s, window):
	return s*window

def window(type='square', N=128):
	if type == square:
		return np.ones(N)
	if type == hamming:
		return signal.hamming(N)
	if type == hanning:
		return signal.hanning(N)
	if type == blackman:
		return signal.blackman(N)
	if type == bartlett:
		return signal.bartlett(N)
	if type == hann:
		return signal.hann(N)
	else:
		raise Exception('Unknown Window')


def filter(s, type=None, window_U=None):
	S = np.fft.rfft(s)
	if type:
		print('here')
		window_U = window(type, N = len(S))
	So = window_signal(S, window_U)
	plt.plot(So)
	so = np.fft.irfft(So)
	return(so)

def norm_window(window):
	return np.linalg.norm(window)


'''
==================
Analysis functions
==================
'''


# Widmo
def spectrum(s, t=None, Fs=128, draw = True):
	'''
		s - Signal
		Fs - probing frequency
		draw - whether to draw the graph or not
	'''
	S = abs(np.fft.rfft(s))**2
	S_freq = np.fft.rfftfreq(len(s), 1/Fs)
	if draw:
		drawArray([(t,s), (S_freq, S)])
	return(S_freq, S)


def multiply(s, T=1, N=1):
	'''
		s - signal
		N - number of times to multiply
	'''
	tN = time(T*N, len(s))
	sN = np.tile(s, N)
	return (tN, sN)

def zero_padding(s, N=2, T=1):
	zeros = np.zeros(len(s)*(N-1))
	s0 = np.concatenate((s,zeros))
	t0 = time(T*N, len(t))
	return (t0, s0)


def power(s, Fs=128, time=True):
	if time:
		return s**2
	else:
		(S_freq, S) = spectrum(s=s, Fs=Fs, draw=False)
		P = S*S.conj()
		P /= Fs
		P = P.real
		if len(s)%2 == 0:
			P[1: -1] *= 2
		else:
			P[1:] *= 2
		return (S_freq, P)

def energy(s, Fs, time=True):
	if time:
		P = power(s)
		dt = 1/Fs
		return np.sum(power(s)*dt)
	else:
		return np.sum(power(s, Fs, time=False))

def mean (s, Fs):
	N = len(s)
	w = window(N = s.shape[1])
	spectrums = np.zeros((N, int(s.shape[1]/2+1)))
	for i, sig in enumerate(s):
		window_signal(s, w)
		(F, P) = power(sig*w, Fs, time=False)
		spectrums[i] = P
	mean = np.mean(spectrums, axis=0)
	trust_interval = np.zeros((len(F), 2)) # [[lower, higher], ...]
	for f in enumerate(F):
		f_0 = f[0]
		trust_interval[f_0] = [
			st.scoreatpercentile(spectrums[:, f_0], 2.5),
			st.scoreatpercentile(spectrums[:, f_0], 97.5)
		]
	return (F, mean, trust_interval)

'''
==================
Testing functions
==================
'''

if __name__ == '__main__':
	N = 20 # liczba realizacji
	T = 1 # 1 s
	Fs = 100 # Hz
	f = 20 # Hz
	realizacje = np.zeros((N, T*Fs)) # tablica na realizacje
	for i in range(N):
	        (t, s) = sin(f=20,T=1, Fs =100) #realizacja sinusa
	        (t, sz) = white_noise(T=1, Fs=100)#realizacja szumu
	        syg = s+sz # sygnał będący sumą powyższych
	        realizacje[i,:] = syg # wkładamy go do tablicy
	(F, mean, trust_interval) = mean(realizacje, Fs)
	plt.plot(F, mean, 'r')
	plt.plot(F, trust_interval[:, 0], 'b--')
	plt.plot(F, trust_interval[:, 1], 'b--')
	plt.show()
