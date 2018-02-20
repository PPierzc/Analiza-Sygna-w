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

def gabor(t0 = 0.5, sigma = 0.1, T = 1.0, f=10, phi = 0, Fs = 128.0):
	t = czas(T,Fs)
	s = np.exp(-((t-t0)/(sigma))**2/2) * np.cos(2*np.pi*f*(t-t0) + phi)
	return s


def falka(f = 10,w=7):
    Fs = 256
    x = np.zeros(2560)
    T= len(x)/Fs
    t = np.arange(0,T,1/Fs)
       
    s = T*f/(2*w)
    xx = np.linspace(-s * 2 * np.pi, s * 2 * np.pi, len(x))
    psi = np.exp(1j * w * xx) * np.exp(-0.5 * (xx**2)) * np.pi**(-0.25) #falka ze skalą odpowiadajacą częstości f
    plt.plot(t,psi.real)
    plt.xlim((0,10))
    plt.show()
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


def ar_model(a, sigma, N):
	'''
    a: współczynniki
    sigma: standardowe odchylenie
    N: liczba próbek w realizacji
    '''
	x = np.zeros(N)
	M = len(a)
	for i in range(M, N):  # kolejno tworzymy próbki w realizacji
		for j, m in enumerate(a):
			x[i] += m * x[i-j-1]
		x[i] += st.norm.rvs(loc=0, scale=sigma)
	return x

def ar_parameters(x,p):
    '''funkcja estymująca parametry modelu AR
    argumenty:
    x- sygnał
    p - rząd modelu
    f. zwraca:
    a - wektor współczynników modelu
    epsilon - estymowana wariancja szumu

    funkcja wymaga zaimportowania modułu numpy as np
    '''
    N = len(x)
    ak = np.correlate(x,x,mode='full')
    norm_ak = np.hstack((np.arange(1,N+1,1),np.arange(N-1,0,-1)))
    ak=ak/norm_ak
    R=ak[N-1:]
    RL  = R[1:1+p]
    RP = np.zeros((p,p))
    for i in range(p):
        aa = ak[N-1-i:N-1-i+p]
        RP[i,:] = aa
    a=np.linalg.solve(RP,RL)
    sigma = (ak[N-1] - np.sum(a*ak[N:N+p]))**0.5
    return a, sigma

def kryterium_AIC(x,M):
    m_range = range(1,M)
    N = len(x)
    AIC = np.zeros(len(m_range))
    for p in m_range:
        a,sigma = ar_parameters(x,p)
        AIC[p-1] = 2*(p-1)/N + np.log(sigma**2)
        print( 'p:', p, ' a:',a,' sigma: ',sigma)
    return AIC

def widmoAR(parametry_a, sigma, N_punktow, Fs):
    f = np.linspace(0,Fs/2,N_punktow)
    z = np.exp(1j*2*np.pi*f/Fs)
    A = -1 * np.ones(N_punktow) + 1j*np.zeros(N_punktow)
    for i in range(len(parametry_a)):
        A += parametry_a[i]*z**(-(i+1))
    H = 1./A
    Sp = H*H.conj()* sigma**2 # widmo
    Sp = Sp/Fs #gęstość widmowa
    Sp = Sp.real
    return f, Sp


def charkterystyki(a,b,f,T,Fs):
    t = np.arange(-T, T, 1/Fs)
    w = 2*np.pi* f/Fs
    w, h = freqz(b, a, w)

    faza = np.unwrap(np.angle(h))
    df = np.diff(faza)
    idx, = np.where(np.abs(df-np.pi)<0.05)
    df[idx] = (df[idx+1]+df[idx-1])/2
    grupowe = - df/np.diff(w)
    opoznienieFazowe = - faza/w

    fig = py.figure()
    py.subplot(3,2,1)
    py.title('moduł transmitancji')
    m = np.abs(h)
    py.plot(f,20*np.log10(m))
    py.ylabel('[dB]')
    py.grid('on')


    py.subplot(3,2,3)
    py.title('opóźnienie fazowe')
    py.plot(f, opoznienieFazowe)
    py.ylabel('próbki')
    py.grid('on')

    py.subplot(3,2,5)
    py.title('opóźnienie grupowe')
    py.plot(f[:-1],grupowe)
    py.ylabel('próbki')
    py.xlabel('Częstość [Hz]')
    py.grid('on')
    py.ylim([0, np.max(grupowe)+1])

    py.subplot(3,2,2)
    py.title('odpowiedź impulsowa')
    x = np.zeros(len(t))
    x[len(t)//2] = 1
    y = lfilter(b,a,x)
    py.plot(t, x)
    py.plot(t, y)
    py.xlim([-T/2,T])
    py.grid('on')

    py.subplot(3,2,4)
    py.title('odpowiedź schodkowa')
    s = np.zeros(len(t))
    s[len(t)//2:] = 1
    ys = lfilter(b,a,s) # przepuszczamy impuls przez filtr i obserwujemy odpowiedź impulsową
    py.plot(t, s)
    py.plot(t, ys)
    py.xlim([-T/2,T])
    py.xlabel('Czas [s]')
    py.grid('on')

    fig.subplots_adjust(hspace=.5)
    py.show()

def spektrogram(x, okno, trans , Fs):
    Nx = len(x)
    No = len(okno)
    okno = okno/np.linalg.norm(okno)
    pozycje_okna = np.arange(0, Nx, trans)
    t = pozycje_okna/Fs
    N_trans = len(pozycje_okna)
    f = np.fft.rfftfreq(No, 1/Fs)
    P = np.zeros((len(f),N_trans))
    z = np.zeros(int(No/2))
    sig = np.concatenate((z,x,z))
    for i,poz in enumerate(pozycje_okna): # iterujemy po możliwych ozycjach ookna
        s = sig[poz: poz+No] # pobierz wycinek sygnału o długości okna i rozpoczynający się w aktualnej pozycji okna
        s = s*okno
        S = np.fft.rfft(s) # oblicz rzeczywistą transformatę zokienkowanego sygnału
        P_tmp = S*S.conj() # obliczam moc
        P_tmp = P_tmp.real/Fs
        if len(s)%2 ==0: # dokładamy moc z ujemnej części widma
            P_tmp[1:-1] *=2
        else:
            P_tmp[1:] *=2
        P[:,i] = P_tmp
    return t,f,P

def skalogram(x,  w=7.0, MinF = 1.0 ,MaxF = 64, df=1.0, Fs=128.):
    '''
    x - sygnał
    w - parametr falki Morleta,
      wiąże się z jej częstością centralną i skalą w nastąpujący sposób:
      f = 2*s*w / T
      gdzie: s-skala,  T-długość sygnału w sek.

    MinF,MaxF - częstości pomiędzy którymi ma być liczony skalogram
    df - odstęp pomiędzy częstościami
    Fs - częstość próbkowania
    '''
    T= len(x)/Fs
    M = len(x)
    t = np.arange(0,T,1./Fs)
    freqs = np.arange(MinF,MaxF,df)
    P = np.zeros((len(freqs),M))
    X = np.fft.fft(x)
    for i,f in enumerate(freqs):
        s = T*f/(2*w)
        xx = np.linspace(-s * 2 * np.pi, s * 2 * np.pi, M)
        psi = np.exp(1j * w * xx) * np.exp(-0.5 * (xx**2)) * np.pi**(-0.25) #falka ze skalą odpowiadajacą częstości f
        Psi = np.fft.fft(psi)# transformata falki
        Psi /= np.sqrt(np.sum(Psi*Psi.conj()))  # normalizujemy transformatę falki
        tmp = np.fft.fftshift(np.fft.ifft(X*Psi)) # liczymy odwrotną transformatę od iloczynu transformaty sygnału i transformaty flaki,
                                                   # czyli liczymy splot sygnału i falki w dziedzinie czasu
        P[i,:] = (tmp*tmp.conj()).real # liczymy moc tego splotu
    return t, freqs, P
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

	a=[0.1, -0.7, 0.3, -0.5, -0.3]
	sigma = .1
	N = 1000
	x = ar_model(a, sigma, N)
	plt.figure(figsize=((20,10)))
	plt.plot(x)
	plt.show()

