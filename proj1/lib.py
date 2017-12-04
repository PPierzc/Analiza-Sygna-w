#Imports TODO: Add a install.py script to check if all dependencies are installed
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
import scipy.io.wavfile as wf
import scipy.signal
from pprint import pprint

class Signal:
    
    def __init__(self, path, name, mode='dev'):
        file = wf.read(path)
        self.Fs = file[0]
        self.s = file[1]
        self.N = self.s.shape[0]
        self.T = self.N/self.Fs
        self.t = np.arange(0,self.T,1/self.Fs)
        self.mode = mode
        self.name = name
        
    
    def plot_signal(self, show = True, invert=False):
        # Show the sound on a plot
        if show: plt.figure(figsize=(20,10))
        if invert: plt.plot(self.s, self.t)
        if not invert: 
            plt.plot(self.t,self.s)
            plt.xlim(0,36)
        if show:
            plt.show()
            
    def plot_fft(self, show = True, invert=True):
        # Show the sound on a plot
        if show: plt.figure(figsize=(20,10))
        if invert:
            plt.plot(abs(np.fft.rfft(self.s)), np.fft.rfftfreq(len(self.s), 1/self.Fs))
            plt.ylim(0,22000)
            plt.gca().invert_xaxis()
        if not invert: 
            plt.plot(np.fft.rfftfreq(len(self.s), 1/self.Fs),np.fft.rfft(self.s))
            plt.xlim(0,36)
        if show:
            plt.show()
    
    def set_window(self,window = lambda n: np.ones(n)):
        if type(window) == type(lambda t: t):
            self.window = window
        else:
            raise TypeError('Invalid type of window, must be a function. eg. lambda x: np.ones(x)')
    
    def find_cut_off(self, W, ratio=0.0001):
        summed = np.sum(W, axis=1)
        summed /= np.linalg.norm(summed)
        max_val = max(summed)
        summed /= max_val
        summed[summed < ratio] = 0
        return np.where(summed == 0)[0][0]
    
    def calc_spectogram(self, n, t_low=0, t_high=-1):
        # cut up the data into n smaller pieces

        # setup the array size
        
        while self.N%n != 0: #Find the first integer division
            n -= 1

        l_probe = int(self.N/n)
        t_probe = l_probe/self.Fs
        self.l_probe = l_probe
        if (self.mode == 'dev'): print('Length of probe: {}, {:.3f}s \nNumber of probes: {}'.format(l_probe, t_probe, n))

        a_minor = self.s.reshape(n, int(self.N/n))
        t_minor = np.arange(0, int(self.N/n)/self.Fs, 1/self.Fs)
        
        #Ok so now we have an array of a cut up signal, we need to window each signal and find the FFT of all trials
        w = self.window(l_probe)
        w_minor = a_minor * w
        self.W_minor = abs(np.fft.rfft(w_minor).astype('float')).T
        self.F_minor = np.fft.rfftfreq(len(w_minor[0]), 1/self.Fs)
    
    def plot_spectogram(self,
                        N_wsp=1000,
                        show = True,
                        scale = 'manual',
                        ratio = 0.01,
                        t_start=0,
                        t_end=-1,
                        f_start=0,
                        f_end=-1,
                        mode = 'full'):
        
        try:
            if self.W_minor == None or self.F_minor == None:
                pass
        except:
            self.calc_spectogram(N_wsp)
            
        #Finding the boundries
        dt = lambda t: abs(self.t - t)
        if t_start != 0:
            t_start = np.where(dt(t_start) == min(dt(t_start)))[0][0]
			t0 = np.ceil(self.t[t_start]/self.l_probe)
            
        if t_end != -1:
            t_end = np.where(dt(t_end) == min(dt(t_end)))[0][0]
            tmax = np.ceil(self.t[t_end]/self.l_probe)
        
        df = lambda f: abs(self.F_minor - f)
        if f_start != 0:
            f_start = np.where(df(f_start) == min(df(f_start)))[0][0]
        if f_end != -1:
            f_end = np.where(df(f_end) == min(df(f_end)))[0][0]
        
        print('time ', t_start, t_end)
        print('frequency ', f_start, f_end)
        print('time ow high', t0, tmax)
        print(self.W_minor.shape)
       
            
        if self.mode == 'dev': print(self.W_minor.shape)
            
        if scale == 'auto': cutoff = self.find_cut_off(self.W_minor, ratio)
        if mode == 'full':
            plt.figure(figsize=((20,20)))
            plt.subplot2grid((3, 3), (0, 0), rowspan=2)
            self.plot_fft(show = False, invert=True)
        
            plt.subplot2grid((3, 3), (0, 1), colspan=2, rowspan=2)
            plt.title(self.name + ' Spectogram')
            plt.ylabel('Frequencies [Hz]')
            plt.xlabel('Time [s]')
            extent=[ self.t[t_start], self.t[t_end], self.F_minor[f_start], self.F_minor[f_end] ]
            aspect=(self.t[t_end]-self.t[t_start])/(self.F_minor[f_end] - self.F_minor[f_start])


            plt.imshow(self.W_minor[f_start:f_end, t0:tmax], extent = extent, aspect = aspect , origin='lower', cmap="hot")
            
            plt.subplot2grid((3, 3), (2, 1), colspan=2)
            self.plot_signal(show = False)
                        
        if mode == 'plain':
            plt.title(self.name + ' Spectogram')
            plt.ylabel('Frequencies [Hz]')
            plt.xlabel('Time [s]')
            plt.imshow(self.W_minor[f_start:f_end, 5:t_end], extent=[ self.t[t_start], self.t[t_end], self.F_minor[f_start], self.F_minor[f_end] ], aspect=(self.t[t_end]-self.t[f_start])/(self.F_minor[f_end] - self.F_minor[f_start]), origin='lower', cmap="hot")
        # plt.colorbar()
        if show:
            plt.show()