import numpy.fft as nf
import numpy as np
import matplotlib.pyplot as plt
import os

class FeatureExtractor(object):

    def __init__(self, target_name, fps, save_path, pick_top_k = 1, cut_slice = 0.15):
        # In frequency domain DC component is a peak
        # and the frequency components are symmetric about half of sample rate (Fs = fps)
        # thus, if you want to find out x princpal component excluding DC components and negative frequencies
        # the pick top k should be set to 2x+1 (one DC component and half negative frequencies)
        self._fps = fps
        self._pick_top_k = pick_top_k * 2 + 1

        self._cut_slice = cut_slice
        self._target_name = target_name
        self._extract_feature = dict()
        self._save_path = save_path
        self._sample_length = None
        self._left_angle = None
        self._right_angle = None
        self._time = None
        
        
        
    def __call__(self, left_angle, right_angle, plot_title):

        self._left_angle = left_angle
        self._right_angle = right_angle

        #print(self._right_angle.size, self._left_angle.size)
        if self._right_angle.size != self._left_angle.size:
            self._extract_feature = {"two angle lists have different length": 0.0}
            return self._extract_feature
        
        assert(self._right_angle.size == self._left_angle.size)
        
        self._time = np.arange(self._left_angle.size)/self._fps
        self._sample_length = self._left_angle.size
        freqs = nf.fftfreq(self._left_angle.size, 1/self._fps)

        plt.figure(plot_title, facecolor='lightgray')
        plt.subplot(211)
        plt.title('Time Domain', fontsize=16)
        plt.ylabel('Angle (Degree °)', fontsize =12)
        plt.grid(linestyle = ':')
        left_line, = plt.plot(self._time, self._left_angle, linewidth = 5)
        right_line, = plt.plot(self._time, self._right_angle, linewidth = 5)

        drop_pre = int(self._left_angle.size * self._cut_slice)
        drop_line = plt.axvline(self._time[drop_pre], color='r', linestyle='--')
        
        plt.legend((left_line, right_line, drop_line),
                        ['left angle', 'right angle', 'cut line'],
                        loc = 'upper right')

        self._left_angle = self._left_angle[drop_pre:]
        self._right_angle = self._right_angle[drop_pre:]
        self._time = self._time[drop_pre:]

        self._sample_length = self._left_angle.size
        freqs = nf.fftfreq(self._left_angle.size, 1/self._fps)
        
        left_freq_domain = nf.fft(self._left_angle)
        right_freq_domain = nf.fft(self._right_angle)

        amplitude_left = np.abs(left_freq_domain)/left_angle.size
        amplitude_right = np.abs(right_freq_domain)/right_angle.size

        phase_left = 180*np.angle(left_freq_domain)/np.pi
        phase_right = 180*np.angle(right_freq_domain)/np.pi
        
        power_left = np.abs(left_freq_domain)**2/left_angle.size
        power_right = np.abs(right_freq_domain)**2/right_angle.size

        filtered_power_left = np.argpartition(power_left, -self._pick_top_k)[:-self._pick_top_k]
        filtered_power_right = np.argpartition(power_right, -self._pick_top_k)[:-self._pick_top_k]

        left_freq_domain[filtered_power_left] = 0
        right_freq_domain[filtered_power_right] = 0
        
        principal_left_freq = np.argmax(amplitude_left[1:int(self._sample_length/2)])+1
        principal_right_freq = np.argmax(amplitude_right[1:int(self._sample_length/2)])+1
        
        principal_amplitude_left = amplitude_left[ principal_left_freq ]
        principal_amplitude_right = amplitude_right[ principal_right_freq ]
        #print(domain_left_freq, domain_right_freq)

        principal_phase_left = phase_left[ principal_left_freq ]
        principal_phase_right = phase_right[ principal_right_freq ]

        filter_left_angle =nf.ifft(left_freq_domain)
        filter_right_angle = nf.ifft(right_freq_domain)

        #Load signal after DFT to txt
        np.savetxt(os.path.join(self._save_path, "left_" + self._target_name + "_DFT_{peak}_{cut}.txt".format(peak = self._pick_top_k, cut = self._cut_slice)),
                                           filter_left_angle.real)
        np.savetxt(os.path.join(self._save_path, "right_" + self._target_name + "_DFT_{peak}_{cut}.txt".format(peak = self._pick_top_k, cut = self._cut_slice)),
                                           filter_right_angle.real)
        
        plt.subplot(212)
        plt.ylabel('Angle after Filter(Degree °)', fontsize =12)
        plt.grid(linestyle = ':')
        left_line, = plt.plot(self._time, filter_left_angle.real, linewidth = 5)
        right_line, = plt.plot(self._time, filter_right_angle.real, linewidth = 5)
        plt.legend((left_line, right_line),
                        ['left angle', 'right angle'],
                        loc = 'upper right')

        plt.savefig(os.path.join(self._save_path, plot_title+'_'+ str((self._pick_top_k - 1)/2) + '.png'))
        plt.clf()
        
        left_min_angle = np.min(filter_left_angle.real)
        left_max_angle = np.max(filter_left_angle.real)
        right_min_angle = np.min(filter_right_angle.real)
        right_max_angle = np.max(filter_right_angle.real)
        '''
        self._extract_feature = [{self._csv_column_prefix+"left principal power frequence":freqs[principal_left_freq],
                                             self._csv_column_prefix+"right principal power frequence":freqs[principal_right_freq],
                                             self._csv_column_prefix+"principal phase difference": np.abs(principal_phase_right - principal_phase_left),
                                             self._csv_column_prefix+"left max angle": left_max_angle,
                                             self._csv_column_prefix+"left_min_angle": left_min_angle,
                                             self._csv_column_prefix+"right max angle": right_max_angle,
                                             self._csv_column_prefix+"right min angle": right_min_angle,
                                             self._csv_column_prefix+"left Flexion range": left_max_angle-left_min_angle,
                                             self._csv_column_prefix+"right Flexion range": right_max_angle-right_min_angle,
                                           }]
        '''
        self._extract_feature = [{"left principal power frequence":freqs[principal_left_freq],
                                             "right principal power frequence":freqs[principal_right_freq],
                                             "principal phase difference": np.abs(principal_phase_right - principal_phase_left),
                                             "left max angle": left_max_angle,
                                             "left min angle": left_min_angle,
                                             "right max angle": right_max_angle,
                                             "right min angle": right_min_angle,
                                             "left Flexion range": left_max_angle-left_min_angle,
                                             "right Flexion range": right_max_angle-right_min_angle,
                                           }]

        return self._extract_feature
        
