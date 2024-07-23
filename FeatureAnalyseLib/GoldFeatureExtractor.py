import numpy.fft as nf
import numpy as np
import matplotlib.pyplot as plt
import copy as cp
import os

class FeatureExtractor(object):

    def __init__(self, target_name, fps, pick_top_k = 2, cut_slice = 0.15, near_side = 'left'):
        # In frequency domain DC component is a peak
        # and the frequency components are symmetric about half of sample rate (Fs = fps)
        # thus, if you want to find out x princpal component excluding DC components and negative frequencies
        # the pick top k should be set to 2x+1 (one DC component and half negative frequencies)
        self._fps = fps
        self._pick_top_k = pick_top_k
        self._inf = 2e18
        self._cut_slice = cut_slice

        self._near_side = near_side
        self._target_name = target_name
        self._prat_name = target_name.split('_')[0]
        self._principal_left_freq = None
        self._principal_right_freq = None
        self._extract_feature = dict()
        # self._save_path = save_path
        self._sample_length = None
        self._left_angle = None
        self._right_angle = None
        self._time = None

    def _RMS_error(self, Asignal, Bsignal):
        assert(len(Asignal) == len(Bsignal))
        size_of_signal = len(Asignal)

        error = [np.abs(Asignal[i] - Bsignal[i]) * np.abs(Asignal[i] - Bsignal[i]) for i in range(size_of_signal)]
        return (np.sum(error)/size_of_signal)**0.5
        
    def _pick_components(self, power_left, power_right, left_freq_domain, right_freq_domain):

        DC_components_left =0
        DC_components_right = 0
        
        picked_left_components = [i + 1 for i in range(self._sample_length>>1)]
        picked_right_components = [i + 1 for i in range(self._sample_length>>1)]
        '''
        picked_left_components = np.argpartition(power_left[1:(self._sample_length>>1)],
                                                                        -self._pick_top_k )[-self._pick_top_k :] + 1
        picked_right_components = np.argpartition(power_right[1:(self._sample_length>>1)],
                                                                        -self._pick_top_k )[-self._pick_top_k :] + 1

        hold_left_freq = [DC_components_left, self._principal_left_freq, self._sample_length - self._principal_left_freq]
        hold_right_freq = [DC_components_right, self._principal_right_freq, self._sample_length - self._principal_right_freq]
        '''
        hold_left_freq = [DC_components_left]
        hold_right_freq = [DC_components_right]

        recovered_left_signal = self._left_angle
        recovered_right_signal = self._right_angle
        #print(picked_left_components)
        for loop in range(self._pick_top_k):

            error_left = self._inf
            error_right = self._inf

            component_with_lowest_loss = None
            signal_with_lowest_loss = None
            #print(hold_left_freq)
            for pick_component in picked_left_components:
                if pick_component in hold_left_freq:
                    continue
                
                temp_hold_left_freq = cp.deepcopy(hold_left_freq)
                temp_hold_left_freq.append(pick_component)
                temp_hold_left_freq.append(self._sample_length - pick_component)
                
                temp_left_freq_domain = np.zeros(self._sample_length)*np.array([0j])
                temp_left_freq_domain[temp_hold_left_freq] = left_freq_domain[temp_hold_left_freq]
                
                temp_recoverd_left_signal = nf.ifft(temp_left_freq_domain)
                #if error_left > np.sum(np.abs(temp_recoverd_left_signal - self._left_angle)):
                if error_left > self._RMS_error(temp_recoverd_left_signal, self._left_angle):
                    
                    error_left = self._RMS_error(temp_recoverd_left_signal, self._left_angle)
                    component_with_lowest_loss = pick_component
                    recovered_left_signal = temp_recoverd_left_signal

            #print(loop, component_with_lowest_loss)
            hold_left_freq.append(component_with_lowest_loss)
            hold_left_freq.append(self._sample_length - component_with_lowest_loss)
            
            for pick_component in picked_right_components:
                if pick_component in hold_right_freq:
                    continue
                
                temp_hold_right_freq = cp.deepcopy(hold_right_freq)
                temp_hold_right_freq.append(pick_component)
                temp_hold_right_freq.append(self._sample_length - pick_component)

                temp_right_freq_domain = np.zeros(self._sample_length)*np.array([0j])
                temp_right_freq_domain[temp_hold_right_freq] = right_freq_domain[temp_hold_right_freq]
                
                temp_recoverd_right_signal = nf.ifft(temp_right_freq_domain)
                
                if error_right > self._RMS_error(temp_recoverd_right_signal, self._right_angle):
                    
                    error_right = self._RMS_error(temp_recoverd_right_signal, self._right_angle)
                    component_with_lowest_loss = pick_component
                    recovered_right_signal = temp_recoverd_right_signal

            hold_right_freq.append(component_with_lowest_loss)
            hold_right_freq.append(self._sample_length - component_with_lowest_loss)

        #recovered_left_signal = np.zeros(self._sample_length)
        return recovered_left_signal, recovered_right_signal
        
    def __call__(self, left_angle, right_angle, plot_title):

        self._left_angle = left_angle
        self._right_angle = right_angle

        #print(self._right_angle.size, self._left_angle.size)
        if self._right_angle.size != self._left_angle.size:
            self._extract_feature = {"two angle lists have different length": 0.0}
            return self._extract_feature
        
        assert(self._right_angle.size == self._left_angle.size)

        self._time = np.arange(self._left_angle.size)/self._fps
        
        plt.figure(plot_title)
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

        '''
        filtered_power_left = np.argpartition(power_left, -self._pick_top_k)[:-self._pick_top_k]
        filtered_power_right = np.argpartition(power_right, -self._pick_top_k)[:-self._pick_top_k]
        picked_power_left = np.argpartition(power_left[1:], -4)[-4:] + 1
        picked_power_right = np.argpartition(power_right[1:], -4)[-4:] + 1

        print(freqs)
        print(picked_power_right)
        print(freqs[picked_power_right])
        print(left_freq_domain[picked_power_right])
        
        left_freq_domain[filtered_power_left] = 0
        right_freq_domain[filtered_power_right] = 0

        print(principal_left_freq)
        print(self._sample_length, np.size(freqs))

        principal_amplitude_left = amplitude_left[ principal_left_freq ]
        principal_amplitude_right = amplitude_right[ principal_right_freq ]
        '''
        self._principal_left_freq = np.argmax(amplitude_left[1:(self._sample_length>>1)])+1
        self._principal_right_freq = np.argmax(amplitude_right[1:(self._sample_length>>1)])+1
        
        principal_phase_left = phase_left[ self._principal_left_freq ]
        principal_phase_right = phase_right[ self._principal_right_freq ]

        #filter_left_angle =nf.ifft(left_freq_domain)
        #filter_right_angle = nf.ifft(right_freq_domain)
        
        # filter_left_angle, filter_right_angle = self._pick_components(power_left, power_right, left_freq_domain, right_freq_domain)
        #Load signal after DFT to txt
        # dropped_item = np.array(drop_pre*[np.nan])
        # np.savetxt(os.path.join(self._save_path, "left_" + self._target_name + ".txt"),
        #                                    np.concatenate((dropped_item, filter_left_angle.real), axis = 0))
        # np.savetxt(os.path.join(self._save_path, "right_" + self._target_name + ".txt"),
        #                                    np.concatenate((dropped_item, filter_right_angle.real), axis = 0))

        # np.savetxt(os.path.join(self._save_path, "left_" + self._target_name + ".txt"),
        #                                    filter_left_angle.real)
        # np.savetxt(os.path.join(self._save_path, "right_" + self._target_name + ".txt"),
        #                                    filter_right_angle.real)
        
        # plt.subplot(212)
        # plt.ylabel('Angle After Filter(Degree °)', fontsize = 12)
        # plt.xlabel('Second', fontsize = 12)
        # plt.grid(linestyle = ':')
        
        # left_line, = plt.plot(self._time, filter_left_angle.real, linewidth = 5)
        # right_line, = plt.plot(self._time, filter_right_angle.real, linewidth = 5)
        # plt.legend((left_line, right_line),
        #                 ['left angle', 'right angle'],
        #                 loc = 'upper right')

        # #plt.savefig(os.path.join(self._save_path, plot_title+'_'+ str(self._pick_top_k ) + '_.eps'), format='eps')
        # plt.savefig(os.path.join(self._save_path, plot_title + '_' + str(self._pick_top_k )  + '_' + str(self._cut_slice)+ '.png'))
        
        # plt.clf()

        # plot energy - frequence here
        '''
        plt.figure("Frequency Domain")
        plt.subplot(211)
        
        plt.title('Frequency Domain', fontsize=16)
        plt.ylabel('Amplitude', fontsize =12)
        plt.grid(linestyle = ':')

        left_energy, = plt.plot(freqs[freqs >0], amplitude_right[freqs >0], linewidth = 5)
        
        plt.subplot(212)
        plt.ylabel('Phase (Degree °)', fontsize =12)
        plt.xlabel('Frequence(Hz)', fontsize = 12)
        plt.grid(linestyle = ':')
        
        left_phase, = plt.plot(freqs[freqs >0], phase_right[freqs >0], linewidth = 5)
        plt.savefig(os.path.join(self._save_path, "FrequencyDomain"+'_'+ str(self._pick_top_k ) + '.eps'))
        plt.clf()
        '''
        left_min_angle = np.min(left_angle)
        left_max_angle = np.max(left_angle)
        right_min_angle = np.min(right_angle)
        right_max_angle = np.max(right_angle)
        
        # self._extract_feature = [{"left principal power frequence":freqs[self._principal_left_freq],
        #                           "right principal power frequence":freqs[self._principal_right_freq],
        #                           "principal phase difference": np.abs(principal_phase_right - principal_phase_left),
        #                           "left max angle": left_max_angle,
        #                           "left min angle": left_min_angle,
        #                           "right max angle": right_max_angle,
        #                           "right min angle": right_min_angle,
        #                           "left Flexion range": left_max_angle-left_min_angle,
        #                           "right Flexion range": right_max_angle-right_min_angle,
        #                         }]

        if self._near_side == 'left':

            # if self._prat_name == "hip":
            #     self._extract_feature = [{
            #                               "near hip max angle": left_max_angle,
            #                               "near hip min angle": left_min_angle,
            #                               "far hip max angle": right_max_angle,
            #                               "far hip min angle": right_min_angle,
            #                               "near hip Flexion range": left_max_angle-left_min_angle,
            #                               "far hip Flexion range": right_max_angle-right_min_angle,
            #                                }]

            # else:
            #     self._extract_feature = [{
            #                               "near knee max angle": left_max_angle,
            #                               "near knee min angle": left_min_angle,
            #                               "far knee max angle": right_max_angle,
            #                               "far knee min angle": right_min_angle,
            #                               "near knee Flexion range": left_max_angle-left_min_angle,
            #                               "far knee Flexion range": right_max_angle-right_min_angle,
            #                                }]

            if self._prat_name == "hip":
                self._extract_feature = [{"stride speed": freqs[self._principal_left_freq],
                                      "near hip Flexion range": left_max_angle-left_min_angle,
                                      "far hip Flexion range": right_max_angle-right_min_angle,
                                       }]
            else:
                self._extract_feature = [{
                                      "near knee Flexion range": left_max_angle-left_min_angle,
                                      "far knee Flexion range": right_max_angle-right_min_angle,
                                       }]
        else:

            # if self._prat_name == "hip":
            #     self._extract_feature = [{
            #                               "near hip max angle": right_max_angle,
            #                               "near hip min angle": right_min_angle,
            #                               "far hip max angle": left_max_angle,
            #                               "far hip min angle": left_min_angle,
            #                               "near hip Flexion range": right_max_angle-right_min_angle,
            #                               "far hip Flexion range": left_max_angle-left_min_angle,
            #                                }]    
            # else:
            #     self._extract_feature = [{
            #                               "near knee max angle": right_max_angle,
            #                               "near knee min angle": right_min_angle,
            #                               "far knee max angle": left_max_angle,
            #                               "far knee min angle": left_min_angle,
            #                               "near knee Flexion range": right_max_angle-right_min_angle,
            #                               "far knee Flexion range": left_max_angle-left_min_angle,
            #                                }]  

            if self._prat_name == "hip":
                self._extract_feature = [{"stride speed": freqs[self._principal_right_freq],
                                      "near hip Flexion range": right_max_angle-right_min_angle,
                                      "far hip Flexion range": left_max_angle-left_min_angle,
                                       }]
            else:
                self._extract_feature = [{
                                      "near knee Flexion range": right_max_angle-right_min_angle,
                                      "far knee Flexion range": left_max_angle-left_min_angle,
                                       }]

        return self._extract_feature
        
