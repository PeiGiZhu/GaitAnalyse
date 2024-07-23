import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

class GetHotelling(object):

    def __init__(self, save_path, mean_path, std_path, all_samples, load_matrix, eig_values, explained_var):

        # load_matrix(14, principal components num)
        self._load_matrix = np.array(pd.read_csv(load_matrix), dtype = complex)
        self._eig_values = np.array(pd.read_csv(eig_values), dtype = complex)
        self._explained_var = np.array(pd.read_csv(explained_var), dtype = complex)
        #self._explained_var = np.abs(np.array(pd.read_csv(explained_var), dtype = complex))
        self._all_samples = np.array(pd.read_csv(all_samples), dtype = complex)
        #self._all_samples = np.abs(np.array(pd.read_csv(all_samples), dtype = complex))
        self._mean = np.loadtxt(mean_path)
        self._std = np.loadtxt(std_path)
        self._save_path = save_path
        self._sample = None
        
    def _Standardization(self):
        self._sample -= self._mean
        self._sample /= self._std
        self._sample = np.where(self._sample ==0, 0.00000001, self._sample)
    
    def _get_Hotelling_T2(self, sample_reduction):

        return np.sum([sample_reduction[item]**2/self._eig_values[item] for item in range(sample_reduction.size) ])
        

    def _get_residual(self, sample_reduction, sample):

        return np.dot((sample - sample_reduction), (sample-sample_reduction).T)

        
    def __call__(self, sample):

        '''
        sample(1, 17) structure as follow:
        [
            age, height, weight,
            knee_phase_different, knee_left_max_angle, knee_left_min_angle,
            knee_right_max_angle, knee_right_min_angle, knee_left_flexion_range,
            knee_right_flexion_range,

            hip_phase_different, hip_left_max_angle, hip_left_min_angle,
            hip_right_max_angle, hip_right_min_angle, hip_left_flexion_range,
            hip_right_flexion_range,
        ]
        '''
        # sample 需要标准化
        self._sample = sample
        self._Standardization()
        '''
        print("--------------")
        print(sample.dtype)
        print(self._load_matrix.dtype)
        '''
        sample_reduction = np.dot(self._sample, self._load_matrix)

        T2 = self._get_Hotelling_T2(sample_reduction.T)
        #residual = self._get_residual(sample_reduction.T, sample)

        '''
        color = ['b' for i in range(0, self._all_samples.shape[0])]
        
        #print(np.abs(self._all_samples))
        #self._all_samples = np.concatenate((self._all_samples, sample_reduction), axis = 0)
        
        ax = plt.figure(figsize=(5,5)).add_subplot(111, projection = '3d')
        ax.set_title('The T2 distance between origin point and target')
        #plt.scatter(self._all_samples[:,0].real, self._all_samples[:,1].real, c = color, edgecolor = 'k')
        ax.scatter(0, 0, 0, c = 'r', marker = '^', edgecolor = 'k', label = 'origin') # original point
        ax.scatter(self._all_samples[:,0].real, self._all_samples[:,1].real, self._all_samples[:,2].real, c = color, edgecolor = 'k', label = 'samples')
        #ax.scatter(self._all_samples[:,0], self._all_samples[:,1], self._all_samples[:,2], c = color, edgecolor = 'k', label = 'samples')
        #print(sample_reduction)
        ax.scatter(sample_reduction[0, 0].real, sample_reduction[0, 1].real, sample_reduction[0, 2].real, c = 'g', edgecolor = 'k', label = 'target')
        #ax.scatter(sample_reduction[0, 0], sample_reduction[0, 1], sample_reduction[0, 2], c = 'g', edgecolor = 'k', label = 'target')
        #ax.text(0, 0, 0, "original point",verticalalignment = 'bottom', horizontalalignment = 'center' )
        ax.text(sample_reduction[0, 0], sample_reduction[0, 1],  sample_reduction[0, 2], "T2 score= {:.2f}".format(float(T2.real)),
                    verticalalignment = 'bottom', horizontalalignment = 'center' )

        plt.legend()
        x_label = 'PC0({:.2%})'.format(float(self._explained_var[0].real))
        y_label = 'PC1({:.2%})'.format(float(self._explained_var[1].real))
        z_label = 'PC2({:.2%})'.format(float(self._explained_var[2].real))

        #x_label = 'PC0({:.2%})'.format(float(self._explained_var[0]))
        #y_label = 'PC1({:.2%})'.format(float(self._explained_var[1]))
        #z_label = 'PC2({:.2%})'.format(float(self._explained_var[2]))
        
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_zlabel(z_label)
        
        #plt.savefig(os.path.join(self._save_path,"PCA_scatter.eps"), transparent=True )
        plt.savefig(os.path.join(self._save_path,"PCA_scatter.png"), transparent=True )
        '''
        return T2
    
        
        
        
    

        
