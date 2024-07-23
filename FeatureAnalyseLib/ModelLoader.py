import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

class ModelLoader(object):

    def __init__(self, explained_var_demand = 0.9):

        self._explained_var_demand = explained_var_demand

        self._save_path = None
        self._data = None
        self._cov = None
        self._eig_values = None
        self._eig_vector = None
        self._explained_var = None

    def _Standardization(self):

        samples_mean = np.mean(self._data, axis = 0, keepdims = True)
        self._data -= samples_mean
        samples_std = np.std(self._data, axis = 0, keepdims = True)
        #if some features have same value, it will cause std = 0.
        samples_std = np.where(samples_std == 0, 0.00000001, samples_std)
        self._data /= samples_std
        self._data = np.where(self._data ==0, 0.00000001, self._data)
        
        samples_mean= np.array(samples_mean)
        samples_std = np.array(samples_std)
        
        np.savetxt(os.path.join(self._save_path,"mean.txt"),samples_mean)
        np.savetxt(os.path.join(self._save_path,"std.txt"),samples_std)

    def _pick_principal_components(self):
        
        indexs_ = None
        for i in range(self._eig_values.size):
        
            indexs_ = np.argsort(-self._eig_values)[:i]
            picked_eig_values = self._eig_values[indexs_]
            if np.sum(np.real(picked_eig_values))/np.sum(np.real(self._eig_values)) >= self._explained_var_demand:
                self._explained_var = picked_eig_values/np.sum(self._eig_values)
                break

        self._eig_vector = self._eig_vector[:, indexs_]
        self._eig_values =  self._eig_values[indexs_]
        
        
    def __call__(self, data, save_path):

        if np.size(data) == 0:
            return False
        
        self._data = data
        #print(self._data)
        self._save_path = save_path
        self._Standardization()
        self._cov = np.dot(self._data.T, self._data)
        self._eig_values, self._eig_vector = np.linalg.eig(self._cov)

        self._pick_principal_components()

        # the standardized sample on reductive dimension is "all_samples_ndim"
        """
        print('-------------')
        print(self._data.shape)
        print(self._eig_vector.shape)
        """
        all_samples_ndim = np.dot(self._data, self._eig_vector)
        all_samples_ndim_df = pd.DataFrame(all_samples_ndim)
        all_samples_ndim_df.to_csv(os.path.join(save_path,'pca_all_samples.csv'), index=False)
        
        load_matrix_df = pd.DataFrame(self._eig_vector)
        load_matrix_df.to_csv(os.path.join(save_path,'pca_load_matrix.csv'), index=False)
        
        eig_values_df = pd.DataFrame(self._eig_values)
        eig_values_df.to_csv(os.path.join(save_path,'pca_eig_values.csv'), index=False)

        explained_var_df = pd.DataFrame(self._explained_var)
        explained_var_df.to_csv(os.path.join(save_path,'pca_explained_var.csv'), index=False)

        # print(os.path.join(save_path,'pca_all_samples.csv'))

        return True
