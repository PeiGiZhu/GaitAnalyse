inf = 1000
global age_divide


global load_matrix_name
global eig_values_name
global explained_var_name
global all_samples_name
global mean_name
global std_name
global age_output_paths


age_divide = [32, 60, 80, inf]
load_matrix_name = "pca_load_matrix.csv"
eig_values_name = "pca_eig_values.csv"
explained_var_name = "pca_explained_var.csv"
all_samples_name = "pca_all_samples.csv"
mean_name = "mean.txt"
std_name = "std.txt"
age_output_paths = ["X_{:d}".format(age_divide[0]),
                     "{:d}_{:d}".format(age_divide[0]+1, age_divide[1]),
                     "{:d}_{:d}".format(age_divide[1]+1, age_divide[2]),
                     "{:d}_X".format(age_divide[2]),
                   ]

