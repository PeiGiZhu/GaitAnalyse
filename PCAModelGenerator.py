import os
import sys
import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tqdm
import argparse
from FeatureAnalyseLib.ModelLoader import *
from FeatureAnalyseLib.BasicParameter import age_output_paths
from FeatureAnalyseLib.BasicParameter import age_divide
root ='.'
#inf = 1000
#age_divide = [20, 30, 50, 70, inf]

def lnitParse(parser):

    parser.descripition = 'This py file can ectract stride speeds, joint angle extension and \
                                    joint angle phase difference during walking from data record by applying mathematical process.\
                                    Those features will be stored in "(input folder path)/Feature.txt" and will be used in futher analyse. '
    
    parser.add_argument("-i", "--input",
                                    help = "This argument is followed by extracted feature record path. \
                                                The default value is './data_record', \
                                                please create a floder to store data record \
                                                before you start work",
                                    dest = "ipath",
                                    type = str,
                                    default = "./PCAmodel" )

    parser.add_argument("-o", "--output",
                                    help = "This argument is followed by output folder path. \
                                                The default value is 'PCAmodel'. ",
                                    dest = "opath",
                                    type = str,
                                    default = "./PCAmodel" )
    
    parser.add_argument("-n", "--filename",
                                    help = "This argument is followed by extracted feature csv file name. \
                                                The default value is 'all_features.csv', \
                                                please create a floder to store data record \
                                                before you start work",
                                    dest = "filename",
                                    type = str,
                                    default = "all_features.csv" )
    
    return parser.parse_args()


def load_data_from_csv(all_features_csv):

    df = pd.read_csv(all_features_csv)
    # df.drop(['left principal power frequence', 'right principal power frequence'], axis= 1, inplace = True)
    # the pandas will automatically generate suffix '.X' if there is same label name in the csv
    # df.drop(['left principal power frequence.1', 'right principal power frequence.1'], axis= 1, inplace = True)
    df.drop(['id'], axis= 1, inplace = True)
    previous_age = 0
    male = []
    female = []
    
    for i, age_group in enumerate(age_divide):

        temp = df[(df.sex == True) & (df.age <= age_group) & (df.age > previous_age)].drop(['sex'],axis=1)
        male.append(np.array(temp))
            
        temp = df[(df.sex == False) & (df.age <= age_group) & (df.age > previous_age)].drop(['sex'],axis=1)
        female.append(np.array(temp))
            
        previous_age = age_group

    #print(male, female)
    return male, female
        

if __name__ == "__main__":

    #init argparse for reding command line
    parser = argparse.ArgumentParser()
    CommandLineArgs = lnitParse(parser)

    input_path = CommandLineArgs.ipath
    output_path = CommandLineArgs.opath
    file_name = CommandLineArgs.filename
    
    if not os.path.exists(input_path):
        print("input_path: " + input_path)
        print("error: cannot find corresponding features folder", file = sys.stderr)
        sys.exit(0)
        
    all_features_csv = os.path.join(input_path, file_name)
    male, female = load_data_from_csv(all_features_csv)
    
    female_output_path = os.path.join(output_path,"female")
    male_output_path = os.path.join(output_path,"male")

    if not os.path.exists(female_output_path):
        os.makedirs(female_output_path)

    if not os.path.exists(male_output_path):
        os.makedirs(male_output_path)

    # age_output_paths are stored in BasicParameter.py 
    
    female_age_output_path = [os.path.join(female_output_path, age_output_path ) for age_output_path in age_output_paths]

    male_age_output_path = [os.path.join(male_output_path, age_output_path ) for age_output_path in age_output_paths]

    
    for folder in female_age_output_path:
        if not os.path.exists(folder):
            os.makedirs(folder)

    for folder in male_age_output_path:
        if not os.path.exists(folder):
            os.makedirs(folder)

    modelloader = ModelLoader(explained_var_demand = 0.9)
    #print(male)
    
    with tqdm.tqdm(total=len(male)+len(female), position=0, leave=True) as pbar:
        
        for i, samples in enumerate(male):

            modelloader(samples, male_age_output_path[i])
            pbar.update()
        
        for i, samples in enumerate(female):

            modelloader(samples, female_age_output_path[i])
            pbar.update()
    
        
    
    
