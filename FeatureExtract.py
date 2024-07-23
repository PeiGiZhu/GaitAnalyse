import numpy as np
import argparse
import io
import os
import sys
import cv2
import tqdm
import pandas as pd
from FeatureAnalyseLib.FeatureExtractor import *
# from FeatureAnalyseLib.FeatureExtractor_copy import *
from FeatureAnalyseLib.GetHotelling import *
from FeatureAnalyseLib.BasicParameter import age_divide
from FeatureAnalyseLib.BasicParameter import age_output_paths
from FeatureAnalyseLib.BasicParameter import load_matrix_name
from FeatureAnalyseLib.BasicParameter import eig_values_name
from FeatureAnalyseLib.BasicParameter import explained_var_name
from FeatureAnalyseLib.BasicParameter import all_samples_name
from FeatureAnalyseLib.BasicParameter import mean_name
from FeatureAnalyseLib.BasicParameter import std_name

near_side = {46: 'left', 47: 'right', 48: 'left', 49: 'right', 50: 'left', 51: 'right', 53: 'left', 54: 'left', 55: 'right',
             38: 'right', 56: 'right', 57: 'right', 58: 'right', 59: 'right', 60: 'right', 61: 'right', 62: 'right', 64: 'right', 
             65: 'right', 39: 'right', 66: 'right', 67: 'right', 68: 'right', 69: 'right', 70: 'right', 71: 'left', 72: 'right', 
             73: 'left', 74: 'right', 75: 'right', 76: 'right', 77: 'right', 78: 'left', 79: 'left', 80: 'right', 81: 'right', 
             82: 'right', 83: 'right', 84: 'right', 85: 'right', 86: 'left', 87: 'right', 88: 'right', 89: 'left', 40: 'right', 
             3: 'right', 4: 'left', 8: 'left', 11: 'left', 12: 'right', 13: 'left', 16: 'right', 17: 'left', 19: 'right', 
             20: 'left', 23: 'left', 25: 'right', 26: 'left', 30: 'right', 32: 'right', 33: 'right', 34: 'right', 41: 'right', 42: 'left', 
             43: 'left', 44: 'right', 45: 'right', 15: 'left', 5: 'right', 18: 'right', 27: 'left', 28: 'left', 31: 'left', 36: 'left', 
             10: 'right', 22: 'left', 24: 'right', 35: 'right', 37: 'right', 0: 'right'}

def lnitParse(parser):

    parser.descripition = 'This py file can ectract stride speeds, joint angle extension and \
                                    joint angle phase difference during walking from data record by applying mathematical process.\
                                    Those features will be stored in "(input folder path)/Feature.txt" and will be used in futher analyse. '
    
    parser.add_argument("-i", "--input",
                                    help = "This argument is followed by input folder name. \
                                                The default value is 'data_record', \
                                                please create a floder to store data record \
                                                before you start work",
                                    dest = "ipath",
                                    type = str,
                                    default = "data_record" )

    parser.add_argument("-o", "--output",
                                    help = "This argument is followed by output folder name. \
                                                The default value is 'extracted_features'. ",
                                    dest = "opath",
                                    type = str,
                                    default = "extracted_features" )
    
    parser.add_argument("-d", "--database",
                                     help = "Add -d to store this test subject information to PCA database. ",
                                    dest = "database",
                                    action = "store_true" )

    parser.add_argument("-dp", "--dpath",
                                    help = "This argument is followed by PCAmodel folder path. \
                                                The default value is './PCAmodel'. ",
                                    dest = "dpath",
                                    type = str,
                                    default = "./PCAmodel" )
    
    parser.add_argument("-t2", "--hotellingT2",
                                     help = "Add -t2 to analyse target(patients) recovery level. \
                                                The hotelling'T2 value reflect the distance between target\
                                                and normal samples.",
                                    dest = "hotellingT2",
                                    action = "store_true" )
    
    parser.add_argument("-n", "--filename",
                                    help = "This argument is followed by input video file name \
                                                for example 'xxx.avi' (this py allows any video format) \
                                                the default value is 'xxx.avi' to avoid no input error",
                                    dest = "filename",
                                    type = str,
                                    default = "xxx.avi")
    
    
    parser.add_argument("-ns", "--near_side",
                                    help = "This argument is followed by the which side of the body is close to the camera \
                                                for example 'left' if the left side leg is near to camera \
                                                the default value is 'left' to avoid no input error",
                                    dest = "near_side",
                                    type = str,
                                    default = "left")


    parser.add_argument("-p", "--peaks",
                                    help = "This argument repersents how many peaks. \
                                                will be picked in power density spectrum \
                                                The default value is 2(pick the principal frequency component) ",
                                    dest = "peaks",
                                    type= int,
                                    default = 5)

    parser.add_argument("-c", "--cutslice",
                                    help = "Set this argument to cut slice from prefix of signal \
                                                of time domain signal plot, because sometime the prefix of\
                                                signal are disturbed by unexpected noises.\
                                                The default value is 0.15(15 percent prefix drop of total length of signal) ",
                                    dest = "cutslice",
                                    type= float,
                                    default = 0.15)
    
    parser.add_argument("-a", "--age",
                                    help = "The test subject age \
                                                The default value is 20 years. ",
                                    dest = "age",
                                    type= int,
                                    default = 21)

    parser.add_argument("-he", "--height",
                                    help = "The test subject height (m). \
                                                The default value is 1.75 m. ",
                                    dest = "height",
                                    type= float,
                                    default = 1.75)

    parser.add_argument("-we", "--weight",
                                    help = "The test subject weight (kg). \
                                                The default value is 75 kg.",
                                    dest = "weight",
                                    type= float,
                                    default = 75)

    parser.add_argument("-s", "--sex",
                                    help = "Add -s if test subject is female. ",
                                    dest = "sex",
                                    action = "store_false" )
    parser.add_argument("-id", "--id",
                                    help = "Add -s if test subject is female. ",
                                    dest = "id",
                                    type = int,
                                    default = 0 )
    
    return parser.parse_args()

def load_subject_info(subject_id,
                      subject_sex,
                      subject_age,
                      subject_height,
                      subject_weight):
    
    info_dict = [{"id": subject_id,
                 "sex": subject_sex,
                 "age": subject_age,
                 "height": subject_height,
                 "weight": subject_weight,}]
    df = pd.DataFrame(info_dict)
    #df.to_csv(os.path.join(output_path,"info.csv"))
    return df

def PCAmodel_exists(PCAmodel_path):
    if not os.path.exists(PCAmodel_path):
        print("PCAmodel_path: " + PCAmodel_path)
        print("error: cannot find corresponding PCAmodel_path", file = sys.stderr)
        sys.exit(0)
    if not os.listdir(PCAmodel_path):
        print("PCAmodel_path: " + PCAmodel_path)
        print("error: PCAmodel_path is empty. No model has been loaded.", file = sys.stderr)
        sys.exit(0)

if __name__ == "__main__":

    #init argparse for reding command line
    parser = argparse.ArgumentParser()
    CommandLineArgs = lnitParse(parser)

    input_path = CommandLineArgs.ipath
    output_path = CommandLineArgs.opath
    database_path = CommandLineArgs.dpath
    file_name = CommandLineArgs.filename
    pick_peaks = CommandLineArgs.peaks
    GetHotellingT2 = CommandLineArgs.hotellingT2
    StoreDatabase = CommandLineArgs.database
    cut_slice = CommandLineArgs.cutslice
    near_side = CommandLineArgs.near_side
    # test subject basic information
    subject_sex = CommandLineArgs.sex
    subject_age = CommandLineArgs.age
    subject_height = CommandLineArgs.height
    subject_weight = CommandLineArgs.weight
    subject_id = CommandLineArgs.id
    # save csv in 'data_record'
    subject_info_df = load_subject_info(subject_id,
                                        subject_sex,
                                        subject_age,
                                        subject_height,
                                        subject_weight)
    
    total_tasks = 3
    input_path = os.path.join(input_path, file_name)
    
    if not os.path.exists(database_path):
        os.makedirs(database_path)

    if not os.path.exists(input_path):
        print("input_path: " + input_path)
        print("error: cannot find corresponding video file", file = sys.stderr)
        sys.exit(0)
    
    left_knee_txt = os.path.join(input_path,"left_knee.txt")
    right_knee_txt = os.path.join(input_path,"right_knee.txt")
    left_hip_txt = os.path.join(input_path,"left_hip.txt")
    right_hip_txt = os.path.join(input_path,"right_hip.txt")
    left_ankle_txt = os.path.join(input_path,"left_ankle.txt")
    right_ankle_txt = os.path.join(input_path,"right_ankle.txt")
    
    lost_angle_frame_txt = os.path.join(input_path,"lost_angle_frame.txt")
    stride_speed_txt = os.path.join(input_path,"stride_speed.txt")
    video_fps_txt = os.path.join(input_path,"fps.txt")
    step_length_lr_txt = os.path.join(input_path,"step_length_lr.txt")
    step_length_rl_txt = os.path.join(input_path,"step_length_rl.txt")
    output_path = os.path.join(input_path, output_path)
    
    lost_angle_frame = np.loadtxt(lost_angle_frame_txt)

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    
    if lost_angle_frame.size >0:
        print("warning: there are some body detection failed frame during first step. \
                Although those frame will be predicted by kalmanfilter\
                (this is acceptable if there is only a small number of detection failed frames),\
                we still suggest you to get a better quality video input and so that there is no lost angle frame.", file = sys.stderr)

    video_fps = None
    with open(video_fps_txt,'r') as file:
        video_fps = eval(file.readline())
    
    
    knee_angle_extractor = FeatureExtractor("knee", video_fps, output_path, pick_peaks, cut_slice, near_side)
    hip_angle_extractor  = FeatureExtractor("hip", video_fps, output_path, pick_peaks, cut_slice, near_side)
    ankle_angle_extractor  = FeatureExtractor("ankle", video_fps, output_path, pick_peaks, cut_slice, near_side)

    with tqdm.tqdm(total=total_tasks, position=0, leave=True) as pbar:
        #get average stride speed
        #less important for treadmill
        '''
        if os.path.exists(stride_speed_txt):
            stride_speed = np.loadtxt(stride_speed_txt)
            
            speed_n = stride_speed.size
            speed_average = np.mean(stride_speed)
            speed_variance = np.sqrt(np.sum([(item - speed_average)**2 for item in stride_speed])/(speed_n - 1))
            # pick out isolated value in stride speed
            # save those value which located in [mean - 1.65variance, mean + 1.65variance]
            # 90% of value in this range
            stride_speed = [item for item in stride_speed if np.abs(item-speed_average) < 1.65*speed_variance]
            speed_average = np.mean(stride_speed)
            stride_speed_feature = [{"stride speed": speed_average}]
            df = pd.DataFrame(stride_speed_feature)
            subject_info_df = pd.concat([subject_info_df, df], axis = 1)
            
        pbar.update()
        '''
        '''
        if os.path.exists(step_length_lr_txt) and os.path.exists(step_length_rl_txt):
            step_length_lr = np.loadtxt(step_length_lr_txt)
            step_length_rl = np.loadtxt(step_length_rl_txt)

            # print(step_length_lr)
            if step_length_lr.size == 0 or step_length_rl.size == 0:
                avg_step_length_lr = 1
                avg_step_length_rl = 1
            else:
                avg_step_length_lr = np.sum(step_length_lr)/step_length_lr.size
                avg_step_length_rl = np.sum(step_length_rl)/step_length_rl.size

            # min_step_length = min(avg_step_length_lr, avg_step_length_rl)
            # max_step_length = max(avg_step_length_lr, avg_step_length_rl)

            # df = pd.DataFrame([{"step length ratio": min_step_length/(min_step_length + max_step_length)}])
            with open(os.path.join(output_path,"step_length_ratio.txt"),'w') as file:
                file.write(str(avg_step_length_lr/avg_step_length_rl))
                file.close()
            df = pd.DataFrame([{"step length ratio": avg_step_length_lr/avg_step_length_rl}])
            subject_info_df = pd.concat([subject_info_df, df], axis = 1)

        pbar.update()
        '''

        #process knee angle plot
        if os.path.exists(left_knee_txt) and os.path.exists(right_knee_txt):
            left_knee = np.loadtxt(left_knee_txt)
            right_knee = np.loadtxt(right_knee_txt)
            knee_extracted_feature = knee_angle_extractor(left_knee, right_knee,"knee_angle")
            df = pd.DataFrame(knee_extracted_feature)
            df.to_csv(os.path.join(output_path,"knee_angle_feature_{peak}_{cut}.csv".format(peak = pick_peaks, cut = cut_slice)))
            '''
            with open(os.path.join(output_path,"knee_angle_feature.txt"),'w') as file:
                
                for key, value in knee_extracted_feature.items():
                    file.write(key+": ")
                    file.write(str(value))
                    file.write("\n")
            '''
            subject_info_df = pd.concat([subject_info_df, df], axis = 1)
        pbar.update()
        
        
        #process hip angle plot
        if os.path.exists(left_hip_txt) and os.path.exists(right_hip_txt):
            left_hip = np.loadtxt(left_hip_txt)
            right_hip = np.loadtxt(right_hip_txt)
            hip_extracted_feature = hip_angle_extractor(left_hip, right_hip, "hip_angle")
            df = pd.DataFrame(hip_extracted_feature)
            df.to_csv(os.path.join(output_path,"hip_angle_feature_{peak}_{cut}.csv".format(peak = pick_peaks, cut = cut_slice)))
            '''
            with open(os.path.join(output_path,"hip_angle_feature.txt"),'w') as file:
                
                for key, value in hip_extracted_feature.items():
                    file.write(key+": ")
                    file.write(str(value))
                    file.write("\n")
            '''
            subject_info_df = pd.concat([subject_info_df, df], axis = 1)
        pbar.update()

        '''
        #process ankle angle plot
        if os.path.exists(left_ankle_txt) and os.path.exists(right_ankle_txt):
            left_ankle = np.loadtxt(left_ankle_txt)
            right_ankle = np.loadtxt(right_ankle_txt)
            ankle_extracted_feature = ankle_angle_extractor(left_ankle, right_ankle, "ankle_angle")
            df = pd.DataFrame(ankle_extracted_feature)
            df.to_csv(os.path.join(output_path,"ankle_angle_feature_{peak}_{cut}.csv".format(peak = pick_peaks, cut = cut_slice)))
            #subject_info_df = pd.concat([subject_info_df, df], axis = 1)
        pbar.update()
        '''
        if StoreDatabase is True:
            header = False if os.path.exists(os.path.join(database_path,"all_features.csv")) else True
            subject_info_df.to_csv(os.path.join(database_path,"all_features.csv"), mode ='a',header = header, index=False)
            
        if GetHotellingT2 is True:
            sex = bool(subject_info_df.sex.bool)
            sex = "female" if sex == False else "male"
            age = int(subject_info_df.age)
            
            for i, age_group in enumerate(age_divide):
                if age_group >= age:
                    age = age_output_paths[i]
                    break
            
            PCAmodel_path = os.path.join(database_path, sex)
            PCAmodel_path = os.path.join(PCAmodel_path, age)
            PCAmodel_exists(PCAmodel_path)

            load_matrix = os.path.join(PCAmodel_path, load_matrix_name)
            eig_values = os.path.join(PCAmodel_path, eig_values_name)
            explained_var = os.path.join(PCAmodel_path, explained_var_name)
            all_samples = os.path.join(PCAmodel_path, all_samples_name)
            mean_path = os.path.join(PCAmodel_path, mean_name)
            std_path = os.path.join(PCAmodel_path, std_name)
            
            getHotelling = GetHotelling(output_path,
                                        mean_path,
                                        std_path,
                                        all_samples,
                                        load_matrix,
                                        eig_values,
                                        explained_var)

            # print(list(subject_info_df))
            # subject_info_df.drop(['id', 'sex', 'left principal power frequence', 'right principal power frequence'], axis= 1, inplace = True)
            # subject_info_df.drop(['id', 'sex', 'far principal power frequence', 'near principal power frequence'], axis= 1, inplace = True)
            subject_info_df.drop(['id', 'sex'], axis= 1, inplace = True)
            # print(subject_info_df)
            subject_info_array = np.array(subject_info_df, dtype = complex)
            T2 = getHotelling(subject_info_array)
            print('!!! Hotelling T2', T2)
            
            
        pbar.update()
    
