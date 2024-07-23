from calendar import c
from ProcessVideoLib.PoseClassification import *
from ProcessVideoLib.PoseEmbedding import *
from ProcessVideoLib.RepetionCounter import *
from ProcessVideoLib.Commons import *
from ProcessVideoLib.ClassificationSmoothing import *
from ProcessVideoLib.Visualizer import *
from ProcessVideoLib.StrideSpeed import *
#PlotAngle will be moved to Visualizer and previous one will be discard soon
from ProcessVideoLib.PlotAngle import *
##########################
from ProcessVideoLib.DataRecorder import *
from ProcessVideoLib.JointAngle import *
from ProcessVideoLib.KalmanFilter import *
# from ProcessVideoLib.KalmanFilter_one import *
# from ProcessVideoLib.UnscentedKF import *
# from ProcessVideoLib.UnscentedKF_cos import *
# from ProcessVideoLib.UnscentedKF_cos_one import *
from ProcessVideoLib.Landmarks import *


import numpy as np
import argparse
import io
import os
import cv2
import tqdm
from mediapipe.python.solutions import pose as mp_pose
from mediapipe.python.solutions import drawing_utils as mp_drawing
from mediapipe.python.solutions import drawing_styles as mp_drawing_styles

#the class name in sample folder
class_name = 'right'
#back ground color for segmentation mask
BG_COLOR = (192, 192, 192) # gray

Test = "cos"

#init argparse for reading command line
def InitParser(parser):
    
    parser.descripition = 'This py file can process walking video input (allow any format)\
                                    and return a video analyse with visual gait parameters.'
    parser.add_argument("-i", "--input",
                                    help = "This argument is followed by input file path. \
                                                The default value is 'vedio_input', \
                                                please create a floder to store video file \
                                                before you start work",
                                    dest = "ipath",
                                    type = str,
                                    default = "video_input" )
    parser.add_argument("-o", "--output",
                                    help = "This argument is followed by result file path. \
                                                The default value is 'vedio_output' ",
                                    dest = "opath",
                                    type = str,
                                    default = "video_output" )
    parser.add_argument("-n", "--filename",
                                    help = "This argument is followed by input video file name \
                                                for example 'xxx.avi' (this py allows any video format) \
                                                the default value is 'xxx.avi' to avoid no input error",
                                    dest = "filename",
                                    type = str,
                                    default = "xxx.avi")
    parser.add_argument("-c", "--csvfile",
                                    help = "This argument is followed by samples folder for KNN \
                                                pose classify. The default value is \
                                                'fitness_poses_csvs_out' ",
                                    dest = "cpath",
                                    type = str,
                                    default = "fitness_poses_csvs_out" )
    parser.add_argument("-s", "--showimg",
                                    help = "Add -s to show processing img result \
                                                pre n(Custom) frames. \
                                                Set 0(default value) to close this function",
                                    dest = "show",
                                    type= int,
                                    default = 0)
    parser.add_argument("-sd", "--savedata",
                                    help = "This argument is followed by folder name for saving joint angle history.\
                                                The joint angle history is used for waveform analysis. \
                                                And the default value is 'data_record' ",
                                    dest = "sdata",
                                    type= str,
                                    default = 'data_record')
    parser.add_argument("-m", "--mask",
                                    help = "Add -m to apply segmentation mask. \
                                                This may improve key point detection preformance \
                                                in some cases, but will be time comsuming",
                                    dest = "mask",
                                    action = "store_true")
    parser.add_argument("-pn", "--near_pnoisecoeff",
                                    help = "This argument repersents coefficient of process noise covariance. \
                                                The default value is 10 ",
                                    dest = "pnoise_n",
                                    type= float,
                                    default = 10) #1e-2
    parser.add_argument("-mn", "--near_mnoisecoeff",
                                    help = "This argument repersents coefficient of measurement noise covariance. \
                                                The default value is 1 ",
                                    dest = "mnoise_n",
                                    type= float,
                                    default = 1) #1e-2

    parser.add_argument("-pf", "--far_pnoisecoeff",
                                    help = "This argument repersents coefficient of process noise covariance. \
                                                The default value is 1e-2 ",
                                    dest = "pnoise_f",
                                    type= float,
                                    default = 1e-2) # 1e-1
    parser.add_argument("-mf", "--far_mnoisecoeff",
                                    help = "This argument repersents coefficient of measurement noise covariance. \
                                                The default value is 1e-3 ",
                                    dest = "mnoise_f",
                                    type= float,
                                    default = 1e-3) # 1e-2

    return parser.parse_args()

def check_csv(csv_path):
    
    storted_files = os.listdir(csv_path)
    sample_files = [file for file in storted_files if 'csv' in file ]

    if len(sample_files) == 0:
        print("error: there is no sample file for pose classification", file = sys.stderr)
        sys.exit(0)

def check_input(input_path):

    if not os.path.exists(input_path):
        print("input_path: " + input_path)
        print("error: cannot find corresponding video file", file = sys.stderr)
        sys.exit(0)

def copy_world_landmarks_for_ankle(world_landmarks, frame_height, frame_width):
  self_landmarks = dict()

  self_landmarks[mp_pose.PoseLandmark.RIGHT_HIP] = Landmarks(world_landmarks[mp_pose.PoseLandmark.RIGHT_HIP])
  self_landmarks[mp_pose.PoseLandmark.LEFT_HIP] = Landmarks(world_landmarks[mp_pose.PoseLandmark.LEFT_HIP])

  self_landmarks[mp_pose.PoseLandmark.RIGHT_KNEE] = Landmarks(world_landmarks[mp_pose.PoseLandmark.RIGHT_KNEE])
  self_landmarks[mp_pose.PoseLandmark.LEFT_KNEE] = Landmarks(world_landmarks[mp_pose.PoseLandmark.LEFT_KNEE])
  
  self_landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE] = Landmarks(world_landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE])
  self_landmarks[mp_pose.PoseLandmark.LEFT_ANKLE] = Landmarks(world_landmarks[mp_pose.PoseLandmark.LEFT_ANKLE])

  self_landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER] = Landmarks(world_landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER])
  self_landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER] = Landmarks(world_landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER])

  self_landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX] = Landmarks(world_landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX])
  self_landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX] = Landmarks(world_landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX])

  for lmk in self_landmarks:
    self_landmarks[lmk].x = self_landmarks[lmk].x * frame_width
    self_landmarks[lmk].y = self_landmarks[lmk].y * frame_height
    self_landmarks[lmk].z = self_landmarks[lmk].z * frame_width

  return self_landmarks

def copy_landmarks_for_hip(landmarks, frame_height, frame_width):
  self_landmarks = dict()

  self_landmarks[mp_pose.PoseLandmark.RIGHT_HIP] = Landmarks(landmarks[mp_pose.PoseLandmark.RIGHT_HIP])
  self_landmarks[mp_pose.PoseLandmark.LEFT_HIP] = Landmarks(landmarks[mp_pose.PoseLandmark.LEFT_HIP])

  self_landmarks[mp_pose.PoseLandmark.RIGHT_KNEE] = Landmarks(landmarks[mp_pose.PoseLandmark.RIGHT_KNEE])
  self_landmarks[mp_pose.PoseLandmark.LEFT_KNEE] = Landmarks(landmarks[mp_pose.PoseLandmark.LEFT_KNEE])
  
  self_landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE] = Landmarks(landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE])
  self_landmarks[mp_pose.PoseLandmark.LEFT_ANKLE] = Landmarks(landmarks[mp_pose.PoseLandmark.LEFT_ANKLE])

  self_landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER] = Landmarks(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER])
  self_landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER] = Landmarks(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER])

  self_landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX] = Landmarks(landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX])
  self_landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX] = Landmarks(landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX])

  for lmk in self_landmarks:
    self_landmarks[lmk].x = self_landmarks[lmk].x * frame_width
    self_landmarks[lmk].y = self_landmarks[lmk].y * frame_height
    self_landmarks[lmk].z = self_landmarks[lmk].z * frame_width

  return self_landmarks

def copy_landmarks(landmarks, frame_height, frame_width, Smoothed_LR_Zpos):
  self_landmarks = dict()

  self_landmarks[mp_pose.PoseLandmark.RIGHT_HIP] = Landmarks(landmarks[mp_pose.PoseLandmark.RIGHT_HIP])
  self_landmarks[mp_pose.PoseLandmark.LEFT_HIP] = Landmarks(landmarks[mp_pose.PoseLandmark.LEFT_HIP])

  self_landmarks[mp_pose.PoseLandmark.RIGHT_KNEE] = Landmarks(landmarks[mp_pose.PoseLandmark.RIGHT_KNEE])
  self_landmarks[mp_pose.PoseLandmark.LEFT_KNEE] = Landmarks(landmarks[mp_pose.PoseLandmark.LEFT_KNEE])
  
  self_landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE] = Landmarks(landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE])
  self_landmarks[mp_pose.PoseLandmark.LEFT_ANKLE] = Landmarks(landmarks[mp_pose.PoseLandmark.LEFT_ANKLE])

  self_landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER] = Landmarks(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER])
  self_landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER] = Landmarks(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER])

  self_landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX] = Landmarks(landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX])
  self_landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX] = Landmarks(landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX])
  

#   for lmk in self_landmarks:
#     self_landmarks[lmk].x = self_landmarks[lmk].x * frame_width
#     self_landmarks[lmk].y = self_landmarks[lmk].y * frame_height
#     self_landmarks[lmk].z = self_landmarks[lmk].z * frame_width

  for lmk in self_landmarks:
    #right side is hidden
    if "RIGHT" in str(lmk):
      if Smoothed_LR_Zpos['right'] > Smoothed_LR_Zpos['left']:
        self_landmarks[lmk].x = self_landmarks[lmk].x * frame_width
        self_landmarks[lmk].y = self_landmarks[lmk].y * frame_height
        self_landmarks[lmk].z = self_landmarks[lmk].z * frame_width
      else:
        pass
    else:
      if Smoothed_LR_Zpos['right'] < Smoothed_LR_Zpos['left']:
        self_landmarks[lmk].x = self_landmarks[lmk].x * frame_width
        self_landmarks[lmk].y = self_landmarks[lmk].y * frame_height
        self_landmarks[lmk].z = self_landmarks[lmk].z * frame_width
      else:
        pass

  return self_landmarks

def get_joint_angle(joint_angle_predictor, far_side_angle_filter, near_side_angle_filter, landmarks, world_landmarks, frame_height, frame_width, Smoothing, frame_idx):

    joint_angle = dict()
    far_side_joint_angle = dict()
    near_side_joint_angle = dict()

    if world_landmarks is None:
        return joint_angle
    
    LR_Zpos = {"left": 10*world_landmarks[mp_pose.PoseLandmark.LEFT_HIP].z, 
                "right": 10*world_landmarks[mp_pose.PoseLandmark.RIGHT_HIP].z}
    
    # print(LR_Zpos)
    Smoothed_LR_Zpos = Smoothing(LR_Zpos)

    # print("knee x ", world_landmarks[mp_pose.PoseLandmark.LEFT_KNEE].x, world_landmarks[mp_pose.PoseLandmark.RIGHT_KNEE].x)
    # print("knee y ", world_landmarks[mp_pose.PoseLandmark.LEFT_KNEE].y, world_landmarks[mp_pose.PoseLandmark.RIGHT_KNEE].y)
    # print("knee z ", world_landmarks[mp_pose.PoseLandmark.LEFT_KNEE].z, world_landmarks[mp_pose.PoseLandmark.RIGHT_KNEE].z)
    
    self_landmarks = copy_landmarks(world_landmarks, frame_height, frame_width, Smoothed_LR_Zpos)

    left_knee_angle = joint_angle_predictor(self_landmarks,
                                            mp_pose.PoseLandmark.LEFT_KNEE,
                                            mp_pose.PoseLandmark.LEFT_HIP,
                                            mp_pose.PoseLandmark.LEFT_ANKLE)
    
    right_knee_angle = joint_angle_predictor(self_landmarks,
                                            mp_pose.PoseLandmark.RIGHT_KNEE,
                                            mp_pose.PoseLandmark.RIGHT_HIP,
                                            mp_pose.PoseLandmark.RIGHT_ANKLE)

    self_landmarks = copy_landmarks_for_hip(world_landmarks, frame_height, frame_width)

    # left_hip_angle = joint_angle_predictor(self_landmarks,
    #                                         mp_pose.PoseLandmark.LEFT_HIP,
    #                                         mp_pose.PoseLandmark.LEFT_KNEE,
    #                                         mp_pose.PoseLandmark.LEFT_SHOULDER)

    # right_hip_angle = joint_angle_predictor(self_landmarks,
    #                                         mp_pose.PoseLandmark.RIGHT_HIP,
    #                                         mp_pose.PoseLandmark.RIGHT_KNEE,
                                            # mp_pose.PoseLandmark.RIGHT_SHOULDER)
    
    left_hip_angle = joint_angle_predictor(self_landmarks,
                                            mp_pose.PoseLandmark.LEFT_HIP,
                                            mp_pose.PoseLandmark.LEFT_KNEE,
                                            None,
                                            use_invisible_point = True)

    right_hip_angle = joint_angle_predictor(self_landmarks,
                                            mp_pose.PoseLandmark.RIGHT_HIP,
                                            mp_pose.PoseLandmark.RIGHT_KNEE,
                                            None,
                                            use_invisible_point = True)
    
    # self_landmarks = copy_world_landmarks_for_ankle(world_landmarks, frame_height, frame_width)
    
    left_ankle_angle = joint_angle_predictor(self_landmarks,
                                            mp_pose.PoseLandmark.LEFT_ANKLE,
                                            mp_pose.PoseLandmark.LEFT_KNEE,
                                            mp_pose.PoseLandmark.LEFT_FOOT_INDEX)

    right_ankle_angle = joint_angle_predictor(self_landmarks,
                                            mp_pose.PoseLandmark.RIGHT_ANKLE,
                                            mp_pose.PoseLandmark.RIGHT_KNEE,
                                            mp_pose.PoseLandmark.RIGHT_FOOT_INDEX)
    
    if Smoothed_LR_Zpos['right'] > Smoothed_LR_Zpos['left']:

        near_side_joint_angle[mp_pose.PoseLandmark.LEFT_KNEE] = left_knee_angle
        near_side_joint_angle[mp_pose.PoseLandmark.LEFT_HIP] = left_hip_angle
        near_side_joint_angle[mp_pose.PoseLandmark.LEFT_ANKLE] = left_ankle_angle

        far_side_joint_angle[mp_pose.PoseLandmark.RIGHT_KNEE] = right_knee_angle
        far_side_joint_angle[mp_pose.PoseLandmark.RIGHT_HIP] = right_hip_angle
        far_side_joint_angle[mp_pose.PoseLandmark.RIGHT_ANKLE] = right_ankle_angle
    
    else:

        far_side_joint_angle[mp_pose.PoseLandmark.LEFT_KNEE] = left_knee_angle
        far_side_joint_angle[mp_pose.PoseLandmark.LEFT_HIP] = left_hip_angle
        far_side_joint_angle[mp_pose.PoseLandmark.LEFT_ANKLE] = left_ankle_angle

        near_side_joint_angle[mp_pose.PoseLandmark.RIGHT_KNEE] = right_knee_angle
        near_side_joint_angle[mp_pose.PoseLandmark.RIGHT_HIP] = right_hip_angle
        near_side_joint_angle[mp_pose.PoseLandmark.RIGHT_ANKLE] = right_ankle_angle

    # joint_angle = {**near_side_joint_angle, **far_side_joint_angle}
    # joint_angle = far_side_angle_filter(joint_angle)
    raw_joint_angle = {**near_side_joint_angle, **far_side_joint_angle}
    # v_list = []
    # a_list = []
    if Test == "cos":
        # print(far_side_joint_angle)
        near_side_joint_angle = near_side_angle_filter(near_side_joint_angle, frame_idx)
        far_side_joint_angle = far_side_angle_filter(far_side_joint_angle, frame_idx)
        
        joint_angle = {**near_side_joint_angle, **far_side_joint_angle}

    elif Test == "cos_one":
        # print("Yes!!")
        joint_angle = {**near_side_joint_angle, **far_side_joint_angle}
        joint_angle = far_side_angle_filter(joint_angle)
    else:
        pass 
    
    return raw_joint_angle, joint_angle, Smoothed_LR_Zpos
    

def get_step_length(world_landmarks, frame_width, frame_height):

    distance_x = frame_width*(world_landmarks[mp_pose.PoseLandmark.LEFT_HEEL].x - world_landmarks[mp_pose.PoseLandmark.RIGHT_HEEL].x)
    distance_y = frame_height*(world_landmarks[mp_pose.PoseLandmark.LEFT_HEEL].y - world_landmarks[mp_pose.PoseLandmark.RIGHT_HEEL].y)
    distance_z = frame_width*(world_landmarks[mp_pose.PoseLandmark.LEFT_HEEL].z - world_landmarks[mp_pose.PoseLandmark.RIGHT_HEEL].z)

    foot_gap_x = frame_width*(world_landmarks[mp_pose.PoseLandmark.LEFT_HIP].x - world_landmarks[mp_pose.PoseLandmark.RIGHT_HIP].x)
    foot_gap_y = frame_height*(world_landmarks[mp_pose.PoseLandmark.LEFT_HIP].y - world_landmarks[mp_pose.PoseLandmark.RIGHT_HIP].y)
    foot_gap_z = frame_width*(world_landmarks[mp_pose.PoseLandmark.LEFT_HIP].z - world_landmarks[mp_pose.PoseLandmark.RIGHT_HIP].z)

    distance = np.array([distance_x, distance_y, distance_z])
    foot_gap = np.array([foot_gap_x, foot_gap_y, foot_gap_z])

    diff = np.linalg.norm(distance)**2 - np.linalg.norm(foot_gap)**2
    step_length = (diff)**0.5 if diff > 0 else 0

    # print(distance_x)
    # print(distance_y)
    # print(distance_z)
    # print(step_length)

    return step_length


def MainWork(pose_tracker,
            pose_embedder,
            pose_classifier,
            pose_classification_filter,
            repetition_counter,
            joint_angle_predictor,
            far_side_angle_filter,
            near_side_angle_filter,
            visualizer,
            speed_calculator,
            show_image,
            out_video,
            show_per_frame,
            seg_mask,
            data_recorder,
            Smoothing):
    
    frame_idx = 0
    output_frame = None
    joint_angle_history = []
    raw_joint_angle_history = []
    stride_speed_history = []
    lost_angle_history = [] # body dectection failed
    step_length_rl = []
    step_length_lr = []
    left_right_step_length = 0
    right_left_step_length = 0

    with tqdm.tqdm(total=video_n_frames, position=0, leave=True) as pbar:
        while True:
            # Get next frame of the video.
            success, input_frame = video_cap.read()
            if not success:
                break
            '''
            image_hight, image_width, _ = input_frame.shape
            if image_hight < 720:
                padding_size_up = (720 - image_hight)//2
                padding_size_down = 720 -  padding_size_up - image_hight
                input_frame = cv2.copyMakeBorder(input_frame,
                                                                     padding_size_up,
                                                                     padding_size_down,
                                                                     0, 0,
                                                                     cv2.BORDER_CONSTANT,
                                                                     value=(0,0,0))
                
            if image_width < 1280:
                padding_size_left = (1280 - image_width)//2
                padding_size_right = 1280 -  padding_size_up - image_width
                input_frame = cv2.copyMakeBorder(input_frame,
                                                                     0, 0,
                                                                     padding_size_left,
                                                                     padding_size_right,
                                                                     cv2.BORDER_CONSTANT,
                                                                     value=(0,0,0))
            '''
            #Draw segmantation mask
            if seg_mask is True:
                
                input_frame = cv2.cvtColor(input_frame, cv2.COLOR_BGR2RGB)
                result = pose_tracker.process(image=input_frame)
                
                condition = False if result.segmentation_mask is None else np.stack((result.segmentation_mask,) * 3, axis=-1) > 0.1
                bg_image = np.zeros(input_frame.shape, dtype=np.uint8)
                bg_image[:] = BG_COLOR
                
                input_frame = np.where(condition, input_frame, bg_image)
                input_frame = cv2.cvtColor(input_frame, cv2.COLOR_RGB2BGR)
                
            # Run pose tracker.
            input_frame = cv2.cvtColor(input_frame, cv2.COLOR_BGR2RGB)
            result = pose_tracker.process(image=input_frame)
            pose_landmarks = result.pose_landmarks
            pose_world_landmarks = result.pose_world_landmarks
            joint_angle = dict()
            raw_joint_angle = dict() # raw joint angle

            # Draw pose prediction.
            output_frame = input_frame.copy()
            if pose_landmarks is not None:
                
                mp_drawing.draw_landmarks(
                    image=output_frame,
                    landmark_list=pose_landmarks,
                    connections=mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

            if pose_landmarks is not None:

                frame_height, frame_width = output_frame.shape[0], output_frame.shape[1]
                # Get joint angle
                raw_joint_angle, joint_angle, Smoothed_LR_Zpos = get_joint_angle(joint_angle_predictor, 
                                                                                 far_side_angle_filter,
                                                                                 near_side_angle_filter,
                                                                                 pose_landmarks.landmark, 
                                                                                 pose_world_landmarks.landmark, 
                                                                                 frame_height, 
                                                                                 frame_width, 
                                                                                 Smoothing,
                                                                                 frame_idx)
                # last_raw_joint_angle = raw_joint_angle                     
                # joint_angle = angle_filter(joint_angle)

                # Get landmarks.
                pose_landmarks_array = np.array([[lmk.x * frame_width, lmk.y * frame_height, lmk.z * frame_width]
                                            for lmk in pose_landmarks.landmark], dtype=np.float32)
                assert pose_landmarks_array.shape == (33, 3), 'Unexpected landmarks shape: {}'.format(pose_landmarks_array.shape)
                # Classify the pose on the current frame.
                pose_classification = pose_classifier(pose_landmarks_array)

                # Smooth classification using EMA.
                pose_classification_filtered = pose_classification_filter(pose_classification)

                # Count repetitions.
                repetitions_count = repetition_counter(pose_classification_filtered)

                # print(pose_classification)
                # print(pose_classification_filtered)

                # Calculate step length
                # near side = left side
                # print(get_step_length(pose_world_landmarks.landmark, frame_width, frame_height))
                # print(pose_world_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HEEL].x - pose_world_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HEEL].x)
                if Smoothed_LR_Zpos['right'] > Smoothed_LR_Zpos['left']:
                    # Left foot front
                    if pose_world_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HEEL].x > pose_world_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HEEL].x:
                        # print("LEFT > RIGH")
                        right_left_step_length = max(right_left_step_length, get_step_length(pose_world_landmarks.landmark, frame_width, frame_height))
                        if left_right_step_length > 0:
                            step_length_lr.append(left_right_step_length)
                            left_right_step_length = -1
                    # Right foot front
                    else:
                        # print("LEFT < RIGHT")
                        left_right_step_length = max(left_right_step_length, get_step_length(pose_world_landmarks.landmark, frame_width, frame_height))
                        if right_left_step_length > 0:
                            step_length_rl.append(right_left_step_length)
                            right_left_step_length = -1
                else:
                    # Right foot front
                    if pose_world_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HEEL].x > pose_world_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HEEL].x:
                        # print("LEFT > RIGH")
                        left_right_step_length = max(left_right_step_length, get_step_length(pose_world_landmarks.landmark, frame_width, frame_height))
                        if right_left_step_length > 0:
                            step_length_rl.append(right_left_step_length)
                            right_left_step_length = -1
                    # Left foot front
                    else:
                        # print("LEFT < RIGHT")
                        right_left_step_length = max(right_left_step_length, get_step_length(pose_world_landmarks.landmark, frame_width, frame_height))
                        if left_right_step_length > 0:
                            step_length_lr.append(left_right_step_length)
                            left_right_step_length = -1


            else:
                # No pose => no classification on current frame.
                pose_classification = None

                # If body detection fail, the kalman filter will use prediction angle to replace masuerment angle
                # joint_angle = far_side_angle_filter(dict())

                raw_joint_angle = {mp_pose.PoseLandmark.LEFT_KNEE: np.nan,
                                   mp_pose.PoseLandmark.LEFT_HIP: np.nan,
                                   mp_pose.PoseLandmark.RIGHT_KNEE: np.nan,
                                   mp_pose.PoseLandmark.RIGHT_HIP: np.nan,
                                   mp_pose.PoseLandmark.LEFT_ANKLE: np.nan,
                                   mp_pose.PoseLandmark.RIGHT_ANKLE: np.nan}

                if Test == "cos":
                    near_side_joint_angle = dict()
                    far_side_joint_angle = dict()
                    near_side_joint_angle = near_side_angle_filter(near_side_joint_angle, frame_idx)
                    far_side_joint_angle = far_side_angle_filter(far_side_joint_angle, frame_idx)
                    joint_angle = {**near_side_joint_angle, **far_side_joint_angle}

                elif Test == "cos_one":
                    # print("Yes!!")
                    joint_angle = far_side_angle_filter(dict())
                else:
                    pass 

                # raw_joint_angle = {mp_pose.PoseLandmark.LEFT_KNEE: None,
                #                    mp_pose.PoseLandmark.LEFT_HIP: None,
                #                    mp_pose.PoseLandmark.RIGHT_KNEE: None,
                #                    mp_pose.PoseLandmark.RIGHT_HIP: None,
                #                    mp_pose.PoseLandmark.LEFT_ANKLE: None,
                #                    mp_pose.PoseLandmark.RIGHT_ANKLE: None}
                
                # Still add empty classification to the filter to maintaing correct
                # smoothing for future frames.
                pose_classification_filtered = pose_classification_filter(dict())
                pose_classification_filtered = None

                # Don't update the counter presuming that person is 'frozen'. Just
                # take the latest repetitions count.
                repetitions_count = repetition_counter.n_repeats
                lost_angle_history.append(frame_idx)

            raw_joint_angle_history.append(raw_joint_angle)
            joint_angle_history.append(joint_angle)
            
            # Calculate stride speed
            stride_speed = speed_calculator(repetitions_count, frame_idx)
            
            if repetitions_count > 0:
                stride_speed_history.append(stride_speed)

            # Draw classification plot and repetition counter.
            if pose_landmarks is not None:
                pose_landmarks = result.pose_landmarks.landmark
                
            output_frame = visualizer(
                frame=output_frame,
                pose_classification=pose_classification,
                pose_classification_filtered=pose_classification_filtered,
                repetitions_count=repetitions_count,
                stride_speed = stride_speed,
                landmarks = pose_landmarks,
                angles = joint_angle)

            # Save the output frame.
            out_video.write(cv2.cvtColor(np.array(output_frame), cv2.COLOR_RGB2BGR))

            # Show intermediate frames of the video to track progress.
            # if show_per_frame != 0 and frame_idx % show_per_frame==0:
            #     show_image(output_frame)

            frame_idx += 1
            pbar.update()

    data_recorder(raw_joint_angle_history, joint_angle_history, lost_angle_history, stride_speed_history, step_length_rl, step_length_lr)
    '''
    np.savetxt('va_left_knee.txt',org_left_knee)
    np.savetxt('va_right_knee.txt',org_right_knee)
    '''
    
if __name__ == "__main__":

    #init argparse for reading command line
    parser = argparse.ArgumentParser()
    CommandLineArgs = InitParser(parser)
    
    input_path = CommandLineArgs.ipath
    output_path = CommandLineArgs.opath
    file_name = CommandLineArgs.filename
    pose_samples_folder = CommandLineArgs.cpath
    show_per_frame = CommandLineArgs.show
    data_record_folder = CommandLineArgs.sdata
    seg_mask = CommandLineArgs.mask
    pnoise_n = CommandLineArgs.pnoise_n
    mnoise_n = CommandLineArgs.mnoise_n
    pnoise_f = CommandLineArgs.pnoise_f
    mnoise_f = CommandLineArgs.mnoise_f

    input_path = os.path.join(input_path, file_name)

    #check input parameter from command line
    check_csv(pose_samples_folder)
    check_input(input_path)
    
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    output_path = os.path.join(output_path, file_name.split('.')[0]+'.mp4')

    if not os.path.exists(data_record_folder):
        os.makedirs(data_record_folder)

    readme_file = os.path.join(data_record_folder, "readme.txt")
    np.savetxt(readme_file, np.array([pnoise_n, mnoise_n, pnoise_f, mnoise_f]))

    data_record_folder = os.path.join(data_record_folder,file_name)
    if not os.path.exists(data_record_folder):
        os.makedirs(data_record_folder)

    # Open the video.
    video_cap = cv2.VideoCapture(input_path)

    # Get some video parameters to generate output video with classificaiton.
    video_n_frames = video_cap.get(cv2.CAP_PROP_FRAME_COUNT)
    video_fps = video_cap.get(cv2.CAP_PROP_FPS)
    video_width = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    #init all component for processing video input
    
    # Initialize tracker.  min_tracking_confidence=0.5, detection_confidence = 0.3
    pose_tracker = mp_pose.Pose(
        enable_segmentation=seg_mask,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.3,
        model_complexity=2)

    # Initialize embedder.
    pose_embedder = FullBodyPoseEmbedder()

    # Initialize classifier.
    # Ceck that you are using the same parameters as during bootstrapping.
    pose_classifier = PoseClassifier(
        pose_samples_folder=pose_samples_folder,
        pose_embedder=pose_embedder,
        top_n_by_max_distance=30,
        top_n_by_mean_distance=10)

    # Initialize EMA smoothing & speed prediction smoothing.
    pose_classification_filter = EMADictSmoothing(
        window_size=10,
        alpha=0.2)

    # Initialize counter.
    repetition_counter = RepetitionCounter(
        class_name=class_name,
        enter_threshold=6,
        exit_threshold=4)
    
    # Initialize data recorder
    data_recorder = DataRecorder(data_record_folder, video_fps, video_n_frames)
    
    # Initialize JointAnglePredictor & AnglePlotter.
    joint_angle_predictor = JointAngle()

    #Initialize Kalman filter
    far_side_angle_filter = KalmanFilter(video_fps, pnoise = pnoise_f, mnoise = mnoise_f)
    near_side_angle_filter = KalmanFilter(video_fps, pnoise = pnoise_n, mnoise = mnoise_n)
    # Initialize renderer.
    visualizer = Visualizer(
        class_name=class_name,
        plot_x_max=video_n_frames,
        # Graphic looks nicer if it's the same as `top_n_by_mean_distance`.
        plot_y_max=10)

    #Initialize StrideSpeed
    speed_calculator = StrideSpeed(
        video_fps,
        count_windows = 1)

    #Initialize ShowImage
    show_image = ShowImage(figsize=(20,20))

    # Open output video.
    #video_width = video_width if video_width >1280 else 1280
    #video_height = video_height if video_height > 720 else 720
    # out_video = None
    out_video = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), video_fps, (video_width, video_height))
    # out_video = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), video_fps, (1280, 720))
    
    Smoothing = EMADictSmoothing()

    # process each frames
    MainWork(pose_tracker,
                     pose_embedder,
                     pose_classifier,
                     pose_classification_filter,
                     repetition_counter,
                     joint_angle_predictor,
                     far_side_angle_filter,
                     near_side_angle_filter,
                     visualizer,
                     speed_calculator,
                     show_image,
                     out_video,
                     show_per_frame,
                     seg_mask,
                     data_recorder,
                     Smoothing)
    
    # Close output video.
    # out_video.release()

    # Release MediaPipe resources.
    pose_tracker.close()
