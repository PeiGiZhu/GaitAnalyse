import numpy as np
import os

class DataRecorder(object):

    def __init__(self, save_path, fps, frames):
        
        self._save_path = save_path
        self._fps = int(fps)
        self._frames = frames

    def __call__(self, raw_joint_angle_history, joint_angle_history, lost_angle_frame, stride_speed_history, step_length_lr, step_length_rl):

        left_knee=[]
        right_knee=[]
        left_hip=[]
        right_hip=[]
        left_ankle=[]
        right_ankle=[]

        raw_left_knee = []
        raw_right_knee = []
        raw_left_hip = []
        raw_right_hip = []

        for angles in raw_joint_angle_history:
            
            for center, angle in angles.items():
                if "LEFT_HIP" in str(center):
                    raw_left_hip.append(angle)
                if "RIGHT_HIP" in str(center):
                    raw_right_hip.append(angle)
                if "LEFT_KNEE" in str(center):
                    raw_left_knee.append(angle)
                if "RIGHT_KNEE" in str(center):
                    raw_right_knee.append(angle)
        
        for angles in joint_angle_history:

            for center, angle in angles.items():
                if "LEFT_HIP" in str(center):
                    left_hip.append(angle)
                if "RIGHT_HIP" in str(center):
                    right_hip.append(angle)
                if "LEFT_KNEE" in str(center):
                    left_knee.append(angle)
                if "RIGHT_KNEE" in str(center):
                    right_knee.append(angle)
                if "LEFT_ANKLE" in str(center):
                    left_ankle.append(angle)
                if "RIGHT_ANKLE" in str(center):
                    right_ankle.append(angle)
                
        left_knee = np.array(left_knee)
        right_knee = np.array(right_knee)
        left_hip = np.array(left_hip)
        right_hip = np.array(right_hip)
        left_ankle = np.array(left_ankle)
        right_ankle = np.array(right_ankle)
        lost_angle_frame = np.array(lost_angle_frame)
        
        raw_left_knee = np.array(raw_left_knee)
        raw_right_knee = np.array(raw_right_knee)
        raw_left_hip = np.array(raw_left_hip)
        raw_right_hip = np.array(raw_right_hip)
        
        print("saving joint angle history as txt...")
        
        with open(os.path.join(self._save_path,"fps.txt"), 'w') as file:
            file.write(str(self._fps))
        
        with open(os.path.join(self._save_path,"frames.txt"), 'w') as file:
            file.write(str(self._frames))
            
        
        np.savetxt(os.path.join(self._save_path,"raw_left_knee.txt"), raw_left_knee)
        np.savetxt(os.path.join(self._save_path,"raw_right_knee.txt"), raw_right_knee)
        np.savetxt(os.path.join(self._save_path,"raw_left_hip.txt"), raw_left_hip)
        np.savetxt(os.path.join(self._save_path,"raw_right_hip.txt"), raw_right_hip)
        
        # with open(os.path.join(self._save_path,"raw_left_knee.txt"), 'w') as file:
        #     file.writelines(str(raw_left_knee))

        # with open(os.path.join(self._save_path,"raw_right_knee.txt"), 'w') as file:
        #     file.writelines(str(raw_right_knee))

        # with open(os.path.join(self._save_path,"raw_left_hip.txt"), 'w') as file:
        #     file.writelines(str(raw_left_hip))

        # with open(os.path.join(self._save_path,"raw_right_hip.txt"), 'w') as file:
        #     file.writelines(str(raw_right_hip))

        # print(step_length_lr, step_length_rl)
        if step_length_lr == None:
            pass
        else:
            np.savetxt(os.path.join(self._save_path,"step_length_lr.txt"), step_length_lr)
        
        if step_length_rl == None:
            pass
        else:
            np.savetxt(os.path.join(self._save_path,"step_length_rl.txt"), step_length_rl)
        
        if stride_speed_history == None:
            pass
        else:
            np.savetxt(os.path.join(self._save_path,"stride_speed.txt"), stride_speed_history)

        np.savetxt(os.path.join(self._save_path,"left_knee.txt"), left_knee)
        np.savetxt(os.path.join(self._save_path,"right_knee.txt"), right_knee)
        np.savetxt(os.path.join(self._save_path,"left_hip.txt"), left_hip)
        np.savetxt(os.path.join(self._save_path,"right_hip.txt"), right_hip)
        np.savetxt(os.path.join(self._save_path,"left_ankle.txt"), left_ankle)
        np.savetxt(os.path.join(self._save_path,"right_ankle.txt"), right_ankle)
        np.savetxt(os.path.join(self._save_path,"lost_angle_frame.txt"), lost_angle_frame)
        
        
        print("Done!")

