import numpy as np
from numpy import linalg

from mediapipe.python.solutions import pose as mp_pose
from mediapipe.python.solutions import drawing_utils as mp_drawing
from mediapipe.python.solutions import drawing_styles as mp_drawing_styles

class JointAngle(object):
    """Get joint Angle from 3D landmark"""

    def __init__(self):
        
        self._center = None
        self._pointA = None
        self._pointB_or_vector =None
        
        self._angle =None
        self._use_invisible_point = False
        self._world_landmarks = None
        self._view_side = None


    def _CheckVisibility(self, threshold):
        if self._world_landmarks[self._center].visibility >= threshold                                      \
        and self._world_landmarks[self._pointA].visibility >= threshold                                  \
        and ( self._use_invisible_point or self._world_landmarks[self._pointB].visibility >= threshold):
            return True
        else:
            return False
        
    def _GetAngle(self):
        
        center_x = self._world_landmarks[self._center].x
        center_y = self._world_landmarks[self._center].y
        center_z = self._world_landmarks[self._center].z

        pointA_x = self._world_landmarks[self._pointA].x
        pointA_y = self._world_landmarks[self._pointA].y
        pointA_z = self._world_landmarks[self._pointA].z

        center = np.array([center_x,center_y,center_z])
        pointA = np.array([pointA_x,pointA_y,pointA_z])
        center_pointA = pointA - center

        if self._use_invisible_point is False:
            
            pointB_x = self._world_landmarks[self._pointB].x
            pointB_y = self._world_landmarks[self._pointB].y
            pointB_z = self._world_landmarks[self._pointB].z

            pointB = np.array([pointB_x,pointB_y,pointB_z])
            center_pointB = pointB - center

            # angle = atan2(vector2.y, vector2.x) - atan2(vector1.y, vector1.x)
            cos_angle = center_pointA.dot(center_pointB.T)/(np.linalg.norm(center_pointA)*np.linalg.norm(center_pointB))
            self._angle = np.rad2deg(np.arccos(cos_angle))
            
        else:
            
            invisible_vector = self._get_invisible_vector()
            cos_angle = center_pointA.dot(invisible_vector.T)/(np.linalg.norm(center_pointA)*np.linalg.norm(invisible_vector))
            self._angle = np.rad2deg(np.arccos(cos_angle))
            
                

            
        
    def _get_invisible_vector(self):

        left_right_x = self._world_landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].x     \
                                    - self._world_landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x

        left_right_y = self._world_landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y     \
                                    - self._world_landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y

        left_right_z = self._world_landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].z     \
                                    - self._world_landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].z

        left_right = np.array([left_right_x, left_right_y, left_right_z])

        rotate_around_y_axis = np.array([[0, 0, -1],
                                         [0, 1, 0],
                                         [1, 0, 0]])

        rotated_left_right = np.dot(rotate_around_y_axis, left_right.T)

        #print(rotated_left_right)

        projection_on_xzplane = np.array([rotated_left_right[0], 0, rotated_left_right[2]] )

        # projection_on_xzplane = np.array([-1,0,0])

        return projection_on_xzplane
    

    
    def __call__(self, world_landmarks, center, pointA, pointB, use_invisible_point = False):

        self._world_landmarks = world_landmarks
        self._use_invisible_point = use_invisible_point
        self._center = center
        self._pointA = pointA
        self._pointB = None if use_invisible_point is True else pointB

        if self._CheckVisibility(threshold = 0.4) is True:
            self._GetAngle()
            return self._angle
        else:
            # return None
            return np.nan




        
