import numpy as np
from numpy import linalg

class JointAngle(object):
    """Get joint Angle from 3D landmark"""

    def __init__(self):
        
        self._center = None
        self._pointA = None

        self._angle =None
        self._use_invisible_point = False
        self._invisible_left = None #LEFT_SHOULDER
        self._invisible_right = None #RIGHT_SHOULDER

        
    def _GetAngle(self):
        
        center_x = self._center.x
        center_y = self._center.y
        center_z = self._center.z

        pointA_x = self._pointA.x
        pointA_y = self._pointA.y
        pointA_z = self._pointA.z

        center = np.array([center_x,center_y,center_z])
        pointA = np.array([pointA_x,pointA_y,pointA_z])
        center_pointA = pointA - center

        if self._use_invisible_point is False:
            
            pointB_x = self._pointB.x
            pointB_y = self._pointB.y
            pointB_z = self._pointB.z

            pointB = np.array([pointB_x,pointB_y,pointB_z])
            center_pointB = pointB - center

            cos_angle = center_pointA.dot(center_pointB.T)/(np.linalg.norm(center_pointA)*np.linalg.norm(center_pointB))
            self._angle = np.rad2deg(np.arccos(cos_angle))
            
        else:
            
            invisible_vector = self._get_invisible_vector()
            cos_angle = center_pointA.dot(invisible_vector.T)/(np.linalg.norm(center_pointA)*np.linalg.norm(invisible_vector))
            self._angle = np.rad2deg(np.arccos(cos_angle))
            
                

            
    #Look here: https://www.cnblogs.com/meteoric_cry/p/7987548.html
    def _get_invisible_vector(self):

        left_right_x = self._invisible_right.x     \
                                    - self._invisible_left.x

        left_right_y = self._invisible_right.y     \
                                    - self._invisible_left.y

        left_right_z = self._invisible_right.z     \
                                    - self._invisible_left.z

        left_right = np.array([left_right_x, left_right_y, left_right_z])
        '''
        rotate_around_y_axis = np.array([[0, 0, -1],
                                                           [0, 1, 0],
                                                           [1, 0, 0]])
        '''
        rotate_around_z_axis = np.array([[0, -1, 0],
                                                           [1, 0, 0],
                                                           [0, 0, 1]])
        
        rotated_left_right = np.dot(rotate_around_z_axis, left_right.T)

        #print(rotated_left_right)

        projection_on_xyplane = np.array([rotated_left_right[0], rotated_left_right[1], 0] )

        return projection_on_xyplane
    

    
    def __call__(self, center, pointA, pointB, use_invisible_point = False, invisible_left = None, invisible_right = None ):

        self._use_invisible_point = use_invisible_point
        self._center = center
        self._pointA = pointA
        self._pointB = None if use_invisible_point is True else pointB
        self._invisible_left = invisible_left
        self._invisible_right = invisible_right
        
        self._GetAngle()
        return self._angle





        
