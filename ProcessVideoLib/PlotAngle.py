# discard this class py

import cv2

class PlotAngle(object):

    def __init__(self):
        
        self._bias_x = 0.02
        self._bias_y = 0.02

        self._angle = dict()
        self._image = None
        self._landmarks = None

        self._RightPartColor = (255,255,0)
        self._LeftPartColor = (100,150,255)
        
        
    def __call__(self, image, landmarks, angles):

        self._image = image
        self._angles = angles
        image_hight, image_width, _ = self._image.shape
        self._landmarks = landmarks

        for center, angle in self._angles.items():

            if angle is None:
                continue
            
            location_x = (self._landmarks[center].x - self._bias_x) * image_width
            location_y = (self._landmarks[center].y + self._bias_y) * image_hight

            if "RIGHT" in str(center):
                cv2.putText(self._image, '{:.2f}'.format(angle), (int(location_x), int(location_y)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, self._RightPartColor, 2)
            else:
                cv2.putText(self._image, '{:.2f}'.format(angle), (int(location_x), int(location_y)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, self._LeftPartColor, 2)

        return self._image
