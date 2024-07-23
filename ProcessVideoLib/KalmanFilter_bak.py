from cv2 import KalmanFilter as KF
import numpy as np

class KalmanFilter(object):

    def __init__(self, fps, dynamParams = 12, measureParams = 4, controlParams = 0, pnoise = 1e-3, mnoise = 1e-3):
        '''
        Kalman filter is applied to correct measurement error by
        cobmining last predicted value and current measurement value
        accorrding to kalman gain.
        The fomular as below:
        
        z_t = H * x_t + v_t
        x_t|t-1 = A * xu_t-1 + w_t-1
        Pt|t-1 = At|t-1 * P_t-1 * A'_t|t-1 + Q_t-1
        K_t = Pt|t-1 * H' * invert(H * P_t|t-1 * H' + R_t)
        xu_t = x_t|t-1 + K_t * (z_t - H * x_t|t-1)

        where X' means the transpose matrix of X,
        and invert(X) means the invert matrix of X.
        
        H: measurement Matrix
        A: state-transition Matrix
        P: error estimate covariance matrix
        Q: process noise covariance matrix
        R: measurement noise covariance matrix
        K: Kalman gain
        
        x: system state Matrix
        z: measured state
        xu: system state after combining predicted value and current measurement value
        v: measurement noise v ~ N(0, R)
        w: process noise w ~ N(0, Q)
        
        '''
        self._deltaT = 1/fps
        self._deltaT2 = 1/(fps*fps)
        self._KF = KF(dynamParams, measureParams, controlParams)
        self._process_noise_coefficient = pnoise
        self._measurement_noise_coefficient = mnoise
        # the state is [angle1, angle2, angle3, angle4, speed1, speed2, speed3, speed4]
        # self._KF.statePost = np.float32( np.random.normal(loc = 0.0, scale = 0.1, size = (8, 1) ) ).reshape(8,1)
        # this is matrix H in kalman formulas 
        '''
        self._KF.measurementMatrix = np.array([[1, 0, 0, 0, 0, 0, 0, 0],
                                                                      [0, 1, 0, 0, 0, 0, 0, 0],
                                                                      [0, 0, 1, 0, 0, 0, 0, 0],
                                                                      [0, 0, 0, 1, 0, 0, 0, 0]], np.float32)
        '''
        self._KF.measurementMatrix = np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                               [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                               [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                               [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]], np.float32)
        
        self._measurement = None
        
        # this is matrix A or F in kalman formulas
        # the state updata as following:
        # angle_t = angle_t-1 + angular speed_t - 1
        # angular speed_t = angular speed_t - 1
        '''
        self._KF.transitionMatrix =np.array([[1, 0, 0, 0, 1, 0, 0, 0],
                                                              [0, 1, 0, 0, 0, 1, 0, 0],
                                                              [0, 0, 1, 0, 0, 0, 1, 0],
                                                              [0, 0, 0, 1, 0, 0, 0, 1],
                                                              [0, 0, 0, 0, 1, 0, 0, 0],
                                                              [0, 0, 0, 0, 0, 1, 0, 0],
                                                              [0, 0, 0, 0, 0, 0, 1, 0],
                                                              [0, 0, 0, 0, 0, 0, 0, 1]], np.float32)
        '''
        
        self._KF.transitionMatrix =np.array([[1, 0, 0, 0, self._deltaT, 0, 0, 0, self._deltaT2/2, 0, 0, 0],
                                             [0, 1, 0, 0, 0, self._deltaT, 0, 0, 0, self._deltaT2/2, 0, 0],
                                             [0, 0, 1, 0, 0, 0, self._deltaT, 0, 0, 0, self._deltaT2/2, 0],
                                             [0, 0, 0, 1, 0, 0, 0, self._deltaT, 0, 0, 0, self._deltaT2/2],
                                             [0, 0, 0, 0, 1, 0, 0, 0, self._deltaT, 0, 0, 0],
                                             [0, 0, 0, 0, 0, 1, 0, 0, 0, self._deltaT, 0, 0],
                                             [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, self._deltaT, 0],
                                             [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, self._deltaT],
                                             [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                                             [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                                             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                                             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], ], np.float32)
        
        self._KF.processNoiseCov = self._process_noise_coefficient * np.eye(dynamParams, dtype = np.float32) #1e-5 5e-4
        
        self._KF.measurementNoiseCov = self._measurement_noise_coefficient * np.eye(measureParams, dtype = np.float32) #1e-3
        
        self._KF.errorCovPost = 100 * np.eye(dynamParams, dtype = np.float32)

        self._correct_angle = dict()


    def __call__(self, measurement):

        self._mesurement = [angle for _, angle in measurement.items()]

        x = self._KF.predict()
        #If the joint_angle(measurement) dict is empty then replace it by prediction from previous frame
        if len(measurement) == 0:
            self._mesurement = x[0:4]
        #If some angle in joint_angle(measurement) dict is None then replace it by corresponding value from prediction 
        else:
            for index, value in enumerate(self._mesurement):
                if value is None:
                    self._mesurement[index] = x[index]
                    
        self._mesurement = np.array(self._mesurement, np.float32)
        xu = self._KF.correct(self._mesurement)

        Index = 0
        self._correct_angle = dict()
        for key, _ in measurement.items():
            self._correct_angle[key] = float(xu[Index])
            Index = Index +1

        return self._correct_angle
        
        
        
        



