from filterpy.kalman import UnscentedKalmanFilter, MerweScaledSigmaPoints
from filterpy.common import Q_continuous_white_noise
from filterpy.common import Q_discrete_white_noise
import numpy as np

class KalmanFilter(object):

    def _fx(self, x, dt):
        
        # bias = 6
        for i in range(0, 48, 8):
            # x[i] = x[i + bias] + np.cos(x[i + 2*bias]*dt + x[i + 3*bias]) + np.sin(x[i + 4*bias]*dt + x[i + 5*bias])
            x[i] = x[i + 1] + x[i + 2] * np.sin(x[i + 3]*dt + x[i + 4]) + x[i + 5] * np.sin (x[i + 6]*dt + x[i + 7])

        # for i in range(5, 36):
        #     x[i] = x[i]

        return x

    def _hx(self, x):
        
        # return x[0:6]
        return [x[0], x[8], x[16], x[24], x[32], x[40]]

    def __init__(self, fps, pnoise = 1e-3, mnoise = 1e-3, dynamParams = 48, measureParams = 6, controlParams = 0):
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
        # self._deltaT2 = 1/(fps*fps)
        # self._KF = KF(dynamParams, measureParams, controlParams)
        self._process_noise_coefficient = pnoise
        self._measurement_noise_coefficient = mnoise
        self._points = MerweScaledSigmaPoints(dynamParams, alpha = .1, beta = 2., kappa = 3-dynamParams)
        self._UKF = UnscentedKalmanFilter(dim_x = dynamParams, dim_z = measureParams, dt = self._deltaT, fx = self._fx, hx = self._hx, points = self._points)

        # the state is [angle1, angle2, angle3, angle4, speed1, speed2, speed3, speed4]
        # self._KF.statePost = np.float32( np.random.normal(loc = 0.0, scale = 0.1, size = (8, 1) ) ).reshape(8,1)
        # self._UKF.x = np.float32( np.random.normal(loc = 0.0, scale = 0.1, size = (dynamParams, 1) ) ).reshape(dynamParams,)
        self._UKF.x = np.zeros(dynamParams)
        # this is matrix H in kalman formulas 

        # self._measurementMatrix = np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        #                                        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        #                                        [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        #                                        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        #                                        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        #                                        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],], np.float32)
        
        # self._measurement = np.float32( np.random.normal(loc = 0.0, scale = 0.1, size = (measureParams, 1) ) ).reshape(measureParams,)
        self._measurement = np.zeros(measureParams)

        # this is matrix A or F in kalman formulas
        # the state updata as following:
        # angle_t = angle_t-1 + angular speed_t - 1
        # angular speed_t = angular speed_t - 1
        
        # self._transitionMatrix =np.array([[1, 0, 0, 0, 0, 0, self._deltaT, 0, 0, 0, 0, 0, self._deltaT2/2, 0, 0, 0, 0, 0],
        #                                      [0, 1, 0, 0, 0, 0, 0, self._deltaT, 0, 0, 0, 0, 0, self._deltaT2/2, 0, 0, 0, 0],
        #                                      [0, 0, 1, 0, 0, 0, 0, 0, self._deltaT, 0, 0, 0, 0, 0, self._deltaT2/2, 0, 0, 0],
        #                                      [0, 0, 0, 1, 0, 0, 0, 0, 0, self._deltaT, 0, 0, 0, 0, 0, self._deltaT2/2, 0, 0],
        #                                      [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, self._deltaT, 0, 0, 0, 0, 0, self._deltaT2/2, 0],
        #                                      [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, self._deltaT, 0, 0, 0, 0, 0, self._deltaT2/2],
        #                                      [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, self._deltaT, 0, 0, 0, 0, 0],
        #                                      [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, self._deltaT, 0, 0, 0, 0],
        #                                      [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, self._deltaT, 0, 0, 0],
        #                                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, self._deltaT, 0, 0],
        #                                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, self._deltaT, 0],
        #                                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, self._deltaT],
        #                                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        #                                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        #                                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
        #                                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
        #                                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
        #                                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], ], np.float32)
        
        # self._KF.processNoiseCov = self._process_noise_coefficient * np.eye(dynamParams, dtype = np.float32) #1e-5 5e-4
        
        # self._KF.measurementNoiseCov = self._measurement_noise_coefficient * np.eye(measureParams, dtype = np.float32) #1e-3
        
        # self._KF.errorCovPost = 100 * np.eye(dynamParams, dtype = np.float32)

        self._UKF.Q = self._process_noise_coefficient * np.eye(dynamParams, dtype = np.float32)
        # self._UKF.Q = Q_discrete_white_noise(dim = 6, dt = self._deltaT, var = 1, block_size=6)

        self._UKF.R = self._measurement_noise_coefficient * np.eye(measureParams, dtype = np.float32)

        self._UKF.P = 100

        self._correct_angle = dict()

    def __call__(self, measurement):

        self._mesurement = [angle for _, angle in measurement.items()]

        # x = self._KF.predict()
        #If the joint_angle(measurement) dict is empty then replace it by prediction from previous frame
        if len(measurement) == 0:
            # self._mesurement = self._UKF.x[0:6]
            self._mesurement = self._hx(self._UKF.x)
        #If some angle in joint_angle(measurement) dict is None then replace it by corresponding value from prediction 
        else:
            for index, value in enumerate(self._mesurement):
                # if value is None:
                if np.isnan(value):
                    self._mesurement[index] = self._UKF.x[8*index]
                    
        self._mesurement = np.array(self._mesurement, np.float32)
        
        self._UKF.predict()
        self._UKF.update(self._mesurement)

        Index = 0
        self._correct_angle = dict()
        for key, _ in measurement.items():
            self._correct_angle[key] = float(self._UKF.x[8*Index])
            Index = Index +1

        return self._correct_angle
        
        
        
        



