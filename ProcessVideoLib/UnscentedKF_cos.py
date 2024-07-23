from filterpy.kalman import UnscentedKalmanFilter, MerweScaledSigmaPoints
from filterpy.common import Q_continuous_white_noise
from filterpy.common import Q_discrete_white_noise
import numpy as np

class KalmanFilter(object):

    def _fx_24(self, x, dt):
        
        # for i in range(0, 24, 8):
        #     x[i] = x[i + 1] + x[i + 2] * np.sin(x[i + 3]*dt + x[i + 4]) + x[i + 5] * np.sin(x[i + 6]*dt + x[i + 7])
        #     x[i] = x[i] + x[i + 1] * np.sin(x[i + 2]*dt + x[i + 3]) + x[i + 4] * np.sin(x[i + 5]*dt + x[i + 6])
            # print(x[i + 1], x[i + 2], x[i + 3], x[i + 4], x[i + 5], x[i + 6], x[i + 7])
        t = dt*self._frame
        for i in range(0, 24, 8):
            x[i] = x[i + 1] + x[i + 2] * np.sin(x[i + 3]*t + x[i + 4]) + x[i + 5] * np.sin(x[i + 6]*t + x[i + 7])
            # print(x[i + 1], x[i + 2], x[i + 3], x[i + 4], x[i + 5], x[i + 6], x[i + 7])

        return x

    def _fx_9(self, x, dt):

        tm = np.array([[1, self._deltaT, self._deltaT2, 0, 0, 0, 0, 0, 0],
                       [0, 1, self._deltaT, 0, 0, 0, 0, 0, 0],
                       [0, 0, 1, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 1, self._deltaT, self._deltaT2, 0, 0, 0],
                       [0, 0, 0, 0, 1, self._deltaT, 0, 0, 0],
                       [0, 0, 0, 0, 0, 1, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 1, self._deltaT, self._deltaT2],
                       [0, 0, 0, 0, 0, 0, 0, 1, self._deltaT],
                       [0, 0, 0, 0, 0, 0, 0, 0, 1], ], np.float32)

        return np.dot(tm, x)

    def _hx_24(self, x):
        return [x[0], x[8], x[16]]

    def _hx_9(self, x):
        return [x[0], x[3], x[6]]

    def _fx_test(self, x, dt):
        
        return x

    def _hx_test(self, x):

        T = self._deltaT*self._frame
        result = []
        for i in range(0, 21, 7):
            result.append(x[i] + x[i + 1] * np.sin(x[i + 2]*T + x[i + 3]) + x[i + 4] * np.sin(x[i + 5]*T + x[i + 6]))
        
        return result

    def __init__(self, fps, pnoise = 1e-3, mnoise = 1e-3, dynamParams = 9, measureParams = 3, controlParams = 0):
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
        self._frame = 0
        self._deltaT = 1/fps
        self._deltaT2 = 1/(2*(fps*fps))
        # self._KF = KF(dynamParams, measureParams, controlParams)
        self._dynamParams = dynamParams
        self._process_noise_coefficient = pnoise
        self._measurement_noise_coefficient = mnoise
        self._points = MerweScaledSigmaPoints(dynamParams, alpha = .1, beta = 2., kappa = 0)

        self._UKF = UnscentedKalmanFilter(dim_x = dynamParams, 
                                            dim_z = measureParams, 
                                            dt = self._deltaT, 
                                            fx = self._fx_9, 
                                            hx = self._hx_9, 
                                            points = self._points)

        # if dynamParams == 24:
        #     self._UKF = UnscentedKalmanFilter(dim_x = dynamParams, 
        #                                     dim_z = measureParams, 
        #                                     dt = self._deltaT, 
        #                                     fx = self._fx_24, 
        #                                     hx = self._hx_24, 
        #                                     points = self._points)

        # elif dynamParams == 9:
        #     self._UKF = UnscentedKalmanFilter(dim_x = dynamParams, 
        #                                     dim_z = measureParams, 
        #                                     dt = self._deltaT, 
        #                                     fx = self._fx_9, 
        #                                     hx = self._hx_9, 
        #                                     points = self._points)
        # else: pass

        # the state is [angle1, angle2, angle3, angle4, speed1, speed2, speed3, speed4]
        # self._KF.statePost = np.float32( np.random.normal(loc = 0.0, scale = 0.1, size = (8, 1) ) ).reshape(8,1)
        # self._UKF.x = np.float32( np.random.normal(loc = 0.0, scale = 0.1, size = (dynamParams, 1) ) ).reshape(dynamParams,)
        self._UKF.x = np.zeros(dynamParams)
        # self._UKF.x = np.array([100, 20, 6.28, 3.14, 20, 6.28, 3.14, 100, 20, 6.28, 3.14, 20, 6.28, 3.14, 100, 20, 6.28, 3.14, 20, 6.28, 3.14,])
        # this is matrix H in kalman formulas 

        # self._measurementMatrix = np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        #                                        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        #                                        [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        #                                        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        #                                        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        #                                        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],], np.float32)
        
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

        # self._UKF.Q = Q_discrete_white_noise(dim = 3, dt = self._deltaT, var = pnoise, block_size=3)
        # self._UKF.Q = Q_continuous_white_noise(dim = 3, dt = self._deltaT, spectral_density = pnoise, block_size=3)

        self._UKF.Q = self._process_noise_coefficient * np.eye(dynamParams, dtype = np.float32)

        self._UKF.R = self._measurement_noise_coefficient * np.eye(measureParams, dtype = np.float32)

        self._UKF.P = 100 * np.eye(dynamParams, dtype = np.float32)

        self._correct_angle = dict()

    def __call__(self, measurement, frame = 0):

        self._mesurement = [angle for _, angle in measurement.items()]
        self._frame = frame

        # x = self._KF.predict()
        #If the joint_angle(measurement) dict is empty then replace it by prediction from previous frame
        if len(measurement) == 0:
            # self._mesurement = self._hx_24(self._UKF.x) if self._dynamParams == 24 else self._hx_9(self._UKF.x)
            self._mesurement = self._hx_9(self._UKF.x)
        #If some angle in joint_angle(measurement) dict is None then replace it by corresponding value from prediction 
        else:
            for index, value in enumerate(self._mesurement):
                if np.isnan(value):
                # if value is None:
                    # self._mesurement[index] = self._hx_24(self._UKF.x)[index] if self._dynamParams == 24 else self._hx_9(self._UKF.x)[index]
                    self._mesurement[index] = self._hx_9(self._UKF.x)[index]
                    # self._mesurement[index] = self._UKF.x[8*index]
                    
        self._mesurement = np.array(self._mesurement, np.float32)
        
        # print("one P")
        # print(self._UKF.P)
        
        self._UKF.predict()
        self._UKF.update(self._mesurement)

        Index = 0
        self._correct_angle = dict()
        for key, _ in measurement.items():
            # self._correct_angle[key] = float(self._hx_24(self._UKF.x)[Index]) if self._dynamParams == 24 else float(self._hx_9(self._UKF.x)[Index])
            self._correct_angle[key] = float(self._hx_9(self._UKF.x)[Index])
            Index = Index +1

        return self._correct_angle
        
        
        
        



