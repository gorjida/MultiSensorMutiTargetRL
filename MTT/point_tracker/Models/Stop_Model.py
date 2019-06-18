import numpy as np
class Stop_Model:
    def __init__(self, sample_time, use_vel = True,velocity_noise=None,init_state=None,init_cov=None):
        self.T = sample_time
        self.Q = velocity_noise
        self.x_k_km1 = np.zeros([4, 1])
        self.p_k_km1 = np.ones([4, 4])
        if init_state is None:
            self.x_k_km1 = np.zeros([4,1])
        else:
            self.x_k_km1 = np.array(init_state).reshape(4,1)

        if init_cov is None:
            self.p_k_km1 = np.ones([4,4])
        else:
            self.p_k_km1 = init_cov

        if use_vel:
            self.predicted_meas = np.zeros([3,1])
            self.S_k = np.eye(3)
            self.S_k_R = np.eye(3)
        else:
            self.predicted_meas = np.zeros([2, 1])
            self.S_k = np.eye(2)
            self.S_k_R = np.eye(2)

    def get_linearized_matrices(self, estimate=None):
        A = np.array([1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1])
        #B = np.array([[self.T ** 2 / 2.0, 0], [0, self.T ** 2 / 2.0], [self.T, 0], [0, self.T]])
        return (A)

    def motion_prediction(self,estimate,estimate_cov):
        A = self.get_linearized_matrices()
        x_k_km1 = A.dot(estimate)
        p_k_km1 = estimate_cov
        return (x_k_km1,p_k_km1)