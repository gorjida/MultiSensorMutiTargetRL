import numpy as np


class CV_Model:
    def __init__(self, sample_time, velocity_noise,use_vel = True,init_state = None,init_cov = None):
        self.T = sample_time
        self.Q = velocity_noise
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
        A = np.array([1,0,self.T,0],[0,1,0,self.T],[0,0,1,0],[0,0,0,1])
        B = np.array([[self.T ** 2 / 2.0, 0], [0, self.T ** 2 / 2.0], [self.T, 0], [0, self.T]])
        return (A, B)

    def motion_prediction(self,estimate,estimate_cov):
        A,B = self.get_linearized_matrices()
        predicted_noise_covariance = (B.dot(self.Q)).dot(B.transpose())
        x_k_km1 = A.dot(estimate)
        p_k_km1 = (A.dot(estimate_cov)).dot(A.transpose()) + predicted_noise_covariance
        self.x_k_km1 = x_k_km1
        self.p_k_km1 = p_k_km1
        return (x_k_km1,p_k_km1)
    """

    def constant_turn(self, heading_rate):
        A = np.array(
            [[1, 0, np.sin(heading_rate * self.T) / heading_rate, (np.cos(heading_rate * self.T) - 1) / heading_rate]
                ,
             [0, 1, (1 - np.cos(heading_rate * self.T)) / heading_rate, np.sin(heading_rate * self.T) / heading_rate],
             [0, 0, np.cos(heading_rate * self.T), -np.sin(heading_rate * self.T)],
             [0, 0, np.sin(heading_rate * self.T), np.cos(heading_rate * self.T)]])
        B = np.array([[self.T ** 2 / 2.0, 0], [0, self.T ** 2 / 2.0], [self.T, 0], [0, self.T]])

        # Linearization
        return (A, B)

    def constant_accelaration(self):
        A = np.array([[1, 0, self.T, 0, self.T ** 2 / 2.0, 0], [0, 1, 0, self.T, 0, self.T ** 2 / 2.0],
                      [0, 0, 1, 0, self.T, 0], [0, 0, 0, 1, 0, self.T], [0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 1]])
        B = np.array([[self.T ** 2 / 2.0, 0], [0, self.T ** 2 / 2.0], [self.T, 0], [0, self.T], [1, 0], [0, 1]])

        return (A, B)

    def binary_command(self, command):
        command1 = 0
        command2 = 0
        if command == 0:
            command1 = 1
        elif command == 1:
            command2 = 1
        else:
            command1 = 1
            command2 = 1

        A = np.array([[1, 0, self.T * (command1), 0], [0, 1, 0, self.T * (command2)], [0, 0, 1, 0], [0, 0, 0, 1]])
        B = np.array(
            [[self.T ** 2 / 2.0 * (command1), 0], [0, self.T ** 2 / 2.0 * (command2)], [self.T * (command1), 0],
             [0, self.T * (command2)]])

        return (A, B)
    # def constant_turn(self):
"""