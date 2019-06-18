
import numpy as np
from scipy.stats import norm


class IMM_Tracker:
    def __init__(self,init_estimate,init_covariance,motion_models,scenario,track_id,mode_transition= None,use_velicty=True):

        self.init_estimate = init_estimate
        self.init_covariance = init_covariance
        self.bearing_var = (scenario.bearing_std)**2
        self.range_var = (scenario.range_std)**2
        self.vel_var = (scenario.vel_std)**2
        if mode_transition is None:
            mode_transition = ((1-.95)/(len(motion_models)-1))*np.ones([len(motion_models),len(motion_models)])
            for mode in range(0,len(motion_models)): mode_transition[mode,mode] = .95
        self.mode_transition = mode_transition
        self.mode_probability = (1/len(motion_models))*np.ones([len(motion_models),1])
        self.motion_models = motion_models

        self.x_k_k = np.array(init_estimate).reshape(len(init_estimate),1)
        self.x_k_km1 = self.x_k_k
        self.p_k_k = init_covariance
        self.p_k_km1 = init_covariance
        self.S_k = 1E-5
        self.meas_vec = []
        self.meas_matrix = []
        self.innovation_list = []
        self.innovation_var = []
        self.gain = []
        self.uncertainty = []
        self.assignment = []
        self.assignment_uncertainty = []
        self.status = 0
        self.use_vel = use_velicty
        self.track_id = track_id

    def state_interaction(self):
        #total probability for transition from each single-mode to other modes
        total_mode_transition \
            = (self.mode_probability.transpose().dot(self.mode_transition)).transpose()
        (M,N) = np.shape(self.mode_transition)

        conditional_mode_transition = np.zeros([M,N])
        interact_mean = []
        interact_cov = []

        for j in range(0,N):
            tmp_state = np.array(self.motion_models[j].x_k_km1)
            tmp_cov = np.array(self.motion_models[j].p_k_km1)

            cond_mod = np.zeros([M,1])
            for i in range(0, M):
                #Probability of transition to "j" from i
                conditional_mode_transition \
                    = self.mode_transition[i,j]*self.mode_probability[i]/total_mode_transition[j]
                cond_mod[i] = conditional_mode_transition

                tmp_state+= conditional_mode_transition*self.motion_models[i].x_k_km1

            #Now calculate covariance as well
            for i in range(0,M):
                tmp_cov += cond_mod[i]\
                           *(self.motion_models[i].p_k_km1
                             +(self.motion_models[i].x_k_km1-tmp_state)
                             .dot((self.motion_models[i].x_k_km1-tmp_state).transpose()))


            interact_mean.append(tmp_state)
            interact_cov.append(tmp_cov)

        #Update all mean and covariances
        for j in range(0, N):
            self.motion_models[j].x_k_km1 = interact_mean[j]
            self.motion_models[j].p_k_km1 = interact_cov[j]


    def get_linearized_measurment_matrix(self,target_state,sensor_state):
        """
        Generate the Jacobian matrix (for range/azimuth/doppler measurements)
        :param target_state:
        :param sensor_state:
        :return:
        """
        relative_location = np.array(target_state[0:2]).reshape(2,1) - \
                            np.array(sensor_state[0:2]).reshape(2,1)  ##[x-x_s,y-y_s]

        relative_velocity = np.array(target_state[2:]).reshape(2,1) - \
                            np.array(sensor_state[2:]).reshape(2,1)
        range = np.linalg.norm(relative_location)
        range_dot = (np.sum(relative_velocity*relative_location))/(range)

        measurement_vector_range = np.array([relative_location[0]/range,
                                            relative_location[1]/range, [0], [0]])
        #Jacobian with-respect-to the azimuth
        #azimuth = np.arctan2(relative_location[1] ,relative_location[0])
        azimuth = np.arctan2(relative_location[0], relative_location[1])
        #if azimuth < 0: azimuth+= 2 * np.pi
        #measurement_vector_azimuth = np.array([-relative_location[1] / ((np.linalg.norm(relative_location)) ** 2),
                                       #relative_location[0] / ((np.linalg.norm(relative_location)) ** 2), [0], [0]])
        measurement_vector_azimuth = np.array([relative_location[1] / ((np.linalg.norm(relative_location)) ** 2),
                                               -relative_location[0] / ((np.linalg.norm(relative_location)) ** 2), [0],
                                               [0]])
        #Jacobian with-respect-to the velocity
        with_x = (relative_velocity[0]*range-range_dot*relative_location[0])/(range**2)
        with_y = (relative_velocity[1] * range - range_dot * relative_location[1]) / (range ** 2)
        with_xdot = relative_location[0]/range
        with_ydot = relative_location[1]/range
        measurement_vector_vel = np.array([with_x,with_y,with_xdot,with_ydot])

        measurement_vector_range = measurement_vector_range.transpose()
        measurement_vector_azimuth = measurement_vector_azimuth.transpose()
        measurement_vector_vel = measurement_vector_vel.transpose()

        vec1 = np.array([1,0,0,0]).reshape(1,4)
        vec2 = np.array([0,1,0,0]).reshape(1,4)


        if self.use_vel:
            measurement_matrix = np.concatenate([measurement_vector_range,measurement_vector_azimuth
                                                    ,measurement_vector_vel],axis=0)
            #CHANGED BY ALI
            #measurement_matrix = np.concatenate([vec1,vec2,measurement_vector_vel],0)
        else:
            measurement_matrix = np.concatenate([measurement_vector_range, measurement_vector_azimuth], axis=0)

        return (measurement_matrix,range,azimuth,range_dot)
        #CHANGED
        #return (measurement_matrix,target_state[0][0],target_state[1][0],range_dot)

    def linearized_predicted_measurement(self,sensor_state):
        sensor_state = np.array(sensor_state).reshape(len(sensor_state),1)
        measurement_vector = self.get_linearized_measurment_vector(self.x_k_km1,sensor_state)#Linearize the measurement model
        #predicted_measurement = measurement_vector.dot(np.array(self.x_k_km1))
        predicted_measurement =  np.arctan2(self.x_k_km1[1]-sensor_state[1],self.x_k_km1[0]-sensor_state[0])

        if predicted_measurement<0:predicted_measurement+= 2*np.pi
        return (predicted_measurement,measurement_vector)

    def linearized_predicted_measurement_vector(self,predicted_state,sensor_state):
        """
        Generates the Jacobian + predicted measurement-vector
        :param sensor_state:
        :return:
        """
        sensor_state = np.array(sensor_state).reshape(len(sensor_state),1)
        measurement_matrix,range,azimuth,range_dot = \
            self.get_linearized_measurment_matrix(predicted_state,sensor_state)#Linearize the measurement model

        if self.use_vel:
        #predicted_measurement = measurement_vector.dot(np.array(self.x_k_km1))
            predicted_measurement = np.array([range,azimuth,range_dot]).reshape(3,1)
        else:
            predicted_measurement = np.array([range, azimuth]).reshape(2, 1)

        return (predicted_measurement,measurement_matrix)

    def do_prediction(self,sensor_state):

        # TASK1: go through all the modes and mix probabilities (get x(k+1|k) amd P(k+1|k)) for each mode
        self.state_interaction()
        #TASK2: for each model, estimate
        for model in self.motion_models:
            #Predicted measurement based on each mode
            predicted_measurement, measurement_matrix \
                = self.linearized_predicted_measurement_vector(model.x_k_km1,sensor_state)
            if self.use_vel:
                S_k = (measurement_matrix.dot(model.p_k_km1)).dot(measurement_matrix.transpose())
                S_k_R = S_k + np.diag([self.range_var, self.bearing_var, self.vel_var])

            else:
                S_k = (measurement_matrix.dot(model.p_k_km1)).dot(measurement_matrix.transpose())
                S_k_R = self.S_k + np.diag([self.range_var, self.bearing_var])
            model.S_k = S_k
            model.S_k_R = S_k_R
            model.predicted_meas = predicted_measurement




    def update_states(self,sensor_state,measurement):
        if not self.use_vel: measurement = measurement[0:2]
        measurement = np.array(measurement).reshape(len(measurement),1)

        #self.predicted_state(sensor_state)#prediction-phase
        self.innovation_list.append(measurement - self.predicted_measurement)
        self.innovation_var.append(self.S_k)

        measurement_matrix,range,azimuth,vel = self.get_linearized_measurment_matrix(self.x_k_km1,sensor_state)  # Linearize the measurement model
        #calculate Kalman gain
        kalman_gain = (self.p_k_km1.dot(measurement_matrix.transpose()))\
            .dot(np.linalg.inv(self.S_k))


        self.x_k_k = self.x_k_km1 + kalman_gain.dot(self.innovation_list[-1])
        self.p_k_k = self.p_k_km1 - (kalman_gain.dot(measurement_matrix)).dot(self.p_k_km1)
        self.gain.append(kalman_gain)
