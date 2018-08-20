from target import target
#from sensor import sensor
from sensor_known_motion import sensor
from measurement import measurement
import numpy as np
import random
import sys
from scenario import scenario
from scipy.stats import norm
#import matplotlib.pyplot as plt
import operator
from multiprocessing import Pool
import os



class EKF_tracker:
    def __init__(self,init_estimate,init_covariance,A,B,x_var,y_var,bearing_var):

        self.init_estimate = init_estimate
        self.init_covariance = init_covariance
        self.bearing_var = bearing_var
        self.A = A
        self.B = B
        self.x_var = x_var
        self.y_var = y_var

        self.x_k_k = np.array(init_estimate).reshape(len(init_estimate),1)
        self.x_k_km1 = self.x_k_k
        self.p_k_k = init_covariance
        self.p_k_km1 = init_covariance
        self.S_k = 1E-5
        self.meas_vec = []

        self.innovation_list = []
        self.innovation_var = []
        self.gain = []


    def get_linearized_measurment_vector(self,target_state,sensor_state):
        relative_location = target_state[0:2] - np.array(sensor_state[0:2]).reshape(2,1)  ##[x-x_s,y-y_s]
        measurement_vector = np.array([-relative_location[1] / ((np.linalg.norm(relative_location)) ** 2),
                                       relative_location[0] / ((np.linalg.norm(relative_location)) ** 2), [0], [0]])
        measurement_vector = measurement_vector.transpose()
        return (measurement_vector)

    def linearized_predicted_measurement(self,sensor_state):
        sensor_state = np.array(sensor_state).reshape(len(sensor_state),1)
        measurement_vector = self.get_linearized_measurment_vector(self.x_k_km1,sensor_state)#Linearize the measurement model
        #predicted_measurement = measurement_vector.dot(np.array(self.x_k_km1))
        predicted_measurement =  np.arctan2(self.x_k_km1[1]-sensor_state[1],self.x_k_km1[0]-sensor_state[0])
        if predicted_measurement<0:predicted_measurement+= 2*np.pi
        return (predicted_measurement,measurement_vector)

    def predicted_state(self,sensor_state,measurement):

        Q = np.eye(2)
        Q[0,0] = .1
        Q[1,1] = .1

        #Q[0,0] = 5
        #Q[1,1] = 5
        predicted_noise_covariance = (self.B.dot(Q)).dot(self.B.transpose())
        self.x_k_km1 = self.A.dot(self.x_k_k)
        self.p_k_km1 = (self.A.dot(self.p_k_k)).dot(self.A.transpose()) + predicted_noise_covariance
        predicted_measurement, measurement_vector = self.linearized_predicted_measurement(sensor_state)
        self.meas_vec.append(measurement_vector)
        #measurement_vector = measurement_vector.reshape(1,len(measurement_vector))
        self.S_k = (measurement_vector.dot(self.p_k_km1)).dot(measurement_vector.transpose()) + self.bearing_var
        self.innovation_list.append(measurement - predicted_measurement)
        self.innovation_var.append(self.S_k)


    def update_states(self,sensor_state,measurement):
        self.predicted_state(sensor_state,measurement)#prediction-phase
        measurement_vector = self.get_linearized_measurment_vector(self.x_k_km1,sensor_state)  # Linearize the measurement model
        #calculate Kalman gain
        kalman_gain = (self.p_k_km1.dot(measurement_vector.transpose()))/self.S_k

        self.x_k_k = self.x_k_km1 + kalman_gain*self.innovation_list[-1]
        self.p_k_k = self.p_k_km1 - (kalman_gain.dot(measurement_vector)).dot(self.p_k_km1)
        self.gain.append(kalman_gain)

class smc_tracker:
    def __init__(self,A,B,x_var,y_var,bearing_var,N,initial_state):
        self.bearing_var = bearing_var
        self.A = A
        self.B = B
        self.x_var = x_var
        self.y_var = y_var
        self.num_particles = N
        self.innovation = None

        scen = scenario(1,1)
        loc_min = np.array([scen.x_min, scen.y_min]).reshape(2,1)
        loc_max = np.array([scen.x_max - scen.x_min, scen.y_max - scen.y_min]).reshape(2, 1)
        a = np.kron(loc_min, np.ones([1, self.num_particles]))
        b = np.kron(loc_max, np.ones([1, self.num_particles]))

        vel_min = np.array([scen.vel_min, scen.vel_min]).reshape(2, 1)
        vel_max = np.array([scen.vel_max - scen.vel_min, scen.vel_max - scen.vel_min]).reshape(2, 1)
        aa = np.kron(vel_min, np.ones([1, self.num_particles]))
        bb = np.kron(vel_max, np.ones([1, self.num_particles]))

        #initial_loc_particles = b*np.random.rand(2,self.num_particles)+a
        temp_cov = np.eye(2)
        temp_cov[0, 0] = 0
        temp_cov[1, 1] = 0
        initial_loc_particles = np.kron(np.array(initial_state[0:2]).reshape(2,1),np.ones([1, self.num_particles])) + 0*np.random.multivariate_normal(np.zeros([2]),temp_cov,self.num_particles).transpose()
        initial_vel_particles = bb*np.random.rand(2,self.num_particles)+aa
        initial_state_particles = np.concatenate((initial_loc_particles,initial_vel_particles))



        """
        temp_cov = np.eye(4)
        temp_cov[0,0] = 20
        temp_cov[1,1] = 20
        temp_cov[2,2] = 1
        temp_cov[3,3] = 1
        initial_state = initial_state.reshape(4,1)
        a = np.kron(initial_state,np.ones([1, self.num_particles]))
        initial_state_particles = a + np.random.multivariate_normal(np.zeros([4]),temp_cov,self.num_particles).transpose()
        """



        self.particles_k_km1 = initial_state_particles
        self.particles_k_k = initial_state_particles
        self.bearing_k_km1 = np.zeros([self.num_particles,1])
        self.weight_k_km1 = (1.0/self.num_particles)*np.ones([self.num_particles,1])
        self.weight_k_k = (1.0 / self.num_particles) * np.ones([self.num_particles, 1])

        Q = np.eye(2)
        Q[0, 0] = .01
        Q[1, 1] = .01


        self.predicted_noise_covariance = (self.B.dot(Q)).dot(self.B.transpose())
        Z = np.eye(4)
        Z[2,2] = .01
        Z[3,3] = .01
        Z[0,0] = 5
        Z[1,1] = 5
        #self.predicted_noise_covariance = Z

    def residual_resample(self,weights):
        N = len(weights)
        indexes = np.zeros(N, 'i')

        # take int(N*w) copies of each weight, which ensures particles with the
        # same weight are drawn uniformly
        num_copies = (np.floor(N * np.asarray(weights))).astype(int)
        k = 0
        for i in range(N):
            for _ in range(num_copies[i]):  # make n copies
                indexes[k] = i
                k += 1

        # use multinormal resample on the residual to fill up the rest. This
        # maximizes the variance of the samples
        residual = weights - num_copies  # get fractional part
        residual /= sum(residual)  # normalize
        cumulative_sum = np.cumsum(residual)
        cumulative_sum[-1] = 1.  # avoid round-off errors: ensures sum is exactly one
        indexes[k:N] = np.searchsorted(cumulative_sum, np.random.random(N - k))

        return indexes

    def predict_update(self,sensor_loc,measurement):
        """
        :Generate particles using proposal function
        :return:
        """

        (num_state,num_particles) = np.shape(self.particles_k_k)

        error_list = []
        for n in range(0,num_particles):
            particle = self.particles_k_k[:,n]
            predicted_particle = self.A.dot(particle)
            next_particle = (np.random.multivariate_normal(predicted_particle, self.predicted_noise_covariance))
            #print(next_particle)
            self.particles_k_km1[:,n] = next_particle
            #also generate predicted measurement and innovation

            #predicted_measurement = np.arctan((next_particle[1] - sensor_loc[1]) / (next_particle[0] - sensor_loc[0])) #this is the predicted bearing
            predicted_measurement = np.arctan2(next_particle[1] - sensor_loc[1],next_particle[0] - sensor_loc[0])  # this is the predicted bearing
            if predicted_measurement<0: predicted_measurement+= 2*np.pi

            error = measurement - predicted_measurement
            error_list.append((error**2))
            #print(error)
            self.weight_k_km1[n] = norm.pdf(error/(np.sqrt(self.bearing_var)))*self.weight_k_k[n] #this is the weight of the predicted particle

        self.innovation = np.mean(error_list)

        self.weight_k_km1 = self.weight_k_km1/np.sum(self.weight_k_km1)
        #Next step is resampling

        effective_sample_size = 1.0/(np.sum(self.weight_k_km1**2))
        if effective_sample_size<self.num_particles/2.0:
        #if True:
            #print("resampling...")
            resampled_indexes = self.residual_resample(self.weight_k_km1)
            #print(resampled_indexes)

            self.particles_k_k = self.particles_k_km1[:,resampled_indexes]
            #self.particles_k_k = self.particles_k_km1

            self.weight_k_k = (1.0/num_particles)*np.ones([self.num_particles,1])
            #self.weight_k_k = self.weight_k_km1
        else:
            self.particles_k_k = self.particles_k_km1
            self.weight_k_k = self.weight_k_km1


def gen_learning_rate(iteration,l_max,l_min,N_max):
    if iteration>N_max: return (l_min)
    alpha = 2*l_max
    beta = np.log((alpha/l_min-1))/N_max
    return (alpha/(1+np.exp(beta*iteration)))

base_path = "/dev/resoures/DeepSensorManagement-original/"
#if __name__=="__main__":
def run(args):
    # initialize parameters of interest
    # Method:
    # 0: linear policy
    # 1: RBF policy
    # 2: MLP policy


    vel_var = args[0]
    heading_rate = args[1]
    experiment_folder_name = args[2]
    file = open(base_path+"/"+experiment_folder_name+"/best_data_"+str(heading_rate),"w")
    # initialize actor parameters
    MAX_UNCERTAINTY = 1E9

    num_states = 6
    weight = np.random.normal(0, 1, [2, num_states])

    sigma_max = 1
    num_episodes = []
    gamma = .99

    episode_length = 1500
    learning_rate = 1E-3
    min_learning_rate = 1E-6
    N_max = 200

    window_size = 50
    window_lag = 10
    return_saver = []

    weight_saver1 = []
    weight_saver2 = []

    total_error = {}
    total_error_variance = {}
    total_reward = {}
    #for episode_counter in range(0,N_max):
    #Training parameters
    print("heading-rate="+str(heading_rate))
    for xdot_sensor in np.arange(-15,16,1):
        for ydot_sensor in np.arange(-15,16,1):

            episode_counter = 0
            avg_reward = []
            var_reward = []
            error_saver = []
            while episode_counter<N_max:
                sigma = gen_learning_rate(episode_counter,sigma_max,.1,5000)
                sigma = sigma_max
                discounted_return = np.array([])
                discount_vector = np.array([])
                #print(episodes_counter)
                scen = scenario(1,1)
                bearing_var = 1E-2#variance of bearing measurement
                #Target information
                x = 10000*random.random()-5000#initial x-location
                y = 10000 * random.random() - 5000#initial y-location
                xdot = 10*random.random()-5#initial xdot-value
                ydot = 10 * random.random() - 5#initial ydot-value
                #x = 250; y = 50; xdot = 7; ydot = -5

                init_target_state = [x,y,xdot,ydot]#initialize target state
                init_for_smc = [x+np.random.normal(0,5),y+np.random.normal(0,5),np.random.normal(0,5),np.random.normal(0,5)]#init state for the tracker (tracker doesn't know about the initial state)
                #init_for_smc = [x, y, xdot, ydot]
                init_sensor_state = [10000*random.random()-5000,10000 * random.random() - 5000,3,-2]#initial sensor-state

                temp_loc = np.array(init_target_state[0:2]).reshape(2,1)
                init_location_estimate = temp_loc+0*np.random.normal(np.zeros([2,1]),10)
                init_location_estimate = [init_location_estimate[0][0],init_location_estimate[1][0]]
                init_velocity_estimate = [6*random.random()-3,6*random.random()-3]
                init_velocity_estimate = [init_target_state[2],init_target_state[3]]


                init_estimate = init_location_estimate+init_velocity_estimate
                init_covariance = np.diag([MAX_UNCERTAINTY,MAX_UNCERTAINTY,MAX_UNCERTAINTY,MAX_UNCERTAINTY])#initial covariance of state estimation

                t = target(init_target_state[0:2], init_target_state[2], init_target_state[3], vel_var, vel_var, "CONS_V")#constant-velocity model for target motion
                A, B = t.constant_velocity(1E-10)#Get motion model
                x_var = t.x_var
                y_var = t.y_var

                tracker_object = EKF_tracker(init_for_smc, init_covariance, A,B,x_var,y_var,bearing_var)#create tracker object
                #smc_object = smc_tracker(A,B,x_var,y_var,bearing_var,1000,np.array(init_for_smc))

                s = sensor("CONS_V",[0,0],[xdot_sensor,ydot_sensor],heading_rate)#create sensor object (stochastic policy)
                #s = sensor("CONS_V")
                measure = measurement(bearing_var)#create measurement object

                m = []
                x_est = []; y_est = []; x_vel_est = []; y_vel_est = []
                x_truth = [];
                y_truth = [];
                x_vel_truth = [];
                y_vel_truth = []
                uncertainty = []

                vel_error = []
                pos_error = []
                iteration = []
                innovation = []

                reward = []
                episode_condition = True
                n=0
                violation = 0
                #store required information
                episode_state = []
                episode_actions = []

                while episode_condition:

                    #if n>50: episode_condition=False
                    #update location of target and sensor + generate new measurement
                    #Also, run tracker object
                    t.update_location()
                    m.append(measure.generate_bearing(t.current_location,s.current_location))
                    tracker_object.update_states(s.current_location, m[-1])
                    #if len(tracker_object.meas_vec)>20:
                     #   tmp = np.zeros([2,2])
                      #  for n in range(0,10):
                       #     vector = tracker_object.meas_vec[-1-n]
                        #    cov = (vector.transpose().dot(vector))/bearing_var
                         #   sliced_cov = np.array([[cov[0,0],cov[0,1]],[cov[1,0],cov[1,1]]])
                          #  tmp+= sliced_cov

                        #Fisher_matrix = tmp/10.0
                        #crlb = np.linalg.inv(Fisher_matrix)
                        #print(crlb.diagonal())

                    #create state-vector

                    normalized_innovation = (tracker_object.innovation_list[-1])/tracker_object.innovation_var[-1]



                    #print(normalized_innovation)
                    #if (normalized_innovation<1E-4 or n<10) and n<200:
                        #end of episode
                    current_state = list(tracker_object.x_k_k.reshape(len(tracker_object.x_k_k))) + list(s.current_location)

                    #print(current_state)
                    #state normalization
                    x_slope = 2.0/(scen.x_max-scen.x_min)
                    y_slope = 2.0 / (scen.y_max - scen.y_min)
                    vel_slope = 2.0/(scen.vel_max-scen.vel_min)
                    #normalization
                    current_state[0] = -1+x_slope*(current_state[0]-scen.x_min)
                    current_state[1] = -1 + y_slope * (current_state[1] - scen.y_min)
                    current_state[2] = -1 + vel_slope * (current_state[2] - scen.vel_min)
                    current_state[3] = -1 + vel_slope * (current_state[3] - scen.vel_min)
                    current_state[4] = -1 + x_slope * (current_state[4] - scen.x_min)
                    current_state[5] = -1 + y_slope * (current_state[5] - scen.y_min)
                    s.update_location(weight, sigma, np.array(current_state))
                    estimate = tracker_object.x_k_k
                    episode_state.append(current_state)


                    truth = t.current_location
                    x_est.append(estimate[0])
                    y_est.append(estimate[1])
                    x_vel_est.append(estimate[2])
                    y_vel_est.append(estimate[3])

                    x_truth.append(truth[0])
                    y_truth.append(truth[1])

                    x_vel_truth.append(t.current_velocity[0])
                    y_vel_truth.append(t.current_velocity[1])

                    #print(estimate[-1])
                    #print(np.linalg.norm(estimate[2:4]-np.array([t.current_velocity[0],t.current_velocity[1]])))
                    vel_error.append(np.linalg.norm(estimate[2:4]-np.array([t.current_velocity[0],t.current_velocity[1]]).reshape(2,1)))
                    pos_error.append(np.linalg.norm(estimate[0:2]-np.array(truth).reshape(2,1)))
                    innovation.append(normalized_innovation[0])

                    unormalized_uncertainty = np.sum(tracker_object.p_k_k.diagonal())
                    #if unormalized_uncertainty>MAX_UNCERTAINTY:
                     #   normalized_uncertainty = 1
                    #else:
                     #   normalized_uncertainty = (1.0/MAX_UNCERTAINTY)*unormalized_uncertainty

                    uncertainty.append((1.0/MAX_UNCERTAINTY)*unormalized_uncertainty)
                    if len(uncertainty)<window_size+window_lag:
                        reward.append(0)
                    else:
                        current_avg = np.mean(uncertainty[-window_size:])
                        prev_avg = np.mean(uncertainty[-(window_size+window_lag):-window_lag])
                        if current_avg<prev_avg or uncertainty[-1]<.1:
                        #if current_avg < prev_avg:
                            reward.append(1)
                        else:
                            reward.append(0)

                    #reward.append(-1*uncertainty[-1])
                    #update return

                    discount_vector = gamma*np.array(discount_vector)
                    #discount_vector = list(discount_vector)
                    #discount_vector.append(1)

                    discounted_return+= (1.0*reward[-1])*discount_vector
                    new_return = 1.0*reward[-1]
                    list_discounted_return = list(discounted_return)
                    list_discounted_return.append(new_return)
                    discounted_return = np.array(list_discounted_return)

                    list_discount_vector = list(discount_vector)
                    list_discount_vector.append(1)
                    discount_vector = np.array(list_discount_vector)


                    iteration.append(n)
                    if n>episode_length: break
                    n+=1
                num_episodes.append(n)
                error_saver.append(np.mean(pos_error))
                return_saver.append(sum(reward))
                episode_counter += 1


            total_error[str(xdot_sensor)+"|"+str(ydot_sensor)] = np.mean(sorted(error_saver)[0:int(.95*N_max)])
            total_reward[str(xdot_sensor)+"|"+str(ydot_sensor)] = np.mean(sorted(return_saver,reverse=True)[0:int(.95*N_max)])
            total_error_variance[str(xdot_sensor)+"|"+str(ydot_sensor)] = np.var(sorted(error_saver)[0:int(.95*N_max)])


    sorted_error = sorted(total_error.items(),key=operator.itemgetter(1))
    key = sorted_error[0][0]

    file.write("Min Error="+str(sorted_error[0][1])+"\n")
    file.write("Best params="+str(key)+"\n")
    file.close()


if __name__=="__main__":

    heading_rates = [1E-6,1E-5,5*1E-5,1E-4,5*1E-4,1E-3,5*1E-3]
    vel_var = .01
    p = Pool(len(heading_rates))
    experiment_folder_name = "constant_turn_sensor_single_target"+"_"+str(vel_var)

    if not os.path.exists(base_path+experiment_folder_name):
        os.makedirs(base_path+experiment_folder_name)
    method = 0
    RBF_components = 20
    MLP_neurons = 50
    #vel_var = .001

    job_args = [(vel_var,heading_rates[i],experiment_folder_name) for i in range(0,len(heading_rates))]
    p.map(run,job_args)
    #run([vel_var,heading_rates[0],experiment_folder_name])
