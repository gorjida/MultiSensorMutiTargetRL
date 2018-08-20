from target import target
from sensor import sensor
from measurement import measurement
import numpy as np
import random
import sys
from scenario import scenario
from scipy.stats import norm
#import matplotlib.pyplot as plt

import sklearn.pipeline
from sklearn.kernel_approximation import RBFSampler
import os

from multiprocessing import Pool

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


#Set general parameters
MAX_UNCERTAINTY = 1E9
num_states = 6
num_states_layer2 = 6
sigma_max = 1
num_episodes = []
gamma = .99
episode_length = 1500
learning_rate = 1E-3
N_max = 10000
window_size = 50
window_lag = 10
rbf_var = 1

base_path = "/dev/resoures/DeepSensorManagement-original/"

#list_of_states = []
#with open("raw_states_for_rbf.txt","r") as f:
 #   for line in f:
  #      data = line.strip().split("\t")
   #     dd = []
    #    [dd.append(float(x)) for x in data]
     #   list_of_states.append(dd)
#def run(method,RBF_components,MLP_neurons,process_index,folder_name):
def run(args):
#if __name__=="__main__":
    # initialize parameters of interest
    # Method:
    # 0: linear policy
    # 1: RBF policy
    # 2: MLP policy

    #method = args[0]
    #RBF_components = args[1]
    #MLP_neurons = args[2]
    process_index = args[3]
    folder_name = args[4]
    np.random.seed(process_index+100)
    #process_index = 0
    #np.random.seed(process_index + 100)
    #vel_var = args[5]
    #num_targets = args[6]

    method = 0
    RBF_components = 20
    MLP_neurons = 50
    vel_var = .001
    num_targets = min(6,max(2,np.random.poisson(3)))
    num_targets = np.random.randint(2,5)
    #num_targets = 4


    print("Starting Thread:" + str(process_index))

    #Initialize all the parameters
    params ={0:{},1:{},2:{}}
    if method==0:
        params[0]["weight2"] = np.random.normal(0, 1, [2, num_states_layer2])
        #params[0]["weight2"] = np.array([[  3.97573312,   0.4639474 ,   2.27280486,  12.9085868 ,
         #   3.45722461,   6.36735166],
         #[-11.87940874,   2.59549414,  -5.68556954,   2.87746786,
          #  7.08059984,   5.5631133 ]])

        params[0]["weight"] = np.array([[7.18777985, -13.68815256, 1.69010242, -5.62483187,
                           -4.30451483, 10.09592853],
                         [13.33104057, 13.60537864, 3.46939294, 0.8446329,
                         -14.79733566, -4.78599648]])

        #params[0]["weight"] = np.array([[ 1.45702249, -1.17664153, -0.11593174,  1.02967173, -0.25321044,
         #0.09052774],
       #[ 0.67730786,  0.3213561 ,  0.99580938, -2.39007038, -1.16340594,
        #-1.77515938]])
    elif method==1:
        featurizer = sklearn.pipeline.FeatureUnion([("rbf1", RBFSampler(gamma=rbf_var, n_components=RBF_components, random_state=1))])
        featurizer.fit(np.array(list_of_states))  # Use this featurizer for normalization
        params[1]["weight"] = np.random.normal(0, 1, [2, RBF_components])
    elif method==2:
        params[2]["weigh1"] = np.random.normal(0, 1, [MLP_neurons, num_states])
        params[2]["bias1"] = np.random.normal(0,1,[MLP_neurons,1])
        params[2]["weigh2"] = np.random.normal(0, 1, [2, MLP_neurons])
        params[2]["bias2"] = np.random.normal(0, 1, [2, 1])

    return_saver = []
    error_saver = []
    episode_counter = 0
    weight_saver1 = []
    weight_saver2 = []
    weight_saver2_1 = []
    weight_saver2_2 = []
    #for episode_counter in range(0,N_max):
    #Training parameters
    avg_reward = []
    avg_error = []
    var_reward = []
    training = True


    result_folder = base_path+folder_name+"/"
    reward_file = open(result_folder+"reward_noise:"+str(vel_var)+"_"+str(process_index)+  "_linear_6states.txt","a")
    error_file = open(result_folder + "error_noise:" + str(vel_var) +"_"+str(process_index)+ "_linear_6states.txt", "a")
    error_file_median = open(result_folder + "error_median_noise:" + str(vel_var) + "_" + str(process_index) + "_linear_6states.txt",
                      "a")
    var_file = open(result_folder + "var_noise:" + str(vel_var) +"_"+str(process_index)+ "_linear_6states.txt", "a")
    var_error_file = open(result_folder + "var_error_noise:" + str(vel_var) +"_"+str(process_index)+ "_linear_6states.txt", "a")
    weight_file = open(result_folder + "weight_noise:" + str(vel_var) +"_"+str(process_index)+ "_linear_6states.txt", "a")

    #flatten initial weight and store the values
    if method==0:
        weight = params[0]['weight']
        flatted_weights = list(weight[0, :]) + list(weight[1, :])
        temp = []
        [temp.append(str(x)) for x in flatted_weights]
        weight_file.write("\t".join(temp)+"\n")
    elif method==1:
        weight = params[1]['weight']
        flatted_weights = list(weight[0, :]) + list(weight[1, :])
        temp = []
        [temp.append(str(x)) for x in flatted_weights]
        weight_file.write("\t".join(temp) + "\n")
    elif method==2:
        pass

    #weight = np.reshape(np.array(weights[0]), [2, 6])

    while episode_counter<N_max:
        sigma = gen_learning_rate(episode_counter,sigma_max,.1,5000)
        sigma = sigma_max
        discounted_return = np.array([])
        discount_vector = np.array([])
        #print(episodes_counter)
        scen = scenario(1,1)
        bearing_var = 1E-2#variance of bearing measurement
        #Target information
        x = 10000*np.random.random([num_targets])-5000#initial x-location
        y = 10000 * np.random.random([num_targets]) - 5000#initial y-location
        xdot = 10*np.random.random([num_targets])-5#initial xdot-value
        ydot = 10 * np.random.random([num_targets]) - 5#initial ydot-value

        #TEMP
        #x = [2000,-2000]
        #y = [2000,2000]
        #xdot = [1,1]
        #ydot = [-1,-1]

        init_target_state = []
        init_for_smc = []
        for target_counter in range(0,num_targets):
            init_target_state.append([x[target_counter],y[target_counter],xdot[target_counter],ydot[target_counter]])#initialize target state
            init_for_smc.append([x[target_counter]+np.random.normal(0,5),y[target_counter]
                                 +np.random.normal(0,5),np.random.normal(0,5),np.random.normal(0,5)])#init state for the tracker (tracker doesn't know about the initial state)


        #temp_loc = np.array(init_target_state[0:2]).reshape(2,1)
        #init_location_estimate = temp_loc+0*np.random.normal(np.zeros([2,1]),10)
        #init_location_estimate = [init_location_estimate[0][0],init_location_estimate[1][0]]
        #init_velocity_estimate = [6*random.random()-3,6*random.random()-3]
        #init_velocity_estimate = [init_target_state[2],init_target_state[3]]
        #init_estimate = init_location_estimate+init_velocity_estimate
        init_covariance = np.diag([MAX_UNCERTAINTY,MAX_UNCERTAINTY,MAX_UNCERTAINTY,MAX_UNCERTAINTY])#initial covariance of state estimation
        t = []
        for i in range(0,num_targets):
            t.append(target(init_target_state[i][0:2], init_target_state[i][2],
                            init_target_state[i][3], vel_var, vel_var, "CONS_V"))#constant-velocity model for target motion
        A, B = t[0].constant_velocity(1E-10)#Get motion model
        x_var = t[0].x_var
        y_var = t[0].y_var

        tracker_object = []
        for i in range(0,num_targets):
            tracker_object.append(EKF_tracker(init_for_smc[i], np.array(init_covariance), A,B,x_var,y_var,bearing_var))#create tracker object
            #smc_object = smc_tracker(A,B,x_var,y_var,bearing_var,1000,np.array(init_for_smc))

        #Initialize sensor object
        if method==0:
            s = sensor("POLICY_COMM_LINEAR")#create sensor object (stochastic policy)
        elif method==1:
            s = sensor("POLICY_COMM_RBF")
        elif method==2:
            s = sensor("POLICY_COMM_MLP")
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
        for i in range(0,num_targets):
            x_truth.append([])
            y_truth.append([])
            x_vel_truth.append([])
            y_vel_truth.append([])
            uncertainty.append([])
            vel_error.append([])
            x_est.append([])
            y_est.append([])
            x_vel_est.append([])
            y_vel_est.append([])
            pos_error.append([])
            innovation.append([])
        reward = []
        episode_condition = True
        n=0
        violation = 0
        #store required information
        episode_state = []
        episode_state_out_layer = []
        episode_MLP_state = []
        episode_actions = []
        avg_uncertainty= []
        max_uncertainty = []

        while episode_condition:
            temp_m = []
            input_state_temp = []
            for i in range(0,num_targets):
                t[i].update_location()
                temp_m.append(measure.generate_bearing(t[i].current_location,s.current_location))

            m.append(temp_m)
            temp_reward = []
            target_actions = []
            for i in range(0,num_targets):
                tracker_object[i].update_states(s.current_location, m[-1][i])
                normalized_innovation = (tracker_object[i].innovation_list[-1])/tracker_object[i].innovation_var[-1]
                #print(normalized_innovation)
                #if (normalized_innovation<1E-4 or n<10) and n<200:
                    #end of episode
                current_state = list(tracker_object[i].x_k_k.reshape(len(tracker_object[i].x_k_k))) + list(s.current_location)

                #print(current_state)
                #state normalization
                x_slope = 2.0/(scen.x_max-scen.x_min)
                y_slope = 2.0 / (scen.y_max - scen.y_min)

                x_slope_sensor = 2.0 / (40000)
                y_slope_sensor = 2.0 / (40000)

                vel_slope = 2.0/(scen.vel_max-scen.vel_min)
                #normalization
                current_state[0] = -1+x_slope*(current_state[0]-scen.x_min)
                current_state[1] = -1 + y_slope * (current_state[1] - scen.y_min)
                current_state[2] = -1 + vel_slope * (current_state[2] - scen.vel_min)
                current_state[3] = -1 + vel_slope * (current_state[3] - scen.vel_min)
                current_state[4] = -1 + x_slope * (current_state[4] -scen.x_min)
                current_state[5] = -1 + y_slope * (current_state[5] - scen.y_min)


                #Refactor states based on the usage
                if method==0 or method==2:
                    input_state = current_state
                    input_state_temp.append(input_state) #store input-sates
                elif method==1:
                    #Generate states for the RBF input
                    input_state =  featurizer.transform(np.array(current_state).reshape(1,len(current_state)))
                    input_state = list(input_state[0])


                target_actions.append(s.generate_action(params,input_state,.01))
                estimate = tracker_object[i].x_k_k
                episode_state.append(input_state) ####Neeed to get modified
                if method==2: episode_MLP_state.append(extra_information) #need to get modified
                truth = t[i].current_location
                x_est[i].append(estimate[0])
                y_est[i].append(estimate[1])
                x_vel_est[i].append(estimate[2])
                y_vel_est[i].append(estimate[3])
                x_truth[i].append(truth[0])
                y_truth[i].append(truth[1])
                x_vel_truth[i].append(t[i].current_velocity[0])
                y_vel_truth[i].append(t[i].current_velocity[1])
                vel_error[i].append(np.linalg.norm(estimate[2:4]-np.array([t[i].current_velocity[0],t[i].current_velocity[1]]).reshape(2,1)))
                pos_error[i].append(np.linalg.norm(estimate[0:2]-np.array(truth).reshape(2,1)))
                innovation[i].append(normalized_innovation[0])
                unormalized_uncertainty = np.sum(tracker_object[i].p_k_k.diagonal())
                #if unormalized_uncertainty>MAX_UNCERTAINTY:
                #   normalized_uncertainty = 1
                #else:
                #   normalized_uncertainty = (1.0/MAX_UNCERTAINTY)*unormalized_uncertainty
                uncertainty[i].append((1.0 / MAX_UNCERTAINTY) * unormalized_uncertainty)
                #if len(uncertainty[i])<window_size+window_lag:
                 #   temp_reward.append(0)
                #else:
                 #   current_avg = np.mean(uncertainty[i][-window_size:])
                  #  prev_avg = np.mean(uncertainty[i][-(window_size+window_lag):-window_lag])
                   # if current_avg<prev_avg or uncertainty[i][-1]<.1:
                    #if current_avg < prev_avg:
                    #    temp_reward.append(1)
                    #else:
                     #   temp_reward.append(0)

            this_uncertainty = []
            [this_uncertainty.append(uncertainty[x][-1]) for x in range(0, num_targets)]
            avg_uncertainty.append(np.mean(this_uncertainty))
            max_uncertainty.append(np.max(this_uncertainty))
            if len(avg_uncertainty) < window_size + window_lag:
                reward.append(0)
            else:
                current_avg = np.mean(avg_uncertainty[-window_size:])
                prev_avg = np.mean(avg_uncertainty[-(window_size + window_lag):-window_lag])
                if current_avg < prev_avg or avg_uncertainty[-1] < .1:
                    # if current_avg < prev_avg:
                    reward.append(1)
                else:
                    reward.append(0)

            #voting
            #if np.mean(temp_reward)>.5:
             #   reward.append(np.mean(temp_reward))
            #else:
             #   reward.append(np.mean(temp_reward))

            #if sum(reward)>1100 and num_targets>2: sys.exit(1)

            #Do something on target_actions
            #Create feature-vector from generated target actions

            normalized_state,index_matrix1,index_matrix2,slope = s.update_location_decentralized(target_actions,sigma,params) #Update the sensor location based on all individual actions
            #index_matrix: an n_s \times T matrix that shows the derivative of state in the output layer to the action space in the internal-layer

            backpropagated_to_internal_1 = index_matrix1.dot(np.array(input_state_temp))#8 by 6
            backpropagated_to_internal_2 = index_matrix2.dot(np.array(input_state_temp))# 8 by 6

            episode_state_out_layer.append(normalized_state)
            episode_state.append([backpropagated_to_internal_1,backpropagated_to_internal_2]) #each entry would be a T \times 6 matrix with T being the number of targets
            #reward.append(-1*uncertainty[-1])
            #update return
            discount_vector = gamma*np.array(discount_vector)
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

        #Based on the return from the episode, update parameters of the policy model
        #Normalize returns by the length of episode
        #if episode_counter%10==0 and episode_counter>0: print(weight_saver[-1])

        prev_params = dict(params)
        condition = True
        for i in range(0,num_targets):
            if np.mean(pos_error[i])>10000:
                condition = False
                break
                episode_condition = False
                episode_counter-=1

        if not condition:
            #print("OOPSSSS...")
            continue
        #if episode_counter%100==0 and training:
            #print("Starting the evaluation phase...")
            #training = False
            #episode_condition = False


        condition = True
        if episode_condition and training:
            normalized_discounted_return = discounted_return
            episode_actions = s.sensor_actions
            #init_weight = np.array(weight)
            rate = gen_learning_rate(episode_counter,learning_rate,1E-8,10000)
            internal_rate = gen_learning_rate(episode_counter, 3*1E-5, 1E-10, 10000)
            total_adjustment = np.zeros(np.shape(weight))
            for e in range(0,len(episode_actions)):
                #calculate gradiant
                #state = np.array(episode_state[e]).reshape(len(episode_state[e]),1)
                out_state = np.array(episode_state_out_layer[e]).reshape(len(episode_state_out_layer[e]),1)
                backpropagated_terms = episode_state[e]

                #calculate gradient
                if method==0:
                    deriv_with_out_state = (episode_actions[e].reshape(2, 1) - params[0]['weight2'].dot(out_state)).transpose().dot(params[0]['weight2']) #1 by n_s==> derivative of F with respect to the output state-vector
                    internal_gradiant1 = deriv_with_out_state.dot(backpropagated_terms[0]) #1 by 6
                    internal_gradiant2 = deriv_with_out_state.dot(backpropagated_terms[1]) #1 by 6
                    internal_gradiant = np.concatenate([internal_gradiant1,internal_gradiant2])

                    #gradiant = ((episode_actions[e].reshape(2,1)-params[0]['weight'].dot(state)).dot(state.transpose()))/sigma**2#This is the gradiant
                    gradiant_out_layer = ((episode_actions[e].reshape(2, 1) - params[0]['weight2'].dot(out_state)).dot(
                        out_state.transpose())) / sigma ** 2  # This is the gradiant
                elif method==1:
                    gradiant = ((episode_actions[e].reshape(2, 1) - params[1]['weight'].dot(state)).dot(
                        state.transpose())) / sigma ** 2  # This is the gradiant
                elif method==2:
                    #Gradient for MLP
                    pass

                if np.max(np.abs(gradiant_out_layer))>1E2 or np.max(np.abs(internal_gradiant))>1E2:
                    #print("OOPPSSSS...")
                    continue #clip large gradients

                if method==0:
                    adjustment_term_out_layer = gradiant_out_layer*normalized_discounted_return[e]#an unbiased sample of return
                    adjustment_term_internal_layer = internal_gradiant*normalized_discounted_return[e]
                    params[0]['weight2'] += rate * adjustment_term_out_layer
                    params[0]['weight'] += internal_rate* adjustment_term_internal_layer
                elif method==1:
                    adjustment_term = gradiant * normalized_discounted_return[e]  # an unbiased sample of return
                    params[1]['weight'] += rate * adjustment_term
                elif method==2:
                    #Gradient for MLP
                    pass

            #if not condition:
             #   weight = prev_weight
              #  continue

            episode_counter+=1
            #flatted_weights = list(weight[0, :]) + list(weight[1, :])
            #temp = []
            #[temp.append(str(x)) for x in flatted_weights]
            #weight_file.write("\t".join(temp)+"\n")
            weight_saver1.append(params[0]['weight'][0][0])
            weight_saver2.append(params[0]['weight'][1][0])

            weight_saver2_1.append(params[0]['weight2'][0][0])
            weight_saver2_2.append(params[0]['weight2'][1][0])
        else:
            #print("garbage trajectory: no-update")
            pass


        #if not training:
        return_saver.append(sum(reward))

        error_saver.append(np.mean(pos_error))

        #print(len(return_saver),n)
        if episode_counter%100 == 0 and episode_counter>0:
            # if episode_counter%100==0 and episode_counter>0:
            print(episode_counter, np.mean(return_saver), sigma)
            #print(params[method]['weight'])
            #weight = np.reshape(np.array(weights[episode_counter]), [2, 6])
            #print(weight)
            reward_file.write(str(np.mean(sorted(return_saver,reverse=True)[0:int(.95*len(return_saver))]))+"\n")
            error_file.write(str(np.mean(sorted(error_saver)[0:int(.95*len(error_saver))])) + "\n")
            error_file_median.write(str(np.median(sorted(error_saver)[0:int(.95*len(error_saver))])) + "\n")
            var_error_file.write(str(np.var(sorted(error_saver)[0:int(.95*len(error_saver))])) + "\n")
            var_file.write(str(np.var(sorted(return_saver,reverse=True)[0:int(.95*len(return_saver))]))+"\n")
            #weight_file.write(str(np.mean(return_saver)) + "\n")

            avg_reward.append(np.mean(sorted(return_saver)[0:int(.95*len(return_saver))]))
            avg_error.append(np.mean(sorted(error_saver)[0:int(.95*len(error_saver))]))
            var_reward.append(np.var(return_saver))
            reward_file.close()
            var_file.close()
            error_file.close()
            error_file_median.close()
            var_error_file.close()
            weight_file.close()

            reward_file = open(
                result_folder + "reward_noise:" + str(vel_var) + "_" + str(process_index) + "_linear_6states.txt", "a")
            error_file = open(
                result_folder + "error_noise:" + str(vel_var) + "_" + str(process_index) + "_linear_6states.txt", "a")
            var_file = open(
                result_folder + "var_noise:" + str(vel_var) + "_" + str(process_index) + "_linear_6states.txt", "a")
            var_error_file = open(
                result_folder + "var_error_noise:" + str(vel_var) + "_" + str(process_index) + "_linear_6states.txt",
                "a")
            weight_file = open(
                result_folder + "weight_noise:" + str(vel_var) + "_" + str(process_index) + "_linear_6states.txt", "a")
            error_file_median = open(
                result_folder + "error_median_noise:" + str(vel_var) + "_" + str(process_index) + "_linear_6states.txt",
                "a")

            return_saver = []
            error_saver = []
        num_episodes.append(n)


if __name__=="__main__":

    p = Pool(10)
    experiment_folder_name = "linear_policy_discrete_reward_multiple_T25"

    if not os.path.exists(base_path+experiment_folder_name):
        os.makedirs(base_path+experiment_folder_name)
    method = 0
    RBF_components = 20
    MLP_neurons = 50
    vel_var = .01
    num_targets = 5

    job_args = [(method,RBF_components,MLP_neurons,i,experiment_folder_name,vel_var,num_targets) for i in range(0,10)]
    #run(job_args[0])
    p.map(run,job_args)
    #run(0,RBF_components,MLP_neurons,0,experiment_folder_name)





