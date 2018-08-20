from target import target
from sensor import sensor
from measurement import measurement
import numpy as np
import random
import sys
from scenario import scenario
from scipy.stats import norm
import matplotlib.pyplot as plt



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

if __name__=="__main__":

    # initialize actor parameters
    MAX_UNCERTAINTY = 1E9

    num_states = 6
    weight = np.random.normal(0, 1, [2, num_states])
    #weight = np.array([[ -2.25574896, -10.93715588,   2.70823657,   2.762694  ,
     #    -8.7198743 ,  -1.23711939],
      # [  6.21577201,   4.21497013,   3.32356947,  -1.97588774,
       #  -2.21677106, -11.19916791]])
    #weight = np.array([[ 26.77370351, -10.82684799,   8.26656711,  -3.44698487,-0.42696796,  23.45291762],
     #         [ 17.68385971,  33.84914358,  -7.8356591 ,  -6.29215758,-23.09265287,  -8.53755379]])

    weight = np.array([[ 1.45702249, -1.17664153, -0.11593174,  1.02967173, -0.25321044,
         0.09052774],
       [ 0.67730786,  0.3213561 ,  0.99580938, -2.39007038, -1.16340594,
        -1.77515938]])


    #weight = np.array([[7.18777985, -13.68815256, 1.69010242, -5.62483187,
     #                   -4.30451483, 10.09592853],
      #                 [13.33104057, 13.60537864, 3.46939294, 0.8446329,
       #                -14.79733566, -4.78599648]])




    #weight = np.array([[ 0.77289836, -3.77779476,  1.15820171,  0.45819553, -4.26852939,
     #   -0.48236381],
      # [ 4.58947818,  4.83301283, -2.20658465,  2.31762746,  0.52358363,
       # -3.20550963]])

    #weight = np.array([[11.94112467, 12.66720195, 6.69872756, 9.54047939,
     #                   0.29620499, -9.1414007],
      #                 [-8.91840775, 8.02832091, -5.78963127, 1.14847764,
       #                 6.99658831, -6.19396654]])

    sigma_max = 1
    num_episodes = []
    gamma = .99

    episode_length = 1500
    learning_rate = 1E-3
    min_learning_rate = 1E-6
    N_max = 10000

    window_size = 50
    window_lag = 10
    return_saver = []
    error_saver = []

    episode_counter = 0
    weight_saver1 = []
    weight_saver2 = []
    #for episode_counter in range(0,N_max):
    #Training parameters
    avg_reward = []
    avg_error = []
    var_reward = []
    training = True

    result_folder = "/Users/u6042446/Downloads/DeepSensorManagement-original/results/"
    reward_file = open(result_folder+"reward_noise:"+str(.1)+"_linear_6states_varying_var.txt","a")
    error_file = open(result_folder + "error_noise:" + str(.1) + "_linear_6states_varying_var.txt", "a")
    var_file = open(result_folder + "var_noise:" + str(.1) + "_linear_6states_varying_var.txt", "a")
    var_error_file = open(result_folder + "var_error_noise:" + str(.1) + "_linear_6states_varying_var.txt", "a")
    weight_file = open(result_folder + "weight_noise:" + str(.1) + "_linear_6states_varying_var.txt", "a")

    while episode_counter<N_max:
        sigma = gen_learning_rate(episode_counter,sigma_max,.1,10000)
        if episode_counter%500==0 and episode_counter>0:
            sigma-= .1
            sigma = min(sigma,.1)

        #sigma = sigma_max
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
        #xdot = 5
        #ydot = 5
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

        t = target(init_target_state[0:2], init_target_state[2], init_target_state[3], .1, .1, "CONS_V")#constant-velocity model for target motion
        A, B = t.constant_velocity(1E-10)#Get motion model
        x_var = t.x_var
        y_var = t.y_var

        tracker_object = EKF_tracker(init_for_smc, init_covariance, A,B,x_var,y_var,bearing_var)#create tracker object
        #smc_object = smc_tracker(A,B,x_var,y_var,bearing_var,1000,np.array(init_for_smc))

        s = sensor("POLICY_COMM")#create sensor object (stochastic policy)
        #s = sensor("CONS_A")
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


            #if n<1000:
             #   target_vel_x = t.current_velocity[0]
              #  target_vel_y = t.current_velocity[1]
            #else:
             #   target_vel_x = 5 + (1.0/5000)*(n-1000)
              #  target_vel_y = 5 - (1.0/5000)*(n-1000)
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
            #if normalized_innovation>1 or np.linalg.norm(s.sensor_actions[-1])>100:
             #   episode_condition = False

            #Check for episode termination condition
            #L = len(uncertainty)
            #if n>200:
             #   if np.mean(uncertainty[-20:L-1])>np.mean(uncertainty[-21:L-2]):
              #      violation+=1
               #     if violation>100: break
               # else:
                #    violation = 0


            #print(pos_error[-1])
            iteration.append(n)
            if n>episode_length: break
            n+=1


        #Based on the return from the episode, update parameters of the policy model
        #Normalize returns by the length of episode


        #if episode_counter%10==0 and episode_counter>0: print(weight_saver[-1])

        prev_weight = np.array(weight)
        condition = True
        if np.mean(pos_error)>10000:
            continue
            episode_condition = False
            episode_counter-=1

        if episode_counter%100==0 and training:
            print("Starting the evaluation phase...")
            training = False
            episode_condition = False

        if episode_condition and training:
            normalized_discounted_return = discounted_return
            episode_actions = s.sensor_actions
            #init_weight = np.array(weight)
            rate = gen_learning_rate(episode_counter,learning_rate,1E-8,10000)
            total_adjustment = np.zeros(np.shape(weight))
            for e in range(0,len(episode_actions)):
                #calculate gradiant
                state = np.array(episode_state[e]).reshape(6,1)

                gradiant = ((episode_actions[e].reshape(2,1)-weight.dot(state)).dot(state.transpose()))/sigma**2#This is the gradiant
                if np.isnan(gradiant[0][0]) or np.linalg.norm(weight[0,:])>1E3 or np.linalg.norm(weight[1,:])>1E3:
                    condition = False
                    break
                adjustment_term = gradiant*normalized_discounted_return[e]#an unbiased sample of return
                total_adjustment+= adjustment_term

                weight+= rate*adjustment_term

            if not condition:
                weight = prev_weight
                continue
            episode_counter+=1
            flatted_weights = list(weight[0, :]) + list(weight[1, :])
            temp = []
            [temp.append(str(x)) for x in flatted_weights]
            weight_file.write("\t".join(temp)+"\n")
            weight_saver1.append(weight[0][0])
            weight_saver2.append(weight[0][1])
        else:
            #print("garbage trajectory: no-update")
            pass

        if not training:
            return_saver.append(sum(reward))
            error_saver.append(np.mean(pos_error))

        #print(len(return_saver),n)
        if len(return_saver) == 50:
            # if episode_counter%100==0 and episode_counter>0:
            print(episode_counter, np.mean(return_saver), sigma)
            reward_file.write(str(np.mean(return_saver))+"\n")
            error_file.write(str(np.mean(error_saver)) + "\n")
            var_error_file.write(str(np.var(error_saver)) + "\n")
            var_file.write(str(np.var(return_saver))+"\n")
            #weight_file.write(str(np.mean(return_saver)) + "\n")

            avg_reward.append(np.mean(return_saver))
            avg_error.append(np.mean(error_saver))
            var_reward.append(np.var(return_saver))
            reward_file.close()
            var_file.close()
            error_file.close()
            var_error_file.close()
            weight_file.close()

            reward_file = open(result_folder + "reward_noise:" + str(.1) + "_linear_6states_varying_var.txt", "a")
            error_file = open(result_folder + "error_noise:" + str(.1) + "_linear_6states_varying_var.txt", "a")
            var_file = open(result_folder + "var_noise:" + str(.1) + "_linear_6states_varying_var.txt", "a")
            var_error_file = open(result_folder + "var_error_noise:" + str(.1) + "_linear_6states_varying_var.txt", "a")
            weight_file = open(result_folder + "weight_noise:" + str(.1) + "_linear_6states_varying_var.txt", "a")
            return_saver = []
            error_saver = []
            training = True
            episode_counter+=1

        #sys.exit(1)

        #print(weight[0][0])

        #weight += rate * total_adjustment
        #print(total_adjustment)

        num_episodes.append(n)
        #print(weight)


    plt.subplot(4, 1, 1)
    plt1, = plt.plot(iteration, vel_error, "bs-", linewidth=3)
    plt.ylabel("Velocity Estimate Error", size=15)
    plt.grid(True)
    plt.subplot(4, 1, 2)
    plt2, = plt.plot(iteration, pos_error, "rd-", linewidth=3)
    plt.ylabel("Position Estimate Error", size=15)
    plt.grid(True)
    plt.subplot(4, 1, 3)
    plt3, = plt.plot(iteration, innovation, "mo-", linewidth=3)
    plt.xlabel("iteration", size=15)
    plt.ylabel("Predicted Innovation", size=15)
    plt.grid(True)

    plt.subplot(4, 1, 4)
    plt3, = plt.plot(iteration, reward, "ko-", linewidth=3)
    plt.xlabel("iteration", size=15)
    plt.ylabel("Reward function", size=15)
    plt.grid(True)
    plt.show()


    x_t = []
    y_t = []
    x_s = []
    y_s = []

    [x_t.append(x[0]) for x in t.historical_location]
    [y_t.append(x[1]) for x in t.historical_location]

    [x_s.append(x[0]) for x in s.historical_location]
    [y_s.append(x[1]) for x in s.historical_location]

    plt1, = plt.plot(x_t, y_t, "bs-", linewidth=3)
    plt2, = plt.plot(x_s, y_s, "ro--", linewidth=3)
    plt.xlabel("X", size=15)
    plt.ylabel("Y", size=15)
    plt.grid(True)
    plt.legend([plt1, plt2], ["Target", "Sensor"])
    plt.show()
    sys.exit(1)

    plt.subplot(3, 1, 1)
    plt1, = plt.plot(iteration, vel_error, "bs-", linewidth=3)
    plt.ylabel("Velocity Estimate Error", size=15)
    plt.grid(True)
    plt.subplot(3, 1, 2)
    plt2, = plt.plot(iteration, pos_error, "rd-", linewidth=3)
    plt.ylabel("Position Estimate Error", size=15)
    plt.grid(True)
    plt.subplot(3, 1, 3)
    plt3, = plt.plot(iteration, innovation, "mo-", linewidth=3)
    plt.xlabel("iteration", size=15)
    plt.ylabel("Predicted Innovation", size=15)
    plt.grid(True)
    plt.show()
    sys.exit(1)


    sys.exit(1)

        #tracker_object.update_states(s.current_location,m[-1])

        # plot both trajectories



    plt1, = plt.plot(x_truth,y_truth,"bs-",linewidth=3)
    plt2, = plt.plot(x_est, y_est, "rd-", linewidth=3)
    plt.xlabel("X", size=15)
    plt.ylabel("Y", size=15)
    plt.grid(True)
    plt.legend([plt1, plt2], ["Ground Truth", "Estimate"])
    plt.show()
    sys.exit(1)

    x_t = []
    y_t = []
    x_s = []
    y_s = []

    [x_t.append(x[0]) for x in t.historical_location]
    [y_t.append(x[1]) for x in t.historical_location]

    [x_s.append(x[0]) for x in s.historical_location]
    [y_s.append(x[1]) for x in s.historical_location]

    plt1, = plt.plot(x_t, y_t, "bs-", linewidth=3)
    plt2, = plt.plot(x_s, y_s, "ro--", linewidth=3)
    plt.xlabel("X", size=15)
    plt.ylabel("Y", size=15)
    plt.grid(True)
    plt.legend([plt1, plt2], ["Target", "Sensor"])
    plt.show()
    sys.exit(1)


