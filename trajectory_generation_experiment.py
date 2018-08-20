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
N_max = 200
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
#def run(args):
if __name__=="__main__":
    # initialize parameters of interest
    # Method:
    # 0: linear policy
    # 1: RBF policy
    # 2: MLP policy

    #method = args[0]
    #RBF_components = args[1]
    #MLP_neurons = args[2]
    #process_index = args[3]
    #folder_name = args[4]
    #np.random.seed(process_index+100)
    #process_index = 0
    #np.random.seed(process_index + 100)
    #vel_var = args[5]
    #num_targets = args[6]

    method = 0
    RBF_components = 20
    MLP_neurons = 50
    vel_var = .001
    num_targets = min(6,max(2,np.random.poisson(3)))
    num_targets = 1
    #num_targets = 4

    #create parameters for arctan limitter
    coeff = .9
    v_max = 15
    c = np.tan(coeff * np.pi / 2)
    c_ = np.tan(-coeff*np.pi/2)
    alpha1 = (coeff*np.pi/(2*v_max))*(c**2)
    alpha2 = c - alpha1 * v_max
    alpha1_ = (coeff * np.pi / (2 * v_max)) * (c_ ** 2)
    alpha2_ = c_ + alpha1 * v_max


    #print("Starting Thread:" + str(process_index))

    #Initialize all the parameters
    params ={0:{},1:{},2:{}}
    ###
    #0: multiple targets
    #1: single-target (unconstrained)
    #2: single-target (constrained)

    params[0]["weight2"] = np.array([[-15.0995999 ,  -3.30260383,  -8.12472926,  -2.78509237,
         12.82038819,   5.7735278 ],
       [ -3.00869935,  -4.62431591,  -4.38137369,  -5.63819881,
        -15.59276633, -10.58080936]])

    params[0]["weight"] = np.array([[  4.97659072, -12.8438154 ,   0.81581003,  -7.32680964,
         -3.1707998 ,   9.40878054],
       [ 10.79033716,  15.01036494,   3.11112251,  -2.04943493,
        -11.87343093,  -4.86822482]])



    params[1]["weight"] = np.array([[7.18777985, -13.68815256, 1.69010242, -5.62483187,
                         -4.30451483, 10.09592853],
                        [13.33104057, 13.60537864, 3.46939294, 0.8446329,
                        -14.79733566, -4.78599648]])
    params[2]["weight"] = np.array([[  9.02497513,  -9.82799354,   4.19640778,  -5.42538468,
         -2.43472988,  14.13928278],
       [ 11.61938151,   9.20475081,   9.86998635,   4.52637783,
        -14.31141922,  -2.16144722]])


    #Unconstrained
    """
    params[1]["weight"] = np.array([[  5.75349369, -13.59334294,   0.94308009,  -6.80638952,
         -5.44568898,   8.51273695],
       [ 16.50495031,  12.68794607,   4.57148743,   0.38857003,
        -17.48407913,  -6.18174848]])

    params[2]["weight"] = np.array([[ -9.77540916,  -4.85682238,  -6.76424045,  -6.2528791 ,
         11.74783978,   3.35471872],
       [  6.91669456, -12.23859983,  -2.25310555,  -0.116628  ,
        -17.12004592,  -8.76317268]])
    """

    #params[2]["weight"] = np.random.normal(0, .3, [2, num_states])



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


    #weight = np.reshape(np.array(weights[0]), [2, 6])
    ii = "3"
    #target_file = open("trajectories/truth_"+ii+".txt","w")
    #sensor_file = open("trajectories/sensor_single_"+ii+".txt","w")
    #pcrlb_file = open("trajectories/pcrlb_single_" + ii + ".txt", "w")

    #average_error_file = open("trajectories/average_error_" + ii + ".txt", "w")

    #single_target_based_index = 3
    total_pos_error = [[],[],[],[]]
    effective = 0
    total_avg_pcrlb = np.zeros([1502])
    while episode_counter<N_max:
        sensor_locations  = {}
        for weight_index in range(1, 2):
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

            x = np.array([-2000,2000])
            y = np.array([-4000,-4000])
            xdot = [2,-2]
            ydot = [2,2]



            #x = [3000]
            #y = [3000]
            #xdot = [2]
            #ydot = [2]

            #x = [3000]
            #y = [-3000]
            #xdot = [2]
            #ydot = [-2]

            #x = [-3000]
            #y = [-3000]
            #xdot = [-2]
            #ydot = [-2]

            #x = [-3000]
            #y = [3000]
            #xdot = [-2]
            #ydot = [2]

            #x = [0]
            #y = [1000]
            #xdot = [-5]
            #ydot = [5]



            PCRLB = []
            pcrlb_x = [[],[],[],[]]
            pcrlb_y = [[],[],[],[]]
            pcrlb = [[],[],[],[]]
            avg_pcrlb = []

            for i in range(0,num_targets):PCRLB.append(np.diag([1.0/10,1.0/10,1.0/1,1.0/1]))


            init_target_state = []
            init_for_smc = []
            for target_counter in range(0,num_targets):
                init_target_state.append([x[target_counter],y[target_counter],xdot[target_counter],ydot[target_counter]])#initialize target state
                init_for_smc.append([x[target_counter]+np.random.normal(0,5),y[target_counter]
                                     +np.random.normal(0,5),np.random.normal(0,5),np.random.normal(0,5)])#init state for the tracker (tracker doesn't know about the initial state)
            init_covariance = np.diag([MAX_UNCERTAINTY,MAX_UNCERTAINTY,MAX_UNCERTAINTY,MAX_UNCERTAINTY])#initial covariance of state estimation
            t = []
            for i in range(0,num_targets):
                t.append(target(init_target_state[i][0:2], init_target_state[i][2],
                                init_target_state[i][3], vel_var, vel_var, "CONS_V"))#constant-velocity model for target motion
            A, B = t[0].constant_velocity(1E-10)#Get motion model
            Q = B.dot(np.diag([vel_var**2,vel_var**2])).dot(B.transpose())
            D11 = ((A.transpose()).dot(Q)).dot(A)
            D12 = -(A.transpose()).dot(Q)
            x_var = t[0].x_var
            y_var = t[0].y_var

            tracker_object = []
            for i in range(0,num_targets):
                tracker_object.append(EKF_tracker(init_for_smc[i], np.array(init_covariance), A,B,x_var,y_var,bearing_var))#create tracker object
                #smc_object = smc_tracker(A,B,x_var,y_var,bearing_var,1000,np.array(init_for_smc))

            s = sensor("POLICY_COMM_LINEAR")  # create sensor object (stochastic policy)
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
            normalized_innovation = []
            normalized_innovation_max = []
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
                normalized_innovation.append([])
                normalized_innovation_max.append([])
            reward = []
            episode_condition = True
            n=0
            violation = 70
            #store required information
            episode_state = []
            episode_state_out_layer = []
            episode_MLP_state = []
            episode_actions = []
            avg_uncertainty= []
            max_uncertainty = []

            delta_change_point = list(10000*np.ones([num_targets]))
            while episode_condition:

                temp_m = []
                input_state_temp = []
                for i in range(0,num_targets):
                    t[i].update_location()
                    temp_m.append(measure.generate_bearing(t[i].current_location,s.current_location))

                m.append(temp_m)
                temp_reward = []
                target_actions = []
                records = []
                Deltas = [[],[]]
                for i in range(0,num_targets):

                    ####
                    #Change points
                    #if i==0:
                     #   if n==500: t[i].current_velocity[0]+=3
                    #else:
                     #   if n == 500: t[i].current_velocity[0] -= 3

                    if n == 500: t[i].current_velocity[0] += 1
                    if n==1000: t[i].current_velocity[1]-=1

                    ####


                    tracker_object[i].update_states(s.current_location, m[-1][i])

                    #######
                    temp = ((tracker_object[i].innovation_list[-1])) ** 2 / tracker_object[i].innovation_var[-1]
                    #temp2 = ((tracker_object[i].innovation_list[-1])) ** 2 / tracker_object[i].innovation_var_max[-1]

                    #prob = (1.0 / (np.sqrt(np.pi) * bearing_var)) * np.exp(-.5 * temp[0])
                    #prob2 = (1.0 / (np.sqrt(np.pi * tracker_object[i].innovation_var_max[-1]))) * np.exp(-.5 * temp2[0])

                    #np1 = prob / (prob2 + prob)
                    # if np.isnan(np1[0][0]): np1[0][0] = 1
                    normalized_innovation[i].append(temp[0])

                    if n > 20:
                        m1 = np.mean(normalized_innovation[i][n - 10:n])
                        m2 = np.mean(normalized_innovation[i][n - 20:n - 10])
                        normalized_innovation_max[i].append(m1 / m2)

                    delta_change_point[i] += 1
                    if n > 20 and delta_change_point[i] > 50:
                        if normalized_innovation_max[i][-1] > 10:
                            # print(n)
                            tracker_object[i].p_k_k += .05 * np.diag(
                                [MAX_UNCERTAINTY, MAX_UNCERTAINTY, MAX_UNCERTAINTY, MAX_UNCERTAINTY])
                            delta_change_point[i] = 0

                    #########

                    PCRLB[i] =  np.linalg.inv(Q+(A.dot(np.linalg.inv(PCRLB[i]))).dot(A.transpose())) \
                                + (tracker_object[i].meas_vec[-1].transpose()).dot(tracker_object[i].meas_vec[-1])/bearing_var**2

                    F = np.diag(np.linalg.inv(PCRLB[i]))
                    pcrlb_x[i].append(np.sqrt(F[0]))
                    pcrlb_y[i].append(np.sqrt(F[1]))
                    pcrlb[i].append(np.sqrt(F[0]+F[1]))
                    if i==2:
                        records.append(np.sqrt(F[0]+F[1]))
                    else:
                        records.append(np.sqrt(F[0] + F[1]))

                    #normalized_innovation = (tracker_object[i].innovation_list[-1])/tracker_object[i].innovation_var[-1]
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

                    input_state = current_state
                    input_state_temp.append(input_state)  # store input-sates
                    if weight_index==0:
                        target_actions.append(s.generate_action(params,input_state,.01))
                    elif weight_index==1:
                        extra_information = s.update_location_new(params, input_state, .01,weight_index)
                    else:

                        #Delta = s.update_location_new_limit_v2(params, input_state, .01, v_max, coeff,alpha1, alpha2, alpha1_, alpha2_,weight_index)
                        Delta = s.update_location_new_limit(params, input_state, .1, v_max, coeff, alpha1, alpha2,
                                                               alpha1_, alpha2_, weight_index)

                        #Deltas[0].append(Delta[0])
                        #Deltas[1].append(Delta[1])



                    estimate = tracker_object[i].x_k_k
                    episode_state.append(input_state) ####Neeed to get modified
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
                    #innovation[i].append(normalized_innovation[0])
                    unormalized_uncertainty = np.sum(tracker_object[i].p_k_k.diagonal())
                    #if unormalized_uncertainty>MAX_UNCERTAINTY:
                    #   normalized_uncertainty = 1
                    #else:
                    #   normalized_uncertainty = (1.0/MAX_UNCERTAINTY)*unormalized_uncertainty
                    uncertainty[i].append((1.0 / MAX_UNCERTAINTY) * unormalized_uncertainty)

                #if weight_index==2:
                 #   mean_x_action = np.std(Deltas[0])
                  #  mean_y_action = np.std(Deltas[1])
                   # Delta = [mean_x_action,mean_y_action]
                    #s.sensor_actions.append(Delta)
                    #new_x = s.current_location[0] + Delta[0]
                    #new_y = s.current_location[1] + Delta[1]
                    #s.current_location = [new_x, new_y]
                    #s.historical_location.append(s.current_location)

                avg_pcrlb.append(np.mean(records))
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


                if weight_index==0:
                    #normalized_state,index_matrix1,index_matrix2,slope = \
                     #   s.update_location_decentralized_limit(target_actions,1,params,
                      #                                        v_max,coeff,alpha1,alpha2,alpha1_,alpha2_,weight_index) #Update the sensor location based on all individual actions

                    s.update_location_decentralized(target_actions,.1,params)
                    #index_matrix: an n_s \times T matrix that shows the derivative of state in the output layer to the action space in the internal-layer

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
            #sys.exit(1)
            #print(sum(reward), np.mean(pos_error))
            if (np.mean(pos_error[0]))<100 and (np.mean(pos_error[1]))<100:

                effective+=1
                for i in range(0,num_targets):
                    if not total_pos_error[i]:
                        total_pos_error[i] = pos_error[i]
                    else:
                        total_pos_error[i] = list(np.array(total_pos_error[i])+np.array(pos_error[i]))
            error_saver.append(np.mean(pos_error))
            sensor_locations[weight_index] = s.historical_location

        episode_counter+=1
        total_avg_pcrlb+= np.array(avg_pcrlb)
        print(episode_counter)
        sys.exit(1)

    sys.exit(1)
    avg_error = (np.array(total_pos_error[0])/effective+np.array(total_pos_error[1])/effective+
    np.array(total_pos_error[2]) / effective+np.array(total_pos_error[3])/effective)/4.0
    #for s in avg_error: average_error_file.write(str(s)+"\n")
    #average_error_file.close()
    #sys.exit(1)

    ss = np.array(sensor_locations[weight_index])


    xs = ss[:, 0]
    ys = ss[:, 1]
    for i in range(0,len(x_truth[0])):
        #target_file.write(str(x_truth[0][i])+"\t"+str(y_truth[0][i])+"\n")
        sensor_file.write(str(xs[i]) + "\t" + str(ys[i]) + "\n")
        pcrlb_file.write(str(avg_pcrlb[i])+"\n")

    #target_file.close()
    sensor_file.close()
    pcrlb_file.close()
    sys.exit(1)
    legends = ["Target Trajectory","Unconstrained",r"Constrained($v^o_{max}=15\frac{m}{s}$)"]
    line_style = ["-k","-b", "-r","c"]
    plt1, = plt.plot(x_truth[0], y_truth[0], line_style[0], linewidth=2)

    plts = []
    plts.append(plt1)
    for w in range(2,3):
        ss = np.array(sensor_locations[w])
        xs = ss[:,0];ys = ss[:,1]
        plt1, = plt.plot(xs,ys,line_style[w+1],linewidth=1)
        plts.append(plt1)

    plt.xlabel("X (m)", size=15)
    plt.ylabel("Y (m)", size=15)
    plt.grid(True)
    plt.legend(plts, legends)
    plt.show()






"""
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
"""



