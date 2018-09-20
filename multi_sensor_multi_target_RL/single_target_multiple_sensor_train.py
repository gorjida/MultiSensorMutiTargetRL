"""
Training for a single-target single-sensor scenario
"""

from target import target
from sensor import sensor
from measurement import measurement
from clean_tracker_agent import clean_tracker_agent
import numpy as np
import random
import sys
from scenario import scenario
from EKF_tracker import EKF_tracker
from metric import metric
from scipy.stats import norm
from sensor import get_limit
import matplotlib.pyplot as plt
from FUSION_Agent import centralized_fusion
random.seed(1)

#import sklearn.pipeline
#from sklearn.kernel_approximation import RBFSampler
#import os

from multiprocessing import Pool


def gen_learning_rate(iteration, l_max, l_min, N_max):
    if iteration > N_max: return (l_min)
    alpha = 2 * l_max
    beta = np.log((alpha / l_min - 1)) / N_max
    return (alpha / (1 + np.exp(beta * iteration)))


# Set general parameters
MAX_UNCERTAINTY = 1E9
num_states = 6
sigma_max = .001
num_episodes = []
gamma = .99
episode_length = 1500
learning_rate = 1E-3
N_max = 10000
window_size = 50
window_lag = 10
rbf_var = 1

base_path = "/dev/resoures/DeepSensorManagement-original/"

#def run(args):
if __name__=="__main__":
    num_sensors = 2
    v_max = 20
    coeff = .9
    vel_var = .001

    # create parameters for arctan limitter
    c = np.tan(coeff * np.pi / 2)
    c_ = np.tan(-coeff * np.pi / 2)
    alpha1 = (coeff * np.pi / (2 * v_max)) * (c ** 2)
    alpha2 = c - alpha1 * v_max
    alpha1_ = (coeff * np.pi / (2 * v_max)) * (c_ ** 2)
    alpha2_ = c_ + alpha1 * v_max

    #np.random.seed(process_index)
    #print("Starting Thread:" + str(process_index))

    # Random initialization of policy weights
    pre_trained_weights =  np.array([[7.8298383, 10.37983478, 3.35204969, 9.91446941,
                        -2.84844313, -13.96745699],
                      [-14.93342663, 9.27361261, -4.04988106, 0.17954491,
                        12.16543779, -4.48418833]])
    sensor_params = []
    #for sensor_index in range(0,num_sensors):sensor_params.append(np.random.normal(0, .3, [2, num_states]))
    for sensor_index in range(0,num_sensors): sensor_params.append(pre_trained_weights)

    #params[0]["weight"] = np.array([[ 1.45702249, -1.17664153, -0.11593174,  1.02967173, -0.25321044,0.09052774],
    #[ 0.67730786,  0.3213561 ,  0.99580938, -2.39007038, -1.16340594,
    #-1.77515938]])
    #params[0]["weight"] = np.array([[7.8298383, 10.37983478, 3.35204969, 9.91446941,
     #                   -2.84844313, -13.96745699],
      #                [-14.93342663, 9.27361261, -4.04988106, 0.17954491,
       #                 12.16543779, -4.48418833]])

    return_saver = []
    error_saver = []
    episode_counter = 0
    weight_saver1 = []
    weight_saver2 = []
    # for episode_counter in range(0,N_max):
    # Training parameters
    avg_reward = []
    avg_error = []
    var_reward = []
    training = True

    """
    result_folder = base_path + folder_name + "/"
    reward_file = open(
        result_folder + "reward_noise:" + str(vel_var) + "_" + str(process_index) + "_linear_6states.txt", "a")
    error_file = open(result_folder + "error_noise:" + str(vel_var) + "_" + str(process_index) + "_linear_6states.txt",
                      "a")
    error_file_median = open(
        result_folder + "error_median_noise:" + str(vel_var) + "_" + str(process_index) + "_linear_6states.txt",
        "a")
    var_file = open(result_folder + "var_noise:" + str(vel_var) + "_" + str(process_index) + "_linear_6states.txt", "a")
    var_error_file = open(
        result_folder + "var_error_noise:" + str(vel_var) + "_" + str(process_index) + "_linear_6states.txt", "a")
    weight_file = open(
        result_folder + "weight_noise:" + str(vel_var) + "_" + str(process_index) + "_linear_6states.txt", "a")
    """

    # flatten initial weight and store the values
    """
    if method == 0:
        weight = params[0]['weight']
        flatted_weights = list(weight[0, :]) + list(weight[1, :])
        temp = []
        [temp.append(str(x)) for x in flatted_weights]
        weight_file.write("\t".join(temp) + "\n")
    elif method == 1:
        weight = params[1]['weight']
        flatted_weights = list(weight[0, :]) + list(weight[1, :])
        temp = []
        [temp.append(str(x)) for x in flatted_weights]
        weight_file.write("\t".join(temp) + "\n")
    elif method == 2:
        pass
    """

    # weight = np.reshape(np.array(weights[0]), [2, 6])
    sigma = sigma_max

    mean_pos_error = []
    while episode_counter < N_max:
        # sigma = gen_learning_rate(episode_counter,sigma_max,.1,20000)
        # if episode_counter%1500==0 and episode_counter>0:
        #   sigma-= .15
        #  sigma = max(.1,sigma)
        if episode_counter % 1500 == 0 and episode_counter > 0:
            sigma = sigma_max
            sigma = max(.1, sigma)
        sigma = sigma_max
        discounted_return = np.array([])
        discount_vector = np.array([])
        # print(episodes_counter)
        scen = scenario(1, 1)
        bearing_var = 1E-1  # variance of bearing measurement
        # Randomly initialize target location
        x = 20000 * random.random() -10000  # initial x-location
        y = 20000 * random.random() -10000  # initial y-location
        xdot = 20 * random.random() -10  # initial xdot-value
        ydot = 20 * random.random() -10  # initial ydot-value
        t = [target([x,y], xdot,ydot, vel_var, vel_var,"CONS_V")]

        init_target_state = [x, y, xdot, ydot]  # initialize target state
        init_covariance = np.diag([MAX_UNCERTAINTY, MAX_UNCERTAINTY, MAX_UNCERTAINTY,
                                   MAX_UNCERTAINTY])  # initial covariance of state estimation
        s = []
        init_target_estimate_for_fusion = []
        init_target_cov_for_fusion = []
        init_target_estimate_for_fusion.append(
            np.array([x + np.random.normal(0, 5), y + np.random.normal(0, 5), np.random.normal(0, 5),
             np.random.normal(0, 5)]).reshape(4,1))
        MAX_UNCERTAINTY_FUSION = 1E12
        init_target_cov_for_fusion.append(np.diag([MAX_UNCERTAINTY_FUSION, MAX_UNCERTAINTY_FUSION, MAX_UNCERTAINTY_FUSION,
                                   MAX_UNCERTAINTY_FUSION]))

        ###
        temp_sensor_state = [[-5000,5000,0,0],[5000,-5000,0,0],[-5000,-5000,0,0]]
        for sensor_index in range(0,num_sensors):
            init_sensor_state = [10000 * random.random() - 5000, 10000 * random.random() - 5000, 3, -2]
            init_sensor_state = temp_sensor_state[sensor_index]
            temp_sensor_object = sensor("POLICY_COMM_LINEAR",init_sensor_state[0]
                                        , init_sensor_state[1])

            init_for_tracker = [x + np.random.normal(0, 5), y + np.random.normal(0, 5), np.random.normal(0, 5),
                            np.random.normal(0, 5)]

            A, B = t[0].constant_velocity(1E-10)  # Get motion model
            x_var = t[0].x_var
            y_var = t[0].y_var
            tracker_object = EKF_tracker(init_for_tracker, init_covariance, A, B, x_var, y_var,bearing_var)
            clean_agent = clean_tracker_agent([tracker_object])
            temp_sensor_object.set_tracker_objects(clean_agent)
            s.append(temp_sensor_object)

        #Finally, initialize fusion object
        fusion_agent = centralized_fusion(window_size , window_lag, MAX_UNCERTAINTY,num_sensors,init_target_estimate_for_fusion,init_target_cov_for_fusion)

        #measure = measurement(bearing_var)

        episode_condition = True
        n=0
        metric_obj = metric(1,num_sensors)
        episode_state = []
        for sensor_index in range(0,num_sensors): episode_state.append([])
        while episode_condition:
            t[0].update_location()
            #m.append(measure.generate_bearing(t.current_location, s.current_location))
            for sensor_index in range(0,num_sensors):
                s[sensor_index].gen_measurements(t, measurement(bearing_var), 1, 0)
                s[sensor_index].update_track_estimaes()

                #Move the sensor
                input_states = s[sensor_index].move_sensor(scen, sensor_params[sensor_index], v_max, coeff,
                                                           alpha1, alpha2, alpha1_, alpha2_,sigma)
                episode_state[sensor_index].append(input_states[0])
                #generate local reward
                s[sensor_index].gen_sensor_reward(MAX_UNCERTAINTY, window_size, window_lag)

            #Update estimate of global

            #if n==500: sys.exit(0)
            #fusion_agent.update_global(s,False)
            fusion_agent.update_global_memoryless(s,feedback=True)

            metric_obj.update_truth_estimate_metrics(t, s)
            metric_obj.update_truth_fusion_estimate_metrics(t,fusion_agent)

            #discount_vector = gamma * np.array(discount_vector)
            #discounted_return += (1.0 * s.reward[-1]) * discount_vector
            #new_return = 1.0 * s.reward[-1]
            #list_discounted_return = list(discounted_return)
            #list_discounted_return.append(new_return)
            #discounted_return = np.array(list_discounted_return)

            #list_discount_vector = list(discount_vector)
            #list_discount_vector.append(1)
            #discount_vector = np.array(list_discount_vector)
            n += 1
            if n > episode_length: episode_condition = False

        #sys.exit(0)
        if np.mean(metric_obj.pos_error_fusion[0])>1000: continue
        mean_pos_error.append(np.mean(metric_obj.pos_error_fusion[0]))
        if episode_counter>100: break
        print(episode_counter)
        episode_counter+=1
        continue
        #MODIFIED by HERE

        #TRAINING
        if np.mean(metric_obj.pos_error[0][0]) > 10000:
            print("Passing")
            continue
            #NO UPDATE
            #episode_condition = False
            #episode_counter -= 1

        normalized_discounted_return = discounted_return
        episode_actions = s.sensor_actions
        rate = gen_learning_rate(episode_counter, learning_rate, 1E-8, 10000)
        #total_adjustment = np.zeros(np.shape(weight))
        for e in range(0, len(episode_actions)):
            # calculate gradiant
            state = np.array(episode_state[e]).reshape(len(episode_state[e]), 1)
            predicted_action = params[0]['weight'].dot(state)
            limitted_action1, grad1 = get_limit(v_max, coeff, alpha1, alpha2, alpha1_, alpha2_,
                                                predicted_action[0][0])
            limitted_action2, grad2 = get_limit(v_max, coeff, alpha1, alpha2, alpha1_, alpha2_,
                                                predicted_action[1][0])
            predicted_action[0][0] = limitted_action1
            predicted_action[1][0] = limitted_action2
            limit_grad = np.array([grad1, grad2]).reshape([2, 1])
            gradiant = ((episode_actions[e].reshape(2, 1) - predicted_action) * limit_grad).dot(
                state.transpose()) / sigma ** 2


            if np.max(np.abs(gradiant)) > 1E2: continue  # clip large gradients
            adjustment_term = gradiant * normalized_discounted_return[e]  # an unbiased sample of return
            params[0]['weight'] += rate * adjustment_term

        return_saver.append(sum(s.reward))
        error_saver.append(np.mean(metric_obj.pos_error[0][0]))
        episode_counter+=1

        # print(len(return_saver),n)
        if episode_counter % 100 == 0 and episode_counter > 0:
            #print(episode_counter, np.mean(return_saver), sigma)
            # if episode_counter%100==0 and episode_counter>0:
            avg_reward.append(np.mean(sorted(return_saver)[0:int(.95 * len(return_saver))]))
            avg_error.append(np.mean(sorted(error_saver)[0:int(.95 * len(error_saver))]))
            var_reward.append(np.var(return_saver))
            return_saver = []
            error_saver = []
            print(avg_reward)


"""
if __name__ == "__main__":

    coeff = .9
    v_max = 15

    #p = Pool(15)
    #experiment_folder_name = "linear_policy_discrete_reward_initial_condition_limit_vmax10_coeff9_varying_var"

    #if not os.path.exists(base_path + experiment_folder_name):
     #   os.makedirs(base_path + experiment_folder_name)
    method = 0
    RBF_components = 20
    MLP_neurons = 50
    vel_var = .001

    job_args = [(method, RBF_components, MLP_neurons, i, experiment_folder_name, vel_var, coeff, v_max) for i in
                range(0, 15)]
    p.map(run, job_args)
    run(0, RBF_components, MLP_neurons, 0, experiment_folder_name)
"""





