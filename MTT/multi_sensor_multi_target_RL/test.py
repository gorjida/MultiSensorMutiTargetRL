"""
Training for a single-target single-sensor scenario
"""

from target import target,truth
from sensor import sensor
from measurement import measurement
from clean_tracker_agent import clean_tracker_agent
import numpy as np
import random
import sys
from scenario import scenario
from JPDAF_agent import JPDAF_agent
from track_initiation import single_scan_initiation
from track_management import m_n_logic_management
from EKF_tracker import EKF_tracker
from metric import metric
from scipy.stats import norm
from sensor import get_limit
import matplotlib.pyplot as plt
from Metric import Metric


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
MAX_UNCERTAINTY = 1E6
num_states = 6
sigma_max = 2
num_episodes = []
gamma = .99
episode_length = 1500
learning_rate = 1E-3
N_max = 10000
window_size = 50
window_lag = 10
rbf_var = 1

base_path = "/dev/resoures/DeepSensorManagement-original/"

def truth_initiation(vel_var):
    x = 200 * random.random() - 100  # initial x-location
    y = 200 * random.random() - 100  # initial y-location
    x = 100
    y = 100
    xdot = 400 * random.random() - .2  # initial xdot-value
    ydot = .2 * random.random() - .2  # initial ydot-value


    x = 100
    y = 100
    xdot = 120
    ydot = -120
    t = target([x, y], xdot, ydot, vel_var, vel_var, "CONS_V",.06,1E-6)
    return (t)
#def run(args):
if __name__=="__main__":

    #General parameters
    T_max = 200
    target_intervals = [[0,1200],[300,1000],[700,900]]
    target_intervals = [[0,1500]]
    total_num_targets = len(target_intervals)
    vel_var = .01
    bearing_std = 1E-2
    range_std = 10
    vel_std = 5
    pd = 1
    landa = 0
    M_logic = 3
    N_logic = 5

    rmse = np.zeros([T_max])
    num = 0
    for mcmc in range(0,50):
        #create required objects
        dummy_target = truth_initiation(vel_var)
        A, B = dummy_target.constant_velocity(1E-10)
        scen = scenario(bearing_std,range_std,vel_std,pd,landa)
        measure = measurement(scen)
        sensor_object = sensor("POLICY_COMM_LINEAR", 0, 0,scen)


        num_targets_list = []
        num_measurements_list = []
        truth_index = -1
        truth_map = {}
        for step in range(0,T_max):

            if len(num_targets_list)==0:
                scan_num_targets = 0
            else:
                scan_num_targets = num_targets_list[step-1]
            #create ground-truths
            for target_index in range(0,total_num_targets):
                if step==target_intervals[target_index][0]:
                    #Truth initiation
                    target_obj = truth_initiation(vel_var)
                    scan_num_targets+=1

                    truth_map[target_index] = truth(target_obj)
                elif step==target_intervals[target_index][1]:
                    #Truth death
                    scan_num_targets-=1
                    truth_map[target_index].status = False
                elif (step>target_intervals[target_index][0]
                    and step<target_intervals[target_index][1]):
                    #update the truth
                    truth_map[target_index].target.update_location()

            #generate measurements at this step
            active_targets = []
            [active_targets.append(truth_map[index].target) for index in truth_map if truth_map[index].status]

            sensor_object.gen_measurements(active_targets, measure, scen.pd, scen.landa, True)
            num_measurements_list.append(len(sensor_object.m[-1]))
            num_targets_list.append(scan_num_targets)


        #Tracking with EKF
        metric = Metric(num_targets_list, truth_map,target_intervals,T_max)
        init_estiamte = np.array([truth_map[0].target.historical_location[0][0]+np.random.normal(0,5),
                         truth_map[0].target.historical_location[0][1]+np.random.normal(0,5),
                         0,0])
        init_cov = np.diag([MAX_UNCERTAINTY,MAX_UNCERTAINTY,MAX_UNCERTAINTY,MAX_UNCERTAINTY])
        tracker_obj = EKF_tracker(init_estiamte,init_cov,A,B,vel_var,vel_var,scen,0,use_velicty=True)

        #Tracking
        x_estimates = []
        y_estimates = []
        xdot_estimates = []
        ydot_estimates = []

        x_truth = []
        y_truth = []
        xdot_truth = []
        ydot_truth = []
        error = []
        tracker_obj.status = 1
        for step in range(0, T_max):
            if step==0: sys.exit(0)
            tracker_obj.update_states(np.array([0,0,0,0]),sensor_object.m[step][0])

            #sys.exit(1)
            est = tracker_obj.x_k_k
            x_estimates.append(est[0][0])
            y_estimates.append(est[1][0])
            xdot_estimates.append(est[2][0])
            ydot_estimates.append(est[3][0])

            truth_loc = truth_map[0].target.historical_location[step]
            truth_vel = truth_map[0].target.historical_velocity[step]
            x_truth.append(truth_loc[0])
            y_truth.append(truth_loc[1])
            xdot_truth.append(truth_vel[0])
            ydot_truth.append(truth_vel[1])

            e = np.array([x_truth[-1]-x_estimates[-1],
                          y_truth[-1]-y_estimates[-1]])
            error.append(np.linalg.norm(e))

            association_matrix = metric.update_estimates([tracker_obj],18.99,step)
            #sys.exit(1)


        if np.mean(error)<100:
            rmse+= np.array(error)
            num+=1
            print(num)


    plt.plot(rmse/num)
    plt.show()