"""
Training for a single-target single-sensor scenario
"""

from sensor_simplified import *
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
from scipy.stats import chi2
from Metric import Metric
import json
import time
from Cluster_agent import DBscan_cluster
from track_initiation import *
import operator
from utils import *
from scenario import *
from Cluster_agent import *
from multiprocessing import Pool


def plot_truth(truth_map):
    plts = []
    labels = []
    for index in truth_map:
        target = truth_map[index].target
        x = []
        y = []
        for loc in target.historical_location:
            x.append(loc[0])
            y.append(loc[1])

        plts.append(plt.plot(x,y,"o"))
        labels.append("target "+str(index))

    plt.show()



def gen_learning_rate(iteration, l_max, l_min, N_max):
    if iteration > N_max: return (l_min)
    alpha = 2 * l_max
    beta = np.log((alpha / l_min - 1)) / N_max
    return (alpha / (1 + np.exp(beta * iteration)))


# Set general parameters
MAX_UNCERTAINTY = 100
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


def truth_initiation(vel_var,sample_time,initial_state,rate):
    x = 500 * random.random() - 250  # initial x-location
    y = 500 * random.random() - 250 # initial y-location
    xdot = 5 * random.random() - 2.5  # initial xdot-value
    ydot = 5 * random.random() - 2.5  # initial ydot-value

    #x = 100
    #y = 100
    #xdot = 2
    #ydot = -2
    t = target([initial_state[0], initial_state[1]], initial_state[2],
               initial_state[3], vel_var, vel_var, "CONS_V",sample_time,rate)
    return (t)
#def run(args):

#Units are all in centimeters

def find_gate_threshold(meas_size,pg):
    for x in np.arange(0,50,.1):
        if chi2.cdf(x,meas_size)>=pg:
            return (x)


def raw_data_reader(file_path):
    raw_data = json.load(open(file_path, "r"))["method_calls"]
    num_scans = len(raw_data)

    moving_measurements = []
    static_measurements = []
    moving_measurements_cartz = []
    static_measurements_cartz = []
    for scan_number,scan_data in enumerate(raw_data):

        moving_data = scan_data['values_in_keyword']['movingTargets']
        static_data = scan_data['values_in_keyword']['staticTargets']

        mv = []
        stc = []
        mv_cartz = []
        stc_cartz = []
        for m in moving_data:
            if not m:
                mv.append([])
                mv_cartz.append([])
                continue

            vel = m[0]*100
            azimuth = m[2]
            range = m[3]*100
            #convert everything to CM
            mv.append([range,azimuth,vel])

            mv_cartz.append([range*np.cos(azimuth),range*np.sin(azimuth),vel])

        #if scan_number==49: print(mv_cartz)
        #the same for static measurements
        for m in static_data:
            if not m:
                stc.append([])
                stc_cartz.append([])
                continue

            vel = m[0]*100
            azimuth = m[2]
            range = m[3]*100
            stc.append([range,azimuth,vel])
            stc_cartz.append([range*np.cos(azimuth),range*np.sin(azimuth),vel])

        moving_measurements.append(mv)
        moving_measurements_cartz.append(mv_cartz)
        static_measurements.append(stc)
        static_measurements_cartz.append(stc_cartz)

    return (moving_measurements,moving_measurements_cartz,static_measurements,static_measurements_cartz)

def data_visualizer(data_cartz):
    for scan in data_cartz:
        X = []; Y = []
        [X.append(d[0]) for d in scan]
        [Y.append(d[1]) for d in scan]
        plt.plot(X,Y,"bo")
        plt.xlabel("X (cm)",size = 20)
        plt.ylabel("Y (cm)", size = 20)
        plt.pause(.01)
        plt.clf()
    plt.show()

def gen_measurement_no_vel(stc_raw,mv_raw,scan,cluster_agent,use_vel):
    full_data = []
    all_data = stc_raw[scan] + mv_raw[scan]
    if len(all_data)>0:
        labels, instances = cluster_agent.cluster_data(all_data, 1,use_vel=use_vel)

        for x in instances: full_data.append(np.array(x))
    # Set measurements in sensor-object
    return (full_data)


def gen_measurement(stc_raw,mv_raw,scan,scen):
    """
    Generating data at the current scan
    :param stc_raw:
    :param mv_raw:
    :param scan:
    :param cluster_agent:
    :return:
    """
    static_data = stc_raw[scan]
    #if len(static_data) != 0:
     #   labels, instances = cluster_agent.cluster_data(static_data, 1,True)
    #else:
     #   instances = []
    mod_stc_data = static_data
    mv_data = mv_raw[scan]
    #if len(mv_data) !=0:
     #   labels, instances = cluster_agent.cluster_data(mv_data, 1)
   # else:
    #    instances = []
    mod_mv_data = mv_data
    full_data_mv = []
    full_data_stc = []
    for x in mod_stc_data: full_data_stc.append(np.array(x))
    for x in mod_mv_data:
        #Check for boundary
        X = x[0]*np.sin(x[1])
        Y = x[0]*np.cos(x[1])
        if X>scen.x_max or X<scen.x_min or Y>scen.y_max or Y<scen.y_min: continue
        full_data_mv.append(np.array(x))
    # Set measurements in sensor-object
    return (full_data_mv,full_data_stc)

if __name__=="__main__":

    # General parameters
    #T_max = 300
    #target_intervals = [[0, 1200], [300, 1000], [700, 900]]
    #target_intervals = [[0, 100], [50, 200], [10, 300]]
    targets_initial_states = [[10, 10, 5, 5]]
    #total_num_targets = len(target_intervals)
    vel_var = 1
    bearing_std = (1*np.pi)/180
    range_std = 10
    vel_std = 100
    pd = .9
    pg = .999
    landa = 2
    M_logic = 3
    N_logic = 5
    M_logic_init = 2
    N_logic_init = 2
    sample_time = .06
    use_vel = True
    gate_threshold = find_gate_threshold(3, pg)
    #gate_threshold = 50
    # create required objects
    dummy_target = truth_initiation(vel_var, sample_time, targets_initial_states[0],1E-10)
    A, B = dummy_target.constant_velocity(1E-10)
    scen = scenario(bearing_std, range_std, vel_std, pd, landa)
    #Extract point-clouds
    sensor_object = sensor_simplified("POLICY_COMM_LINEAR", 0, 0, scen)
    #Load raw-data
    file_path = "/home/ali/SharedFolder/tracker_input/tracker_input/data_logger._Logger_set000.json"
    mv_raw, mv_cartz, stc_raw, stc_cartz = raw_data_reader(file_path)
    num_scans = len(mv_raw)
    max_track_id = 0
    num_tracks  = []
    for scan in range(0,num_scans):
        #print(scan)
        #Get all the measurements (both static and large movements)
        full_data_mv,full_data_stc = gen_measurement(stc_raw,mv_raw,scan,scen)
        #Append data to sensor-object
        sensor_object.m_mv.append(full_data_mv)
        sensor_object.m_stc.append(full_data_stc)
        #Update estimates

        large_mv_track_to_meas_assignment, static_track_to_meas_gate, unassigned_measurements = sensor_object.nearest_neighbor_association_Eddy(scan,gate_threshold,use_vel=True)

        sensor_object.update_track_estimaes_eddy(scan,large_mv_track_to_meas_assignment,static_track_to_meas_gate)
        #unassociated_measurements = filter_measurements_for_initiation(unassociated_measurements,True,scen)

        initial_tracks, max_track_id = single_scan_initiation_eddy(unassigned_measurements, scen, vel_var, A, B,max_track_id)
        #if len(initial_tracks)>0: sys.exit(1)
        tracker_obj = []
        for track in initial_tracks: sensor_object.tracker_object.append(track)
        #sensor_object.set_tracker_objects(tracker_obj)
        MAX_UNCERTAINTY = scen.MAX_UNCERTAINTY

        trackers = m_n_logic_management(sensor_object.tracker_object, M_logic, N_logic,
                                                    M_logic_init, N_logic_init,
                                        MAX_UNCERTAINTY, min(.7, pd - .1),scen.vel_threshold_for_static,scan)
        #jpdaf_object.tracks += initial_tracks
        sensor_object.set_tracker_objects(trackers)
        tmp = 0
        for track in sensor_object.tracker_object:
            if track.track.status==1: tmp+=1
        num_tracks.append(tmp)
        if scan==300: sys.exit(1)
