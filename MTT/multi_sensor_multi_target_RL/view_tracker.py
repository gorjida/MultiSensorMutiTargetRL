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


def gen_measurement(stc_raw,mv_raw,scan,cluster_agent):

    static_data = stc_raw[scan]
    if len(static_data) != 0:
        labels, instances = cluster_agent.cluster_data(static_data, 1,True)
        # cluster_agent.plot_cluster(stc_cartz[scan])
    else:
        instances = []
    mod_stc_data = instances

    mv_data = mv_raw[scan]
    if len(mv_data) !=0:
        labels, instances = cluster_agent.cluster_data(mv_data, 1)
    else:
        instances = []
    mod_mv_data = instances


    full_data_mv = []
    full_data_stc = []
    for x in mod_stc_data: full_data_stc.append(np.array(x))
    for x in mv_data: full_data_mv.append(np.array(x))
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
    bearing_std = (10*np.pi)/180
    range_std = 100/3.46
    vel_std = 100
    pd = .9
    pg = .999
    landa = 2
    M_logic = 3
    N_logic = 5
    M_logic_init = 2
    N_logic_init = 2
    sample_time = .05
    use_vel = True
    gate_threshold = find_gate_threshold(3, pg)
    # create required objects
    dummy_target = truth_initiation(vel_var, sample_time, targets_initial_states[0],1E-10)
    A, B = dummy_target.constant_velocity(1E-10)
    scen = scenario(bearing_std, range_std, vel_std, pd, landa)

    #Extract point-clouds

    points = parse_ti_data("two_determ.3tils.txt")
    sensor_object = sensor_simplified("POLICY_COMM_LINEAR", 0, 0, scen)
    num_scans = len(points) #Total number of scans
    max_id = 0
    test = []
    for scan in range(0,num_scans):
        current_point = points[scan]
        sensor_object.m.append(current_point)
        #Association using nearest-neighbor
        measurement_to_track_association,track_to_meas_map,association_map \
            = sensor_object.nearest_neighbor_association(scan,gate_threshold)

        #print(association_map)
        #Update tracks

        sensor_object.update_track_estimaes(scan,track_to_meas_map)
        #Form unassigned measurements
        unassociated_measurements = []
        for meas_index in measurement_to_track_association:
            if measurement_to_track_association[meas_index]==-1:
                unassociated_measurements.append(current_point[meas_index])

        #Track initiation
        #if len(unassociated_measurements)>0 and scan>5: sys.exit(0)
        #if scan==3: sys.exit(1)
        initial_tracks, max_track_id = \
               single_scan_initiation_dbscan(unassociated_measurements, scen, vel_var, A, B, 0)
        for track in initial_tracks: sensor_object.tracker_object.append(track)
        if len(initial_tracks)>0: sys.exit(1)
        try:
            a = sensor_object.tracker_object[0].track.x_k_k[0:2]
            test.append([a[0][0],a[1][0]])
        except:
            pass
        #sys.exit(1)
    file_path = "/home/ali/SharedFolder/tracker_input/tracker_input/data_logger._Logger_set000.json"
    mv_raw, mv_cartz, stc_raw, stc_cartz = raw_data_reader(file_path)
    #cluster_agent = DBscan_cluster(3, cov=cluster_cov, include_vel=True)
    #num_scans = len(mv_raw)


    sensor_object = sensor("POLICY_COMM_LINEAR", 0, 0, scen)

    #Covariance for clustering
    range_var_for_cluster = 5**2
    angle_var_for_cluster = (30*np.pi/180)**2
    vel_var_for_cluster = .1**2

    range_var_for_cluster = 10**2
    angle_var_for_cluster = 10**2
    vel_var_for_cluster = .1**2

    measurement_size_for_clustering = 3
    if measurement_size_for_clustering==3:
        cluster_cov = np.diag([range_var_for_cluster,angle_var_for_cluster,vel_var_for_cluster])
    else:
        cluster_cov = np.diag([range_var_for_cluster, angle_var_for_cluster])


    ######
    # FOR TEST
    """
    num_clusters_per_scan = []
    for scan in range(0,num_scans):
        print(scan)
        all_data = stc_raw[scan] + mv_raw[scan]
        if not all_data: continue
        all_data = np.array(all_data)[:,0:2]
        labels, inst = cluster_agent.cluster_data(all_data, 1)
        num_clusters_per_scan.append(len(inst))
        if scan==200: sys.exit(1)
    ######
    """

    max_track_id = 0

    jpdaf_object = JPDAF_agent([], gate_threshold, pd, 1E-8, (1.0 /scen.volume), scen
                               , use_velocity=use_vel)

    num_active_tracks_per_scan = []


    for scan in range(0,num_scans):
        #print(scan)
        #Get all the measurements (both static and large movements)
        full_data_mv,full_data_stc = gen_measurement(stc_cartz,mv_cartz,scan,cluster_agent)
        #Append data to sensor-object
        sensor_object.m_mv.append(full_data_mv)
        sensor_object.m_stc.append(full_data_stc)
        #Update estimates

        unassociated_measurements = sensor_object.update_track_estimaes(scan)

        unassociated_measurements = filter_measurements_for_initiation(unassociated_measurements,True,scen)
        if len(unassociated_measurements)>0:
            print("Track initialized...")
        initial_tracks, max_track_id = single_scan_initiation(unassociated_measurements, scen, vel_var, A, B,max_track_id,use_vel=use_vel)
        MAX_UNCERTAINTY = scen.MAX_UNCERTAINTY


        jpdaf_object = m_n_logic_management(jpdaf_object, M_logic, N_logic,
                                                    M_logic_init, N_logic_init, MAX_UNCERTAINTY, min(.7, pd - .1),scen.vel_threshold_for_static,scan)
        jpdaf_object.tracks += initial_tracks
        sensor_object.set_tracker_objects(jpdaf_object)

        n = 0
        for track in jpdaf_object.tracks:
            if track.status==1: n+=1

        num_active_tracks_per_scan.append(n)
        print(scan,n)

        if scan==70: sys.exit(1)
        #if scan==126: sys.exit(1)