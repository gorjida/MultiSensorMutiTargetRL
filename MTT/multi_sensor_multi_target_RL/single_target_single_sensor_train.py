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
from scipy.stats import chi2
from Metric import Metric


#import sklearn.pipeline
#from sklearn.kernel_approximation import RBFSampler
#import os

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

        plts.append(plt.plot(x,y,"x"))
        labels.append("True target "+str(index))

    #plt.show()



def gen_learning_rate(iteration, l_max, l_min, N_max):
    if iteration > N_max: return (l_min)
    alpha = 2 * l_max
    beta = np.log((alpha / l_min - 1)) / N_max
    return (alpha / (1 + np.exp(beta * iteration)))


# Set general parameters
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
RATE = 1E-1

base_path = "/dev/resoures/DeepSensorManagement-original/"


def truth_initiation(vel_var,sample_time,initial_state):
    x = 500 * random.random() - 250  # initial x-location
    y = 500 * random.random() - 250 # initial y-location
    xdot = 400 * random.random() - 200  # initial xdot-value
    ydot = 400 * random.random() - 200  # initial ydot-value

    #x = 100
    #y = 100
    #xdot = 2
    #ydot = -2
    t = target([initial_state[0], initial_state[1]], initial_state[2],
               initial_state[3], vel_var, vel_var, "CONS_V",sample_time, RATE)
    return (t)
#def run(args):

#Units are all in centimeters

def find_gate_threshold(meas_size,pg):
    for x in np.arange(0,50,.1):
        if chi2.cdf(x,meas_size)>=pg:
            return (x)
if __name__=="__main__":
#def main(mcmc,landa_in=1,pd_in=1):
    #General parameters
    T_max = 400
    target_intervals = [[0,1200],[300,1000]]
    target_intervals = [[0,200],[50,300],[120,220]]#,[100,400]]
    targets_initial_states = [[100,100,120,-120],[150,-40,-100,110],[-80,180,120,-15]]#,[-500,-500,130,160]]
    total_num_targets = len(target_intervals)
    vel_var = .01
    bearing_std = 1E-2
    range_std = 10
    vel_std = 5
    pd = .9
    pg = .999
    landa = 2
    M_logic = 3
    N_logic = 5

    M_logig_init = 2
    N_logic_init = 2
    use_vel = True
    sample_time = .06

    #create required objects
    dummy_target = truth_initiation(vel_var,sample_time,targets_initial_states[0])
    A, B = dummy_target.constant_velocity(RATE)
    scen = scenario(bearing_std,range_std,vel_std,pd,landa)
    measure = measurement(scen)
    sensor_object = sensor("POLICY_COMM_LINEAR", 0, 0,scen)

    #calculate gate threshold
    if use_vel:
        gate_threshold = find_gate_threshold(3,pg)
        measurement_size = 3
    else:
        gate_threshold = find_gate_threshold(2,pg)
        measurement_size = 2

    track_threshold = find_gate_threshold(measurement_size,.999)

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
                target_obj = truth_initiation(vel_var,sample_time,targets_initial_states[target_index])
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

        sensor_object.gen_measurements(active_targets, measure, scen.pd, scen.landa, use_vel)
        num_measurements_list.append(len(sensor_object.m[-1]))
        num_targets_list.append(scan_num_targets)
        metric = Metric(num_targets_list,truth_map,target_intervals,T_max)





    #TRACKING
    #Main objects:
    #sensor_object: has the list of measurements at each time-step

    #jpdaf_object = JPDAF_agent(tracker_object, .004, pd, 1E-5)
    #tracks_list = [] #list of all the tracks (tentative, active, terminated)
    #sensor_object.set_tracker_objects(tracks_list)

    jpdaf_object = JPDAF_agent([], gate_threshold, pd, 1E-6,(1.0/scen.volume), scen
                               , use_velocity=use_vel)
    x_truth = []
    y_truth = []
    xdot_truth = []
    ydot_truth = []

    x_estimates = []
    y_estimates = []
    xdot_estimates = []
    ydot_estimates = []

    init_estiamte = np.array([truth_map[0].target.historical_location[0][0] + np.random.normal(0, 5),
                              truth_map[0].target.historical_location[0][1] + np.random.normal(0, 5),
                              0, 0])

    error = []

    counter = 0
    max_track_id = 0

    for measurement_time_index in np.arange(0,len(sensor_object.m),1):

        #if measurement_time_index%5!=0:
         #   continue
        #if measurement_time_index==110: sys.exit(0)
        unassociated_measurements = sensor_object.update_track_estimaes(measurement_time_index)
        #if not not unassociated_measurements and measurement_time_index>0: sys.exit(1)
        initial_tracks, max_track_id = single_scan_initiation(unassociated_measurements,scen,vel_var,A,B,max_track_id,
                                                                use_vel=use_vel)

        #Track-management
        MAX_UNCERTAINTY = scen.MAX_UNCERTAINTY
        jpdaf_object = m_n_logic_management(jpdaf_object,M_logic,N_logic,
                                                    M_logig_init,N_logic_init,MAX_UNCERTAINTY,min(.7,pd-.1),scen.vel_threshold_for_static)
        jpdaf_object.tracks+= initial_tracks
        sensor_object.set_tracker_objects(jpdaf_object)
        #print(jpdaf.tracker_object.tracks[0].)


        metric.update_num_targets(jpdaf_object)
        assoc_matrix = metric.update_estimates(jpdaf_object.tracks,track_threshold,measurement_time_index)
        #if measurement_time_index==4: sys.exit(1)
        #if len(jpdaf_object.tracks)>0:
         #   if len(jpdaf_object.tracks[0].assignment)>5:
          #      if sum(jpdaf_object.tracks[0].assignment[-4:])<3: sys.exit(1)
    #return (metric)

"""
if __name__=="__main__":
    estimated_num_tracks = []
    estimated_pos_error = {}
    estimated_vel_error = {}
    #truths = {}
    #estimated_locs = {}
    #etric = main(0, landa=3, pd=.9)
    #sys.exit(1)
    for mcmc in range(0,50):
        print(mcmc)
        #metric = main(mcmc, landa=3, pd=.9)
        try:
            metric = main(mcmc,landa_in=1,pd_in=.8)
            pos_error = metric.pos_error
            vel_error = metric.vel_error
            for index in pos_error:
                if index not in estimated_pos_error:
                    estimated_pos_error[index] = []
                    estimated_vel_error[index] = []

                estimated_pos_error[index].append(pos_error[index])
                estimated_vel_error[index].append(vel_error[index])
                if not estimated_num_tracks:
                    estimated_num_tracks = [metric.active_num_targets]
                else:
                    estimated_num_tracks.append(metric.active_num_targets)

        except:
            print("There is an error...")
            mcmc-=1
            continue

    (run,scan_len) = np.shape(estimated_num_tracks)
    plt1, = plt.plot(range(1,scan_len+1),np.mean(estimated_num_tracks,axis=0),linewidth=3)
    plt2, = plt.plot(range(1, scan_len + 1), metric.num_targets_list,"r")
    plt.xlabel("Scan",size= 20)
    plt.ylabel("Number of targets",size = 20)
    plt.legend([plt1,plt2],["Estimate","Truth"])
    plt.grid(True)
    plt.show()

    #plot errors
    plts = []
    legs = []
    colors = ["b","r","k","m","g"]
    for index in estimated_pos_error:
        T_max = np.shape(estimated_pos_error[index])[-1]
        plts.append(plt.plot(range(0,T_max),np.nanmean(estimated_pos_error[index],axis=0),colors[index]))
        legs.append("Target "+str(index))
    plt.title("Position Estimation Error",size=15)
    plt.xlabel("Scan",size=20)
    plt.ylabel("RMSE (cm)",size=20)
    plt.grid(True)
    plt.legend(plts,legs)
    plt.show()

    # plot errors
    plts = []
    legs = []
    colors = ["b", "r", "k", "m", "g"]
    for index in estimated_vel_error:
        T_max = np.shape(estimated_vel_error[index])[-1]
        plts.append(plt.plot(range(0, T_max), np.nanmean(estimated_vel_error[index], axis=0), colors[index]))
        legs.append("Target " + str(index))
    plt.title("Velocity Estimation Error",size=15)
    plt.xlabel("Scan", size=20)
    plt.ylabel("RMSE (cm/sec)", size=20)
    plt.grid(True)
    plt.legend(plts, legs)
    plt.show()



    sys.exit(1)

    #plot results
    #scans = np.arange(0,len(num_measurements_list),1)
    #plt.plot(scans,num_measurements_list,"bo")
    #plt.xlabel("Measurement Scan",size=20)
    #plt.ylabel("Number of measurements",size=20)
    #plt.grid(True)
    #plt.show()

    #Plot truth and estimates
    pos = metric.estimates_pos
    plts =[]
    legends = []
    colors = ["b","r","k","m","g"]
    index = 0
    for id in pos:
        pp = np.array(pos[id])
        plts.append(plt.plot(pp[:,0],pp[:,1],colors[index]+"o"))
        legends.append("Tracker:"+str(id))
        index+=1
    plt.legend(plts,legends)
    plot_truth(truth_map)
    plt.show()
    scans = np.arange(0,len(num_measurements_list),1)
    plt1, = plt.plot(np.arange(0,len(metric.active_num_targets),1),metric.active_num_targets,"b")
    plt2, = plt.plot(np.arange(0,len(num_targets_list),1),num_targets_list,"r")
    plt.xlabel("Measurement Scan",size=20)
    plt.ylabel("Number of Targets",size=20)
    plt.grid(True)
    plt.legend([plt1,plt2],["Estimates","Truth"])
    plt.show()


"""



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





