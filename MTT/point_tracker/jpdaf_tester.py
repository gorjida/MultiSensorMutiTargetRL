
import os
import numpy as np
from EKF_tracker import *
from scenario import *
from JPDAF_agent import *
from utils import *
from evaluate import *
import matplotlib.pyplot as plt



def fetch_files(base_path,landa_in,mc_in):
    measurements = {}
    truths = {}
    for file in os.listdir(base_path):
        splits = file.replace(".txt","").split("_")
        pd = float(splits[1])
        landa = int(splits[2])
        if landa!=landa_in: continue
        mc = int(splits[3])
        if not mc in mc_in: continue
        print(file)
        if splits[0]=="truth":
            if not mc in truths: truths[mc] = {}
            if not landa in truths[mc]: truths[mc][landa] = {}
            with open(base_path+"/"+file,"r") as f:
                for index,line in enumerate(f):
                    if index ==0: continue
                    line = line.strip()
                    if line == "": continue
                    data = line.strip().split("\t")
                    scan = int(data[0])
                    target_index = int(data[1])
                    x = float(data[2])
                    y = float(data[3])
                    if not target_index in truths[mc][landa]: truths[mc][landa][target_index] = [[],[]]
                    truths[mc][landa][target_index][0].append(x)
                    truths[mc][landa][target_index][1].append(y)
        else:
            if not mc in measurements: measurements[mc] = {}
            if not landa in measurements[mc]: measurements[mc][landa] = []

            tmp_meas = []
            with open(base_path+"/"+file,"r") as f:
                for index,line in enumerate(f):
                    if index ==0: continue
                    line = line.strip()
                    if line == "":
                        measurements[mc][landa].append(tmp_meas)
                        tmp_meas = []
                        continue
                    data = line.strip().split("\t")

                    range = float(data[1])
                    azimuth = float(data[2])
                    vel = float(data[3])
                    tmp_meas.append([range,azimuth,vel])

            if not not tmp_meas: measurements[mc][landa].append(tmp_meas)

    return (truths,measurements)

def initiate_tracks(truths):

    init_state = []
    for truth_index in truths:
        init_x = truths[truth_index][0][0] + np.random.normal(5)
        init_y = truths[truth_index][1][0] + np.random.normal(5)
        init_state.append([init_x,init_y,0,0])

    return (init_state)

if __name__=="__main__":

    #load data
    landa = 10
    num_mcmc = 100
    evaluate_obj = evaluate(5)
    sensor_state = np.array([0,0,0,0])
    base_path = "/home/ali/MTT/multi_sensor_multi_target_RL/scenarios/"
    truths,meas = fetch_files(base_path,landa,np.arange(0,num_mcmc,1))
    init_covariance = np.diag([100,100,100,100])
    x_var = 1E-2
    y_var = 1E-2
    pd = .95

    bearing_std = (5/180)*np.pi
    range_std = 3
    vel_std = 5

    #scenario object
    gate_threshold = 16
    scen = scenario(bearing_std, range_std, vel_std, pd, landa)
    dummy_target = truth_initiation(x_var, 1, [0, 0, 0, 0],
                                    "CONS_V", 1E-4)
    A = dummy_target.motion_params["A"]
    B = dummy_target.motion_params["B"]
    jpdaf_object = JPDAF_agent([], gate_threshold, pd, 1E-8, 1E-6, scen
                               , use_velocity=False)

    #Assumption: a noisy estimate of targets is available
    landas = [landa]

    pos_error = np.zeros([50, 5])
    hyps = np.zeros(50)
    swaps = np.zeros(5)
    for mc in meas:
        # load data
        evaluate_obj = evaluate(5)
        for landa in landas:

            truth_for_test = truths[mc][landa]
            estimated_tracks = {}

            measurements = meas[mc][landa]
            ground_truth = truths[mc][landa]
            initial_track_states = initiate_tracks(ground_truth)
            #Form initial JPDAF tracks
            tracker_objects = []


            for i in range(0, len(initial_track_states)):
                initial_track = EKF_tracker(initial_track_states[i], np.array(init_covariance),
                                            A, B, x_var, y_var, scen, i,
                                            use_velicty=True)

                tracker_objects.append(initial_track)
            #Form JPDAF object
            jpdaf_object = JPDAF_agent(tracker_objects, gate_threshold, pd, 1E-6, (1.0 / scen.volume), scen
                                       , use_velocity=True)

            for scan,measurement in enumerate(measurements):

                #Form truth-matrix (or append nothing if the truth does not exist)
                truth_X = []
                truth_Y = []
                for truth_index in truth_for_test:
                    truth = truth_for_test[truth_index]
                    truth_X.append(truth[0][scan])
                    truth_Y.append(truth[1][scan])

                #Metric extraction
                for track_index,track in enumerate(jpdaf_object.tracks):
                    if not track_index in estimated_tracks: estimated_tracks[track_index] = [[],[]]
                    estimate = track.x_k_k
                    estimated_tracks[track_index][0].append(estimate[0][0])
                    estimated_tracks[track_index][1].append(estimate[1][0])

                evaluate_obj.add_tracks(jpdaf_object.tracks,truth_X,truth_Y)
                #Prediction-phase for all the tracks
                for track in jpdaf_object.tracks: track.predicted_state(sensor_state)
                #JPDAF-phase
                #Phase1: jpdaf_gating
                gate_map, target_measurement_score_assignment, distance_map_targets = jpdaf_object.get_gate_map(measurement)
                #Phase2: jpdaf_assignment
                track_meas_prob = jpdaf_object\
                    .target_to_measurement_probability(gate_map,target_measurement_score_assignment)

                num_hyp = jpdaf_object.num_hypotheses
                hyps[scan]+= num_hyp

                #Phase3: jpdaf_update
                jpdaf_object.update_target_states(sensor_state,
                                                  measurement, track_meas_prob, jpdaf_object.tracks)

            #plot results
            #truth_to_track_assignment = evaluate_obj.truth_to_track_id
            #for truth_id in truth_for_test:
             #   truth = truth_for_test[truth_id]
              #  plt.plot(truth[0],truth[1],linewidth=2)

            #estimates = evaluate_obj.truth_estimates
            #plt.scatter(0,0)
            #for estimate_id in estimates:
             #   estimate = np.array(estimates[estimate_id])
              #  plt.scatter(estimate[:,0],estimate[:,1])

        #plt.show()
        truth_to_track_assignment = evaluate_obj.truth_to_track_id
        this_scan_swaps = evaluate_obj.gen_track_swap(truth_to_track_assignment)
        swaps+= np.array(this_scan_swaps)
        error = np.array(evaluate_obj.truth_estimate_error)
        pos_error+= error
        #print(error[0])
        print(mc)
    pos_error/= num_mcmc
    swaps = swaps/num_mcmc
    hyps = hyps/num_mcmc

    #Plot results

    (n,m) = np.shape(pos_error)
    scans = np.arange(0, n, 1)

    plt.plot(scans,hyps,"b--",linewidth=3)
    plt.xlabel("Scan",size = 20)
    plt.ylabel("Number of Hypotheses",size = 20)
    plt.grid(True)
    plt.show()

    style = ["b-o","r-s","k-d","m-^","g--"]
    plts = []
    for index in range(0,m):
         plt.plot(scans,pos_error[:,index],style[index])

    plt.xlabel("Scan",size = 20)
    plt.ylabel("Location RMSE (m^2)",size = 20)
    plt.grid(True)
    plt.show()

