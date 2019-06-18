from utils import *
import numpy as np
from sensor_simplified import sensor_simplified
from Distance_Cluster import *
from track_initiation import *
from track_management import *

def main(points):
#if __name__=="__main__":
    base_path = "/home/ali/SharedFolder/view_meas_June5th/"
    #List of parameters
    vel_var = 1 #noise on the motion-model
    bearing_std = (10 * np.pi) / 180
    range_std = 100 / 3.46
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
    # create target-object
    dummy_target = truth_initiation(vel_var,sample_time, [0,0,0,0],"CONS_V", 1E-10)
    A = dummy_target.motion_params["A"]
    B = dummy_target.motion_params["B"]
    #create scenario object
    scen = scenario(bearing_std, range_std, vel_std, pd, landa)
    #create sensor-object
    sensor_object = sensor_simplified("POLICY_COMM_LINEAR", 0, 0, scen)

    #Read data
    #points = parse_ti_data(base_path, "two_stop.txt")
    scans = len(points)
    max_distance = find_gate_threshold(2, .99)
    max_id = 0

    X = []
    Y = []

    estimated_tracks = {}
    scan_based_tracks = []
    for scan in range(0,scans):
        cloud = points[scan]
        #if len(cloud)==0: continue
        sensor_object.m.append(np.array(cloud))
        track_meas_assignment,track_to_meas_gate,unassigned_meas = sensor_object.nearest_neighbor_association(scan,gate_threshold,True)
        sensor_object.set_measurements_to_tracks(cloud,track_meas_assignment,track_to_meas_gate)
        #Update states
        sensor_object.update_track_estimaes_point_cloud()
        #if scan==1000: sys.exit(1)

        initial_tracks,max_id = single_scan_initiation(unassigned_meas, scen,vel_var, A, B, max_id, True)

        for track in initial_tracks: sensor_object.tracker_object.append(track)


        list_of_new_tracks = m_n_logic_management(sensor_object.tracker_object,M_logic,N_logic,M_logic_init,N_logic_init)
        sensor_object.tracker_object = list_of_new_tracks
        #if len(initial_tracks) > 0: sys.exit(1)
        tmp_scan_estimates = []
        for id,track in enumerate(sensor_object.tracker_object):
            if track.is_active!=1: continue
            track_id = track.track.track_id
            if not track_id in estimated_tracks: estimated_tracks[track_id] = [[],[]]
            x = track.track.x_k_k[0][0]
            y = track.track.x_k_k[1][0]
            estimated_tracks[track_id][0].append(x)
            estimated_tracks[track_id][1].append(y)
            tmp_scan_estimates.append([[x,y],track.track.track_id])
        #if len(sensor_object.tracker_object)==0 and scan>50: sys.exit(1)
        #if scan ==180: sys.exit(1)
        scan_based_tracks.append(tmp_scan_estimates)

    return (estimated_tracks,scan_based_tracks)