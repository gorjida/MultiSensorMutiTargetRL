import numpy as np
from EKF_tracker import EKF_tracker
import operator
from utils import *
from scenario import *
#from Cluster_agent import *
from scipy.stats import mode
from utils import *
from Distance_Cluster import *
from Track_Cluster import *


def single_scan_initiation(list_of_measuremetns,scen,
                           motion_var,motion_A,motion_B,max_id,use_vel):
    """
    Single-scan initiation based on DB-scan on the point-clouds
    :param list_of_measuremetns:
    :param scen:
    :param motion_var:
    :param motion_A:
    :param motion_B:
    :param max_id:
    :return:
    """
    if len(list_of_measuremetns)==0:
        return ([],max_id)
    #Run clustering over unassigned measurements
    max_distance = find_gate_threshold(2, .99)
    distance_cluster = Distance_Cluster(max_distance, list_of_measuremetns)
    clusters, distance_matrix = distance_cluster.run_cluster(list_of_measuremetns, Centroid_Type.max)
    filtered_clusters = distance_cluster.filter_clusters(clusters)
    init_covariance = np.diag([scen.MAX_UNCERTAINTY, scen.MAX_UNCERTAINTY
                                  , scen.MAX_UNCERTAINTY, scen.MAX_UNCERTAINTY])
    initial_tracks = []
    for cluster in filtered_clusters:
        centroid = cluster.centroid
        range = centroid.range
        azimuth = centroid.azimuth
        x = range*np.sin(azimuth)
        y = range*np.cos(azimuth)
        init_estimate = [x,y,0,0]
        initial_track = EKF_tracker(init_estimate, np.array(init_covariance),
                                    motion_A, motion_B, motion_var, motion_var, scen, max_id, use_velicty=use_vel)
        new_track_cluster = Track_Cluster(scen,list_of_measuremetns,centroid,initial_track,use_vel)
        initial_tracks.append(new_track_cluster)
        max_id+=1
    return (initial_tracks,max_id)






if __name__=="__main__":
    points = parse_ti_data("two_determ.3tils.txt")
    pp = points[501]
    scen = scenario(.1, 10, 10, .9, 1)
    for m in points[0:200]:
        c = customized_cluster_measurements(m,scen)
        print(len(c))
    #instances = cluster_based_initiation_dbscan(pp,scen,0)
    #clusters,max_id = cluster_based_initiation(pp,scen,0)





