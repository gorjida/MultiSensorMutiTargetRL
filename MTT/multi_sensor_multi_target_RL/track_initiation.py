import numpy as np
from EKF_tracker import EKF_tracker
import operator
from utils import *
from scenario import *
from Cluster_agent import *
from scipy.stats import mode

class Custom_Cluster:
    def __init__(self,r,theta,vel):
        self.centroid = [r*np.cos(theta),r*np.sin(theta)]
        self.centroid_polar = [r,theta]
        self.vels = [vel]

class Cluster:
    def __init__(self,measurements):
        self.measurements = np.array(measurements)
        self.initiate_eddy()
        self.num_assigned_measurements = [len(measurements)]
        self.num_measurements_in_gate = [len(measurements)]
        self.alpha = .01
        self.association = []

    def initiate(self):
        meas = np.mean(self.measurements[0:2],axis=0)
        vel = np.mean(self.measurements[:,-1])
        self.centroid_polar = [meas[0],meas[1],vel]
        x = meas[0]*np.cos(meas[1])
        y = meas[0]*np.sin(meas[1])
        self.centroid_cartz = [x,y,vel]

        #Calculate sample-covariance
        sample_cov = np.cov(self.measurements.transpose())
        self.dispersion = sample_cov + np.diag([1**2,(10/180)*np.pi,5**2])

    def initiate_eddy(self):
        self.centroid_polar = self.measurements
        x = self.measurements[0] * np.sin(self.measurements[1])
        y = self.measurements[0] * np.cos(self.measurements[1])
        self.centroid_cartz = [x, y, self.measurements[2]]

        self.dispersion = np.diag([0,0,0])


    def update_state(self,scen,measurements,sensor_state):
        """
        Set new measurments and update number of assigned measurements and dispesion matrix for this cluster
        :param measurements:
        :return:
        """


        #print(np.shape(np.array(measurements)))

        mean_meas = np.mean(np.array(measurements),axis=0)
        NA = len(measurements)
        print(NA)

        (m,n) = np.shape(self.track.S_k)
        if m==3:
            measurement_noise_cov = \
                np.diag([scen.range_std**2,scen.bearing_std**2
                            ,scen.vel_std**2])/NA
        else:
            measurement_noise_cov = \
                np.diag([scen.range_std ** 2, scen.bearing_std ** 2]) / NA

        self.track.S_k += measurement_noise_cov
        self.track.update_states(sensor_state, np.array(mean_meas))

    def update_state_with_static(self,scen,measurement,sensor_state):
        """
        Set new measurments and update number of assigned measurements and dispesion matrix for this cluster
        :param measurements:
        :return:
        """

        (m, n) = np.shape(self.track.S_k)
        if m==3:
            measurement_noise_cov = \
                np.diag([scen.range_std**2,scen.bearing_std**2
                            ,scen.vel_std**2])
        else:
            measurement_noise_cov = \
                np.diag([scen.range_std ** 2, scen.bearing_std ** 2])


        self.track.S_k += measurement_noise_cov

    def update_state_with_no_assignment(self,scen,measurement,sensor_state):
        """
        Set new measurments and update number of assigned measurements and dispesion matrix for this cluster
        :param measurements:
        :return:
        """
        (m, n) = np.shape(self.track.S_k)
        if m==3:
            measurement_noise_cov = \
                np.diag([scen.range_std**2,scen.bearing_std**2
                            ,scen.vel_std**2])
        else:
            measurement_noise_cov = \
                np.diag([scen.range_std ** 2, scen.bearing_std ** 2])

        self.track.S_k += measurement_noise_cov
        self.track.x_k_k = self.track.x_k_km1
        self.track.p_k_k = self.track.p_k_km1



    def update_with_new_measurements(self,scen,measurements,sensor_state):
        """
        Set new measurments and update number of assigned measurements and dispesion matrix for this cluster
        :param measurements:
        :return:
        """
        measurement_noise_cov = \
            np.diag([scen.range_std**2,scen.bearing_std**2
                        ,scen.vel_std**2])
        self.measurements = measurements
        NA = len(measurements)

        self.num_assigned_measurements.append(len(measurements))
        if NA>0:
            N_hat = self.num_measurements_in_gate[-1]
            if N_hat>1:
                detection_ratio = (N_hat - NA) / ((N_hat - 1) * NA)
            else:
                detection_ratio = 1
            if len(measurements)>1:
                sample_cov = np.cov(self.measurements.transpose())
                self.dispersion = (1-self.alpha)*self.dispersion + \
                                self.alpha*sample_cov
            #Update centroid
            meas = np.mean(self.measurements, axis=0)
            #vel = np.median(self.measurements[:, -1])
            self.centroid_polar = meas
            x = meas[0] * np.cos(meas[1])
            y = meas[0] * np.sin(meas[1])
            self.centroid_cartz = [x, y, meas[-1]]
            #Now, update with the new centroid
            self.track.S_k+= measurement_noise_cov/NA + self.dispersion*detection_ratio
            self.track.update_states(sensor_state,np.array(self.centroid_polar))
        else:
            #Other actions
            self.track.x_k_k = self.track.x_k_km1
            self.track.p_k_k = self.track.p_k_km1
            pass

    def set_track(self,track):
        self.track = track






def _to_cart(meas):
    conv_meas = [meas[0]*np.cos(meas[1]),meas[0]*np.sin(meas[1]),meas[2]]
    return (conv_meas)



def filter_measurements_for_initiation(list_of_measurements,user_vel,scen):
    filtered_list = []
    if user_vel:
        for x in list_of_measurements:
            #Each term has two dimensions: [[range,angle,average_vel],[min_val,max_val,avg_vel]]
            vels = [x[-1]]
            logics = []
            [logics.append(np.abs(y)>scen.vel_threshold) for y in vels]
            if sum(logics)>0:
                filtered_list.append(x)
    else:
        filtered_list = list_of_measurements

    return (filtered_list)

def single_scan_initiation(list_of_measuremetns,scen,
                           motion_var,motion_A,motion_B,max_id,use_vel = False,init_estimate=None):

    initial_tracks = []
    for m in list_of_measuremetns:
        #mod_m = m[0,:]
        max_id+=1
        range = m[0]
        azimuth = m[1]
        #if use_vel: velocity = mod_m[]

        #if init_estimate is None:

        init_x = range*np.cos(azimuth)
        init_y = range*np.sin(azimuth)
        #CHANGED
        init_x = range
        init_y = azimuth
        init_x_dot = 0
        init_y_dot = 0
        init_estimate = [init_x,init_y,init_x_dot,init_y_dot]

        init_covariance = np.diag([scen.MAX_UNCERTAINTY,scen.MAX_UNCERTAINTY
                                      ,scen.MAX_UNCERTAINTY,scen.MAX_UNCERTAINTY])

        initial_track = EKF_tracker(init_estimate, np.array(init_covariance),
                    motion_A, motion_B, motion_var, motion_var, scen, max_id,use_velicty=use_vel)
        initial_tracks.append(initial_track)

    return (initial_tracks,max_id)




def single_scan_initiation_dbscan(list_of_measuremetns,scen,
                           motion_var,motion_A,motion_B,max_id):
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
    if not list_of_measuremetns:
        return ([],max_id)
    init_covariance = np.diag([scen.MAX_UNCERTAINTY, scen.MAX_UNCERTAINTY
                                  , scen.MAX_UNCERTAINTY, scen.MAX_UNCERTAINTY])
    clusters = cluster_based_initiation_dbscan(list_of_measuremetns,scen)
    for id in clusters:
        c = clusters[id]
        init_estimate = [c.centroid_cartz[0],
                         c.centroid_cartz[1],0,0]
        initial_track = EKF_tracker(init_estimate, np.array(init_covariance),
                                    motion_A, motion_B, motion_var, motion_var, scen, max_id)
        c.set_track(initial_track)
        max_id+=1

    initial_tracks = []
    for x in clusters: initial_tracks.append(clusters[x])
    return (initial_tracks,max_id)

def single_scan_initiation_eddy(list_of_measuremetns,scen,
                           motion_var,motion_A,motion_B,max_id):
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
    if not list_of_measuremetns:
        return ([],max_id)
    init_covariance = np.diag([scen.MAX_UNCERTAINTY, scen.MAX_UNCERTAINTY
                                  , scen.MAX_UNCERTAINTY**2, scen.MAX_UNCERTAINTY**2])
    clusters = []
    for meas_index,m in enumerate(list_of_measuremetns):
        clusters.append(Cluster(m))
    for c in clusters:
        #c = clusters[id]
        init_estimate = [c.centroid_cartz[0],
                         c.centroid_cartz[1],0,0]
        initial_track = EKF_tracker(init_estimate, np.array(init_covariance),
                                    motion_A, motion_B, motion_var, motion_var, scen, max_id, use_velicty=True)
        c.set_track(initial_track)
        max_id+=1

    initial_tracks = []
    for c in clusters: initial_tracks.append(c)
    return (initial_tracks,max_id)


def cluster_based_initiation_dbscan(list_of_measuremetns,scen):
    cluster_obj = DBscan_cluster(scen.clustering_max_meas_distance,vel_distance=scen.clustering_max_vel_distance)
    labels = cluster_obj.cluster_data(list_of_measuremetns,scen.clustering_min_num_points)
    cluster_instances = {}
    for idx,label in enumerate(labels):
        if label == -1: continue
        if label not in cluster_instances: cluster_instances[label] = []
        cluster_instances[label].append(list_of_measuremetns[idx])

    filtered_clusters = {}
    for cluster_id in cluster_instances:
        c = Cluster(cluster_instances[cluster_id])
        #remove very low velocities
        if np.abs(c.centroid_polar[-1])<scen.clustering_min_vel_threshold: continue
        filtered_clusters[cluster_id] = c
    return (filtered_clusters)

def customized_cluster_measurements(list_of_measurements,scen):

    clusters = []

    for m in list_of_measurements:
        cartz = _to_cart(m)
        if len(clusters)==0:

            init_cluster = Custom_Cluster(m[0],m[1],m[2])
            clusters.append(init_cluster)
        else:
            score_map = {}
            for index,existing_cluster in enumerate(clusters):
                #check for velocity
                vmin = np.min(existing_cluster.vels)
                vmax = np.max(existing_cluster.vels)
                vel_dist_max = np.abs(m[-1]-vmax)
                vel_dist_min = np.abs(m[-1]-vmin)
                if vel_dist_max<scen.clustering_max_vel_distance \
                        and vel_dist_min<scen.clustering_max_vel_distance:
                    #Check for distance
                    distance = np.linalg.norm(np.array(cartz[0:2])-np.array([existing_cluster.centroid]))
                    if distance<scen.clustering_max_meas_distance:
                        score_map[index] = distance
                else:
                    score_map[index] = 1E10
            if not score_map: continue
            sorted_score_map = sorted(score_map.items(),key=operator.itemgetter(1))
            if sorted_score_map[0][-1]==1E10:
                #No assignment (form a new cluster)
                init_cluster = Custom_Cluster(m[0], m[1], m[2])
                clusters.append(init_cluster)
            else:
                assigned_index = sorted_score_map[0][0]
                assigned_cluster = clusters[assigned_index]
                num_assignments = len(assigned_cluster.vels)
                current_centroid = np.array(assigned_cluster.centroid)
                current_centroid_polar = np.array(assigned_cluster.centroid_polar)
                current_point = np.array(cartz[0:2])
                current_point_polar = np.array(m[0:2])
                assigned_cluster.centroid_polar = (num_assignments*current_centroid_polar+current_point_polar)/(num_assignments+1)
                assigned_cluster.centroid = (num_assignments*current_centroid+current_point)/(num_assignments+1)
                assigned_cluster.vels.append(m[-1])

    #Final check for clusters
    out_clusters = []
    for c in clusters:
        mod_vel = mode(c.vels)[0][0]
        if len(c.vels)>=scen.clustering_min_num_points and np.abs(mod_vel)>=scen.clustering_min_vel_threshold:
            out_clusters.append(c)

    return (out_clusters)

if __name__=="__main__":
    points = parse_ti_data("two_determ.3tils.txt")
    pp = points[501]
    scen = scenario(.1, 10, 10, .9, 1)
    for m in points[0:200]:
        c = customized_cluster_measurements(m,scen)
        print(len(c))
    #instances = cluster_based_initiation_dbscan(pp,scen,0)
    #clusters,max_id = cluster_based_initiation(pp,scen,0)





