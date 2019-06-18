import numpy as np
from utils import *
from enum import Enum
from scenario import static_params

class Centroid_Type(Enum):
    max = "MAX"
    weighted_average = "W_AVG"


class centroid:
    def __init__(self,range,azimuth,vel):
        if type(range)==list: range = range[0]
        if type(azimuth)==list: azimuth = azimuth[0]
        if type(vel)==list: vel = vel[0]
        self.vel = vel
        self.range = range
        self.azimuth = azimuth

class cluster_point:
    def __init__(self,max_distance):
        self.point_ids = []
        self.num_points = 0
        self.max_distance = max_distance
        self.total_snr = 0
        self.centroid = []

    def add_point(self,point_id,point_snr):
        self.point_ids.append(point_id)
        self.total_snr+= point_snr
        self.num_points+=1

    def check_point_assignment(self,point_id,distance_matrix):
        for cluster_point_id in self.point_ids:
            if cluster_point_id<point_id:
                distance = distance_matrix[cluster_point_id,point_id]
            else:
                distance = distance_matrix[point_id,cluster_point_id]

            if distance<self.max_distance:
                return (True)
        return (False)

    def set_centroid(self,points,type):
        raw_indexes = self.point_ids
        points = np.array(points)[raw_indexes,:]
        snrs = points[:,-1]
        velocities = points[:,-2]
        range_azimuths = points[:,0:2]
        if type == Centroid_Type.max:
            centroid_vel = self.max_snr_centroid(velocities,snrs)
        elif type == Centroid_Type.weighted_average:
            centroid_vel = self.weighted_average_centroid(velocities,snrs)

        centroid_range_azimuth = np.mean(range_azimuths,axis=0)
        self.centroid = centroid(centroid_range_azimuth[0],centroid_range_azimuth[1],centroid_vel)


    def max_snr_centroid(self,velocities,snrs):
        max_index = np.where(snrs== max(snrs))[0][0]
        return ([velocities[max_index]])

    def weighted_average_centroid(self,velocities,snrs):
        normalized_weights = snrs/np.linalg.norm(snrs)
        return ([np.sum(velocities*normalized_weights)])


class Distance_Cluster(static_params):
    def __init__(self,max_distance,points):
        static_params.__init__(self)
        self.max_distance = max_distance
        self.covariance = np.diag([50,10*np.pi/180,20])
        self.points = np.array(points)
        self.r_index = 0
        self.az_index = 1
        self.dopp_index = 2
        self.snr_index = 3

    def arrange_points(self,points,use_dopp=False):
        points = np.array(points)
        points_to_process = []
        for p in points:
            if use_dopp:
                points_to_process.append(p[0:3])
            else:
                points_to_process.append(p[0:2])
                self.covariance = self.covariance[0:2, 0:2]
        return (np.array(points_to_process))
    def run_cluster(self,points,type,use_dopp=False):
        if len(points)==0: return ([])
        points_to_process = self.arrange_points(points,use_dopp)
        distance_matrix  = self.create_distance_matrix(np.array(points_to_process))
        pool_of_clusters = self.cluster_points(distance_matrix,type)
        return (pool_of_clusters,distance_matrix)

    def cluster_points(self,distance_matrix,type):
        (row,row) = np.shape(distance_matrix)
        visited_points = set([])
        pool_of_points = range(0,row)
        pool_of_clusters = []
        while len(pool_of_points)>0:
            point = pool_of_points[0]
            pool_of_points = pool_of_points[1:]
            #check if the point has been already visited
            if point in visited_points: continue
            #check if it can belong to any of clusters?
            for cluster in pool_of_clusters:
                if cluster.check_point_assignment(point,distance_matrix):
                    cluster.add_point(point,self.points[point,-1])
                    visited_points.add(point)
                    break
            if point in visited_points: continue
            #Form a new cluster
            pool_of_clusters.append(cluster_point(self.max_distance))
            pool_of_clusters[-1].add_point(point,self.points[point,-1])

        for cluster in pool_of_clusters:cluster.set_centroid(self.points, type)
        return (pool_of_clusters)

    def create_distance_matrix(self,points):
        """
        Calculate distance_matrix based on the points
        :param points:
        :return:
        """
        (row,col) = np.shape(points)
        distance_matrix = np.zeros([row,row])
        for i in range(0,row):
            for j in range(i+1,row):
                #Calculate the normalized distance
                p1 = points[i,:].reshape(len(points[i,:]),1)
                p2 = points[j,:].reshape(len(points[j,:]),1)
                error = p1 - p2
                distance = error.transpose().dot(np.linalg.inv(self.covariance)).dot(error)
                #check for absolute value of velocity
                diff_vel = np.abs(self.points[i,-2]-self.points[j,-2])
                if diff_vel<self.clustering_max_vel_distance:
                    distance_matrix[i,j] = np.sqrt(distance)
                else:
                    distance_matrix[i,j] = np.inf
        return (distance_matrix)

    def filter_clusters(self,pool_of_clusters):
        passed_clusters = []
        for cluster in pool_of_clusters:
            if cluster.num_points>=self.clustering_min_num_points and \
                    np.abs(cluster.centroid.vel)>=self.clustering_min_vel_threshold and cluster.total_snr>=self.clustering_min_total_snr:
                passed_clusters.append(cluster)
        return (passed_clusters)




if __name__=="__main__":

    base_path = "/home/ali/SharedFolder/view_meas_June5th/"
    points = parse_ti_data("sngl_smpl.txt")

    num_clusters = []
    for p in points:
        if len(p)==0:
            num_clusters.append(0)
            continue
        max_distance = find_gate_threshold(2,.99)
        distance_cluster = Distance_Cluster(max_distance,p)
        clusters, distance_matrix = distance_cluster.run_cluster(p,Centroid_Type.max)

        filtered_clusters = distance_cluster.filter_clusters(clusters)
        num_clusters.append(len(filtered_clusters))


    #num_clusters = []
    #for p in points:
     #   num_clusters.append(len(distance_cluster.run_cluster(p)))
    #pool_of_clusters = distance_cluster.run_cluster(points[132])
    #distance_matrix = distance_cluster.create_distance_matrix()

