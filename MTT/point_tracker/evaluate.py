import numpy as np
from munkres import *

class evaluate:
    def __init__(self,total_num_truths):
        self.truth_to_track_id = []
        self.truth_estimate_error = []
        self.total_num_truths = total_num_truths
        self.truth_estimates = {}
        self.munkres_obj = Munkres()
        for truth_id in range(0,total_num_truths): self.truth_estimates[truth_id] = []

    def add_tracks(self,tracks,truth_X,truth_Y):
        if len(truth_X)!=self.total_num_truths:
            print("Number of truths does not match...")
            sys.exit(1)

        track_id_map = {}
        num_truths = len(truth_X)
        num_estimates = len(tracks)
        distance_matrix = 1E10*np.ones([num_estimates,num_truths])
        error_matrix = 1E10*np.ones([num_estimates,num_truths])

        this_scan_to_track_assignment = np.zeros(num_truths)
        this_scan_error = -1*np.ones(num_truths)

        for truth_id in range(0, self.total_num_truths):
            if truth_X[truth_id]==-1:
                #Truth does not exist
                this_scan_to_track_assignment.append(-1)
                this_scan_error.append(np.nan)

            truth_loc = [truth_X[truth_id], truth_Y[truth_id]]
            best_estimate_map = -1
            best_error = np.inf

            for index, track in enumerate(tracks):

                track_id_map[index] = track.track_id
                estimate = track.x_k_k[0:2]
                if truth_id==0:
                    self.truth_estimates[index].append([estimate[0][0],estimate[1][0]])
                covariance = track.p_k_k[0:2, 0:2]
                error = np.array(estimate).reshape(2, 1) - np.array(truth_loc).reshape(2, 1)
                normalized_distance = error.transpose().dot(np.linalg.inv(covariance)).dot(error)
                distance_matrix[index,truth_id] = normalized_distance
                sqrt_error = np.linalg.norm(error)
                error_matrix[index, truth_id] = sqrt_error
                #if sqrt_error<best_error:
                 #   best_error = sqrt_error
                  #  best_estimate_map = index

            #this_scan_to_track_assignment.append(best_estimate_map)
            #this_scan_error.append(best_error)

        best_assignment = self.munkres_obj.compute(distance_matrix)
        for tuple in best_assignment:
            track_id = tuple[0]
            truth_id = tuple[1]
            if distance_matrix[track_id,truth_id]<1E10:
                this_scan_to_track_assignment[truth_id] = track_id_map[track_id]
                this_scan_error[truth_id] = error_matrix[track_id,truth_id]
            else:
                this_scan_to_track_assignment[truth_id] = -1

            this_scan_to_track_assignment[truth_id] = truth_id
            this_scan_error[truth_id] = error_matrix[truth_id,truth_id]

        self.truth_to_track_id.append(this_scan_to_track_assignment)
        self.truth_estimate_error.append(this_scan_error)

    def gen_track_swap(self,truth_to_track_assignment):
        truth_to_track_assignment = np.array(truth_to_track_assignment)
        (num_scans,num_truths) = np.shape(truth_to_track_assignment)

        swaps = []
        for id in range(0,num_truths):
            assigned_ids = truth_to_track_assignment[:,id]
            swap = 0
            for s in range(0,num_scans-1):
                if assigned_ids[s]!=assigned_ids[s+1]: swap+=1

            swaps.append(swap/num_scans)
        return (swaps)









