
import numpy as np
from munkres import Munkres

class Metric:
    def __init__(self,num_targets_list,truth_map,target_intervals,T_max):
        self.num_targets_list = num_targets_list
        self.active_num_targets = []
        self.tentative_num_targets = []
        self.estimates_pos = {}
        self.estimates_vel = {}

        self.target_state = {}
        self.m = Munkres()

        self.pos_error = {}
        self.vel_error = {}

        for target_index in truth_map:
            self.pos_error[target_index] = []
            self.vel_error[target_index] = []
            target_obj = truth_map[target_index].target
            if target_index not in self.target_state: self.target_state[target_index] = []
            start_index = target_intervals[target_index][0]
            end_index = target_intervals[target_index][1]
            prefix_state = []
            postfix_state = []
            [prefix_state.append([]) for c in range(0,start_index)]
            [postfix_state.append([]) for c in range(end_index+1,T_max+1)]
            for loc,vel in zip(target_obj.historical_location,target_obj.historical_velocity):
                current_state = loc + vel
                self.target_state[target_index].append(current_state)


            self.target_state[target_index] = prefix_state + \
                                              self.target_state[target_index] + postfix_state

            [self.pos_error[target_index].append(np.nan) for x in range(0,T_max)]
            [self.vel_error[target_index].append(np.nan) for x in range(0, T_max)]
    def update_num_targets(self,jpdaf_obj):
        active = 0
        tentative = 0
        for track in jpdaf_obj.tracks:
            if track.status==1:
                active+=1
            elif track.status==0:
                tentative+=1

        self.active_num_targets.append(active)
        self.tentative_num_targets.append(tentative)

    def update_estimates(self,tracks,distance_threshold,scan):
        association_matrix = []
        track_id_map = {}
        num_active_tracks = 0
        for track in tracks:
            if track.status!=1: continue
            id = track.track_id
            track_id_map[num_active_tracks] = id
            num_active_tracks+=1
            estimate = track.x_k_k
            cov = track.p_k_k

            truth_association = self.track_to_truth_vector(estimate,cov,distance_threshold,scan)

            association_matrix.append(truth_association)
            if id not in self.estimates_pos:
                self.estimates_pos[id] = []
                self.estimates_vel[id] = []
            self.estimates_pos[id]\
                .append([estimate[0][0],estimate[1][0]])
            self.estimates_vel[id].append([estimate[2][0],estimate[3][0]])


        #Global NearestNeighbor for truth-to-track association
        association_matrix = np.array(association_matrix)

        #print(scan)
        #print(self.target_state[1][scan])
       # print(association_matrix)

        if len(association_matrix)>0:
            (n_row,n_col) = np.shape(association_matrix)
            if n_row<=n_col:
                best_assignment = self.m.compute(np.array(association_matrix))
            else:
                best_assignment = self.m.compute(np.array(association_matrix).transpose())
                mod_best_assignment = []
                for t in best_assignment:
                    new_t = (t[1],t[0])
                    mod_best_assignment.append(new_t)

                best_assignment = mod_best_assignment
        else:
            best_assignment = []

        #print(association_matrix)
        #print(best_assignment)
        #print(scan)
        #print("\n\n\n")
        track_to_target_assignment = []
        for track_id,truth_id in best_assignment:
            id = track_id_map[track_id]
            #print(association_matrix[track_id,truth_id])
            if association_matrix[track_id,truth_id]>1E4:
                continue
            else:
                #this is a valid association
                #if scan>=len(self.target_state[truth_id]): continue
                #print(np.array(self.target_state[truth_id][scan]))
                pos_error = np.array(self.estimates_pos[id][-1]).reshape(2,1) - \
                            np.array(self.target_state[truth_id][scan])[0:2].reshape(2,1)
                vel_error = np.array(self.estimates_vel[id][-1]).reshape(2, 1) - \
                            np.array(self.target_state[truth_id][scan])[2:].reshape(2, 1)

                self.pos_error[truth_id][scan]= np.linalg.norm(pos_error)
                self.vel_error[truth_id][scan] = np.linalg.norm(vel_error)
                track_to_target_assignment.append([track_id,truth_id])


        return (track_to_target_assignment)

    def track_to_truth_vector(self,track_estimate,track_cov,distance_threshold,scan):
        distance_vec = []
        for target_index in range(0,len(self.target_state)):
        #for target_index in self.target_state:
            #print(scan)
            #print(target_index)
            #print(distance_vec)


            #if target_index==1: print(self.target_state[target_index][scan])
            if True:
                if not self.target_state[target_index][scan]:
                    distance_vec.append(1E6)
                    continue
                state = np.array(self.target_state[target_index][scan]).reshape(4,1)
                error = (state-track_estimate).reshape(4,1)

                #Calculate normalized error
                distance = error.transpose().dot(np.linalg.inv(track_cov)).dot(error)[0][0]

                #distance = error.transpose().dot(error)[0][0]
                if distance<distance_threshold:
                    distance_vec.append(distance)
                else:
                    #Put a number instead of INF
                    distance_vec.append(1E6)
            #else:
             #   distance_vec.append(1E6)

        #print("\n\n\n")
        return (distance_vec)

















