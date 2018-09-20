
import numpy as np
class centralized_fusion:
    def __init__(self,window_size , window_lag, MAX_UNCERTAINTY,num_sensors,initial_x_k_k,initial_p_k_k):
        self.initial_fusion_uncertainty = 1E18
        self.num_targets = len(initial_x_k_k)
        self.window_size = window_size
        self.window_lag = window_lag
        self.MAX_UNCERTAINTY = MAX_UNCERTAINTY

        self.num_sensors = num_sensors
        self.global_x_k_k = list(initial_x_k_k)
        self.global_p_k_k = list(initial_p_k_k)

        self.sensors_target_uncertainty = []
        self.sensors_avg_uncertainty = []
        self.reward = []
        for n in range(0,self.num_sensors):
            self.sensors_target_uncertainty.append([])
            self.sensors_avg_uncertainty.append([])
            self.reward.append([])
        for n in range(0,self.num_sensors):
            for t in range(0,len(self.global_x_k_k)):
                self.sensors_target_uncertainty[n].append([])


    def calculate_senosr_reward(self,diff_uncertainty):
        if diff_uncertainty<0:
            return (1)
        else:
            return (0)

    #For now, let's go with known assignment (this is not realisitic but just for sanity checks)
    def form_2d_assignment(self,local_x_k_k_s,local_p_k_k_s,global_x_k_k_s,global_p_k_k_s):
        local_to_global_map = {}
        for n in range(0,self.num_targets):
            local_to_global_map[n] = n
        return (local_to_global_map)

    def update_global(self,sensors,memory=True):

        self.cumulative_x_k_k = []
        for t in range(0, len(self.global_x_k_k)):
            temp = np.linalg.inv(self.global_p_k_k[t]).dot(self.global_x_k_k[t]).reshape([4, 1])
            self.cumulative_x_k_k.append(temp)

        if not memory:
            for target_index in range(0,self.num_targets):
                self.global_p_k_k[target_index] = np.diag([self.initial_fusion_uncertainty,self.initial_fusion_uncertainty,
                                         self.initial_fusion_uncertainty,self.initial_fusion_uncertainty])
        #Sequential update over all the sensors
        for sensor_index in range(0,self.num_sensors):
            current_sensor_object = sensors[sensor_index]
            #Get all the local estimates
            local_tracks = current_sensor_object.tracker_object.tracks #List of all local tracks
            local_x_k_k_s = []
            local_p_k_k_s = []
            for track in local_tracks:
                local_x_k_k_s.append(track.x_k_k)
                local_p_k_k_s.append(track.p_k_k)
            #Temporary
            local_to_global_map = self.form_2d_assignment(local_x_k_k_s,local_p_k_k_s)
            #Update global track estimates based on the estimates coming from each sensor
            temp_target_uncertainty = []
            for target_index in range(0,self.num_targets):
                local_x_k_k = local_x_k_k_s[target_index]
                local_p_k_k = local_p_k_k_s[target_index]

                global_index = local_to_global_map[target_index]
                global_x_k_k = self.global_x_k_k[global_index]
                global_p_k_k = self.global_p_k_k[global_index]
                #Ignore correlation between tracks (sub-optimal fusion)
                inv_matrix_global = np.linalg.inv(global_p_k_k)
                inv_matrix_local = np.linalg.inv(local_p_k_k)
                sensor_specific_p = np.linalg.inv(inv_matrix_local+inv_matrix_global)
                self.cumulative_x_k_k[target_index]+= np.linalg.inv(local_p_k_k).dot(local_x_k_k)

                temp_target_uncertainty.append(np.trace(sensor_specific_p)/self.MAX_UNCERTAINTY)
                self.sensors_target_uncertainty[sensor_index][target_index]\
                    .append(np.trace(sensor_specific_p)/self.MAX_UNCERTAINTY)
                #update globals
                self.global_p_k_k[target_index] = sensor_specific_p
                self.global_x_k_k[target_index] = (sensor_specific_p.dot(self.cumulative_x_k_k[target_index]))
            self.sensors_avg_uncertainty[sensor_index].append(np.mean(temp_target_uncertainty))
            #Calculate global reward assigned to each sensor
            if len(self.sensors_avg_uncertainty[sensor_index]) < self.window_size + self.window_lag:
                self.reward[sensor_index].append(0)
            else:
                current_avg = np.mean(self.sensors_avg_uncertainty[sensor_index][-self.window_size:])
                prev_avg = np.mean(self.sensors_avg_uncertainty[sensor_index][-(self.window_size
                                                                                + self.window_lag):-self.window_lag])
                if current_avg < prev_avg or self.sensors_avg_uncertainty[sensor_index][-1] < .1:
                    # if current_avg < prev_avg:
                    self.reward[sensor_index].append(1)
                else:
                    self.reward[sensor_index].append(0)

    def update_global_memoryless(self,sensors,feedback=False):
        cum_cov = {}
        cum_mean = {}
        for t in range(0, len(self.global_x_k_k)):
            #temp = np.linalg.inv(self.global_p_k_k[t]).dot(self.global_x_k_k[t]).reshape([4, 1])

            cum_cov[t] = []
            cum_mean[t] = []

        sensor_to_track_assignment = {}
        target_uncertainty = {}
        for sensor_index in range(0, self.num_sensors):
            sensor_to_track_assignment[sensor_index] = {}
            current_sensor_object = sensors[sensor_index]
            local_tracks = current_sensor_object.tracker_object.tracks #Set of all the tracks
            local_x_k_k_s = [] #set of local means
            local_p_k_k_s = [] #set of local covariance matrices
            for track in local_tracks:
                local_x_k_k_s.append(track.x_k_k)
                local_p_k_k_s.append(track.p_k_k)
            local_to_global_map = self.form_2d_assignment(local_x_k_k_s, local_p_k_k_s,cum_mean,cum_cov)
            #TEmporary: each observer has N same tracks and each track is assigned to its own one
            sensor_to_track_assignment[sensor_index] = local_to_global_map

            temp_target_uncertainty = []
            for target_index in range(0, self.num_targets):
                if target_index in sensor_to_track_assignment[sensor_index]:
                    assigned_target_index = sensor_to_track_assignment[sensor_index][target_index]
                    local_x_k_k = local_x_k_k_s[assigned_target_index]
                    local_p_k_k = local_p_k_k_s[assigned_target_index]

                    if not cum_cov[target_index]:
                        cum_cov[target_index].append(np.linalg.inv(local_p_k_k))
                        cum_mean[target_index].append(np.linalg.inv(local_p_k_k).dot(local_x_k_k))
                    else:
                        cum_cov[target_index].append(np.linalg.inv(local_p_k_k)+cum_cov[target_index][-1])
                        cum_mean[target_index].append(np.linalg.inv(local_p_k_k).dot(local_x_k_k)+cum_mean[target_index][-1])
                        #temp_target_uncertainty.append((np.trace(np.linalg.inv(cum_cov[target_index][-1]))-
                         #                               np.trace(np.linalg.inv(cum_cov[target_index][-2]))))

                    temp_target_uncertainty.append(np.trace(np.linalg.inv(cum_cov[target_index][-1]))/self.MAX_UNCERTAINTY)
                    #Store uncertainy
                    #self.sensors_target_uncertainty[sensor_index][target_index].append(np.trace(np.linalg.inv(cum_cov[target_index][-1])))

            self.sensors_avg_uncertainty[sensor_index].append(np.mean(temp_target_uncertainty))
            #calculate reward
            if len(self.sensors_avg_uncertainty[sensor_index]) < self.window_size + self.window_lag:
                self.reward[sensor_index].append(0)
            else:
                current_avg = np.mean(self.sensors_avg_uncertainty[sensor_index][-self.window_size:])
                prev_avg = np.mean(self.sensors_avg_uncertainty[sensor_index][-(self.window_size
                                                                                + self.window_lag):-self.window_lag])
                if current_avg < prev_avg:# or self.sensors_avg_uncertainty[sensor_index][-1] < .1:
                    # if current_avg < prev_avg:
                    self.reward[sensor_index].append(1)
                else:
                    self.reward[sensor_index].append(0)
            #target_uncertainty[sensor_index] = temp_target_uncertainty

        #print(target_uncertainty)
        #for sensor_index in range(0, self.num_sensors):
         #   if not not target_uncertainty[sensor_index]:
          #      rewards = []
           #     for u in target_uncertainty[sensor_index]: rewards.append(self.calculate_senosr_reward(u))
                #Use max-vote
            #    if sum(rewards)/(1.0*len(rewards))>.5:
             #       self.reward[sensor_index].append(1)
              #  else:
               #     self.reward[sensor_index].append(-1)

        for t in range(0, len(self.global_x_k_k)):
            self.global_p_k_k[t] = np.linalg.inv(cum_cov[t][-1])
            self.global_x_k_k[t] = self.global_p_k_k[t].dot(cum_mean[t][-1])
            if feedback:
                for sensor_index in range(0,self.num_sensors):
                    current_sensor_object = sensors[sensor_index]
                    if t in sensor_to_track_assignment[sensor_index]:
                        assigned_target_index = sensor_to_track_assignment[t]
                        sensors[sensor_index].tracker_object.tracks[t].x_k_k = self.global_x_k_k[t]
                        #sensors[sensor_index].tracker_object.tracks[t].p_k_k = self.global_p_k_k[t]



    def update_global_sequential(self,sensors):
        for sensor_index in range(0, self.num_sensors):
            current_sensor_object = sensors[sensor_index]
            local_tracks = current_sensor_object.tracker_object.tracks
            local_x_k_k_s = []
            local_p_k_k_s = []
            for track in local_tracks:
                local_x_k_k_s.append(track.x_k_k)
                local_p_k_k_s.append(track.p_k_k)

            local_to_global_map = self.form_2d_assignment(local_x_k_k_s, local_p_k_k_s)
            for target_index in range(0, self.num_targets):
                #New measurement and associated covariance
                local_x_k_k = local_x_k_k_s[target_index]
                local_p_k_k = local_p_k_k_s[target_index]
                #Form innovation covariances
                S = self.global_p_k_k[target_index]+local_p_k_k
                #Form kalman gain
                Gain = self.global_p_k_k[target_index].dot(np.linalg.inv(self.global_p_k_k[target_index]+local_p_k_k))
                #Form error-term
                estimate_error = local_x_k_k-self.global_x_k_k[target_index]
                print(Gain)
                #Update rules
                self.global_x_k_k[target_index] += Gain.dot(estimate_error)
                self.global_p_k_k[target_index] = self.global_p_k_k[target_index]-Gain.dot(self.global_p_k_k[target_index])

            


