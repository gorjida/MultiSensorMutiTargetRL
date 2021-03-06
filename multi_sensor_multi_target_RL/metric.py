
import numpy as np

def expand_list(x,num_expand):
    for n in range(0,num_expand):
        x.append([])

    return (x)
class metric():
    def __init__(self,num_targets,num_sensors):
        self.num_targets = num_targets
        self.num_sensors = num_sensors
        self.x_est = expand_list([],num_sensors);
        self.y_est = expand_list([],num_sensors);

        self.x_vel_est = expand_list([],num_sensors);
        self.y_vel_est = expand_list([],num_sensors);

        self.x_truth = [];
        self.y_truth = [];

        self.x_vel_truth = [];
        self.y_vel_truth = []

        self.x_est_fusion = []
        self.y_est_fusion = []

        self.x_vel_est_fusion = []
        self.y_vel_est_fusion = []

        self.vel_error = expand_list([],num_sensors);
        self.pos_error = expand_list([],num_sensors);
        self.pos_error_fusion = []
        self.vels_error_fusion = []

        for i in range(0, num_targets):
            self.x_truth.append([])
            self.y_truth.append([])
            self.x_vel_truth.append([])
            self.y_vel_truth.append([])

            self.x_est_fusion.append([])
            self.y_est_fusion.append([])
            self.x_vel_est_fusion.append([])
            self.y_vel_est_fusion.append([])
            self.pos_error_fusion.append([])
            self.vels_error_fusion.append([])


        for s in range(0,num_sensors):
            for i in range(0,num_targets):
                self.x_est[s].append([])
                self.y_est[s].append([])
                self.x_vel_est[s].append([])
                self.y_vel_est[s].append([])
                self.vel_error[s].append([])
                self.pos_error[s].append([])



    def update_truth_estimate_metrics(self,t,s):
        for i in range(0, self.num_targets):
            truth = t[i].current_location
            self.x_truth[i].append(truth[0])
            self.y_truth[i].append(truth[1])
            self.x_vel_truth[i].append(t[i].current_velocity[0])
            self.y_vel_truth[i].append(t[i].current_velocity[1])
            for sensor_index in range(0, self.num_sensors):
                estimate = s[sensor_index].tracker_object.tracks[i].x_k_k
                self.x_est[sensor_index][i].append(estimate[0])
                self.y_est[sensor_index][i].append(estimate[1])
                self.x_vel_est[sensor_index][i].append(estimate[2])
                self.y_vel_est[sensor_index][i].append(estimate[3])
                self.pos_error[sensor_index][i].append(np.linalg.norm(estimate[0:2] - np.array(truth).reshape(2, 1)))
                self.vel_error[sensor_index][i].append(np.linalg.norm(estimate[2:4] - np.array([t[i].current_velocity[0], t[i].current_velocity[1]]).reshape(2, 1)))

    def update_truth_fusion_estimate_metrics(self,t,fusion_agent):
        for i in range(0, self.num_targets):
            truth = t[i].current_location
            self.x_truth[i].append(truth[0])
            self.y_truth[i].append(truth[1])
            self.x_vel_truth[i].append(t[i].current_velocity[0])
            self.y_vel_truth[i].append(t[i].current_velocity[1])
            self.x_est_fusion[i].append(fusion_agent.global_x_k_k[i][0])
            self.y_est_fusion[i].append(fusion_agent.global_x_k_k[i][1])
            self.x_vel_est_fusion[i].append(fusion_agent.global_x_k_k[i][2])
            self.y_vel_est_fusion[i].append(fusion_agent.global_x_k_k[i][3])
            self.pos_error_fusion[i].append(np.linalg.norm(np.array([fusion_agent.global_x_k_k[i][0],fusion_agent.global_x_k_k[i][1]]) -
                                                           np.array(truth).reshape(2, 1)))
            #self.vels_error_fusion[i].append(np.linalg.norm(np.array([fusion_agent.x_k_k[2],fusion_agent.x_k_k[3]]) - np.array(truth).reshape(2, 1)))