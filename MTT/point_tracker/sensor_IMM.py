import numpy as np
import random
from motion_model import motion_model
import motion_init_object
from motion_init_object import motion_init_object
import operator
from munkres import Munkres
import matplotlib.pyplot as plt
#from JPDAF_agent import JPDAF_agent


class sensor_IMM(motion_model, motion_init_object):
    def __init__(self, type, init_x, init_y, scen):

        motion_model.__init__(self, 1)
        motion_init_object.__init__(self)

        initial_location = [init_x, init_y]
        # initial_location = [X, Y]
        mean_x_vel = self.init_xdot
        mean_y_vel = self.init_ydot
        mean_x_acc = self.init_xdotdot
        mean_y_acc = self.init_ydotdot
        x_var = self.x_var
        y_var = self.y_var
        self.scen = scen
        self.optimizer = Munkres()
        self.initial_location = initial_location
        self.current_location = self.initial_location
        self.historical_location = [self.initial_location]

        self.initial_velocity = [0, 0]
        self.current_velocity = self.initial_velocity
        self.historical_velocity = [self.initial_velocity]
        self.x_var = x_var
        self.y_var = y_var

        # For constant accelaration model
        self.initial_acc = [mean_x_acc, mean_y_acc]
        self.current_acc = self.initial_acc
        self.historical_acc = [self.initial_acc]

        # For constant-turn model
        self.initial_speed = [self.init_speed]
        self.current_speed = self.initial_speed
        self.historical_speed = [self.initial_speed]

        self.initial_heading = [self.init_heading]
        self.current_heading = self.initial_heading
        self.historical_heading = [self.initial_heading]
        # generate an initial command

        self.initial_command = np.random.multinomial(1, np.array([1, 1, 1]) / 3.0).argmax()
        # current command
        self.current_command = self.initial_command
        self.historical_command = [self.initial_command]

        self.motion_type = type
        self.sensor_actions = []

        # list of measurements
        self.m_mv = []
        self.m_stc = []
        self.m = []
        self.reward = []
        self.avg_uncertainty = []
        self.uncertainty = []
        self.tracker_object = []

    def set_tracker_objects(self, tracker_objects):
        self.tracker_object = tracker_objects
        for i in range(0, len(self.tracker_object)): self.uncertainty.append([])

    def run_prediction(self):
        """
        Prediction for centroid of each cluster
        :return:
        """
        if not not self.tracker_object:
            for cluster in self.tracker_object:
                #prediction for each track
                cluster.track.do_prediction(self.current_location + self.current_velocity)

    def get_association_matrix(self,measurements,gate_threshold,use_vel=True):
        """
        TASK find the association cost between measurements and tracks
        :param measurements:
        :param gate_threshold:
        :param use_vel:
        :return:
        """

        #get noise_covariance
        measurements = np.array(measurements)
        # Association: between tracks and generated measurements
        association_matrix = np.zeros([len(self.tracker_object)
                                          , len(measurements)])
        track_to_meas_gate = {}
        meas_to_track_gate = {}

        if use_vel:
            measurements = measurements[:,0:3]
            measurement_noise_cov = \
                np.diag([self.scen.range_std ** 2, self.scen.bearing_std ** 2, self.scen.vel_std ** 2])
        else:
            measurements = measurements[:,0:2]
            measurement_noise_cov = \
                np.diag([self.scen.range_std ** 2, self.scen.bearing_std ** 2])


        for meas_index, m in enumerate(measurements): meas_to_track_gate[meas_index] = []
        for cluster_index, cluster in enumerate(self.tracker_object): track_to_meas_gate[cluster_index] = []
        #Loop over all the tracks (aka group tracks)
        for cluster_index, cluster in enumerate(self.tracker_object):
            num_meas_in_gate = 0
            # S_k: it only has the impact of target maneuver, add dispersion and measurement covariance

            #Think about this (adding estimate of extent as well)
            innov_cov = cluster.track.S_k + \
                        cluster.measurement_noise_cov + cluster.dispersion
            inv_gate_cov = np.linalg.inv(innov_cov)
            # Loop over all the measurements
            for meas_index, m in enumerate(measurements):
                # calculate the normalized distance
                len_meas = len(cluster.track.predicted_measurement)
                #form error-vector
                error = (np.array(m[0:len_meas]).reshape(len_meas, 1) - cluster.track.predicted_measurement.reshape(len_meas, 1))
                #Normalize the error by the inverse of innovation covariance
                normalized_distance = error.transpose().dot(inv_gate_cov).dot(error)
                #print(normalized_distance)
                if normalized_distance > gate_threshold:
                    association_matrix[cluster_index, meas_index] = 1E6
                else:
                    track_to_meas_gate[cluster_index].append(meas_index)
                    association_matrix[cluster_index, meas_index] \
                        = normalized_distance + np.log(np.linalg.det(innov_cov))

                    meas_to_track_gate[meas_index].append(cluster_index)
        return (association_matrix,track_to_meas_gate,meas_to_track_gate)

    def measurement_to_track_assignment(self,association_matrix):
        (num_tracks,num_meas) = np.shape(association_matrix)
        track_to_meas_assignment = {}
        for index in range(0,num_meas):
            track_min_assignment = association_matrix[:,index].argmin()
            if association_matrix[track_min_assignment,index]<100:
                if not track_min_assignment in track_to_meas_assignment: track_to_meas_assignment[track_min_assignment] = []
                track_to_meas_assignment[track_min_assignment].append(index)

        return (track_to_meas_assignment)
    def run_optimizer(self,association_matrix):
        """
        One-to-one assignment
        :param association_matrix:
        :return:
        """
        # Optimization
        (row,col) = np.shape(association_matrix)
        if row>0 and col>0:
            (n_row, n_col) = np.shape(association_matrix)
            if n_row <= n_col:
                best_assignment = self.optimizer.compute(np.array(association_matrix))
            else:
                best_assignment = self.optimizer.compute(np.array(association_matrix).transpose())
                mod_best_assignment = []
                for t in best_assignment:
                    new_t = (t[1], t[0])
                    mod_best_assignment.append(new_t)

                best_assignment = mod_best_assignment
        else:
            best_assignment = []

        track_to_meas_assignment = {}
        for cluster_id,cluster in enumerate(self.tracker_object): track_to_meas_assignment[cluster_id] = -1
        for b in best_assignment:
            track_id = b[0]
            meas_id = b[1]
            if association_matrix[track_id,meas_id]<1E6:
                track_to_meas_assignment[track_id] = meas_id
        return (track_to_meas_assignment)

    def nearest_neighbor_association(self,measurement_index,gate_threshold,use_vel=True):
        """
        Global nearest-neighbor association
        :param measurement_index: measurement-index
        :param gate_threshold: threshold-gate
        :param use_vel: flag for using/not-using velocity
        :return:
        track_to_measurement_assignment: key=track_id, list of measurement_indexes assigned to each track_id
        track_to_measurement_gate: key=track_id, list of measurement_indexes falling in the gate
        unassigned_measurement: list of all unassigned measurements
        """

        #Prediction over all the tracks
        measurements = self.m[measurement_index]
        num_tracks = len(self.tracker_object)
        num_meas = len(measurements)
        if num_tracks>0:
            self.run_prediction()
            if num_meas>0:
                association_matrix,track_to_meas_gate,meas_to_track_gate \
                    = self.get_association_matrix(measurements,gate_threshold,use_vel=use_vel)
                #Best track to measurement assignment
                track_to_meas_assignment = self.measurement_to_track_assignment(association_matrix)

                #List of unassigned measurements (large-movements with no-in-gate assignment)
                unassigned_measurements = []
                for meas_index in meas_to_track_gate:
                    if len(meas_to_track_gate[meas_index])==0: unassigned_measurements.append(self.m[measurement_index][meas_index])
            else:
                track_to_meas_gate = {}
                track_to_meas_assignment = {}
                unassigned_measurements = self.m[measurement_index]
        else:
            track_to_meas_gate = {}
            track_to_meas_assignment = {}
            unassigned_measurements = self.m[measurement_index]

        return(track_to_meas_assignment,track_to_meas_gate,unassigned_measurements)

    def set_measurements_to_tracks(self,cloud,track_meas_assignment,track_to_meas_gate):
        array_cloud = np.array(cloud)
        # Update each track
        for track,track_obj in enumerate(self.tracker_object):
            if track in track_meas_assignment:
                track_specific_assigned_measurements = array_cloud[track_meas_assignment[track], :]
            else:
                track_specific_assigned_measurements = np.array([])

            if track in track_to_meas_gate:
                track_specific_gates = array_cloud[track_to_meas_gate[track], :]
            else:
                track_specific_gates = np.array([])

            track_obj.set_new_measurements(track_specific_assigned_measurements, track_specific_gates)

    def update_track_estimaes_point_cloud(self):
        """
        TASK updates all estimates based on the latest measurements
        :param measurement_index: current scan
        :return: List of measurements falling outside of the gate (for track initiation)
        """

        #Three different cases:
        #1) there is a large-movement assigned to a track: update track
        #2) there is not large-movement assigned but there is a static movement: no-update, but maintain
        #3) there is neither large nor static movements: use prediction, but no assignment
        #print(mv_track_to_meas_assoc)
        for cluster_index, cluster in enumerate(self.tracker_object):
            cluster.update_with_new_measurements(self.current_location+self.current_velocity)

    def update_track_estimaes(self, measurement_index,track_to_meas_assoc):
        """
        TASK updates all estimates based on the latest measurements
        :param measurement_index: current scan
        :return: List of measurements falling outside of the gate (for track initiation)
        """
        measurements = np.array(self.m[measurement_index])
        for track_id in range(0,len(self.tracker_object)):
            if not track_id in track_to_meas_assoc:
                association = []
            else:
                association = track_to_meas_assoc[track_id]
            #Associated measurements
            associated_measurements = measurements[association,:]

            #Update the tracker_object parameters (dispersion, centroid, etc)
            self.tracker_object[track_id].\
                update_with_new_measurements(self.scen,associated_measurements
                                             ,self.current_location+self.current_velocity)

            #Update the track based on the centroid
            if not not association:
                self.tracker_object[track_id].association.append(1)
            else:
                self.tracker_object[track_id].association.append(0)


    def gen_measurements(self, t, measure, pd, landa, use_vel):
        """
        TASK generate measurements for the current sensor
        :param t: list of target objects
        :param measure: measruement object
        :param pd: probability of detection
        :param landa: rate of false-alarm
        :return: Generated list of measurements
        """
        temp_m = []
        input_state_temp = []
        num_targets = len(t)
        for i in np.arange(0, num_targets, 1):
            # Consider impact of pd (probability of detection)
            bearing = measure.generate_bearing(t[i].current_location, self.current_location)
            range_ = measure.generate_range(t[i].current_location, self.current_location)
            velocity = measure.generate_radial_velocity(t[i].current_location
                                                        , self.current_location, t[i].current_velocity,
                                                        self.current_velocity)
            if use_vel:
                meas_vector = \
                    np.array([range_, bearing, velocity])
            else:
                meas_vector = \
                    np.array([range_, bearing])

            if np.random.random() < pd: temp_m.append(meas_vector)
        # Now add False-alarms
        num_false_alrams = np.random.poisson(landa)
        false_measures = []
        for false_index in np.arange(0, num_false_alrams, 1):
            # generate x,y randomly
            random_x = (self.scen.x_max - self.scen.x_min) * np.random.random() + self.scen.x_min
            random_y = (self.scen.y_max - self.scen.y_min) * np.random.random() + self.scen.y_min
            # Low-velocity false-alarms
            random_xdot = 2 * np.random.random() - 1
            random_ydot = 2 * np.random.random() - 1
            false_measures.append([random_x, random_y])
            bearing = measure.generate_bearing([random_x, random_y], self.current_location)
            range = measure.generate_range([random_x, random_y], self.current_location)
            velocity = measure.generate_radial_velocity([random_x, random_y]
                                                        , self.current_location,
                                                        [random_xdot, random_ydot], self.current_velocity)
            if use_vel:
                meas_vector = np.array([range, bearing, velocity])
            else:
                meas_vector = np.array([range, bearing])
            temp_m.append(meas_vector)
        self.m.append(temp_m)  # Number of measurements is not necessarily equal to that of targets

    def sigmoid(self, x, derivative=False):
        return self.sigmoid(x) * (1 - self.sigmoid(x)) if derivative else 1 / (1 + np.exp(-x))



    def find_index(self, array, val):
        return (np.argwhere(array == val)[0][0])



if __name__ == "__main__":
    s = sensor([500, 500], 3, -2, .01, .01)

    for n in range(0, 500):
        s.update_location()





