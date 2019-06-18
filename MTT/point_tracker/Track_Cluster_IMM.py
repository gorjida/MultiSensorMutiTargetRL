import numpy as np
from Distance_Cluster import *
from EKF_tracker import EKF_tracker
import operator
from utils import *
from scenario import *
from scipy.stats import mode


class Custom_Cluster:
    def __init__(self, r, theta, vel):
        self.centroid = [r * np.cos(theta), r * np.sin(theta)]
        self.centroid_polar = [r, theta]
        self.vels = [vel]


class Track_Cluster_IMM:
    def __init__(self, scen, assigned_measurements, centroid, track, use_vel=True):
        self.use_vel = use_vel
        self.scen = scen
        assigned_measurements = np.array(assigned_measurements)
        self.refine_measurements(assigned_measurements)

        # Initialize each centroid
        self.initiate(centroid)
        self.num_assigned_measurements = [len(assigned_measurements)]
        self.num_measurements_in_gate = [len(assigned_measurements)]
        self.track = track
        self.alpha = .05
        self.track_assignment = []

        self.is_initiated = 1
        self.is_tentative = 0
        self.is_active = 0

    def refine_measurements(self, assigned_measurements):
        if self.use_vel:
            self.measurement_noise_cov = \
                np.diag([self.scen.range_std ** 2, self.scen.bearing_std ** 2
                            , self.scen.vel_std ** 2])
        else:
            self.measurement_noise_cov = \
                np.diag([self.scen.range_std ** 2, self.scen.bearing_std ** 2])

        if np.shape(assigned_measurements)[0] > 0:
            # Extract point SNRs
            self.snrs = assigned_measurements[:, -1]
            # Slice measurements and get covariance matrix
            if self.use_vel:
                self.measurements = assigned_measurements[:, 0:3]

            else:
                self.measurements = assigned_measurements[:, 0:2]
        else:
            self.snrs = np.array([])
            self.measurements = np.array([])

    def set_new_measurements(self, assigned_measurements, gated_measurements):

        self.num_measurements_in_gate.append(len(gated_measurements))
        self.num_assigned_measurements.append(len(assigned_measurements))
        self.refine_measurements(assigned_measurements)

    def mean_measurements(self, measurements):
        return (np.mean(measurements, axis=0))

    def weighted_mean_measurements(self, measurements):
        weighted_snr = self.snrs / np.sum(self.snrs)
        return (np.average(measurements, weights=weighted_snr, axis=0))

    def sample_cov_unweighted(self, measurements):
        return (np.cov(measurements.transpose()))

    def sample_cov_weighted(self, measurements):
        return (np.cov(measurements.transpose(), aweights=self.snrs))

    def initiate(self, centroid):
        """
        Initialize the cluster with the centroid
        :return:
        """

        x = centroid.range * np.sin(centroid.azimuth)
        y = centroid.range * np.cos(centroid.azimuth)

        self.centroid_cartz = [x, y]
        self.centroid_polar = [centroid.range, centroid.azimuth]
        if self.use_vel:
            vel = centroid.vel
            self.centroid_cartz += [vel]
            self.centroid_polar += [vel]

        # Calculate sample-covariance (for range/azimuth/vel)
        # sample_cov = self.sample_cov_unweighted(self.measurements)
        sample_cov = self.sample_cov_weighted(self.measurements)
        self.dispersion = np.diag([0, 0, 0])
        self.dispersion = sample_cov

    """
    def initiate_eddy(self):
        self.centroid_polar = self.measurements
        x = self.measurements[0] * np.sin(self.measurements[1])
        y = self.measurements[0] * np.cos(self.measurements[1])
        self.centroid_cartz = [x, y, self.measurements[2]]

        self.dispersion = np.diag([0,0,0])
    """

    """
    def update_state_with_static(self,scen,measurement,sensor_state,use_vel=True):

        if use_vel:
            measurement_noise_cov = \
                np.diag([scen.range_std**2,scen.bearing_std**2
                            ,scen.vel_std**2])
        else:
            measurement_noise_cov = \
                np.diag([scen.range_std ** 2, scen.bearing_std ** 2])


        self.track.S_k = self.track.S_k +  measurement_noise_cov + self.dispersion
    """

    def update_state_with_no_assignment(self, scen, measurement, sensor_state, use_vel=True):
        """
        Set new measurments and update number of assigned measurements and dispesion matrix for this cluster
        :param measurements:
        :return:
        """

        # Use prediction
        self.track.S_k = self.track.S_k + self.measurement_noise_cov + self.dispersion
        self.track.x_k_k = self.track.x_k_km1
        self.track.p_k_k = self.track.p_k_km1

    def update_with_new_measurements(self, sensor_state):
        """
        Set new measurments and update number of assigned measurements and dispesion matrix for this cluster
        :param measurements:
        :return:
        """
        NA = self.num_assigned_measurements[-1]
        if NA > 0:
            self.track_assignment.append(1)
            N_hat = self.num_measurements_in_gate[-1]
            if N_hat > 1:
                detection_ratio = (N_hat - NA) / ((N_hat - 1) * NA)
            else:
                detection_ratio = 1
            if len(self.measurements) > 1:
                sample_cov = self.sample_cov_weighted(self.measurements)
                self.dispersion = (1 - self.alpha) * self.dispersion + \
                                  self.alpha * sample_cov

            # Update centroid
            mean_meas = self.mean_measurements(self.measurements)

            self.centroid_polar = mean_meas
            x = mean_meas[0] * np.sin(mean_meas[1])
            y = mean_meas[0] * np.sin(mean_meas[1])
            self.centroid_cartz = [x, y]
            if self.use_vel: self.centroid_cartz += [mean_meas[-1]]
            # Now, update with the new centroid
            self.track.S_k = self.track.S_k + self.measurement_noise_cov / NA + self.dispersion * detection_ratio
            self.track.update_states(sensor_state, np.array(self.centroid_polar))
        else:
            # Other actions
            self.track.x_k_k = self.track.x_k_km1
            self.track.p_k_k = self.track.p_k_km1
            self.track_assignment.append(0)

    def set_track(self, track):
        self.track = track