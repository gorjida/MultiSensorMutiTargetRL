
class static_params:
    def __init__(self):
        # Parameters for track-initiation
        self.clustering_max_vel_distance = 200
        # Maximum distance between points at each cluster
        self.clustering_max_meas_distance = 400
        # Minimum velocity for allocation of centroid
        self.clustering_min_vel_threshold = 10
        # Minimum number of points within the allocation set
        self.clustering_min_num_points = 3
        self.clustering_min_total_snr = 100

#Units are all in centimeters
class scenario:
    def __init__(self,bearing_std,range_std,vel_std,pd,landa):
        self.x_min = -2000
        self.x_max = 2000
        self.y_min = 0
        self.y_max = 2000
        self.volume = 3.14*(self.x_max)**2
        self.vel_min = -200
        self.vel_max = 200
        #self.num_targets = num_targets
        #self.num_sensors = num_sensors

        self.MIN_RANGE = 0
        self.MAX_RANGE = 100
        self.MAX_UNCERTAINTY = 10000
        self.bearing_std = bearing_std
        self.range_std = range_std
        self.vel_std = vel_std

        self.pd = pd
        self.landa = landa
        self.vel_threshold = 0
        self.vel_threshold_for_static = 5





