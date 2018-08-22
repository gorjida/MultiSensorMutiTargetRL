class clean_tracker_agent:
    def __init__(self,init_tracks):
        self.tracks = init_tracks

    def update_target_states(self,sensor_state,measurement):
        for tracker in self.tracks:
            tracker.update_states(sensor_state,measurement)