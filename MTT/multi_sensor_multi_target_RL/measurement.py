
import numpy as np

class measurement:
    def __init__(self,scenario):
        self.bearing_std = scenario.bearing_std
        self.range_std = scenario.range_std
        self.vel_std = scenario.vel_std
        self.MIN_RANGE = scenario.MIN_RANGE
        self.MAX_RANGE = scenario.MAX_RANGE

    def generate_bearing(self,target_loc,sensor_loc):
        noiseless_bearing = np.arctan2(target_loc[0]-sensor_loc[0],target_loc[1]-sensor_loc[1])
        #if noiseless_bearing<0:
         #   noiseless_bearing+= 2*np.pi
        return (noiseless_bearing+np.random.normal(0,self.bearing_std))

    def generate_range(self,target_loc,sensor_loc):
        diff = np.array(target_loc) - np.array(sensor_loc)
        noiseles_range = np.linalg.norm(diff)
        noise_range = noiseles_range+np.random.normal(0,self.range_std)
        #noiseles_range = min(max(noiseles_range,self.MIN_RANGE),self.MAX_RANGE)
        return (noiseles_range+ np.random.normal(0,self.range_std))

    def generate_radial_velocity(self,target_loc,sensor_loc,target_vel,sensor_vel):
        diff = np.array(target_loc) - np.array(sensor_loc)
        diff_vel = np.array(target_vel) - np.array(sensor_vel)
        noiseless_radial_vel = (np.sum(diff*diff_vel))/np.linalg.norm(diff)
        return (noiseless_radial_vel + np.random.normal(0,self.vel_std))