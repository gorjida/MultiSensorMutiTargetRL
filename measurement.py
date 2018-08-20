
import numpy as np
class measurement:
    def __init__(self,bearing_std):
        self.model_var = bearing_std**2

    def generate_bearing(self,target_loc,sensor_loc):
        noiseless_bearing = np.arctan2(target_loc[1]-sensor_loc[1],target_loc[0]-sensor_loc[0])
        if noiseless_bearing<0:
            noiseless_bearing+= 2*np.pi
        return (noiseless_bearing+np.random.normal(0,self.model_var))