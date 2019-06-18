from utils import *
import numpy as np

if __name__=="__main__":

    #List of parameters
    vel_var = 1 #noise on the motion-model
    bearing_std = (10 * np.pi) / 180
    range_std = 100 / 3.46
    vel_std = 100
    pd = .9
    pg = .999
    landa = 2
    M_logic = 3
    N_logic = 5
    M_logic_init = 2
    N_logic_init = 2
    sample_time = .05
    use_vel = True
    gate_threshold = find_gate_threshold(3, pg)
    # create required objects
    dummy_target = truth_initiation(vel_var, sample_time, targets_initial_states[0], 1E-10)
    A, B = dummy_target.constant_velocity(1E-10)
    scen = scenario(bearing_std, range_std, vel_std, pd, landa)