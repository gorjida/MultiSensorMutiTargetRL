

import numpy as np
import re
from scipy.stats import chi2
import random
from target import target

base_path = "/home/ali/SharedFolder/TI_scenarioes/"
base_path = "/home/ali/SharedFolder/view_meas_June5th/"

def truth_initiation(vel_var,sample_time,initial_state,type,rate):
    x = 500 * random.random() - 250  # initial x-location
    y = 500 * random.random() - 250 # initial y-location
    xdot = 5 * random.random() - 2.5  # initial xdot-value
    ydot = 5 * random.random() - 2.5  # initial ydot-value

    #x = 100
    #y = 100
    #xdot = 2
    #ydot = -2
    t = target([initial_state[0], initial_state[1]], initial_state[2],
               initial_state[3], vel_var, vel_var, type,sample_time,rate)
    return (t)

def find_gate_threshold(meas_size,pg):
    """
    Finding the gate threshold based on the probability of gate and dimension of measurements
    :param meas_size:
    :param pg:
    :return:
    """
    for x in np.arange(0,50,.1):
        if chi2.cdf(x,meas_size)>=pg:
            return (x)

def parse_ti_data(base_path,file_path):
    frame_index = -1
    new_frame = False

    frame_points = []
    this_frame_points = []
    with open(base_path+file_path,"r") as f:
        for line in f:
            data = line.strip()
            if data.startswith("Frame:"):
                frame = int(re.findall("\d+",data)[0])
                print("Frame:"+str(frame))
                frame_index+=1
                new_frame = True
                if frame>0: frame_points.append(this_frame_points)
                this_frame_points = []

            if data.startswith("point"):
                #str_point = data.replace("point ( ","").split("( ")
                points = re.findall(r"[-+]?\d*\.\d+|\d+",data)
                point = []
                for index,x in enumerate(points):
                    value= float(x)
                    if index==0 or index==2: value*= 100
                    point.append(value)
                this_frame_points.append(point)

    return (frame_points)


if __name__=="__main__":
    points = parse_ti_data("sngl_smpl.txt")