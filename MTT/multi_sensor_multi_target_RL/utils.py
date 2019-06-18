

import numpy as np
import re
from scipy.stats import chi2


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