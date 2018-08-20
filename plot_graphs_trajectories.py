import matplotlib.pyplot as plt
import numpy as np
import os

folder_path = "ResultsMultipleTarget/"
if __name__=="__main__":


    truth1 = folder_path+"t1_truth.txt"
    truth2 = folder_path + "t2_truth.txt"
    est1 = folder_path + "landa5_t1_est.txt"
    est2 = folder_path + "landa5_t2_est.txt"
    sensor = folder_path + "landa5_sensor.txt"
    false = folder_path+"landa5_false_measurements.txt"

    t1 = []
    t2 = []
    e1 = []
    e2 = []
    ff = []
    s = []

    with open(truth1,"r") as f:
        for line in f:
            data = line.strip().split("\t")
            t1.append([float(data[0]),float(data[1])])

    with open(truth2,"r") as f:
        for line in f:
            data = line.strip().split("\t")
            t2.append([float(data[0]),float(data[1])])

    with open(est1,"r") as f:
        for line in f:
            data = line.strip().split("\t")
            e1.append([float(data[0].replace("[","").replace("]","")),float(data[1].replace("[","").replace("]",""))])

    with open(est2,"r") as f:
        for line in f:
            data = line.strip().split("\t")
            e2.append([float(data[0].replace("[","").replace("]","")),float(data[1].replace("[","").replace("]",""))])

    with open(sensor,"r") as f:
        for line in f:
            data = line.strip().split("\t")
            s.append([float(data[0]),float(data[1])])

    with open(false,"r") as f:
        for line in f:
            data = line.strip().split("\t")
            ff.append([float(data[0]),float(data[1])])


    t1 = np.array(t1)
    t2 = np.array(t2)
    plt1, = plt.plot(t1[:,0],t1[:,1],"k",linewidth=3)
    plt2, = plt.plot(t2[:, 0], t2[:, 1], "k", linewidth=3)

    e1 = np.array(e1)
    e2 = np.array(e2)
    plt3, = plt.plot(e1[:,0],e1[:,1],"r--",linewidth=2)
    plt4, = plt.plot(e2[:, 0], e2[:, 1], "r--", linewidth=2)

    s = np.array(s)
    plt5, = plt.plot(s[:,0],s[:,1],"b-",linewidth=3)

    ff = np.array(ff)
    plt6, = plt.plot(ff[:,0],ff[:,1],"g.",markersize=1)

    plt.annotate('Observer Starting Point', xy=(0, 0), xycoords='data',
              xytext=(-200, 400)
              )

    plt.xlabel("X(m)", size=20)
    plt.ylabel("Y(m)", size=20)
    plt.xlim([-10000,10000])
    plt.ylim([-10000,10000])
    plt.legend([plt1,plt3,plt5,plt6],["Truths","Estimates","Optimal Observer Trajectory","False Measurements"])
    plt.grid(True)
    plt.show()

