import matplotlib.pyplot as plt
import numpy as np

base_dir = "trajectories/"
if __name__=="__main__":


    plts = []
    legends = ["Single-target Planner","Multi-target Planner"]
    line_style = ["b", "r","k","m","c"]

    for index in range(0,5):
        pcrlb_file = base_dir + "pcrlb_single_" + str(index) + ".txt"
        error_file = base_dir+"average_error_"+str(index)+".txt"
        pclb = []
        with open(pcrlb_file, "r") as f:
            for line in f:
                data = line.strip()
                pclb.append(float(data))

            if index==3 or index==4:
                plt1, = plt.plot(range(1, len(pclb) + 1), pclb, line_style[index-3], linewidth=2)
                plts.append(plt1)

        error = []
        with open(error_file, "r") as f:
            for line in f:
                data = line.strip()
                error.append(float(data))

            #plt2, = plt.plot(range(1, len(pclb) + 1), error, line_style[index-3+1], linewidth=2)
            #plts.append(plt2)



    plt.xlabel("Time Step", size=15)
    plt.ylabel(r"PCRLB ($m^2$)", size=15)
    plt.grid(True)
    plt.legend(plts, legends)
    plt.show()

    sys.exit(1)


    plts = []
    legends = ["Target1","Target2","Target3","Target4"]
    line_style = ["-b","-r","-k","-c"]

    for index in range(0,4):
        pcrlb_file = base_dir+ "pcrlb_single_"+str(index)+".txt"
        pclb = []
        with open(pcrlb_file,"r") as f:
            for line in f:
                data = line.strip()
                pclb.append(float(data))

            plt1, = plt.plot(range(1,len(pclb)+1), pclb, line_style[index], linewidth=2)
            plts.append(plt1)

    plt.xlabel("Time Step", size=15)
    plt.ylabel(r"Sample PCRLB ($m^2$)", size=15)
    plt.grid(True)
    plt.legend(plts, legends)
    plt.show()

    sys.exit(1)



    plts = []
    legends = ["Target1", "Observer1","Target2", "Observer2","Target3", "Observer3","Target4", "Observer4","Multi-target Observer"]
    line_style = ["-b","--b","-r","--r","-k","--k","-c","--c","--m"]

    indexes = [0,1,2,3,4]
    for idx,index in enumerate(indexes):
        truth_file = base_dir+ "truth_"+str(index)+".txt"
        sensor_file = base_dir+"sensor_single_"+str(index)+".txt"

        x_tuth = []; y_truth = []
        x_observer = []; y_observer = []
        if index<4:
            with open(truth_file,"r") as f:
                for line in f:
                    data = line.strip().split("\t")
                    x_tuth.append(float(data[0]))
                    y_truth.append(float(data[1]))

        with open(sensor_file,"r") as f:
            for line in f:
                data = line.strip().split("\t")
                x_observer.append(float(data[0]))
                y_observer.append(float(data[1]))

        if index<4:
            plt1, = plt.plot(x_tuth,y_truth,line_style[2*idx],linewidth = 2)
            plt2, = plt.plot(x_observer,y_observer,line_style[2*idx+1],linewidth=2)
            plts.append(plt1)
            plts.append(plt2)
        else:
            plt2, = plt.plot(x_observer, y_observer, line_style[2*idx], linewidth=4)
            plts.append(plt2)

    plt.xlabel("X (m)", size=15)
    plt.ylabel("Y (m)", size=15)
    plt.grid(True)
    plt.legend(plts, legends,loc=2)
    plt.show()


