from target import target
import numpy as np
import matplotlib.pyplot as plt
from measurement import measurement
from scenario import scenario


x_min =  -1000
x_max =  1000
y_min =  -5000
y_max = 5000
max_scans = 50

initial_target_locs = [[100,300],[100,450]
    ,[150,100],[150,520],[387.5,80]]

initial_target_vels = [[15,0],[15,-10],
                       [15,10],[15,0],[0,15]]
num_targets = int(len(initial_target_locs))

targets = []
rates = np.array([1E-6,1E-6,1E-6,1E-6,1E-6])


def writer(pd,landa,mcmc,data):
    file = "scenarios/scenario_" + str(pd) + "_"+str(landa) + "_" + str(mcmc)+".txt"
    writer = open(file,"w")
    writer.write("scan\trange\tbearing\tvelocity\n")
    for scan,d in enumerate(data):
        for meas in d:
            tmp = [str(scan)]
            [tmp.append(str(x)) for x in meas]
            writer.write("\t".join(tmp)+"\n")

        writer.write("\n")
    writer.close()

def writer_truth(pd,landa,mcmc,X,Y):
    file = "scenarios/truth_" + str(pd) + "_"+str(landa) + "_" + str(mcmc)+".txt"
    writer = open(file,"w")
    writer.write("scan\ttarget_index\tX\tY\n")
    for scan,d in enumerate(X):
        for index,x in enumerate(d):
            tmp = [str(scan),str(index),str(x),str(Y[scan][index])]
            writer.write("\t".join(tmp)+"\n")
        writer.write("\n")
    writer.close()

if __name__=="__main__":

    pd = .95
    landa = 10
    bearing_std = (5/180)*np.pi
    range_std = 3
    vel_std = .2
    mcmc = 0
    for mcmc in np.arange(0, 100, 1):
        scen = scenario(bearing_std, range_std, vel_std, pd, landa)
        meas = measurement(scen)

        X = []
        VX = []
        Y = []
        VY = []
        targets = []
        for index in np.arange(0,num_targets,1):
            t = target(initial_target_locs[index], initial_target_vels[index][0],
                       initial_target_vels[index][1], .001, .001, "CONS_V", 1, rates[index])
            targets.append(t)

        num_crossing_targets = []
        for scan in np.arange(0,max_scans,1):

            distance_matrix = np.zeros([5, 5])
            tmp_x = []
            tmp_y = []
            tmp_vx = []
            tmp_vy = []
            for index,t in enumerate(targets):

                """
                if scan>40 and scan<50 and index==1:
                    t.rate = .1
                if scan>50 and index==1:
                    t.rate = 1E-6
                    #t.current_velocity = [15,15]

                if scan>40 and scan<50 and index==2:
                    t.rate = -.1
                if scan>50 and index==2:
                    t.rate = 1E-6
                """

                location = t.current_location
                velocity = t.current_velocity
                tmp_x.append(location[0])
                tmp_y.append(location[1])
                tmp_vx.append(velocity[0])
                tmp_vy.append(velocity[1])
                t.update_location()
            X.append(tmp_x)
            Y.append(tmp_y)
            VX.append(tmp_vx)
            VY.append(tmp_vy)

        x_max = np.max(X) + 200
        x_min = np.min(X) - 200
        y_max = np.max(Y) + 200
        y_min = np.min(Y) - 200


    #Generate measurements
        sensor_loc = [0,0]
        sensor_vel = [0,0]
        list_of_measurements = []
        num_measurements = []
        for scan in np.arange(0,max_scans,1):
            this_scan_meas = []
            for t_index in np.arange(0,num_targets,1):
                if np.random.rand()<pd:
                    #Generate measurement for each target
                    target_loc = [X[scan][t_index],Y[scan][t_index]]
                    target_vel = [VX[scan][t_index], VY[scan][t_index]]
                    #measurements
                    range = meas.generate_range(target_loc,sensor_loc)
                    bearing = meas.generate_bearing(target_loc,sensor_loc)
                    vel = meas.generate_radial_velocity(target_loc,sensor_loc,target_vel,sensor_vel)
                    this_scan_meas.append([range,bearing,vel])
            #Generate false measurements
            num_false_alrams = np.random.poisson(landa)
            false_measures = []
            for false_index in np.arange(0, num_false_alrams, 1):
                # generate x,y randomly
                random_x = ( x_max- x_min) * np.random.random() + x_min
                random_y = (y_max - y_min) * np.random.random() + y_min
                # Low-velocity false-alarms
                random_xdot = 2 * np.random.random() - 1
                random_ydot = 2 * np.random.random() - 1
                false_measures.append([random_x, random_y])
                bearing = meas.generate_bearing([random_x, random_y], sensor_loc)
                range = meas.generate_range([random_x, random_y], sensor_loc)
                velocity = meas.generate_radial_velocity([random_x, random_y]
                                                            , sensor_loc,
                                                            [random_xdot, random_ydot], sensor_vel)
                this_scan_meas.append([range, bearing, vel])

            list_of_measurements.append(this_scan_meas)
            num_measurements.append(len(this_scan_meas))

        print(min(num_measurements),max(num_measurements),np.mean(num_measurements))
        writer(pd,landa,mcmc,list_of_measurements)
        writer_truth(pd,landa,mcmc,X,Y)


    sys.exit(0)
    X = np.array(X)
    Y = np.array(Y)
    X = np.array(X)
    Y = np.array(Y)
    plts = []
    legends = []
    colors = ["b","r","k","g","m"]
    for index in np.arange(0,num_targets,1):
        plts.append(plt.plot(X[:,index],Y[:,index],colors[index],label="Target-"+str(index+1)))
        legends.append("Target-"+str(index+1))

    plts.append(plt.plot(0,0,"ks",markersize=10,label="Sensor"))
    legends.append("Sensor")
    plt.legend()
    plt.xlabel("X (m)",size=20)
    plt.ylabel("Y (m)",size=20)
    plt.grid(True)
    plt.show()

