import matplotlib.pyplot as plt
import numpy as np
import os

base_folder_path = "/Users/u6042446/Downloads/DeepSensorManagement-original-2/barriers/"

if __name__=="__main__":
    v_max = [5,7,10,15,"unconstained"]
    post_fix = ["linear_policy_discrete_reward_initial_condition_limit_vmax5_coeff9_with_Barrier",
                "linear_policy_discrete_reward_initial_condition_limit_vmax7_coeff9_with_Barrier",
                "linear_policy_discrete_reward_initial_condition_limit_vmax10_coeff9_with_Barrier_C2000",
                "linear_policy_discrete_reward_initial_condition_limit_vmax15_coeff9_with_Barrier",
                "linear_policy_discrete_reward_initial_condition_limit_vmax1500_coeff9_with_Barrier"]

    indexes = [1,3,9,3,3]
    line_style = ["-.b","-.r","-.c","-k","-m","-g"]

    plts = []
    legends = []
    for ii,p in enumerate(post_fix):
        folder_path = base_folder_path+p
        for filename in os.listdir(folder_path):
            #if filename.startswith("error_noise:"):

            if filename.startswith("reward"):
                index = int(filename.split("_linear_6states.txt")[0].split("_")[-1])
                if indexes[ii]!=index: continue
                vals = []
                with open(folder_path+"/"+filename,"r") as f:
                    for line in f:
                        d = float(line.strip())
                        if not vals:
                            vals.append(d)
                        else:
                            vals.append((vals[-1] + d) / 2.0)
                    plt_obj, = plt.plot(range(1,len(vals)+1),vals,line_style[ii],linewidth=2)
                    plts.append(plt_obj)
                    if ii<len(post_fix)-1:
                        legends.append(r"$v^o_{max}=$"+str(v_max[ii])+r"$\frac{m}{s}$")
                    else:
                        legends.append("Unconstrained")
                    #if np.mean(vals[-10:])<25:
                     #   plt.plot(range(1,len(vals)+1),vals,"b--",linewidth=1)
                    #else:
                     #   plt.plot(range(1, len(vals) + 1), vals, "r-", linewidth=1)


    plt.xlabel(r"Training Iteration ($\times 100$)",size=20)
    plt.ylabel("Average Sum-reward",size=20)
    plt.grid(True)
    plt.legend(plts,legends,fontsize=15)
    plt.show()

    #plt.ylabel("Average RMSE (m)",size=15)
    #plt.grid(True)
    #plt.show()
