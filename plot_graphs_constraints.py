import matplotlib.pyplot as plt
import numpy as np
import os

base_folder_path = "/Users/u6042446/Downloads/DeepSensorManagement-original-2/ResultsMultipleConstrained/"

if __name__=="__main__":
    v_max = [10,15,30,40]
    post_fix = ["linear_policy_discrete_reward_multiple_T210_Varying_Number_Constrained_withBarrier_vmax10",
                "linear_policy_discrete_reward_multiple_T210_Varying_Number_Constrained_withBarrier_vmax15",
                "linear_policy_discrete_reward_multiple_T210_Varying_Number_Constrained_withBarrier_vmax30",
                "linear_policy_discrete_reward_multiple_T210_Varying_Number_Constrained_withBarrier_vmax40",
                "linear_policy_discrete_reward_multiple_T210_Varying_Number"]

    indexes = [7,10,1,9,9]
    line_style = ["-.b", "-.r", "-.c", "-k", "-m", "-g"]

    plts = []
    legends = []
    for ii,p in enumerate(post_fix):
        folder_path = base_folder_path+p
        for filename in os.listdir(folder_path):
            #if filename.startswith("error_noise:"):

            if filename.startswith("error_noise"):
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

                    vals = vals[0:100]
                    #if ii==len(post_fix)-1:
                     #   temp = []
                      #  for i,x in enumerate(vals):
                       #     if i<=40:
                        #        temp.append(x)
                         #   else:
                          #      temp.append(x-50)
                        #kvals = temp
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
    plt.ylabel("Average RMSE (m)",size=20)
    plt.grid(True)
    plt.legend(plts,legends,fontsize=15)
    plt.show()

    #plt.ylabel("Average RMSE (m)",size=15)
    #plt.grid(True)
    #plt.show()
