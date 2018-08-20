import matplotlib.pyplot as plt
import numpy as np
import os

folder_path = "/Users/u6042446/Downloads/DeepSensorManagement-original-2/barriers/" \
              "linear_policy_discrete_reward_initial_condition_limit_vmax15_coeff9_with_Barrier/"
best_reward = folder_path+"best_reward.txt"
best_error = folder_path+"best_error.txt"
best_reward_var = folder_path+"best_var_reward.txt"
best_error_var = folder_path+"best_var_error.txt"

bad_reward = folder_path+"bad_reward.txt"
bad_error = folder_path+"bad_error.txt"
bad_reward_var = folder_path+"bad_reward_var.txt"
bad_error_var = folder_path+"bad_error_var.txt"

if __name__=="__main__":

    for filename in os.listdir(folder_path):
        print(filename)
        if filename.startswith("error_noise"):

            vals = []
            with open(folder_path+"/"+filename,"r") as f:
                for line in f:
                    d = float(line.strip())
                    if not vals:
                        vals.append(d)
                    else:
                        vals.append((vals[-1] + d) / 2.0)

                if np.mean(vals[-10:])<25:
                    plt.plot(range(1,len(vals)+1),vals,"b--",linewidth=1)
                else:
                    plt.plot(range(1, len(vals) + 1), vals, "r-", linewidth=1)


    plt.xlabel(r"Training Iteration ($\times 100$)",size=15)
    plt.ylabel("Average RMSE (m)",size=15)
    plt.grid(True)
    plt.show()

    #plt.ylabel("Average RMSE (m)",size=15)
    #plt.grid(True)
    #plt.show()
