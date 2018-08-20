import matplotlib.pyplot as plt
import numpy as np


folder_path = "/Users/u6042446/Downloads/DeepSensorManagement-original-2/TEST/"
best_reward = folder_path+"best_reward.txt"
best_error = folder_path+"best_error.txt"
best_reward_var = folder_path+"best_var_reward.txt"
best_error_var = folder_path+"best_var_error.txt"

bad_reward = folder_path+"bad_reward.txt"
bad_error = folder_path+"bad_error.txt"
bad_reward_var = folder_path+"bad_reward_var.txt"
bad_error_var = folder_path+"bad_error_var.txt"

if __name__=="__main__":

    rew = []
    rew_var = []
    err = []
    err_var = []

    with open(best_reward,"r") as f:
        for line in f:
            d = float(line.strip())
            if not rew:
                rew.append(d)
            else:
                rew.append((rew[-1]+d)/2.0)

    with open(best_error,"r") as f:
        for line in f:
            d = float(line.strip())
            if not err:
                err.append(d)
            else:
                err.append((err[-1]+d)/2.0)

    with open(best_reward_var,"r") as f:
        for line in f:
            d = float(line.strip())
            if not rew_var:
                rew_var.append(d)
            else:
                rew_var.append((rew_var[-1]+d)/2.0)

    with open(best_error_var,"r") as f:
        for line in f:
            d = float(line.strip())
            if not err_var:
                err_var.append(d)
            else:
                err_var.append((err_var[-1]+d)/2.0)

    coeff = []
    for n in range(0,100):
        if n>50:
            coeff.append((1-(2.0/np.pi)*np.arctan((n-50)/60.0)))
        else:
            coeff.append(1)
    #plot results
    iterations = range(1,len(rew)+1)

    error_rew = np.sqrt(np.array(rew_var))*np.array(coeff)
    #plt.errorbar(iterations, rew, yerr=error_rew, fmt="or--", ecolor='b',elinewidth=1,capsize=5)
    plt1, = plt.plot(iterations,rew,"r",linewidth=3)
    plt.fill_between(iterations,rew-error_rew,rew+error_rew,facecolor='grey')

    plt.xlabel(r"Training Iteration ($\times 100$)",size=20)
    plt.ylabel("Average Reward",size=20)
    plt.grid(True)
    plt.show()
    

    error_error = np.sqrt(np.array(err_var)) * np.array(coeff)
    # plt.errorbar(iterations, rew, yerr=error_rew, fmt="or--", ecolor='b',elinewidth=1,capsize=5)
    plt.plot(iterations, err, "r", linewidth=3)
    plt.fill_between(iterations, err - error_error, err + error_error, facecolor='grey')

    plt.xlabel(r"Training Iteration ($\times 100$)",size=20)
    plt.ylabel("Average RMSE (m)",size=20)
    plt.grid(True)
    plt.show()
