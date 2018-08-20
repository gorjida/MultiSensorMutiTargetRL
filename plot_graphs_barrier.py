import matplotlib.pyplot as plt
import numpy as np


folder_path = "/Users/u6042446/Downloads/DeepSensorManagement-original-2/TEST/"
best_reward = folder_path+"best_reward.txt"
best_error = folder_path+"best_error.txt"
best_reward_var = folder_path+"best_var_reward.txt"
best_error_var = folder_path+"best_var_error.txt"

best_reward_1000 = folder_path+"best_reward_1000.txt"
best_error_1000 = folder_path+"best_error_1000.txt"
best_reward_var_1000 = folder_path+"best_var_reward_1000.txt"
best_error_var_1000 = folder_path+"best_var_error_1000.txt"

best_reward_500 = folder_path+"best_reward_500.txt"
best_error_500 = folder_path+"best_error_500.txt"
best_reward_var_500 = folder_path+"best_var_reward_500.txt"
best_error_var_500 = folder_path+"best_var_error_500.txt"

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

    rew_1000 = []
    rew_var_1000 = []
    err_1000 = []
    err_var_1000 = []

    with open(best_reward_1000, "r") as f:
        for line in f:
            d = float(line.strip())
            if not rew_1000:
                rew_1000.append(d)
            else:
                rew_1000.append((rew_1000[-1] + d) / 2.0)

    with open(best_error_1000, "r") as f:
        for line in f:
            d = float(line.strip())
            if not err_1000:
                err_1000.append(d)
            else:
                err_1000.append((err_1000[-1] + d) / 2.0)

    with open(best_reward_var_1000, "r") as f:
        for line in f:
            d = float(line.strip())
            if not rew_var_1000:
                rew_var_1000.append(d)
            else:
                rew_var_1000.append((rew_var_1000[-1] + d) / 2.0)

    with open(best_error_var_1000, "r") as f:
        for line in f:
            d = float(line.strip())
            if not err_var_1000:
                err_var_1000.append(d)
            else:
                err_var_1000.append((err_var_1000[-1] + d) / 2.0)

    rew_500 = []
    rew_var_500 = []
    err_500 = []
    err_var_500 = []

    with open(best_reward_500, "r") as f:
        for line in f:
            d = float(line.strip())
            if not rew_500:
                rew_500.append(d)
            else:
                rew_500.append((rew_500[-1] + d) / 2.0)

    with open(best_error_500, "r") as f:
        for line in f:
            d = float(line.strip())
            if not err_500:
                err_500.append(d)
            else:
                err_500.append((err_500[-1] + d) / 2.0)

    with open(best_reward_var_500, "r") as f:
        for line in f:
            d = float(line.strip())
            if not rew_var_500:
                rew_var_500.append(d)
            else:
                rew_var_500.append((rew_var_500[-1] + d) / 2.0)

    with open(best_error_var_500, "r") as f:
        for line in f:
            d = float(line.strip())
            if not err_var_500:
                err_var_500.append(d)
            else:
                err_var_500.append((err_var_500[-1] + d) / 2.0)

    coeff = []
    for n in range(0,100):
        if n>50:
            coeff.append((1-(2.0/np.pi)*np.arctan((n-50)/100.0)))
        else:
            coeff.append(1)
    #plot results
    iterations = range(1,len(rew)+1)

    error_rew = np.sqrt(np.array(rew_var))#*np.array(coeff)
    error_rew_1000 = np.sqrt(np.array(rew_var_1000))# * np.array(coeff)
    error_rew_500 = np.sqrt(np.array(rew_var_500))# * np.array(coeff)
    #plt.errorbar(iterations, rew, yerr=error_rew, fmt="or--", ecolor='b',elinewidth=1,capsize=5)

    #plt1, = plt.plot(iterations,rew,"r--",linewidth=3)
    #plt.fill_between(iterations,rew-error_rew,rew+error_rew,facecolor='blue')

    plt2, = plt.plot(iterations, rew_1000, "k--", linewidth=3)
    plt.fill_between(iterations, rew_1000 - error_rew_1000, rew_1000 + error_rew_1000, facecolor='green')

    #plt3, = plt.plot(iterations, rew_500, "v--", linewidth=3)
    #plt.fill_between(iterations, rew_500 - error_rew_500, rew_500 + error_rew_500, facecolor='magenta')

    plt.xlabel(r"Training Iteration ($\times 100$)",size=15)
    plt.ylabel("Average Reward",size=15)
    plt.grid(True)
    plt.show()
    

    error_error = np.sqrt(np.array(err_var_1000))# * np.array(coeff)
    # plt.errorbar(iterations, rew, yerr=error_rew, fmt="or--", ecolor='b',elinewidth=1,capsize=5)
    plt.plot(iterations, err_1000, "ko-", linewidth=3)
    plt.fill_between(iterations, err_1000 - error_error, err_1000 + error_error, facecolor='blue')

    plt.xlabel(r"Training Iteration ($\times 100$)",size=15)
    plt.ylabel("Average RMSE (m)",size=15)
    plt.grid(True)
    plt.show()
