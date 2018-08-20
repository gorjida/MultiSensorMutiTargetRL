
class centralized_fusion:
    def __init__(self,s,initial_x_k_k,initial_p_k_k):
        self.num_targets = len(initial_x_k_k)
        self.sensors = s
        self.global_x_k_k = initial_x_k_k
        self.global_p_k_k = initial_p_k_k


    #For now, let's go with known assignment (this is not realisitic but just for sanity checks)
    def form_2d_assignment(self,local_x_k_k,local_p_k_k):
        local_to_global_map = {}
        for n in range(0,self.num_targets):
            local_to_global_map[n] = n
        return (local_to_global_map)

    def update_global(self,local_x_k_k,local_p_k_k):
        #Temporary
        local_to_global_map = self.form_2d_assignment(local_x_k_k,local_p_k_k)
        #Update global track estimates based on the estimates coming from each sensor
        for target_index in range(0,self.num_targets):
            


