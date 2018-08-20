
class centralized_fusion:
    def __init__(self,s,initial_x_k_k,initial_p_k_k):
        self.sensors = s
        self.global_x_k_k = None
        self.global_p_k_k = None