import numpy as np

class motion_model:
    def __init__(self,sample_time):
        self.constant_velocity_type = "CONS_V"
        self.constant_accelaration_type = "CONS_A"
        self.constant_turn_type = "CONS_TURN"
        self.binary_command_type = "BINARY_COMM"
        self.policy_command_type = "POLICY_COMM"
        self.policy_command_type_linear = "POLICY_COMM_LINEAR"
        self.policy_command_type_RBF = "POLICY_COMM_RBF"
        self.policy_command_type_MLP = "POLICY_COMM_MLP"
        self.policy_command_type_RANDOM = "POLICY_COMM_RANDOM"

        self.T = sample_time

    def constant_velocity(self,heading_rate):
        A = np.array([[1,0,np.sin(heading_rate*self.T)/heading_rate,(np.cos(heading_rate*self.T)-1)/heading_rate]
                         ,[0,1,(1-np.cos(heading_rate*self.T))/heading_rate,np.sin(heading_rate*self.T)/heading_rate],
                      [0,0,np.cos(heading_rate*self.T),-np.sin(heading_rate*self.T)],
                      [0,0,np.sin(heading_rate*self.T),np.cos(heading_rate*self.T)]])
        B = np.array([[self.T**2/2.0,0],[0,self.T**2/2.0],[self.T,0],[0,self.T]])

        return (A,B)

    def constant_accelaration(self):
        A = np.array([[1,0,self.T,0,self.T**2/2.0,0],[0,1,0,self.T,0,self.T**2/2.0],
                      [0,0,1,0,self.T,0],[0,0,0,1,0,self.T],[0,0,0,0,1,0],[0,0,0,0,0,1]])
        B = np.array([[self.T**2/2.0,0],[0,self.T**2/2.0],[self.T,0],[0,self.T],[1,0],[0,1]])

        return (A,B)

    def binary_command(self,command):
        command1 = 0
        command2 = 0
        if command==0:
            command1 = 1
        elif command==1:
            command2 = 1
        else:
            command1 = 1
            command2 = 1

        A = np.array([[1, 0, self.T*(command1), 0], [0, 1, 0, self.T*(command2)], [0, 0, 1, 0], [0, 0, 0, 1]])
        B = np.array([[self.T ** 2 / 2.0*(command1), 0], [0, self.T ** 2 / 2.0*(command2)], [self.T*(command1), 0], [0, self.T*(command2)]])

        return (A,B)
    #def constant_turn(self):

