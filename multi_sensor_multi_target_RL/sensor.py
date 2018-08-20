import numpy as np
import random
from motion_model import motion_model
import motion_init_object
from motion_init_object import motion_init_object
import matplotlib.pyplot as plt
from JPDAF_agent import JPDAF_agent


def get_limit(v_max,coeff,alpha1,alpha2,alpha1_,alpha2_,x):
    K = (2.0*v_max)/np.pi
    if x<=v_max and x>=-v_max:
        return (coeff*x,coeff)
    elif x>0:
        return (K*np.arctan(alpha1*x+alpha2),(K*alpha1)/(1+(alpha1*x+alpha2)**2))
    else:
        return (K*np.arctan(alpha1_*x+alpha2_),(K*alpha1_)/(1+(alpha1_*x+alpha2_)**2))

class sensor(motion_model,motion_init_object):
    def __init__(self,type,init_x,init_y):
        motion_model.__init__(self,1)
        motion_init_object.__init__(self)

        initial_location = [init_x,init_y]
        #initial_location = [X, Y]
        mean_x_vel = self.init_xdot
        mean_y_vel = self.init_ydot
        mean_x_acc = self.init_xdotdot
        mean_y_acc = self.init_ydotdot
        x_var = self.x_var
        y_var = self.y_var

        self.initial_location = initial_location
        self.current_location = self.initial_location
        self.historical_location = [self.initial_location]

        self.initial_velocity = [mean_x_vel, mean_y_vel]
        self.current_velocity = self.initial_velocity
        self.historical_velocity = [self.initial_velocity]
        self.x_var = x_var
        self.y_var = y_var

        #For constant accelaration model
        self.initial_acc = [mean_x_acc, mean_y_acc]
        self.current_acc = self.initial_acc
        self.historical_acc = [self.initial_acc]

        #For constant-turn model
        self.initial_speed = [self.init_speed]
        self.current_speed = self.initial_speed
        self.historical_speed = [self.initial_speed]

        self.initial_heading = [self.init_heading]
        self.current_heading = self.initial_heading
        self.historical_heading = [self.initial_heading]
        #generate an initial command

        self.initial_command = np.random.multinomial(1,np.array([1,1,1])/3.0).argmax()
        #current command
        self.current_command = self.initial_command
        self.historical_command = [self.initial_command]

        self.motion_type = type
        self.sensor_actions = []

        #list of measurements
        self.m = []
        self.reward = []
        self.avg_uncertainty = []
        self.uncertainty = []
        self.tracker_object = None

    def set_tracker_objects(self,tracker_objects):
        self.tracker_object = tracker_objects
        for i in range(0,len(self.tracker_object.tracks)): self.uncertainty.append([])

    def update_track_estimaes(self):self.tracker_object.update_target_states(self.current_location, self.m[-1]) #Update the target estimates


    def move_sensor(self,scen,params,v_max, coeff, alpha1, alpha2, alpha1_, alpha2_):

        track_local_actions = []
        for i in range(0, len(self.tracker_object.tracks)):
            current_state = list(self.tracker_object.tracks[i].x_k_k.reshape(len(self.tracker_object.tracks[i].x_k_k))) + list(self.current_location)

            x_slope = 2.0 / (scen.x_max - scen.x_min)
            y_slope = 2.0 / (scen.y_max - scen.y_min)
            vel_slope = 2.0 / (scen.vel_max - scen.vel_min)
            # normalization
            current_state[0] = -1 + x_slope * (current_state[0] - scen.x_min)
            current_state[1] = -1 + y_slope * (current_state[1] - scen.y_min)
            current_state[2] = -1 + vel_slope * (current_state[2] - scen.vel_min)
            current_state[3] = -1 + vel_slope * (current_state[3] - scen.vel_min)
            current_state[4] = -1 + x_slope * (current_state[4] - scen.x_min)
            current_state[5] = -1 + y_slope * (current_state[5] - scen.y_min)
            input_state = current_state
            #input_state_temp.append(input_state)  # store input-sates
            track_local_actions.append(self.generate_action(params, input_state, .01))

        #Update the location of the sensor
        normalized_state, index_matrix1, index_matrix2, slope = \
            self.update_location_decentralized_limit(track_local_actions, 1, params,
                                                  v_max, coeff, alpha1, alpha2, alpha1_, alpha2_, 0)


    def gen_sensor_reward(self,MAX_UNCERTAINTY,window_size,window_lag):
        """
        TASK update the reward for each sensor
        :param MAX_UNCERTAINTY:
        :param window_size:
        :param window_lag:
        :return:
        """

        for i in range(0, len(self.tracker_object.tracks)):
            unormalized_uncertainty = np.sum(self.tracker_object.tracks[i].p_k_k.diagonal())
            self.uncertainty[i].append((1.0 / MAX_UNCERTAINTY) * unormalized_uncertainty)

        this_uncertainty = []
        [this_uncertainty.append(self.uncertainty[x][-1]) for x in range(0, len(self.tracker_object.tracks))]
        self.avg_uncertainty.append(np.mean(this_uncertainty))

        if len(self.avg_uncertainty) < window_size + window_lag:
            self.reward.append(0)
        else:
            current_avg = np.mean(self.avg_uncertainty[-window_size:])
            prev_avg = np.mean(self.avg_uncertainty[-(window_size + window_lag):-window_lag])
            if current_avg < prev_avg or self.avg_uncertainty[-1] < .1:
                # if current_avg < prev_avg:
                self.reward.append(1)
            else:
                self.reward.append(0)

    def gen_measurements(self,t,measure,pd,landa):
        """
        TASK generate measurements for the current sensor
        :param t: list of target objects
        :param measure: measruement object
        :param pd: probability of detection
        :param landa: rate of false-alarm
        :return: Generated list of measurements
        """
        temp_m = []
        input_state_temp = []
        num_targets = len(t)
        for i in range(0, num_targets):
            # Consider impact of pd (probability of detection)
            if np.random.random() < pd:temp_m.append(measure.generate_bearing(t[i].current_location, self.current_location))
        # Now add False-alarms
        num_false_alrams = np.random.poisson(landa)
        false_measures = []
        for false_index in range(0, num_false_alrams):
            # generate x,y randomly
            random_x = 20000 * np.random.random() - 10000
            random_y = 20000 * np.random.random() - 10000
            false_measures.append([random_x, random_y])
            temp_m.append(measure.generate_bearing([random_x, random_y], self.current_location))
        self.m.append(temp_m)  # Number of measurements is not necessarily equal to that of targets

    def sigmoid(self,x, derivative=False):
        return self.sigmoid(x) * (1 - self.sigmoid(x)) if derivative else 1 / (1 + np.exp(-x))

    def plot_sensor_trajectory(self):
        x = []
        y = []
        [x.append(z[0]) for z in self.historical_location]
        [y.append(z[1]) for z in self.historical_location]
        plot1, = plt.plot(x,y,"b--",linewidth=1)
        plt.xlabel("x",size=15)
        plt.ylabel("y",size=15)
        plt.grid(True)
        plt.show()

    def generate_action(self,params,state,sigma):
        if self.motion_type==self.policy_command_type_linear:
            weight = params[0]['weight']
            Delta = np.random.normal(weight.dot(state), sigma)
            return (Delta)
        elif self.motion_type==self.policy_command_type_RBF:
            weight = params[1]['weight']
            Delta = np.random.normal(weight.dot(state), sigma)
            return (Delta)

        elif self.motion_type==self.policy_command_type_MLP:
            weight1 = params[2]['weight1']
            weight2 = params[2]['weight2']
            bias1 = params[2]['bias1']
            bias2 = params[2]['bias2']

            layer1_output = self.sigmoid(weight1.dot(state)+bias1)
            layer2_output = weight2.dot(layer1_output)+bias2
            Delta = np.random.normal(layer2_output.reshape([2]),sigma)
            return (Delta)

    def find_index(self,array,val):
        return (np.argwhere(array==val)[0][0])

    def update_location_decentralized_limit(self,target_actions,sigma,params,v_max,coeff,alpha1,alpha2,alpha1_,alpha2_,weight_index=0):
        x_actions = np.array(target_actions)[:, 0]
        y_actions = np.array(target_actions)[:, 1]
        num_targets = len(x_actions)
        index_matrix_1 = np.zeros([2*3,num_targets])
        index_matrix_2 = np.zeros([2*3, num_targets])
        #Create index matrix for back-propagating error
        min_x_action = np.min(x_actions)
        index = self.find_index(x_actions,min_x_action)
        index_matrix_1[0,index] = 1
        max_x_action = np.max(x_actions)
        #index = x_actions.searchsorted(max_x_action)
        index = self.find_index(x_actions, max_x_action)
        index_matrix_1[1,index] = 1

        mean_x_action = np.mean(x_actions)
        index_matrix_1[2,:] = (1.0/num_targets)*np.ones([1,num_targets])
        #median_x_action = np.median(x_actions)
        min_y_action = np.min(y_actions)
        index = self.find_index(y_actions,min_y_action)
        index_matrix_2[3,index] = 1
        max_y_action = np.max(y_actions)
        index = self.find_index(y_actions,max_y_action)
        index_matrix_2[4,index] = 1
        mean_y_action = np.mean(y_actions)
        index_matrix_2[5,:] = (1.0/num_targets)*np.ones([1,num_targets])
        #median_y_action = np.median(y_actions)
        #index = self.find_index(y_actions,median_y_action)
        #index_matrix_2[7,index] = 1


        unnormalized_features = [min_x_action,max_x_action,mean_x_action] \
                                + [min_y_action,max_y_action,mean_y_action]
        # normalize features
        MAX_VAL = 20
        MIN_VAL = -20
        slope = (2.0) / (MAX_VAL - MIN_VAL)
        normalized_features = []
        for s in unnormalized_features: normalized_features.append(1.0 + slope * (s - MAX_VAL))

        weight = params[0]['weight2']
        Delta = weight.dot(np.reshape(normalized_features,[len(normalized_features),1]))

        Delta[0], grad = get_limit(v_max, coeff, alpha1, alpha2, alpha1_, alpha2_, Delta[0])
        Delta[1], grad = get_limit(v_max, coeff, alpha1, alpha2, alpha1_, alpha2_, Delta[1])
        Delta = np.random.normal(Delta, sigma)

        self.sensor_actions.append(Delta)
        # Delta = np.random.normal(np.zeros([2]),sigma)
        new_x = self.current_location[0] + Delta[0]
        new_y = self.current_location[1] + Delta[1]
        self.current_location = [new_x[0], new_y[0]]
        self.historical_location.append(self.current_location)
        #final_action = np.mean(actions,axis=0)
        #self.sensor_actions.append(final_action)
        #new_x = self.current_location[0] + final_action[0]
        #new_y = self.current_location[1] + final_action[1]
        #self.current_location = [new_x, new_y]
        #self.historical_location.append(self.current_location)
        return (normalized_features,index_matrix_1*slope,index_matrix_2*slope,slope)

    def update_location_decentralized(self,target_actions,sigma,params):
        x_actions = np.array(target_actions)[:, 0]
        y_actions = np.array(target_actions)[:, 1]

        num_targets = len(x_actions)
        index_matrix_1 = np.zeros([2*3,num_targets])
        index_matrix_2 = np.zeros([2*3, num_targets])

        #Create index matrix for back-propagating error
        min_x_action = np.min(x_actions)
        index = self.find_index(x_actions,min_x_action)
        index_matrix_1[0,index] = 1
        max_x_action = np.max(x_actions)
        #index = x_actions.searchsorted(max_x_action)
        index = self.find_index(x_actions, max_x_action)
        index_matrix_1[1,index] = 1

        mean_x_action = np.mean(x_actions)
        index_matrix_1[2,:] = (1.0/num_targets)*np.ones([1,num_targets])

        #median_x_action = np.median(x_actions)

        #index = self.find_index(x_actions, median_x_action)
        #index_matrix_1[3,index] = 1


        min_y_action = np.min(y_actions)
        index = self.find_index(y_actions,min_y_action)
        index_matrix_2[3,index] = 1
        max_y_action = np.max(y_actions)
        index = self.find_index(y_actions,max_y_action)
        index_matrix_2[4,index] = 1
        mean_y_action = np.mean(y_actions)
        index_matrix_2[5,:] = (1.0/num_targets)*np.ones([1,num_targets])
        #median_y_action = np.median(y_actions)
        #index = self.find_index(y_actions,median_y_action)
        #index_matrix_2[7,index] = 1


        unnormalized_features = [min_x_action,max_x_action,mean_x_action] \
                                + [min_y_action,max_y_action,mean_y_action]
        # normalize features
        MAX_VAL = 20
        MIN_VAL = -20
        slope = (2.0) / (MAX_VAL - MIN_VAL)
        normalized_features = []
        for s in unnormalized_features: normalized_features.append(1.0 + slope * (s - MAX_VAL))

        weight = params[0]['weight2']
        Delta = np.random.normal(weight.dot(np.reshape(normalized_features,[len(normalized_features),1])), sigma)
        self.sensor_actions.append(Delta)
        # Delta = np.random.normal(np.zeros([2]),sigma)
        new_x = self.current_location[0] + Delta[0]
        new_y = self.current_location[1] + Delta[1]
        self.current_location = [new_x, new_y]
        self.historical_location.append(self.current_location)

        #final_action = np.mean(actions,axis=0)
        #self.sensor_actions.append(final_action)
        #new_x = self.current_location[0] + final_action[0]
        #new_y = self.current_location[1] + final_action[1]
        #self.current_location = [new_x, new_y]
        #self.historical_location.append(self.current_location)
        return (normalized_features,index_matrix_1*slope,index_matrix_2*slope,slope)

    def update_location_new_limit_v2(self,params,state,sigma,v_max,coeff,alpha1,alpha2,alpha1_,alpha2_,weight_index=0):
        if self.motion_type==self.policy_command_type_linear:
            weight = params[weight_index]['weight']
            Delta = np.random.normal((2.0*v_max/np.pi)*np.arctan(weight.dot(state)), sigma)
            Delta[0],grad = get_limit(v_max,coeff,alpha1,alpha2,alpha1_,alpha2_,Delta[0])
            Delta[1],grad = get_limit(v_max, coeff, alpha1, alpha2, alpha1_, alpha2_, Delta[1])
            #self.sensor_actions.append(Delta)
            # Delta = np.random.normal(np.zeros([2]),sigma)
            #new_x = self.current_location[0] + Delta[0]
            #new_y = self.current_location[1] + Delta[1]
            #self.current_location = [new_x, new_y]
            #self.historical_location.append(self.current_location)
            return (Delta)

    def update_location_new_limit(self,params,state,sigma,v_max,coeff,alpha1,alpha2,alpha1_,alpha2_,weight_index=0):
        if self.motion_type==self.policy_command_type_linear:
            weight = params[weight_index]['weight']
            Delta = weight.dot(state)
            Delta[0],grad = get_limit(v_max,coeff,alpha1,alpha2,alpha1_,alpha2_,Delta[0])
            Delta[1],grad = get_limit(v_max, coeff, alpha1, alpha2, alpha1_, alpha2_, Delta[1])
            Delta = np.random.normal(Delta, sigma)
            self.sensor_actions.append(Delta)
            # Delta = np.random.normal(np.zeros([2]),sigma)
            new_x = self.current_location[0] + Delta[0]
            new_y = self.current_location[1] + Delta[1]
            self.current_location = [new_x, new_y]
            self.historical_location.append(self.current_location)
            return (None)

    def update_location_new(self,params,state,sigma,weight_index=0):

        if self.motion_type==self.policy_command_type_linear:
            weight = params[weight_index]['weight']
            Delta = np.random.normal(weight.dot(state), sigma)
            self.sensor_actions.append(Delta)
            # Delta = np.random.normal(np.zeros([2]),sigma)
            new_x = self.current_location[0] + Delta[0]
            new_y = self.current_location[1] + Delta[1]
            self.current_location = [new_x, new_y]
            self.historical_location.append(self.current_location)
            return (None)
        elif self.motion_type==self.policy_command_type_RBF:
            weight = params[1]['weight']
            Delta = np.random.normal(weight.dot(state), sigma)
            self.sensor_actions.append(Delta)
            # Delta = np.random.normal(np.zeros([2]),sigma)
            new_x = self.current_location[0] + Delta[0]
            new_y = self.current_location[1] + Delta[1]
            self.current_location = [new_x, new_y]
            self.historical_location.append(self.current_location)
            return (None)

        elif self.motion_type==self.policy_command_type_MLP:
            weight1 = params[2]['weight1']
            weight2 = params[2]['weight2']
            bias1 = params[2]['bias1']
            bias2 = params[2]['bias2']

            layer1_output = self.sigmoid(weight1.dot(state)+bias1)
            layer2_output = weight2.dot(layer1_output)+bias2
            Delta = np.random.normal(layer2_output.reshape([2]),sigma)
            self.sensor_actions.append(Delta)
            # Delta = np.random.normal(np.zeros([2]),sigma)
            new_x = self.current_location[0] + Delta[0]
            new_y = self.current_location[1] + Delta[1]
            self.current_location = [new_x, new_y]
            self.historical_location.append(self.current_location)
            return (layer1_output)

        elif self.motion_type==self.policy_command_type_RANDOM:
            Delta = np.random.normal(np.zeros([2]), sigma)
            self.sensor_actions.append(Delta)
            # Delta = np.random.normal(np.zeros([2]),sigma)
            new_x = self.current_location[0] + Delta[0]
            new_y = self.current_location[1] + Delta[1]
            self.current_location = [new_x, new_y]
            self.historical_location.append(self.current_location)
            return (None)

    def update_location(self,weight,sigma,state):

        new_command = np.random.multinomial(1,np.array([1,1,1])/3.0).argmax()
        A,B = self.binary_command(new_command)

        if self.motion_type==self.constant_turn_type:
            #generate values for both the heading and speed
            heading = self.heading_rate

            #This is constant-turn model
            new_x = self.current_location[0] + self.T*self.current_speed[0]*np.cos(heading)
            new_y = self.current_location[1] + self.T*self.current_speed[0]*np.sin(heading)
            new_speed = self.current_speed[0] + self.speed_std*np.sqrt(self.T)*np.random.normal(0,1)
            #new_heading = self.current_heading[0] + self.heading_std*np.sqrt(self.T)*np.random.normal(0,1)
            new_heading = heading

            self.current_location = [new_x,new_y]
            self.current_speed = [new_speed]
            self.current_heading = [new_heading]
            self.historical_location.append(self.current_location)
            self.historical_speed.append(self.current_speed)
            self.historical_heading.append(self.current_heading)
        elif self.motion_type==self.policy_command_type:
            Delta =  np.random.normal(weight.dot(state),sigma)
            self.sensor_actions.append(Delta)
            #Delta = np.random.normal(np.zeros([2]),sigma)
            new_x = self.current_location[0] + Delta[0]
            new_y = self.current_location[1] + Delta[1]

            self.current_location = [new_x,new_y]
            self.historical_location.append(self.current_location)
        else:
            if self.motion_type==self.constant_velocity_type:
                A,B = self.constant_velocity(self.heading_rate)
            elif self.motion_type==self.constant_accelaration_type:
                A, B = self.constant_accelaration()

            noise_x = np.random.normal(0,self.x_var)
            noise_y = np.random.normal(0,self.y_var)

            if self.motion_type == self.constant_accelaration_type:
                current_state = [self.current_location[0], self.current_location[1], self.current_velocity[0],
                                 self.current_velocity[1],self.current_acc[0],self.current_acc[1]]
            else:
                current_state = [self.current_location[0], self.current_location[1], self.current_velocity[0],
                                self.current_velocity[1]]
            new_state = A.dot(current_state) + B.dot(np.array([noise_x, noise_y]))  # This is the new state

            new_location = [new_state[0], new_state[1]]
            self.current_location = new_location
            self.historical_location.append(self.current_location)

            new_velocity = [new_state[2], new_state[3]]
            self.current_velocity = new_velocity
            self.historical_velocity.append(self.current_velocity)

            if self.motion_type == self.constant_accelaration_type:
                new_acc = [new_state[4], new_state[5]]
                self.current_acc = new_acc
                self.historical_acc.append(self.current_acc)

            self.current_command = new_command
            self.historical_command.append(new_command)

if __name__=="__main__":
    s = sensor([500,500],3,-2,.01,.01)

    for n in range(0,500):
        s.update_location()

    



