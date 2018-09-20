from motion_model import motion_model
import numpy as np

class target(motion_model):
    def __init__(self, initial_location, mean_x_vel, mean_y_vel, x_var, y_var,motion_type):
        motion_model.__init__(self,1)
        self.motion_type = motion_type
        self.initial_location = initial_location
        self.current_location = self.initial_location
        self.historical_location = [self.initial_location]

        self.initial_velocity = [mean_x_vel,mean_y_vel]
        self.current_velocity =  self.initial_velocity
        self.historical_velocity = [self.initial_velocity]
        self.x_var = x_var
        self.y_var = y_var

        # generate an initial command
        #self.initial_command
        # current command
        #self.current_command
        #self.historical_command = [self.initial_command]

    def update_location(self):

        if self.motion_type==self.constant_velocity_type:
            A,B = self.constant_velocity(1E-4)

        #sigma_x = ((self.x_var)*abs(self.initial_velocity[0]))/20.0
        #sigma_y = ((self.y_var) * abs(self.initial_velocity[1])) / 20.0
        noise_x = np.random.normal(0, self.x_var)
        noise_y = np.random.normal(0, self.y_var)

        current_state = [self.current_location[0],self.current_location[1],self.current_velocity[0],self.current_velocity[1]]
        #current_state = [self.current_location[0], self.current_location[1], t_x_vel,
         #                t_y_vel]
        new_state = A.dot(current_state)+B.dot(np.array([noise_x,noise_y]))#This is the new state


        #if new_state[2]>0:
         #   new_state[2] = max(min(new_state[2],self.initial_velocity[0]*(1+self.x_var)),self.initial_velocity[0]*(1-self.x_var))
       # else:
        #    new_state[2] = max(min(new_state[2], self.initial_velocity[0] * (1-self.x_var)), self.initial_velocity[0] * (1+self.x_var))

        #if new_state[3]>0:
         #   new_state[3] = max(min(new_state[3],self.initial_velocity[1]*(1+self.x_var)),self.initial_velocity[1]*(1-self.x_var))
        #else:
         #   new_state[3] = max(min(new_state[3], self.initial_velocity[1] * (1-self.x_var)), self.initial_velocity[1] * (1+self.x_var))


        new_location = [new_state[0],new_state[1]]
        self.current_location = new_location
        self.historical_location.append(self.current_location)

        new_velocity = [new_state[2], new_state[3]]
        self.current_velocity = new_velocity
        self.historical_velocity.append(self.current_velocity)

if __name__=="__main__":
    t = target([100,100],2,2,.1,.1,"CONS_V")
    for n in range(0,500):
        t.update_location()



