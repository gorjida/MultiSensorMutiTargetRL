
import random
import numpy as np

class motion_init_object:
    def __init__(self):

        self.loc_max = 50000
        self.loc_min = -50000
        self.v_max = 20
        self.v_min = -20



        #self.init_x = 10000*random.random()-5000
        #self.init_y = 10000*random.random()-5000

        self.init_x = 0
        self.init_y = 0

        self.init_xdot = 10
        self.init_ydot = -20
        self.init_xdotdot = -.1
        self.init_ydotdot = .2

        self.init_speed = 10
        self.init_heading = -np.pi/6

        self.heading_rate = .01*random.random()-.005
        self.heading_rate = .03

        self.speed_std = 1
        self.heading_std = .001
        self.x_var = 0
        self.y_var = 0