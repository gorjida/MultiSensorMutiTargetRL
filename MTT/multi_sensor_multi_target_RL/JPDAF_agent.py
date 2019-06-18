
import numpy as np
from scipy.stats import norm
from scipy.stats import multivariate_normal

def recursive_permutations(nested_list,current_index,temp_indexes,permutations,num_targets):
    if current_index>len(nested_list)-1:

        is_duplicate = False
        distinct = set([])

        num_measuremet_to_target_assignments = 0
        for x in temp_indexes:
            if x in distinct and x>0:
                is_duplicate =True
                break
            distinct.add(x)
            if x>0: num_measuremet_to_target_assignments+=1
        if not is_duplicate:
            permutations.append(temp_indexes)
            num_targets.append(num_measuremet_to_target_assignments)
        return ()
    else:
        for x in nested_list[current_index]:
           recursive_permutations(nested_list,current_index+1,temp_indexes+[x],permutations,num_targets)

def generate_association_events(gate_map):
    distinct_entries = set([])
    possiblities = []
    for t,v in gate_map.items(): possiblities.append(v)
    permutations = []
    num_targets = []
    temp_indexes = []
    recursive_permutations(possiblities,0,temp_indexes,permutations,num_targets)

    return (permutations,num_targets)



class JPDAF_agent:
    def __init__(self,init_tracks,threshold,target_pd,false_alarm_probability,landa,scen,use_velocity=False):

        self.tracks = init_tracks #estiamted tracks
        self.threshold = threshold #threshold for gating
        self.PD = target_pd #probability of target detection
        self.PG = .999
        self.scen = scen
        self.MAX_UNCERTAINTY = 2*( (scen.x_max)**2+(scen.vel_max**2) )
        self.landa = landa
        self.fa_probability = false_alarm_probability #probability of false-alarm
        self.target_measurement_prob = []
        self.use_velocity = use_velocity
        if use_velocity:
            self.measurement_size = 3
        else:
            self.measurement_size = 2


    def get_gate_map(self,measurements,meas_bias_index):
        num_targets = len(self.tracks)  # number of tracks (or targets)
        num_measurements = len(measurements)  # total number of measurements
        target_measurement_score_assignment = np.zeros([num_targets,
                                                        num_measurements + 1])  # A T by M+1 matrix==> tm-th entry: score of assigning the m-th measurement to the t-th target

        gate_map = {}  # indexes of measurements falling inside the gate of each target (soft-gating)
        distance_map_targets = {}
        measurement_map = {}
        unassociated_measurement_map = {}
        for t in range(0, num_targets):
            target_gate_temp = []
            # prediction for this track
            distance_map = {}
            for m in range(0, num_measurements + 1):
                if m == 0:
                    # Origin is false-alarm
                    target_gate_temp.append(0)
                    # No score is generated because there is no measurement falling in the gate of the current target
                else:
                    if self.use_velocity:
                        meas = measurements[m - 1]
                    else:
                        meas = measurements[m - 1][0:2]
                    if not m in measurement_map: measurement_map[m] = 0
                    if not m in unassociated_measurement_map: unassociated_measurement_map[m] = 1E6
                    # calculate innovation and then apply gating
                    innov = meas.reshape(len(meas), 1) - \
                            self.tracks[t].predicted_measurement

                    distance = innov.transpose().dot(np.linalg
                                                          .inv(self.tracks[t].S_k)).dot(innov)
                    distance_map[m] = distance
                    if distance < self.threshold:
                        assignment_score = multivariate_normal.pdf(innov.reshape(3), mean=np.zeros(3),
                                                                   cov=self.tracks[t].S_k, allow_singular=True) / (
                                               self.landa)
                        target_measurement_score_assignment[
                            t, m] = assignment_score  # likelihood of assigning the m-th measurement to the t-th target
                        target_gate_temp.append(m+meas_bias_index)
                        measurement_map[m] += 1
                    if distance < unassociated_measurement_map[m]:
                        unassociated_measurement_map[m] = distance

            gate_map[t] = target_gate_temp  # list of all potential measurement-to-target assignments
            distance_map_targets[t] = distance_map
        return (gate_map,target_measurement_score_assignment,distance_map_targets)

    def target_to_measurement_probability(self,gate_map,track_to_meas_assignment):
        """
        :param sensor_state: location of the sensor
        :param measurements: vector of received measurements
        :return: Calculates the probability of assigning a measurement to each target
        """
        (num_targets,num_measurements_p1) = np.shape(track_to_meas_assignment)
        num_targets = len(self.tracks) #number of tracks (or targets)
        #num_measurements =  #total number of measurements
        #target_measurement_score_assignment = np.zeros([num_targets,num_measurements+1]) #A T by M+1 matrix==> tm-th entry: score of assigning the m-th measurement to the t-th target
        target_measurement_prob = np.zeros([num_targets,num_measurements_p1]) #probability of assigning the m-th measurement to the t-th target

        #gate_map = {} #indexes of measurements falling inside the gate of each target (soft-gating)
        #measurement_map = {}
        #unassociated_measurement_map = {}

        #for t in range(0,num_targets):

            #Update assignment (check if there is any measurement inside the gate)
            #if sum(gate_map[t])>0:
             #   self.tracks[t].assignment.append(1)
            #else:
             #   self.tracks[t].assignment.append(0)

        #print(gate_map)

        #unassociated_measurements = []
        #Index starts from "1" due to dummy measurement
        #for m in unassociated_measurement_map:
         #   if m==0: continue
         #   if unassociated_measurement_map[m] >self.threshold*1.15: unassociated_measurements.append(m-1)
        #Now, generate events based on the measurement-to-target associations
        #Exampel: gate_map[0] = [0,1], gate_map[1] = [0,1,2]===> permutations = [[0, 0], [0, 1], [0, 2], [1, 0], [1, 2]], num_targets= [0, 1, 1, 1, 2]
        permutations,num_targets_assignment = generate_association_events(gate_map) #generates all possible association events
        self.num_hypotheses = len(permutations)
        #print(permutations)
        #Loop over all possibe association events and calcualte the unnormalized probabilities
        total_event_score = 0
        for idx,permute in enumerate(permutations):
            #print(permute)
            #each entry in the permutations is an "EVENT"; for above example: permute= [0,0],[0,1],...
            #number_of_detected_targets = num_targets_assignment[idx] #Number of targets detected in the current permutation==> for above example: 0,1,1,...
            #number_of_false_measurements = num_measurements - number_of_detected_targets #Number of false measurements in the current event===> assuming 3 measurements, for above example: 3,2,2,...
            event_score = 1
            #calculate probability of this event
            for target_index,measurement_index in enumerate(permute):
                #vector: T \by 1 vector where t-th index denotes the index of the measurement assigned to the t-th target
                if measurement_index>0:
                    #There is a measurement assigned to the t-th target
                    #print(target_measurement_score_assignment[target_index,measurement_index])
                    event_score*= self.PD*track_to_meas_assignment[target_index,measurement_index]
                    #event_score*= ((self.PD)**number_of_detected_targets)*((1-self.PD)**(num_targets-number_of_detected_targets))*(self.fa_probability)**(number_of_false_measurements)
                    #event_score *= (self.PD)
                else:
                    event_score*= (1-self.PD*self.PG)*self.fa_probability

            #print(event_score)
            total_event_score+= event_score
            #assign the aboce score to all the target/measurement probability
            for target_index,measurement_index in enumerate(permute): target_measurement_prob[target_index,measurement_index] += event_score

        #Normalize scores and form a probability
        if total_event_score>0:
            target_measurement_prob/= total_event_score
        #else:

        #print(permutations,target_measurement_prob)
        #self.target_measurement_prob = target_measurement_prob
        #print(target_measurement_prob)
        return (target_measurement_prob)

    def update_target_states(self,sensor_state,measurements,target_measurement_prob, tracks_to_update):
        """
        :param sensor_state: state of the sensor
        :param measurements: list of measurements
        :return: update target states and associatd covariances
        """

        #First, calculate probability of assigning each measurement to a target

        #target_measurement_prob,unassociated_measurements = self.target_to_measurement_probability(sensor_state,measurements)
        #print(target_measurement_prob)
        #self.unassociated_measurements = unassociated_measurements
        #Now, update EKF_states
        #print(target_measurement_prob)
        for idx,track in enumerate(tracks_to_update):
            probs = target_measurement_prob[idx, :]
            #print(idx,probs)
            no_meas_assignment_prob = probs[0]
            meas_assignment_prob = probs[1:]

            measurement_matrix = track.meas_matrix[-1] #measurement vector
            # calculate Kalman gain
            kalman_gain = (track.p_k_km1.dot(measurement_matrix.transpose())) \
                .dot(np.linalg.inv(track.S_k))

            weighted_innov = np.zeros([self.measurement_size,1])
            expected_innov_2 = np.zeros([self.measurement_size,self.measurement_size])

            for meas_index,m in enumerate(measurements):
                if self.use_velocity:
                    m = m.reshape(self.measurement_size,1)
                else:
                    m = m[0:self.measurement_size].reshape(self.measurement_size, 1)

                #expected_2_innov+= meas_assignment_prob[meas_index]*(m-track.predicted_measurement)

                expected_innov_2+= meas_assignment_prob[meas_index]*\
                                   ((m-track.predicted_measurement).dot((m-track.predicted_measurement).transpose()))
                weighted_innov += meas_assignment_prob[meas_index]*(m-track.predicted_measurement)
            #store weighted innovation for this track

            track.innovation_list.append(weighted_innov)
            track.innovation_var.append(track.S_k)

            #update target-state

            track.x_k_k = track.x_k_km1 + kalman_gain.dot(weighted_innov)
            #update covariance

            track.p_k_k =  no_meas_assignment_prob*track.p_k_km1+(1-no_meas_assignment_prob)*(track.p_k_km1 - (kalman_gain.dot(track.S_k)).dot(kalman_gain.transpose())) + \
                           kalman_gain.dot((expected_innov_2-weighted_innov.dot(weighted_innov.transpose()))).dot(kalman_gain.transpose())
            #print(kalman_gain.dot((expected_innov_2-weighted_innov.dot(weighted_innov.transpose()))).dot(kalman_gain.transpose()))
            track.gain.append(kalman_gain)
            track.uncertainty.append(np.sum(np.diag(track.p_k_k)))
            if len(track.uncertainty)>1:
                if track.uncertainty[-1]<track.uncertainty[-2] or track.uncertainty[-1]<self.MAX_UNCERTAINTY/1E3:
                    track.assignment_uncertainty.append(1)
                else:
                    track.assignment_uncertainty.append(0)











