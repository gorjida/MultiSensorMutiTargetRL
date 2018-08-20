
import numpy as np
from scipy.stats import norm

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
    def __init__(self,init_tracks,threshold,target_pd,false_alarm_probability):

        self.tracks = init_tracks #estiamted tracks
        self.threshold = threshold #threshold for gating
        self.PD = target_pd #probability of target detection
        self.fa_probability = false_alarm_probability #probability of false-alarm
        self.target_measurement_prob = []


    def target_to_measurement_probability(self,sensor_state,measurements):
        """
        :param sensor_state: location of the sensor
        :param measurements: vector of received measurements
        :return: Calculates the probability of assigning a measurement to each target
        """
        num_targets = len(self.tracks) #number of tracks (or targets)
        num_measurements = len(measurements) #total number of measurements
        target_measurement_score_assignment = np.zeros([num_targets,num_measurements+1]) #A T by M+1 matrix==> tm-th entry: score of assigning the m-th measurement to the t-th target
        target_measurement_prob = np.zeros([num_targets,num_measurements+1]) #probability of assigning the m-th measurement to the t-th target
        gate_map = {} #indexes of measurements falling inside the gate of each target (soft-gating)


        for t in range(0,num_targets):
            target_gate_temp = []
            #prediction for this track
            self.tracks[t].predicted_state(sensor_state) #Do prediction for each target

            distance_map = {}
            for m in range(0,num_measurements+1):
                if m==0:
                    #Origin is false-alarm
                    target_gate_temp.append(0)
                    #No score is generated because there is no measurement falling in the gate of the current target
                else:
                    #calculate innovation
                    innov = measurements[m-1] - self.tracks[t].predicted_measurement
                    distance = (innov)**2/self.tracks[t].S_k


                    distance_map[m] = distance
                    #if m==1: print(distance)
                    if distance<self.threshold:

                        #gate_matrix[t,m] = 1
                        #calcualte the assignment score
                        assignment_score = norm.pdf(innov,0,np.sqrt(self.tracks[t].S_k)) #score of assigning the target to the measurement

                        target_measurement_score_assignment[t, m] = assignment_score #likelihood of assigning the m-th measurement to the t-th target
                        target_gate_temp.append(m)

            #if len(target_gate_temp)==1:
             #   min_assignment = sorted(distance_map.items(),key=operator.itemgetter(1))[0][1]
              #  target_gate_temp.append(sorted(distance_map.items(),key=operator.itemgetter(1))[0][0])

            gate_map[t] = target_gate_temp #list of all potential measurement-to-target assignments

        #Now, generate events based on the measurement-to-target associations
        #Exampel: gate_map[0] = [0,1], gate_map[1] = [0,1,2]===> permutations = [[0, 0], [0, 1], [0, 2], [1, 0], [1, 2]], num_targets= [0, 1, 1, 1, 2]

        permutations,num_targets_assignment = generate_association_events(gate_map) #generates all possible association events
        self.num_hypotheses = len(permutations)
        #print(permutations)

        #Loop over all possibe association events and calcualte the unnormalized probabilities
        total_event_score = 0
        for idx,permute in enumerate(permutations):
            #each entry in the permutations is an "EVENT"; for above example: permute= [0,0],[0,1],...
            number_of_detected_targets = num_targets_assignment[idx] #Number of targets detected in the current permutation==> for above example: 0,1,1,...
            number_of_false_measurements = num_measurements - number_of_detected_targets #Number of false measurements in the current event===> assuming 3 measurements, for above example: 3,2,2,...

            event_score = 1
            #calculate probability of this event
            for target_index,measurement_index in enumerate(permute):
                #vector: T \by 1 vector where t-th index denotes the index of the measurement assigned to the t-th target
                if measurement_index>0:
                    #There is a measurement assigned to the t-th target
                    #print(target_measurement_score_assignment[target_index,measurement_index])
                    event_score*= target_measurement_score_assignment[target_index,measurement_index]

            event_score*= ((self.PD)**number_of_detected_targets)*((1-self.PD)**(num_targets-number_of_detected_targets))*(self.fa_probability)**(number_of_false_measurements)
            total_event_score+= event_score

            #assign the aboce score to all the target/measurement probability
            for target_index,measurement_index in enumerate(permute): target_measurement_prob[target_index,measurement_index] += event_score

        #Normalize scores and form a probability
        if total_event_score>0:
            target_measurement_prob/= total_event_score
        #else:

        #print(permutations,target_measurement_prob)
        #self.target_measurement_prob = target_measurement_prob
        return (target_measurement_prob)

    def update_target_states(self,sensor_state,measurements):
        """

        :param sensor_state: state of the sensor
        :param measurements: list of measurements
        :return: update target states and associatd covariances
        """

        #First, calculate probability of assigning each measurement to a target
        target_measurement_prob = self.target_to_measurement_probability(sensor_state,measurements)
        #Now, update EKF_states


        for idx,track in enumerate(self.tracks):
            probs = target_measurement_prob[idx, :]
            no_meas_assignment_prob = probs[0]
            meas_assignment_prob = probs[1:]

            measurement_vector = track.meas_vec[-1] #measurement vector
            # calculate Kalman gain
            kalman_gain = (track.p_k_km1.dot(measurement_vector.transpose())) / track.S_k
            #calculate weighted innovation
            weighted_innov = 0
            expected_2_innov = 0
            expected_innov_2 = 0
            for meas_index,m in enumerate(measurements):
                expected_2_innov+= meas_assignment_prob[meas_index]*(m-track.predicted_measurement)
                expected_innov_2+= meas_assignment_prob[meas_index]*((m-track.predicted_measurement)**2)
                weighted_innov += meas_assignment_prob[meas_index]*(m-track.predicted_measurement)
            #store weighted innovation for this track
            track.innovation_list.append(weighted_innov)
            track.innovation_var.append(track.S_k)
            #update target-state
            track.x_k_k = track.x_k_km1 + kalman_gain * weighted_innov
            #update covariance
            track.p_k_k =  no_meas_assignment_prob*track.p_k_km1+(1-no_meas_assignment_prob)*(track.p_k_km1 - (kalman_gain.dot(measurement_vector)).dot(track.p_k_km1)) + \
                           kalman_gain.dot(kalman_gain.transpose())*(expected_innov_2-expected_2_innov**2)
            track.gain.append(kalman_gain)