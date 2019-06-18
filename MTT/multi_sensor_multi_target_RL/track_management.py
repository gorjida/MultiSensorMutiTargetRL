
import numpy as np

def track_normalized_distance(track1,track2):
    error = track1.x_k_k - track2.x_k_k
    cov = track1.p_k_k + track2.p_k_k
    error = error.reshape(4,1)
    normalized_distance = error.transpose().dot(np.linalg.inv(cov)).dot(error)
    return (normalized_distance)

def m_n_logic_management(trackers,M,N,
                         M_init,N_init,MAX_UNCERTAINTY,pd,vel_threshold_for_static,scan):
    list_of_tracks = trackers
    list_of_new_tracks = []

    num_alive = 0
    for track_ in list_of_tracks:
        track = track_.track

        #confirmed track
        if track.status==1:
            if len(track_.association)<N:
                list_of_new_tracks.append(track_)
                continue
            latest_assignments = track_.association[-N:]
            if sum(latest_assignments)>M: list_of_new_tracks.append(track_)

            #if scan>0 and scan%100==0:
             #   if np.mean(track.assignment)>.5: num_alive+=1


            #if np.mean(track.assignment_uncertainty[-50:])>.9:#np.mean(track.assignment[-100:])>.7:
           #     track.status = 1
            #    list_of_new_tracks.append(track_)
            #else:
             #   print("Track id="+str(track.track_id)+" with length="+str(len(track.assignment))+" is killed...")

        elif track.status==0:
            #tentative track (This is very hacky)

            if len(track_.association)<N:
                list_of_new_tracks.append(track_)
                continue
            #if sum(track.assignment_uncertainty) < M: continue
            #Hack (remove static estimates)
            #estimate = track.x_k_k
            #radial_vel = (estimate[0]*estimate[2]+estimate[1]*estimate[3])/np.linalg.norm([estimate[0],estimate[1]])
            #if np.abs(radial_vel)<vel_threshold_for_static: continue
            latest_assignments = track_.association[-N:]
            if (sum(latest_assignments)>=M):
                #print(track.uncertainty)
                track.status = 1
                list_of_new_tracks.append(track_)

        elif track.status==-1:

            #initialized track
            if len(track.assignment)<N_init:
                list_of_new_tracks.append(track_)
                continue
            #if track.uncertainty[-1]>track.uncertainty[-2]: continue
            latest_assignments = track.assignment[-N_init:]
            if (sum(latest_assignments)>=M_init):
                track.status = 0
                list_of_new_tracks.append(track_)

        #if sum(latest_assignments)>=M:
        #if (sum(latest_assignments)>=M):
            #Check uncertainty for tentative tracks
            #if track.status==0:
             #   if np.mean(track.assignment_uncertainty[-N:]) < .7 \
              #          or np.mean(track.uncertainty[-N:])>2*MAX_UNCERTAINTY: continue

            #confirm the track
            #if track.status[-1]!=0: track.status.append(1)
            #uncertainty = track.uncertainty[-1]
            #last_uncertainty = track.uncertainty[-N]
            #Don't add diverged track here

            #if uncertainty<last_uncertainty or uncertainty<MAX_UNCERTAINTY/10:
            #track.status = 1
            #list_of_new_tracks.append(track)
        #else:
         #   track.status.append(track.status[-1])

    #jpdaf_tracks.tracks = list_of_new_tracks
    #print(num_alive)
    return (list_of_new_tracks)