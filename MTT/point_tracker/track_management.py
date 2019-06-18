
import numpy as np

def track_normalized_distance(track1,track2):
    error = track1.x_k_k - track2.x_k_k
    cov = track1.p_k_k + track2.p_k_k
    error = error.reshape(4,1)
    normalized_distance = error.transpose().dot(np.linalg.inv(cov)).dot(error)
    return (normalized_distance)

def m_n_logic_management(trackers,M,N,M_init,N_init):
    list_of_tracks = trackers
    list_of_new_tracks = []

    num_alive = 0
    for track in list_of_tracks:
        if track.is_active==1:
            if len(track.track_assignment)<N:
                list_of_new_tracks.append(track)
                continue
            latest_assignments = track.track_assignment[-N:]
            if sum(latest_assignments)>=M: list_of_new_tracks.append(track)
        elif track.is_tentative==1:
            #tentative track (This is very hacky)
            if len(track.track_assignment)<N:
                list_of_new_tracks.append(track)
                continue
            latest_assignments = track.track_assignment[-N:]
            if (sum(latest_assignments)>=M):
                track.is_tentative = 0
                track.is_active = 1
                list_of_new_tracks.append(track)

        elif track.is_initiated==1:

            #initialized track
            if len(track.track_assignment)<N_init:
                list_of_new_tracks.append(track)
                continue
            #if track.uncertainty[-1]>track.uncertainty[-2]: continue
            latest_assignments = track.track_assignment[-N_init:]
            if (sum(latest_assignments)>=M_init):
                track.is_initiated = 0
                track.is_tentative = 1
                list_of_new_tracks.append(track)
    return (list_of_new_tracks)