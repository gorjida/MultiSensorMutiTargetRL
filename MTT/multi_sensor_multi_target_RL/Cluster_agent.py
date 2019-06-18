
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.datasets.samples_generator import make_blobs
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
class DBscan_cluster:
    def __init__(self,max_distance,
                 vel_distance=None,cov=None):
        self.max_distance = max_distance
        self.cov = cov
        self.vel_distance = vel_distance
        #self.min_samples = min_samples

    def distance_func(self,s1,s2):

        delta_v = np.abs(s1[-1] - s2[-1])
        s1 = s1[0:2]
        s2 = s2[0:2]

        if self.vel_distance is not None:
            if delta_v>self.vel_distance: return (1E6)

        s1 = np.array(s1).reshape(len(s1),1)
        s2 = np.array(s2).reshape(len(s2),1)
        catrz1 = np.array([s1[0]*np.cos(s1[1]),s1[0]*np.sin(s1[1])]).reshape(2,1)
        catrz2 = np.array([s2[0] * np.cos(s2[1]), s2[0] * np.sin(s2[1])]).reshape(2, 1)
        error = catrz1 - catrz2
        if self.cov is not None:
            distance = error.transpose().dot(np.linalg.inv(self.cov)).dot(error)
        else:
            distance = np.linalg.norm(error)
        return (distance)

    def plot_cluster(self,data_in_cartz):

        colors = ["b", "r", "k", "m", "g"]
        for index,point in enumerate(data_in_cartz):
            cluster_index = self.labels[index]
            if cluster_index==-1: continue
            centroid = self.merged_clustered_instances[cluster_index]
            centroid_x = centroid[0]*np.cos(centroid[1])
            centroid_y = centroid[0]*np.sin(centroid[1])
            plt.plot(centroid_x,centroid_y,"s",markerfacecolor=colors[cluster_index],
                        markeredgecolor='k', markersize=20)
            plt.plot(point[0], point[1], 'o', markerfacecolor=colors[cluster_index],
                        markeredgecolor='k', markersize=14)

        plt.show()

    def cluster_data(self,data,min_samples):
        data_save = np.array(data)
        data = np.array(data)[:,0:2] #remove velocity
        self.data = data
        #scale data
        #data = StandardScaler().fit_transform(data)
        dbscan = DBSCAN(eps=self.max_distance
                        ,metric=self.distance_func
                        ,min_samples=min_samples)
        db = dbscan.fit(data)
        self.labels = db.labels_
        #n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
        #unique_labels = set(labels)
        return (self.labels)

        """
        for index,point in enumerate(data_save):
            cluster_index = self.labels[index]
            if cluster_index==-1: continue
            if cluster_index not in clustered_instances: clustered_instances[cluster_index] = []
            clustered_instances[cluster_index].append(point)

        self.merged_clustered_instances = {}
        merged_clustered_instances = []
        for index in clustered_instances:
            mean = np.mean(clustered_instances[index],axis=0)
            vels = np.array(clustered_instances[index])[:, 2]
            vel_stats = [np.min(vels), np.max(vels), np.mean(vels)]
            self.merged_clustered_instances[index] = mean
            merged_clustered_instances.append(mean)
        return (self.labels,merged_clustered_instances)
        """



if __name__=="__main__":
    centers = [[1, 1], [-1, -1], [1, -1]]
    X, labels_true = make_blobs(n_samples=750, centers=centers, cluster_std=0.4,
                                random_state=0)

    X = StandardScaler().fit_transform(X)

    cov = np.cov(X.transpose())
    cluster_agent = DBscan_cluster(.03,5,cov)
    db = cluster_agent.cluster_data(X)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_

    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)

    print('Estimated number of clusters: %d' % n_clusters_)
    print('Estimated number of noise points: %d' % n_noise_)
    print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels_true, labels))
    print("Completeness: %0.3f" % metrics.completeness_score(labels_true, labels))
    print("V-measure: %0.3f" % metrics.v_measure_score(labels_true, labels))
    print("Adjusted Rand Index: %0.3f"
          % metrics.adjusted_rand_score(labels_true, labels))
    print("Adjusted Mutual Information: %0.3f"
          % metrics.adjusted_mutual_info_score(labels_true, labels))
    print("Silhouette Coefficient: %0.3f"
          % metrics.silhouette_score(X, labels))

    import matplotlib.pyplot as plt

    # Black removed and is used for noise instead.
    unique_labels = set(labels)
    colors = [plt.cm.Spectral(each)
              for each in np.linspace(0, 1, len(unique_labels))]
    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise.
            col = [0, 0, 0, 1]

        class_member_mask = (labels == k)

        xy = X[class_member_mask & core_samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                 markeredgecolor='k', markersize=14)

        xy = X[class_member_mask & ~core_samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                 markeredgecolor='k', markersize=6)

    plt.title('Estimated number of clusters: %d' % n_clusters_)
    plt.show()