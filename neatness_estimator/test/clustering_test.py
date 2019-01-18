#!/usr/bin/env python
import numpy as np

class Cluster(object):
    def __init__(self, point, indices=[]):
        self.indices = indices
        self.point = point

class Clustering():

    def calc_distance(self, cluster_indices):
        min_norm = 2 ** 24
        min_pair = []
        for i, cluster_a in enumerate(cluster_indices):
            for j, cluster_b in enumerate(cluster_indices):
                if i == j:
                    continue
                norm = np.linalg.norm(cluster_a.point - cluster_b.point)
                if norm < min_norm:
                    min_norm = norm
                    min_pair = [i, j]

        return min_norm, min_pair

    def clustering(self, cluster_indices, threshold=0.1):
        min_norm, min_pair = self.calc_distance(cluster_indices)
    
        if min_norm > threshold:
            return cluster_indices

        # pop min_pair cluster
        next_cluster_indices = []
        for i, cluster in enumerate(cluster_indices):
            if i in min_pair:
                continue
            next_cluster_indices.append(cluster)

        next_cluster_indices.append(Cluster((cluster_indices[min_pair[0]].point + \
                                             cluster_indices[min_pair[1]].point) * 0.5,
                                            indices=(cluster_indices[min_pair[0]].indices + \
                                                     cluster_indices[min_pair[1]].indices)))

        return self.clustering(next_cluster_indices, threshold=threshold)
    
    def clustering_wrapper(self, points, threshold=1.0):
        n = len(points)
        cluster_indices = [Cluster(points[i, 0], indices=[i]) for i in range(n)]
        return self.clustering(cluster_indices, threshold=threshold)

def test():
    points = np.array([np.array([[7.84609842, 2.63768315, 0.8523097 ],
                                 [0.09129143, 0.1250577 , 0.11914825]]),
                       np.array([[7.84300373, 2.62830658, 0.85508216],
                                 [0.07923746, 0.17941594, 0.12557435]]),
                       np.array([[7.84815388, 2.69635992, 0.85843098],
                                 [0.25818682, 0.17383528, 0.09551519]]),
                       np.array([[7.81059589, 2.61523724, 0.85010148],
                                 [0.06206465, 0.09550524, 0.02151728]]),
                       np.array([[7.86117506, 3.6964224 , 0.88917613],
                                 [0.06942844, 0.12477469, 0.03618002]])])

    clustering = Clustering()
    result = clustering.clustering_wrapper(points, 0.1)
    for cluster in result:
        print(cluster.indices, cluster.point)
        for i in cluster.indices:
            print('--')
            print(points[i])
            
if __name__=='__main__':
    test()
