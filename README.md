# neatness_estimator

ros package for object detection.

## instlation
```
rosdep install -y -r --from-paths --ignore-src .
catkin build neatness_estimator
```

## mask_rcnn instance segmentation and pcl proccessing
```
roslaunch neatness_estimator mask_rcnn_clustering2.launch
```

## multi_euclidean_clustering
 - input
 1. `~input_cluster_indices` (jsk_recognition_msgs/ClusterPointIndices)
 2. `~input_point_cloud` (sensor_msgs/PointCloud2)

 - output
 1. `~output_indices` (jsk_recognition_msgs/ClusterPointIndices)

```
roslaunch neatness_estimator multi_euclidean_clustering.launch
```

## Demo
![image not foud](./images/mask_rcnn_image.png)
![image not foud](./images/mask_rcnn_clustering.png)