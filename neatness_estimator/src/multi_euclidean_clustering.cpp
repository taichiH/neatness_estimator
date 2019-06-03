#include "neatness_estimator/multi_euclidean_clustering.h"

namespace neatness_estimator
{

  void MultiEuclideanClustering::onInit()
  {
    nh_ = getNodeHandle();
    pnh_ = getPrivateNodeHandle();

    pnh_.getParam("min_size", minsize_);
    pnh_.getParam("max_size", maxsize_);
    pnh_.getParam("cluster_tolerance", cluster_tolerance_);
    pnh_.getParam("approximate_sync_", approximate_sync_);
    pnh_.getParam("downsample_step_", step_);

    cloud_.reset(new pcl::PointCloud<pcl::PointXYZ>);
    clustered_cloud_.reset(new pcl::PointCloud<pcl::PointXYZ>);
    filtered_cloud_.reset(new pcl::PointCloud<pcl::PointXYZ>);

    output_cluster_indices_pub_ =
      pnh_.advertise<jsk_recognition_msgs::ClusterPointIndices>("output_indices", 1);

    sub_cluster_indices_.subscribe(pnh_, "input_cluster_indices", 1);
    sub_point_cloud_.subscribe(pnh_, "input_point_cloud", 1);

    if (approximate_sync_){
      async_ = boost::make_shared<message_filters::Synchronizer<ApproximateSyncPolicy> >(1000);
      async_->connectInput(sub_cluster_indices_, sub_point_cloud_);
      async_->registerCallback(boost::bind(&MultiEuclideanClustering::callback, this, _1, _2));
    } else {
      sync_  = boost::make_shared<message_filters::Synchronizer<SyncPolicy> >(1000);
      sync_->connectInput(sub_cluster_indices_, sub_point_cloud_);
      sync_->registerCallback(boost::bind(&MultiEuclideanClustering::callback, this, _1, _2));
    }
  }

  void MultiEuclideanClustering::callback(const jsk_recognition_msgs::ClusterPointIndices::ConstPtr& cluster_indices,
                                          const sensor_msgs::PointCloud2::ConstPtr& point_cloud)
  {
    jsk_recognition_msgs::ClusterPointIndices output_cluster_indices;
    cloud_->clear();
    pcl::fromROSMsg(*point_cloud, *cloud_);

    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered (new pcl::PointCloud<pcl::PointXYZ>);
    cloud_filtered->header = cloud_->header;
    cloud_filtered->is_dense = cloud_->is_dense;
    cloud_filtered->sensor_origin_ = cloud_->sensor_origin_;
    cloud_filtered->sensor_orientation_ = cloud_->sensor_orientation_;

    int index = 0;
    for (int i=0; i<cloud_->points.size(); i++) {
      if (i % step_ == 0) {
        cloud_filtered->points.push_back(cloud_->points.at(i));
      }
    }

    // downsample
    // float resolution = 0.01;
    // pcl::VoxelGrid<pcl::PointXYZ> sor;
    // sor.setInputCloud(cloud_);
    // sor.setLeafSize(resolution, resolution, resolution);
    // sor.filter(*cloud_filtered);

    // get original indices
    // std::vector<int> original_indices(cloud_filtered->points.size());
    // for(int i=0; i<cloud_filtered->points.size(); i++){
    //   original_indices.at(i) = sor.getCentroidIndex(cloud_filtered->points.at(i));
    // }

    if(cloud_filtered->points.size() > 0){
      for(auto point_indices : cluster_indices->cluster_indices){
        // organized pointcloud
        pcl::PointIndices::Ptr nonnan_indices (new pcl::PointIndices);
        for (auto index : point_indices.indices) {
          auto it = original_to_filtered(index);
          pcl::PointXYZ p = cloud_filtered->points[it];
          if (!std::isnan(p.x) && !std::isnan(p.y) && !std::isnan(p.z)) {
            nonnan_indices->indices.push_back(it);
          }
        }

        pcl::search::KdTree<pcl::PointXYZ>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZ>);
        std::vector<pcl::PointIndices> output_indices;
        pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
        tree->setInputCloud(cloud_filtered);
        ec.setClusterTolerance (cluster_tolerance_);
        ec.setMinClusterSize (minsize_);
        ec.setMaxClusterSize (maxsize_);
        ec.setSearchMethod (tree);
        ec.setIndices(nonnan_indices);
        ec.setInputCloud (cloud_filtered);
        ec.extract (output_indices);

        pcl_msgs::PointIndices point_indices_msg;
        int size, index = 0;
        if(output_indices.size() > 0){
          for(int i=0; i < output_indices.size(); i++){
            if(output_indices[i].indices.size() > size){
              size = output_indices[i].indices.size();
              index = i;
            }
          }

          // set extracted indices to ros msg
          for (int i=0; i<output_indices.at(index).indices.size(); i++){
            point_indices_msg.indices.push_back
              (filtered_to_original(output_indices.at(index).indices.at(i)));
          }
          point_indices_msg.header = cluster_indices->header;
        }
        output_cluster_indices.cluster_indices.push_back(point_indices_msg);
      }
      output_cluster_indices.header = cluster_indices->header;
      output_cluster_indices_pub_.publish(output_cluster_indices);

    }
  }
} // namespace neatness_estimator

#include <pluginlib/class_list_macros.h>
PLUGINLIB_EXPORT_CLASS(neatness_estimator::MultiEuclideanClustering, nodelet::Nodelet)
