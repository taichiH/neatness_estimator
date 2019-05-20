#include "neatness_estimator/multi_euclidean_clustering.h"

namespace neatness_estimator
{

  void MultiEuclideanClustering::onInit()
  {
    DiagnosticNodelet::onInit();
    output_cluster_indices_pub_ =
      advertise<jsk_recognition_msgs::ClusterPointIndices>(*pnh_, "output_indices", 1);

    cloud_.reset(new pcl::PointCloud<pcl::PointXYZ>);
    clustered_cloud_.reset(new pcl::PointCloud<pcl::PointXYZ>);
    filtered_cloud_.reset(new pcl::PointCloud<pcl::PointXYZ>);

    onInitPostProcess();
  }

  void MultiEuclideanClustering::subscribe()
  {
    sub_cluster_indices.subscribe(*pnh_, "input_cluster_indices", 1);
    sub_point_cloud.subscribe(*pnh_, "input_point_cloud", 1);

    if (approximate_sync_){
      async_ = boost::make_shared<message_filters::Synchronizer<ApproximateSyncPolicy> >(1000);
      async_->connectInput(sub_cluster_indices, sub_point_cloud);
      async_->registerCallback(boost::bind(&MultiEuclideanClustering::callback, this, _1, _2));
    } else {
      sync_  = boost::make_shared<message_filters::Synchronizer<SyncPolicy> >(1000);
      sync_->connectInput(sub_cluster_indices, sub_point_cloud);
      sync_->registerCallback(boost::bind(&MultiEuclideanClustering::callback, this, _1, _2));
    }
  }

  void MultiEuclideanClustering::unsubscribe()
  {
    sub_cluster_indices.unsubscribe();
    sub_point_cloud.unsubscribe();
  }

  void MultiEuclideanClustering::callback(const jsk_recognition_msgs::ClusterPointIndices::ConstPtr& cluster_indices,
                                         const sensor_msgs::PointCloud2::ConstPtr& point_cloud)
  {
    jsk_recognition_msgs::ClusterPointIndices output_cluster_indices;
    cloud_->clear();
    pcl::fromROSMsg(*point_cloud, *cloud_);

    if(cloud_->points.size() > 0){
      for(auto point_indices : cluster_indices->cluster_indices){

        // organized pointcloud
        pcl::PointIndices::Ptr nonnan_indices (new pcl::PointIndices);
        for (auto index : point_indices.indices) {
          pcl::PointXYZ p = cloud_->points[index];
          if (!std::isnan(p.x) && !std::isnan(p.y) && !std::isnan(p.z)) {
            nonnan_indices->indices.push_back(index);
          }
        }

        pcl::search::KdTree<pcl::PointXYZ>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZ>);
        std::vector<pcl::PointIndices> output_indices;
        pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
        tree->setInputCloud(cloud_);
        ec.setClusterTolerance (cluster_tolerance_);
        ec.setMinClusterSize (minsize_);
        ec.setMaxClusterSize (maxsize_);
        ec.setSearchMethod (tree);
        ec.setIndices(nonnan_indices);
        ec.setInputCloud (cloud_);
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
          point_indices_msg.indices = output_indices.at(index).indices;
          point_indices_msg.header = cluster_indices->header;
        }
        output_cluster_indices.cluster_indices.push_back(point_indices_msg);
      }
      output_cluster_indices.header = cluster_indices->header;
      output_cluster_indices_pub_.publish(output_cluster_indices);

    }
  }
}
