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


  bool MultiEuclideanClustering::extract(const pcl_msgs::PointIndices& point_indices,
                                         pcl_msgs::PointIndices& point_indices_msg,
                                         const std_msgs::Header& header)
  {
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

    int size, index = 0;
    if(output_indices.size() > 0) {
      for(int i=0; i < output_indices.size(); i++) {
        if(output_indices[i].indices.size() > size) {
          size = output_indices[i].indices.size();
          index = i;
        }
      }
      // set extracted indices to ros msg
      point_indices_msg.indices = output_indices.at(index).indices;

    }
    point_indices_msg.header = header;

    return true;
  }

  void MultiEuclideanClustering::thread_func(int i,
                                             const pcl_msgs::PointIndices& point_indices,
                                             const std_msgs::Header& header)
  {
    mtx_.lock();
    pcl_msgs::PointIndices point_indices_msg;
    extract(point_indices, point_indices_msg, header);
    point_indices_map_.emplace(i, point_indices_msg);
    mtx_.unlock();
  }

  void MultiEuclideanClustering::callback(const jsk_recognition_msgs::ClusterPointIndices::ConstPtr& cluster_indices,
                                          const sensor_msgs::PointCloud2::ConstPtr& point_cloud)
  {
    point_indices_map_.clear();

    jsk_recognition_msgs::ClusterPointIndices output_cluster_indices;
    cloud_->clear();
    pcl::fromROSMsg(*point_cloud, *cloud_);

    if(cloud_->points.size() == 0) {
      return;
    }

    std::cerr << "cluster_indices->cluster_indices.size(): "
              << cluster_indices->cluster_indices.size() << std::endl;

    std::vector<std::thread> threads;
    for(int i=0; i<cluster_indices->cluster_indices.size(); ++i) {
      pcl_msgs::PointIndices point_indices = cluster_indices->cluster_indices.at(i);
      std_msgs::Header header = cluster_indices->header;
      threads.push_back(std::thread
                        (&MultiEuclideanClustering::thread_func,
                         this, i, point_indices, header));
    }

    for(std::thread &th : threads) {
      th.join();
    }

    for(auto it = point_indices_map_.begin(); it != point_indices_map_.end(); ++it) {
      output_cluster_indices.cluster_indices.push_back(it->second);
    }

    output_cluster_indices.header = cluster_indices->header;
    output_cluster_indices_pub_.publish(output_cluster_indices);
  }

} // namespace neatness_estimator

#include <pluginlib/class_list_macros.h>
PLUGINLIB_EXPORT_CLASS(neatness_estimator::MultiEuclideanClustering, nodelet::Nodelet)
