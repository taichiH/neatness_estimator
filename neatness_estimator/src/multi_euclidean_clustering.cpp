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
    pnh_.param("downsample", downsample_, false);
    pnh_.param("leaf_size", leaf_size_, 0.01);

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
    boost::mutex::scoped_lock(mutex_);
    jsk_recognition_msgs::ClusterPointIndices output_cluster_indices;
    cloud_->clear();
    pcl::fromROSMsg(*point_cloud, *cloud_);

    pcl::VoxelGrid<pcl::PointXYZ> voxel;
    pcl::PointCloud<pcl::PointXYZ>::Ptr voxel_cloud(
        new pcl::PointCloud<pcl::PointXYZ>);
    if (downsample_) {
      voxel.setLeafSize(leaf_size_, leaf_size_, leaf_size_);
      voxel.setSaveLeafLayout(true);
      voxel.setInputCloud(cloud_);
      voxel.filter(*voxel_cloud);
      preprocessed_cloud_ = voxel_cloud;
      downsample_to_original_indices_.resize(cloud_->points.size());
      original_to_downsample_indices_.resize(cloud_->points.size());
      std::fill(original_to_downsample_indices_.begin(),
                original_to_downsample_indices_.end(),
                -1);
      std::fill(downsample_to_original_indices_.begin(),
                downsample_to_original_indices_.end(),
                std::vector<int>());
      for (int i_point = 0; i_point < cloud_->points.size(); ++i_point) {
        pcl::PointXYZ p = cloud_->points[i_point];
        if (std::isnan(p.x) || std::isnan(p.y) || std::isnan(p.z)) {
          continue;
        }
        int index = voxel.getCentroidIndex(p);
        if (index == -1) {
          continue;
        }
        original_to_downsample_indices_[i_point] = index;
        downsample_to_original_indices_[index].push_back(i_point);
      }
          } else {
      preprocessed_cloud_ = cloud_;
    }

    if(preprocessed_cloud_->points.size() > 0){
      for(auto point_indices : cluster_indices->cluster_indices){

        // organized pointcloud
        pcl::PointIndices::Ptr nonnan_indices (new pcl::PointIndices);
        for (auto original_index : point_indices.indices) {
          int index;
          if (downsample_) {
            index = original_to_downsample_indices_[original_index];
          } else {
            index = original_index;
          }
          if (index == -1) {
            continue;
          }

          pcl::PointXYZ p = preprocessed_cloud_->points[index];
          if (!std::isnan(p.x) && !std::isnan(p.y) && !std::isnan(p.z)) {
            nonnan_indices->indices.push_back(index);
          }
        }
        std::sort(nonnan_indices->indices.begin(), nonnan_indices->indices.end());
        nonnan_indices->indices.erase(std::unique(nonnan_indices->indices.begin(), nonnan_indices->indices.end()), nonnan_indices->indices.end());

        pcl_msgs::PointIndices point_indices_msg;
        if (nonnan_indices->indices.size() > 0) {

          // Create the filtering object
          pcl::PointCloud<pcl::PointXYZ>::Ptr filtered_points(new pcl::PointCloud<pcl::PointXYZ>);
          pcl::ExtractIndices<pcl::PointXYZ> extract;
          extract.setInputCloud (voxel_cloud);
          extract.setIndices (nonnan_indices);
          extract.setNegative(false);
          extract.filter (*filtered_points);

          pcl::search::KdTree<pcl::PointXYZ>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZ>);
          tree->setInputCloud(filtered_points);

          std::vector<pcl::PointIndices> output_indices;
          pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
          ec.setClusterTolerance(cluster_tolerance_);
          ec.setMinClusterSize(minsize_);
          ec.setMaxClusterSize(maxsize_);
          ec.setSearchMethod(tree);
          ec.setInputCloud(filtered_points);
          ec.extract(output_indices);

          if (output_indices.size() > 0) {
            int size = 0;
            int index = 0;
            for(int i=0; i < output_indices.size(); i++){
              if(output_indices[i].indices.size() > size){
                size = output_indices[i].indices.size();
                index = i;
              }
            }

            if (downsample_) {
              for (size_t i_index = 0; i_index < output_indices.at(index).indices.size(); ++i_index) {
                point_indices_msg.indices.insert(
                                                 point_indices_msg.indices.end(),
                                                 downsample_to_original_indices_[
                                                                                 nonnan_indices->indices[output_indices.at(index).indices[i_index]]].begin(),
                                                 downsample_to_original_indices_[nonnan_indices->indices[output_indices.at(index).indices[i_index]]].end()
                                                 );
              }
            } else {
              point_indices_msg.indices = output_indices.at(index).indices;
            }
          }
        }
        // set extracted indices to ros msg
        point_indices_msg.header = point_cloud->header;
        output_cluster_indices.cluster_indices.push_back(point_indices_msg);
      }
      output_cluster_indices.header = point_cloud->header;
      output_cluster_indices_pub_.publish(output_cluster_indices);

    }
  }
} // namespace neatness_estimator

#include <pluginlib/class_list_macros.h>
PLUGINLIB_EXPORT_CLASS(neatness_estimator::MultiEuclideanClustering, nodelet::Nodelet)
