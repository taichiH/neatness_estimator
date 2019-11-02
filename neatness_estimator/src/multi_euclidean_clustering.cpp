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
    pnh_.getParam("downsample", downsample_);
    pnh_.getParam("leaf_size", leaf_size_);
    pnh_.getParam("multi_threading", multi_threading_);
    pnh_.getParam("get_centroid_multi", get_centroid_multi_);
    pnh_.getParam("debug_time", debug_time_);

    int policy = 0;
    pnh_.getParam("policy", policy);
    target_policy_ = static_cast<MultiEuclideanClustering::MODE>(policy);

    cloud_.reset(new pcl::PointCloud<pcl::PointXYZ>);

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


  int MultiEuclideanClustering::get_target_index
  (const std::vector<pcl::PointIndices>& output_indices)
  {
    int index = 0;
    if (target_policy_ == MultiEuclideanClustering::MODE::CLOUDSIZE) {
      int size = 0;
      for (int i=0; i < output_indices.size(); i++) {
        if (output_indices[i].indices.size() > size) {
          size = output_indices[i].indices.size();
          index = i;
        }
      }
    } else {
      // TODO: implement other mode
      // MODE::CENTER
    }

    return index;
  }

  int MultiEuclideanClustering::extract
  (const pcl_msgs::PointIndices& point_indices,
   pcl_msgs::PointIndices& point_indices_msg)
  {
    // organized pointcloud
    pcl::PointIndices::Ptr nonnan_indices (new pcl::PointIndices);
    for (auto original_index : point_indices.indices) {
      int index = downsample_ ?
        original_to_downsample_indices_[original_index] : original_index;
      if (index == -1)
        continue;

      pcl::PointXYZ p = preprocessed_cloud_->points[index];
      if (!std::isnan(p.x) && !std::isnan(p.y) && !std::isnan(p.z))
        nonnan_indices->indices.push_back(index);
    }

    std::sort(nonnan_indices->indices.begin(), nonnan_indices->indices.end());
    nonnan_indices->indices.erase(std::unique(nonnan_indices->indices.begin(),
                                              nonnan_indices->indices.end()),
                                  nonnan_indices->indices.end());

    if (nonnan_indices->indices.size() == 0) {
      return -1;
    }

    // Create the filtering object
    pcl::PointCloud<pcl::PointXYZ>::Ptr filtered_points(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::ExtractIndices<pcl::PointXYZ> extract;
    extract.setInputCloud (preprocessed_cloud_);
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


    if (output_indices.size() == 0) {
      return -1;
    }

    int index = get_target_index(output_indices);

    if (downsample_) {
      for (size_t i_index = 0; i_index < output_indices.at(index).indices.size(); ++i_index) {
        point_indices_msg.indices.insert
          (point_indices_msg.indices.end(),
           downsample_to_original_indices_
           [nonnan_indices->indices[output_indices.at(index).indices[i_index]]].begin(),
           downsample_to_original_indices_
           [nonnan_indices->indices[output_indices.at(index).indices[i_index]]].end());
      }
    } else {
      point_indices_msg.indices = output_indices.at(index).indices;
    }

    return 0;
  }

  void MultiEuclideanClustering::get_centroid_index_multi
  (const int idx, const pcl::PointXYZ& p)
  {
    get_centroid_mutex_.lock();
    int centroid_index = voxel_.getCentroidIndex(p);
    if (centroid_index != -1) {

      centroid_map_.emplace(idx, centroid_index);

    }
    get_centroid_mutex_.unlock();
  }

  void MultiEuclideanClustering::thread_callback
  (int idx,
   const pcl_msgs::PointIndices& point_indices,
   const std_msgs::Header& header)
  {
    mutex_.lock();
    pcl_msgs::PointIndices point_indices_msg;
    extract(point_indices, point_indices_msg);
    point_indices_msg.header = header;
    point_indices_map_.emplace(idx, point_indices_msg);
    mutex_.unlock();
  }

  bool MultiEuclideanClustering::downsample_cloud
  (const pcl::PointCloud<pcl::PointXYZ>::Ptr& orig_cloud,
   pcl::PointCloud<pcl::PointXYZ>::Ptr& sparse_cloud,
   std::vector<std::vector<int> >& sparse_to_orig_indices,
   std::vector<int>& orig_to_sparse_indices)
  {
    std::cerr << __func__ << std::endl;

    voxel_.setLeafSize(leaf_size_, leaf_size_, leaf_size_);
    voxel_.setSaveLeafLayout(true);
    voxel_.setInputCloud(orig_cloud);
    voxel_.filter(*sparse_cloud);

    sparse_to_orig_indices.resize(orig_cloud->points.size());
    orig_to_sparse_indices.resize(orig_cloud->points.size());
    std::fill
      (orig_to_sparse_indices.begin(), orig_to_sparse_indices.end(), -1);
    std::fill
      (sparse_to_orig_indices.begin(), sparse_to_orig_indices.end(), std::vector<int>());

    double start = ros::Time::now().toSec();

    std::string debug_message;
    if (get_centroid_multi_) {
      debug_message = "get_centroid_multi";
      std::cerr << 1 << std::endl;

      std::vector<std::thread> get_centroid_index_threads;
      for (int i_point = 0; i_point < orig_cloud->points.size(); ++i_point) {

        pcl::PointXYZ p = orig_cloud->points.at(i_point);
        if (std::isnan(p.x) || std::isnan(p.y) || std::isnan(p.z))
          continue;

        get_centroid_index_threads.push_back
          (std::thread(&MultiEuclideanClustering::get_centroid_index_multi, this, i_point, p));
      }

      std::cerr << 2 << std::endl;

      for(std::thread &get_centroid_index_thread : get_centroid_index_threads)
        get_centroid_index_thread.join();

      std::cerr << 3 << std::endl;

      for(auto it = centroid_map_.begin(); it != centroid_map_.end(); ++it) {
        orig_to_sparse_indices[it->first] = it->second;
        sparse_to_orig_indices[it->second].push_back(it->first);
      }

      std::cerr << 4 << std::endl;

    } else {
      debug_message = "get_centroid simple";
      for (int i_point = 0; i_point < orig_cloud->points.size(); ++i_point) {
        pcl::PointXYZ p = orig_cloud->points[i_point];
        if (std::isnan(p.x) || std::isnan(p.y) || std::isnan(p.z))
          continue;

        int index = voxel_.getCentroidIndex(p);
        if (index == -1)
          continue;

        orig_to_sparse_indices[i_point] = index;
        sparse_to_orig_indices[index].push_back(i_point);
      }
    }

    if (debug_time_) {
      ROS_WARN("    [downsample_cloud: getCentroidIndex][%s] time: %.5lf [ms]",
               debug_message.c_str(),
               (ros::Time::now().toSec() - start) * 1000);
    }

    return true;
  }

  void MultiEuclideanClustering::callback
  (const jsk_recognition_msgs::ClusterPointIndices::ConstPtr& cluster_indices,
   const sensor_msgs::PointCloud2::ConstPtr& point_cloud)
  {
    double start = ros::Time::now().toSec();

    jsk_recognition_msgs::ClusterPointIndices output_cluster_indices;
    cloud_->clear();
    pcl::fromROSMsg(*point_cloud, *cloud_);
    preprocessed_cloud_.reset(new pcl::PointCloud<pcl::PointXYZ>);

    if (downsample_) {
      downsample_cloud(cloud_, preprocessed_cloud_,
                       downsample_to_original_indices_,
                       original_to_downsample_indices_);
    } else {
      preprocessed_cloud_ = cloud_;
    }

    if(preprocessed_cloud_->points.size() == 0) {
      ROS_WARN("preprocessed_cloud_ size: %d", preprocessed_cloud_->points.size());
      return;
    }

    if (debug_time_) {
      ROS_WARN("[preprocess] time: %.5lf [ms]",
               (ros::Time::now().toSec() - start) * 1000);
    }

    if (multi_threading_) {
      double total_start = ros::Time::now().toSec();

      std::vector<std::thread> threads;
      for(int i=0; i<cluster_indices->cluster_indices.size(); ++i) {
        pcl_msgs::PointIndices point_indices = cluster_indices->cluster_indices.at(i);
        threads.push_back(std::thread
                          (&MultiEuclideanClustering::thread_callback,
                           this, i, point_indices, cluster_indices->header));
      }

      for(std::thread &th : threads)
        th.join();
      for(auto it = point_indices_map_.begin(); it != point_indices_map_.end(); ++it)
        output_cluster_indices.cluster_indices.push_back(it->second);

      if (debug_time_) {
        ROS_WARN("[multi thread] size: %d, extract time: %.5lf [ms]" ,
                 cluster_indices->cluster_indices.size(),
                 (ros::Time::now().toSec() - total_start) * 1000);
      }

    } else {
      double total_start = ros::Time::now().toSec();
      for(auto point_indices : cluster_indices->cluster_indices) {
        pcl_msgs::PointIndices point_indices_msg;
        extract(point_indices, point_indices_msg);
        point_indices_msg.header = cluster_indices->header;
        output_cluster_indices.cluster_indices.push_back(point_indices_msg);
      }

      if (debug_time_) {
        ROS_WARN("[single thread] size: %d, extract time: %.5lf [ms]" ,
                 cluster_indices->cluster_indices.size(),
                 (ros::Time::now().toSec() - total_start) * 1000);
      }
    }

    output_cluster_indices.header = point_cloud->header;
    output_cluster_indices_pub_.publish(output_cluster_indices);

    if (debug_time_) {
      ROS_WARN("[callback] time: %.5lf [ms]" ,
               (ros::Time::now().toSec() - start) * 1000);
    }
  }

} // namespace neatness_estimator

#include <pluginlib/class_list_macros.h>
PLUGINLIB_EXPORT_CLASS(neatness_estimator::MultiEuclideanClustering, nodelet::Nodelet)
