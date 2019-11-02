#ifndef _NEATNESS_ESTIMATOR_MULTI_EUCLIDEAN_CLUSTERING_H_
#define _NEATNESS_ESTIMATOR_MULTI_EUCLIDEAN_CLUSTERING_H_

#include <thread>
#include <mutex>

#include <ros/ros.h>
#include <nodelet/nodelet.h>

#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>

#include <std_msgs/Header.h>
#include <jsk_recognition_msgs/ClusterPointIndices.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl_msgs/PointIndices.h>

#include <pcl/point_types.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/filter.h>
#include <pcl/kdtree/kdtree.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/segmentation/extract_clusters.h>

namespace neatness_estimator
{
  class MultiEuclideanClustering : public nodelet::Nodelet
  {
  public:

    typedef message_filters::sync_policies::ExactTime<
      jsk_recognition_msgs::ClusterPointIndices,
      sensor_msgs::PointCloud2
      > SyncPolicy;

    typedef message_filters::sync_policies::ApproximateTime<
      jsk_recognition_msgs::ClusterPointIndices,
      sensor_msgs::PointCloud2
      > ApproximateSyncPolicy;

  protected:
    enum MODE {
      CLOUDSIZE = 0,
      CENTER = 1,
    };

    virtual void onInit();

    virtual void callback
      (const jsk_recognition_msgs::ClusterPointIndices::ConstPtr& cluster_indices,
       const sensor_msgs::PointCloud2::ConstPtr& point_cloud);

    virtual int extract
      (const pcl_msgs::PointIndices& point_indices,
       pcl_msgs::PointIndices& point_indices_msg);

    virtual void thread_callback
      (int idx,
       const pcl_msgs::PointIndices& point_indices,
       const std_msgs::Header& header);

    virtual void get_centroid_index_multi
      (const int idx, const pcl::PointXYZ& p);

    virtual bool downsample_cloud
      (const pcl::PointCloud<pcl::PointXYZ>::Ptr& orig_cloud,
       pcl::PointCloud<pcl::PointXYZ>::Ptr& sparse_cloud,
       std::vector<std::vector<int> >& sparse_to_orig_indices,
       std::vector<int>& orig_to_sparse_indices);

    virtual int get_target_index
      (const std::vector<pcl::PointIndices>& output_indices);


    ros::NodeHandle nh_;
    ros::NodeHandle pnh_;
    ros::Publisher output_cluster_indices_pub_;

    boost::shared_ptr<message_filters::Synchronizer<SyncPolicy> >sync_;
    boost::shared_ptr<message_filters::Synchronizer<ApproximateSyncPolicy> > async_;
    message_filters::Subscriber<jsk_recognition_msgs::ClusterPointIndices> sub_cluster_indices_;
    message_filters::Subscriber<sensor_msgs::PointCloud2> sub_point_cloud_;

    int target_policy_ = 0;
    float cluster_tolerance_ = 0.01;
    float minsize_ = 10;
    float maxsize_ = 5000;
    double leaf_size_ = 0.01;

    bool debug_time_ = false;
    bool multi_threading_ = false;
    bool get_centroid_multi_ = false;
    bool downsample_ = false;
    bool approximate_sync_ = true;

    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_;
    pcl::PointCloud<pcl::PointXYZ>::Ptr preprocessed_cloud_;

    std::vector<std::vector<int> > downsample_to_original_indices_;
    std::vector<int> original_to_downsample_indices_;
    std::map<int, pcl_msgs::PointIndices> point_indices_map_;
    std::map<int, int> centroid_map_;

    boost::mutex mutex_;
    boost::mutex get_centroid_mutex_;

    pcl::VoxelGrid<pcl::PointXYZ> voxel_;

  private:
  };
} // namespace neatness_estimator

#endif
