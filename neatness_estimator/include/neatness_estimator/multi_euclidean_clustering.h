#ifndef _NEATNESS_ESTIMATOR_MULTI_EUCLIDEAN_CLUSTERING_H_
#define _NEATNESS_ESTIMATOR_MULTI_EUCLIDEAN_CLUSTERING_H_

#include <ros/ros.h>
#include <nodelet/nodelet.h>
#include <thread>
#include <mutex>

#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>

#include <jsk_recognition_msgs/ClusterPointIndices.h>
#include <jsk_recognition_msgs/LabelArray.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl_msgs/PointIndices.h>
#include <std_msgs/Header.h>

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
    virtual void onInit();
    virtual void callback(const jsk_recognition_msgs::ClusterPointIndices::ConstPtr& cluster_indices,
                          const sensor_msgs::PointCloud2::ConstPtr& point_cloud);
    virtual bool extract(const pcl_msgs::PointIndices& point_indices,
                         pcl_msgs::PointIndices& point_indices_msg);

    ros::NodeHandle nh_;
    ros::NodeHandle pnh_;
    ros::Publisher output_cluster_indices_pub_;

    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_;
    pcl::PointCloud<pcl::PointXYZ>::Ptr clustered_cloud_;
    pcl::PointCloud<pcl::PointXYZ>::Ptr filtered_cloud_;

    boost::shared_ptr<message_filters::Synchronizer<SyncPolicy> >sync_;
    boost::shared_ptr<message_filters::Synchronizer<ApproximateSyncPolicy> > async_;

    message_filters::Subscriber<jsk_recognition_msgs::ClusterPointIndices> sub_cluster_indices_;
    message_filters::Subscriber<sensor_msgs::PointCloud2> sub_point_cloud_;

    float cluster_tolerance_ = 0.01;
    float minsize_ = 10;
    float maxsize_ = 5000;
    bool approximate_sync_ = true;

    std::mutex mtx_;

  private:
  };
} // namespace neatness_estimator

#endif
