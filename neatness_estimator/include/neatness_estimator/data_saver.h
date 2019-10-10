#ifndef _NEATNESS_ESTIMATOR_DATA_SAVER_H_
#define _NEATNESS_ESTIMATOR_DATA_SAVER_H_

#include <ros/ros.h>
#include <nodelet/nodelet.h>
#include <rosbag/bag.h>

#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>

#include <jsk_recognition_msgs/ClusterPointIndices.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/Image.h>
#include <std_srvs/SetBool.h>

#include <mutex>
#include <boost/filesystem.hpp>


namespace neatness_estimator
{
  class DataSaver : public nodelet::Nodelet
  {
  public:

    typedef message_filters::sync_policies::ExactTime<
      sensor_msgs::PointCloud2,
      sensor_msgs::Image,
      jsk_recognition_msgs::ClusterPointIndices
      > SyncPolicy;

    typedef message_filters::sync_policies::ApproximateTime<
      sensor_msgs::PointCloud2,
      sensor_msgs::Image,
      jsk_recognition_msgs::ClusterPointIndices
      > ApproximateSyncPolicy;

    enum Topics {
      CLOUD,
      IMAGE,
      CLUSTER
    };

  protected:

    // functions

    virtual void onInit();

    virtual void callback(const sensor_msgs::PointCloud2::ConstPtr& cloud_msg,
                          const sensor_msgs::Image::ConstPtr& image_msg,
                          const jsk_recognition_msgs::ClusterPointIndices::ConstPtr& cluster_msg);

    virtual bool service_callback(std_srvs::SetBool::Request& req,
                                  std_srvs::SetBool::Response& res);

    virtual bool create_save_dir(std::stringstream& ss,
                                 std::string dir_name);


    // variables

    ros::NodeHandle nh_;

    ros::NodeHandle pnh_;

    ros::ServiceServer server_;

    boost::shared_ptr<message_filters::Synchronizer<SyncPolicy> >sync_;

    boost::shared_ptr<message_filters::Synchronizer<ApproximateSyncPolicy> > async_;

    message_filters::Subscriber<sensor_msgs::PointCloud2> sub_point_cloud_;

    message_filters::Subscriber<sensor_msgs::Image> sub_image_;

    message_filters::Subscriber<jsk_recognition_msgs::ClusterPointIndices> sub_cluster_;

    sensor_msgs::PointCloud2::ConstPtr cloud_msg_;

    sensor_msgs::Image::ConstPtr image_msg_;

    jsk_recognition_msgs::ClusterPointIndices::ConstPtr cluster_msg_;

    boost::mutex mutex_;

    std::string prefix_ = "./";

    std::vector<std::string> topics_;

  private:

  };

} // namespace neatness_estimator

#endif
