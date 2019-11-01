#ifndef _NEATNESS_ESTIMATOR_DATA_SAVER_H_
#define _NEATNESS_ESTIMATOR_DATA_SAVER_H_

#include <ros/ros.h>
#include <nodelet/nodelet.h>
#include <rosbag/bag.h>

#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>

#include <neatness_estimator_msgs/GetFeatures.h>
#include <neatness_estimator_msgs/GetDifference.h>

#include <jsk_recognition_msgs/ClusterPointIndices.h>
#include <jsk_recognition_msgs/LabelArray.h>
#include <jsk_recognition_msgs/BoundingBoxArray.h>
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
      jsk_recognition_msgs::ClusterPointIndices,
      jsk_recognition_msgs::LabelArray,
      jsk_recognition_msgs::BoundingBoxArray,
      jsk_recognition_msgs::BoundingBoxArray
      > SyncPolicy;

    typedef message_filters::sync_policies::ApproximateTime<
      sensor_msgs::PointCloud2,
      sensor_msgs::Image,
      jsk_recognition_msgs::ClusterPointIndices,
      jsk_recognition_msgs::LabelArray,
      jsk_recognition_msgs::BoundingBoxArray,
      jsk_recognition_msgs::BoundingBoxArray
      > ApproximateSyncPolicy;

    enum Topics {
      CLOUD = 1,
      IMAGE = 2,
      CLUSTER = 3,
      LABELS = 4,
      INSTANCE_BOXES = 5,
      CLUSTER_BOXES = 6
    };

  protected:

    // functions

    virtual void onInit();

    virtual void callback(const sensor_msgs::PointCloud2::ConstPtr& cloud_msg,
                          const sensor_msgs::Image::ConstPtr& image_msg,
                          const jsk_recognition_msgs::ClusterPointIndices::ConstPtr& cluster_msg,
                          const jsk_recognition_msgs::LabelArray::ConstPtr& labels_msg,
                          const jsk_recognition_msgs::BoundingBoxArray::ConstPtr& instance_boxes_msg,
                          const jsk_recognition_msgs::BoundingBoxArray::ConstPtr& cluster_boxes_msg);

    virtual bool save_service_callback(std_srvs::SetBool::Request& req,
                                       std_srvs::SetBool::Response& res);

    virtual bool call_service_callback(neatness_estimator_msgs::GetDifference::Request& req,
                                       neatness_estimator_msgs::GetDifference::Response& res);

    virtual bool create_save_dir(std::stringstream& ss,
                                 std::string dir_name);

    virtual bool create_features_vec(const neatness_estimator_msgs::Features& fetures);

    // variables

    ros::NodeHandle nh_;
    ros::NodeHandle pnh_;

    ros::ServiceServer save_server_;
    ros::ServiceServer call_server_;

    ros::ServiceClient feature_client_;
    ros::ServiceClient difference_client_;

    boost::shared_ptr<message_filters::Synchronizer<SyncPolicy> >sync_;
    boost::shared_ptr<message_filters::Synchronizer<ApproximateSyncPolicy> > async_;

    message_filters::Subscriber<sensor_msgs::PointCloud2> sub_point_cloud_;
    message_filters::Subscriber<sensor_msgs::Image> sub_image_;
    message_filters::Subscriber<jsk_recognition_msgs::ClusterPointIndices> sub_cluster_;
    message_filters::Subscriber<jsk_recognition_msgs::LabelArray> sub_labels_;
    message_filters::Subscriber<jsk_recognition_msgs::BoundingBoxArray> sub_instance_boxes_;
    message_filters::Subscriber<jsk_recognition_msgs::BoundingBoxArray> sub_cluster_boxes_;

    sensor_msgs::PointCloud2::ConstPtr cloud_msg_;
    sensor_msgs::Image::ConstPtr image_msg_;
    jsk_recognition_msgs::ClusterPointIndices::ConstPtr cluster_msg_;
    jsk_recognition_msgs::LabelArray::ConstPtr labels_msg_;
    jsk_recognition_msgs::BoundingBoxArray::ConstPtr instance_boxes_msg_;
    jsk_recognition_msgs::BoundingBoxArray::ConstPtr cluster_boxes_msg_;

    std::vector<neatness_estimator_msgs::Features> features_vec_;

    boost::mutex mutex_;
    std::string prefix_ = "./";
    std::string difference_reasoner_service_topic_ = "/difference_reasoner/read";
    std::string compare_data_service_topic_ = "/compare_data/compare";
    std::vector<std::string> topics_;

  private:

  };

} // namespace neatness_estimator

#endif
