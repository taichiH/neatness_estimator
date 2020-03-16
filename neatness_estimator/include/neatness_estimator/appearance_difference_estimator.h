#ifndef _NEATNESS_ESTIMATOR_APPEARANCE_DIFFERENCE_ESTIMATOR_H_
#define _NEATNESS_ESTIMATOR_APPEARANCE_DIFFERENCE_ESTIMATOR_H_

#include <ros/ros.h>
#include <nodelet/nodelet.h>

#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <opencv/cv.hpp>

#include <pcl_conversions/pcl_conversions.h>
#include <pcl_msgs/PointIndices.h>
#include <pcl/point_types_conversion.h>
#include <pcl/point_types.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/keypoints/uniform_sampling.h>
#include <pcl/features/cvfh.h>
#include <pcl/features/normal_3d_omp.h>

#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>

#include <jsk_recognition_msgs/ClusterPointIndices.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/Image.h>

#include <neatness_estimator_msgs/AppearanceDifference.h>
#include <neatness_estimator_msgs/GetDifference.h>
#include <neatness_estimator_msgs/Histogram.h>
#include <neatness_estimator_msgs/HistogramArray.h>

#include <mutex>

namespace neatness_estimator
{

  class AppearanceFeature
  {
  public:
    neatness_estimator_msgs::Histogram color;
    neatness_estimator_msgs::Histogram geometry;
    int size;
  };

  class AppearanceDifferenceEstimator : public nodelet::Nodelet
  {
  public:

  protected:

    typedef std::pair<int, int> AppearancePair;

    typedef message_filters::sync_policies::ApproximateTime<
      sensor_msgs::PointCloud2,
      sensor_msgs::Image,
      jsk_recognition_msgs::ClusterPointIndices
      > ApproximateSync;

    typedef message_filters::sync_policies::ExactTime<
      sensor_msgs::PointCloud2,
      sensor_msgs::Image,
      jsk_recognition_msgs::ClusterPointIndices
      > Sync;


    // functions

    virtual void onInit();

    virtual void callback
      (const sensor_msgs::PointCloud2::ConstPtr& cloud_msg,
       const sensor_msgs::Image::ConstPtr& image_msg,
       const jsk_recognition_msgs::ClusterPointIndices::ConstPtr& cluster_msg);

    virtual bool service_callback
      (neatness_estimator_msgs::GetDifference::Request& req,
       neatness_estimator_msgs::GetDifference::Response& res);

    virtual void register_callback();


    virtual bool run
      (neatness_estimator_msgs::AppearanceDifference::Ptr& difference);

    virtual bool compute_appearance_feature
      (int idx,
       AppearanceFeature& feature);

    virtual bool compute_color_histogram
      (const cv::Mat& image,
       const cv::Mat& mask,
       neatness_estimator_msgs::Histogram& color_histogram);

    virtual bool compute_geometry_histogram
      (const pcl::PointCloud<pcl::PointXYZRGB>::Ptr& rgb_cloud,
       neatness_estimator_msgs::Histogram& geometry_histogram);

    virtual bool get_clustered_cloud
      (const pcl::PointCloud<pcl::PointXYZRGB>::Ptr& rgb_cloud,
       jsk_recognition_msgs::ClusterPointIndices::ConstPtr& cluster_indices,
       int index,
       cv::Mat& mask_image,
       pcl::PointCloud<pcl::PointXYZRGB>::Ptr& clustered_cloud);

    virtual bool load_image
      (const sensor_msgs::Image::ConstPtr& input_msg,
       cv::Mat& input_image);

    virtual bool compute_appearance_difference
      (const std::vector<AppearanceFeature>& features,
       neatness_estimator_msgs::AppearanceDifference::Ptr& difference);


    template<typename T>
      inline float calc_histogram_distance
      (std::vector<T> hist1, std::vector<T> hist2)
      {
        if (hist1.size() != hist2.size()) {
          return false;
        }

        float diff = 0;
        float sum = 0;
        for (int i=0; i<hist1.size(); i++) {
          diff += std::abs(hist1.at(i) - hist2.at(i));
          sum += std::abs(hist1.at(i) + hist2.at(i));
        }
        float distance = diff / sum;

        return distance;
      }


    // variables

    boost::shared_ptr<message_filters::Synchronizer<ApproximateSync> > async_;

    boost::shared_ptr<message_filters::Synchronizer<Sync> > sync_;

    ros::NodeHandle nh_;

    ros::NodeHandle pnh_;

    ros::Publisher difference_pub_;

    ros::ServiceServer service_server_;

    message_filters::Subscriber<sensor_msgs::PointCloud2> sub_point_cloud_;

    message_filters::Subscriber<sensor_msgs::Image> sub_image_;

    message_filters::Subscriber<jsk_recognition_msgs::ClusterPointIndices> sub_cluster_;

    sensor_msgs::PointCloud2::ConstPtr cloud_msg_;

    sensor_msgs::Image::ConstPtr image_msg_;

    jsk_recognition_msgs::ClusterPointIndices::ConstPtr cluster_msg_;

    boost::mutex mutex_;

    bool approximate_sync_ = true;

    AppearancePair pair_;

  private:

  };

} // namespace neatness_estimator

#endif
