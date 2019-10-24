#ifndef _NEATNESS_ESTIMATOR_DIFFERENCE_REASONER_H_
#define _NEATNESS_ESTIMATOR_DIFFERENCE_REASONER_H_

#include <ros/ros.h>
#include <nodelet/nodelet.h>
#include <rosbag/bag.h>
#include <rosbag/view.h>

#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>

#include <neatness_estimator_msgs/GetDisplayFeature.h>
#include <neatness_estimator_msgs/Neatness.h>
#include <neatness_estimator_msgs/NeatnessArray.h>
#include <jsk_recognition_msgs/Histogram.h>
#include <jsk_recognition_msgs/ColorHistogram.h>
#include <jsk_recognition_msgs/ColorHistogramArray.h>
#include <jsk_recognition_msgs/ClusterPointIndices.h>
#include <jsk_recognition_msgs/BoundingBoxArray.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/Image.h>
#include <std_srvs/SetBool.h>

#include <pcl_conversions/pcl_conversions.h>
#include <pcl_msgs/PointIndices.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/io.h>
#include <pcl/point_types_conversion.h>
#include <pcl/point_types.h>
#include <pcl/filters/extract_indices.h>
/* #include <pcl/filters/uniform_sampling.h> */
#include <pcl/keypoints/uniform_sampling.h>
#include <pcl/features/vfh.h>
#include <pcl/features/cvfh.h>

#include <pcl/features/normal_3d_omp.h>
/* #include <pcl/features/shot_omp.h> */
#include <pcl/features/shot.h>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <opencv/cv.hpp>

#include <thread>
#include <mutex>
#include <boost/filesystem.hpp>

#include <jsk_recognition_utils/pcl/color_histogram.h>
#include <jsk_pcl_ros/region_adjacency_graph.h>

#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/visualization/histogram_visualizer.h>


namespace neatness_estimator
{

  pcl::visualization::PCLVisualizer::Ptr normalsVis
    (pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr cloud,
     pcl::PointCloud<pcl::Normal>::ConstPtr normals)
    {
      pcl::visualization::PCLVisualizer::Ptr viewer (new pcl::visualization::PCLVisualizer ("3D Viewer"));
      viewer->setBackgroundColor (0, 0, 0);
      pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb(cloud);
      viewer->addPointCloud<pcl::PointXYZRGB>(cloud, rgb, "sample cloud");
      viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "sample cloud");
      viewer->addPointCloudNormals<pcl::PointXYZRGB, pcl::Normal>(cloud, normals, 10, 0.02, "normals");
      viewer->addCoordinateSystem(0.1);
      viewer->setCameraPosition(0,0,0,0,0,0,0,0,0);
      viewer->initCameraParameters();
      return (viewer);
    }

  class Msgs
  {
  public:
    std::vector<jsk_recognition_msgs::ClusterPointIndices::ConstPtr> cluster;
    std::vector<sensor_msgs::PointCloud2::ConstPtr> cloud;
    std::vector<sensor_msgs::Image::ConstPtr> image;
    std::vector<jsk_recognition_msgs::BoundingBoxArray::ConstPtr> instance_boxes;
    std::vector<jsk_recognition_msgs::BoundingBoxArray::ConstPtr> cluster_boxes;
  };

  class DifferenceReasoner : public nodelet::Nodelet
  {
  public:

  protected:

    enum DIR {
      CURT = 0,
      PREV = 1,
    };

    // functions

    virtual void onInit();

    virtual bool service_callback(std_srvs::SetBool::Request& req,
                                  std_srvs::SetBool::Response& res);

    virtual bool read_data();

    virtual bool get_read_dirs();

    virtual bool compute_histograms
      (const pcl::PointCloud<pcl::PointXYZRGB>::Ptr& rgb_cloud,
       jsk_recognition_msgs::ClusterPointIndices::ConstPtr& input_indices,
       jsk_recognition_msgs::ColorHistogramArray& color_histogram_array,
       std::vector<jsk_recognition_msgs::Histogram>& geometry_histogram_array,
       cv::Mat& mask_image,
       cv::Mat& debug_image,
       std::vector<size_t>& labels,
       std::vector<size_t>& sorted_indices);

    virtual bool compute_color_histogram
      (const pcl::PointCloud<pcl::PointXYZRGB>::Ptr& rgb_cloud,
       jsk_recognition_msgs::ColorHistogram& color_histogram);

    virtual bool compute_geometry_histogram
      (const pcl::PointCloud<pcl::PointXYZRGB>::Ptr& rgb_cloud,
       jsk_recognition_msgs::Histogram& geometry_histogram);


    virtual bool save_pcd(std::string save_path,
                          const pcl::PointCloud<pcl::PointXYZRGB>& cloud);

    virtual bool save_image(std::string save_path,
                            const cv::Mat& image,
                            const cv::Mat& mask_image,
                            const cv::Mat& debug_image);

    virtual bool load_image(const sensor_msgs::Image::ConstPtr& input_msg,
                            cv::Mat& input_image);

    virtual bool save_color_histogram
      (std::string save_dir,
       std::vector<size_t> labels,
       const jsk_recognition_msgs::ColorHistogramArray& color_histogram_array);

    virtual bool save_geometry_histogram
      (std::string save_dir,
       std::vector<size_t> labels,
       const std::vector<jsk_recognition_msgs::Histogram>& geometry_histogram_array);

    virtual bool run();

    virtual bool create_sorted_indices
      (const std::vector<jsk_recognition_msgs::BoundingBox> input_boxes,
       std::vector<size_t>& sorted_indices,
       std::vector<size_t>& labels);

    // variables

    ros::NodeHandle nh_;
    ros::NodeHandle pnh_;

    ros::ServiceServer server_;
    ros::ServiceClient display_feature_client_;

    boost::mutex mutex_;
    std::string prefix_ = "./";

    std::string cloud_topic_ = "/openni_camera/point_cloud/cloud_registered/remote";
    std::string image_topic_ = "/openni_camera/rgb/image_rect_color";
    std::string cluster_topic_ = "/multi_euclidean_clustering/output_indices";
    std::string instance_boxes_topic_ = "/labeled_bounding_box_publisher/output/labeled_instance_boxes";
    std::string cluster_boxes_topic_ = "/labeled_bounding_box_publisher/output/labeled_cluster_boxes";

    std::vector<std::string> save_data_dir_;
    std::vector<std::string> log_dir_;
    std::vector<std::string> dir_;

    Msgs msgs;

    int buffer_size_ = 2;
    int index_ = 0;
    bool debug_view_ = false;
    int bin_size_ = 10;
    double white_threshold_ = 0.3;
    double black_threshold_ = 0.2;
    double normal_search_radius_ = 0.01;
    std::vector<size_t> labels_;
    std::vector<size_t> sorted_indices_;

    // 0: HUE, 1: HUE_AND_SATURATION
    jsk_recognition_utils::HistogramPolicy histogram_policy_ = jsk_recognition_utils::HUE;

  private:

  };

} // namespace neatness_estimator

#endif
