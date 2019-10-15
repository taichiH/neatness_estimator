#include "neatness_estimator/difference_reasoner.h"

#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/visualization/histogram_visualizer.h>
#include <thread>
pcl::visualization::PCLVisualizer::Ptr normalsVis (pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr cloud,
                                                   pcl::PointCloud<pcl::Normal>::ConstPtr normals)
{
  pcl::visualization::PCLVisualizer::Ptr viewer (new pcl::visualization::PCLVisualizer ("3D Viewer"));
  viewer->setBackgroundColor (0, 0, 0);
  pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb(cloud);
  viewer->addPointCloud<pcl::PointXYZRGB> (cloud, rgb, "sample cloud");
  viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "sample cloud");
  viewer->addPointCloudNormals<pcl::PointXYZRGB, pcl::Normal> (cloud, normals, 10, 0.05, "normals");
  viewer->addCoordinateSystem (1.0);
  viewer->initCameraParameters ();
  return (viewer);
}


namespace neatness_estimator
{

  void DifferenceReasoner::onInit()
  {
    nh_ = getNodeHandle();
    pnh_ = getPrivateNodeHandle();
    pnh_.getParam("debug_view", debug_view_);
    pnh_.getParam("prefix", prefix_);
    pnh_.getParam("bin_size", bin_size_);
    pnh_.getParam("white_threshold", white_threshold_);
    pnh_.getParam("black_threshold", black_threshold_);
    pnh_.getParam("normal_search_radius", normal_search_radius_);

    int policy;
    pnh_.getParam("histogram_policy", policy);
    if (policy == 1) {
      histogram_policy_ = jsk_recognition_utils::HUE_AND_SATURATION;
    } else {
      histogram_policy_ = jsk_recognition_utils::HUE;
    }

    pnh_.getParam("cloud_topic", cloud_topic_);
    pnh_.getParam("image_topic", image_topic_);
    pnh_.getParam("cluster_topic", cluster_topic_);
    server_ = pnh_.advertiseService("read", &DifferenceReasoner::service_callback, this);

  }

  bool DifferenceReasoner::get_read_dirs(std::string& current_dir, std::string& prev_dir)
  {

    const boost::filesystem::path path(prefix_.c_str());

    std::vector<double> saved_dirs;
    for (const auto& e : boost::make_iterator_range(boost::filesystem::directory_iterator(path), {})) {
      saved_dirs.push_back(std::stod(e.path().filename().string()));
    }
    std::sort(saved_dirs.begin(), saved_dirs.end(), std::greater<double>());

    std::stringstream current_ss;
    current_ss << prefix_ << "/"
               << std::to_string(static_cast<int>(saved_dirs.at(0))) << "/"
               << std::to_string(static_cast<int>(saved_dirs.at(0))) << ".bag";
    current_dir = current_ss.str();

    std::stringstream prev_ss;
    prev_ss << prefix_ << "/"
            << std::to_string(static_cast<int>(saved_dirs.at(1))) << "/"
            << std::to_string(static_cast<int>(saved_dirs.at(1))) << ".bag";
    prev_dir = prev_ss.str();

    return true;
  }


  bool DifferenceReasoner::read_data(std::string& current_dir, std::string& prev_dir)
  {
    current_cluster_.reset(new jsk_recognition_msgs::ClusterPointIndices);
    current_cloud_.reset(new sensor_msgs::PointCloud2);
    current_image_.reset(new sensor_msgs::Image);
    prev_cluster_.reset(new jsk_recognition_msgs::ClusterPointIndices);
    prev_cloud_.reset(new sensor_msgs::PointCloud2);
    prev_image_.reset(new sensor_msgs::Image);

    rosbag::Bag bag;
    try {
      bag.open(current_dir);
      for (rosbag::MessageInstance const m : rosbag::View(bag)) {
        if (m.getTopic() == cluster_topic_)
          current_cluster_ = m.instantiate<jsk_recognition_msgs::ClusterPointIndices>();
        if (m.getTopic() == cloud_topic_)
          current_cloud_ = m.instantiate<sensor_msgs::PointCloud2>();
        if (m.getTopic() == image_topic_)
          current_image_ = m.instantiate<sensor_msgs::Image>();
      }
      bag.close();

      bag.open(prev_dir);
      for (rosbag::MessageInstance const m : rosbag::View(bag)) {
        if (m.getTopic() == cluster_topic_)
          prev_cluster_ = m.instantiate<jsk_recognition_msgs::ClusterPointIndices>();
        if (m.getTopic() == cloud_topic_)
          prev_cloud_ = m.instantiate<sensor_msgs::PointCloud2>();
        if (m.getTopic() == image_topic_)
          prev_image_ = m.instantiate<sensor_msgs::Image>();
      }
      bag.close();

    } catch (rosbag::BagException& e) {
      ROS_ERROR("failed get rosbag data \n %s", e.what());
      return false;
    }

    if (current_cloud_->width * current_cloud_->height == 0) {
      ROS_WARN("current_cloud size: 0");
      return false;
    }

    if (prev_cloud_->width * prev_cloud_->height == 0) {
      ROS_WARN("prev_cloud size: 0");
      return false;
    }

    return true;
  }

  bool DifferenceReasoner::compute_color_histogram
  (const pcl::PointCloud<pcl::PointXYZRGB>::Ptr& rgb_cloud,
   jsk_recognition_msgs::ColorHistogram& color_histogram)
  {
    pcl::PointCloud<pcl::PointXYZHSV>::Ptr hsv_cloud(new pcl::PointCloud<pcl::PointXYZHSV>);
    pcl::PointCloudXYZRGBtoXYZHSV(*rgb_cloud, *hsv_cloud);

    for (size_t i = 0; i < rgb_cloud->points.size(); i++) {
      hsv_cloud->points.at(i).x = rgb_cloud->points.at(i).x;
      hsv_cloud->points.at(i).y = rgb_cloud->points.at(i).y;
      hsv_cloud->points.at(i).z = rgb_cloud->points.at(i).z;
    }

    if (histogram_policy_ == jsk_recognition_utils::HUE) {
      jsk_recognition_utils::computeColorHistogram1d(*hsv_cloud,
                                                     color_histogram.histogram,
                                                     bin_size_,
                                                     white_threshold_,
                                                     black_threshold_);
    } else if (histogram_policy_ == jsk_recognition_utils::HUE_AND_SATURATION) {
      jsk_recognition_utils::computeColorHistogram2d(*hsv_cloud,
                                                     color_histogram.histogram,
                                                     bin_size_,
                                                     white_threshold_,
                                                     black_threshold_);
    } else {
      ROS_WARN("Invalid histogram policy");
      return false;
    }
    return true;
  }


  bool DifferenceReasoner::compute_geometry_histogram
  (const pcl::PointCloud<pcl::PointXYZRGB>::Ptr& rgb_cloud,
   jsk_recognition_msgs::Histogram& geometry_histogram)
  {

    pcl::NormalEstimationOMP<pcl::PointXYZRGB, pcl::Normal> normal_estimation;
    pcl::PointCloud<pcl::Normal>::Ptr cloud_normals(new pcl::PointCloud<pcl::Normal>());
    pcl::search::KdTree<pcl::PointXYZRGB>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZRGB>());
    normal_estimation.setSearchMethod (tree);
    normal_estimation.setRadiusSearch(normal_search_radius_);
    normal_estimation.setInputCloud(rgb_cloud);
    normal_estimation.compute(*cloud_normals);

    pcl::CVFHEstimation<pcl::PointXYZRGB, pcl::Normal, pcl::VFHSignature308> cvfh;
    cvfh.setInputCloud(rgb_cloud);
    cvfh.setInputNormals(cloud_normals);
    cvfh.setSearchMethod(tree);
    cvfh.setEPSAngleThreshold(5.0f / 180.0f * M_PI);
    cvfh.setCurvatureThreshold(1.0f);
    cvfh.setNormalizeBins(false);

    pcl::PointCloud<pcl::VFHSignature308>::Ptr cvfhs
      (new pcl::PointCloud<pcl::VFHSignature308>());
    cvfh.compute(*cvfhs);

    int feature_size = sizeof(pcl::VFHSignature308) / sizeof(cvfhs->points[0].histogram[0]);
    cv::Mat histogram = cv::Mat(sizeof(char), feature_size, CV_32F);
    for (int i = 0; i < histogram.cols; i++) {
      histogram.at<float>(0, i) = cvfhs->points[0].histogram[i];
    }

    float curvature = 0.0f;
    for (int i = 0; i < cloud_normals->size(); i++) {
      curvature += cloud_normals->points[i].curvature;
    }
    curvature /= static_cast<float>(cloud_normals->size());
    cv::normalize(histogram, histogram, 0, 1, cv::NORM_MINMAX, -1, cv::Mat());

    for (int i = 0; i < histogram.cols; i++) {
      geometry_histogram.histogram.push_back(histogram.at<float>(0, i));
    }

    if (debug_view_) {
      pcl::visualization::PCLVisualizer::Ptr viewer;
      viewer = normalsVis(rgb_cloud, cloud_normals);
      viewer->saveScreenshot("/tmp/normal_viewer.png");
      viewer->spin();
    }

    return true;
  }

  bool DifferenceReasoner::compute_histograms
  (const pcl::PointCloud<pcl::PointXYZRGB>::Ptr& rgb_cloud,
   jsk_recognition_msgs::ClusterPointIndices::ConstPtr& input_indices,
   jsk_recognition_msgs::ColorHistogramArray& color_histogram_array,
   std::vector<jsk_recognition_msgs::Histogram>& geometry_histogram_array)
  {

    for (size_t i = 0; i < input_indices->cluster_indices.size(); ++i) {

      pcl::PointIndices::Ptr nonnan_indices (new pcl::PointIndices);
      for (auto index : input_indices->cluster_indices.at(i).indices) {
        pcl::PointXYZRGB p = rgb_cloud->points.at(index);
        if (!std::isnan(p.x) && !std::isnan(p.y) && !std::isnan(p.z)) {
          nonnan_indices->indices.push_back(index);
        }
      }

      pcl::ExtractIndices<pcl::PointXYZRGB> extract;
      extract.setInputCloud(rgb_cloud);
      extract.setIndices(nonnan_indices);
      pcl::PointCloud<pcl::PointXYZRGB>::Ptr clustered_cloud
        (new pcl::PointCloud<pcl::PointXYZRGB>());
      extract.filter(*clustered_cloud);

      jsk_recognition_msgs::ColorHistogram color_histogram;
      compute_color_histogram(clustered_cloud, color_histogram);
      color_histogram_array.histograms.push_back(color_histogram);

      jsk_recognition_msgs::Histogram geometry_histogram;
      compute_geometry_histogram(clustered_cloud, geometry_histogram);
      geometry_histogram_array.push_back(geometry_histogram);

    }

    return true;
  }


  bool DifferenceReasoner::service_callback(std_srvs::SetBool::Request& req,
                                            std_srvs::SetBool::Response& res)
  {
    boost::mutex::scoped_lock lock(mutex_);
    std::string current_dir;
    std::string prev_dir;

    if ( !get_read_dirs(current_dir, prev_dir) ) {
      res.success = false;
      return false;
    }

    if ( !read_data(current_dir, prev_dir) ) {
      res.success = false;
      return false;
    }

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr rgb_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::fromROSMsg(*current_cloud_, *rgb_cloud);

    jsk_recognition_msgs::ColorHistogramArray color_histogram_array;
    std::vector<jsk_recognition_msgs::Histogram> geometry_histogram_array;
    compute_histograms(rgb_cloud,
                       current_cluster_,
                       color_histogram_array,
                       geometry_histogram_array);

    res.success = true;
    return true;
  }

} // namespace neatness_estimator

#include <pluginlib/class_list_macros.h>
PLUGINLIB_EXPORT_CLASS(neatness_estimator::DifferenceReasoner, nodelet::Nodelet)
