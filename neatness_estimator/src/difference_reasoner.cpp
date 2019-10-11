#include "neatness_estimator/difference_reasoner.h"

namespace neatness_estimator
{

  void DifferenceReasoner::onInit()
  {
    nh_ = getNodeHandle();
    pnh_ = getPrivateNodeHandle();
    pnh_.getParam("prefix", prefix_);
    pnh_.getParam("bin_size", bin_size_);
    pnh_.getParam("white_threshold", white_threshold_);
    pnh_.getParam("black_threshold", black_threshold_);

    pnh_.getParam("cloud_topic", cloud_topic_);
    pnh_.getParam("cloud_topic", image_topic_);
    pnh_.getParam("cloud_topic", cluster_topic_);
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
      return false;
    }
    if (prev_cloud_->width * prev_cloud_->height == 0) {
      return false;
    }

    return true;
  }

  bool DifferenceReasoner::compute_color_histogram(sensor_msgs::PointCloud2::ConstPtr& input_cloud,
                                                   jsk_recognition_msgs::ClusterPointIndices::ConstPtr& input_indices,
                                                   jsk_recognition_msgs::ColorHistogramArray& histogram_array)
  {
    std::cerr << __func__ << std::endl;

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr rgb_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::fromROSMsg(*input_cloud, *rgb_cloud);

    pcl::PointCloud<pcl::PointXYZHSV>::Ptr hsv_cloud(new pcl::PointCloud<pcl::PointXYZHSV>);
    pcl::PointCloudXYZRGBtoXYZHSV(*rgb_cloud, *hsv_cloud);

    std::cerr << 1 << std::endl;

    if ( rgb_cloud->points.empty() || hsv_cloud->points.empty() )
      return false;

    std::cerr << 2 << std::endl;

    for (size_t i = 0; i < rgb_cloud->points.size(); i++) {
      hsv_cloud->points.at(i).x = rgb_cloud->points.at(i).x;
      hsv_cloud->points.at(i).y = rgb_cloud->points.at(i).y;
      hsv_cloud->points.at(i).z = rgb_cloud->points.at(i).z;
    }

    std::cerr << 3 << std::endl;

    pcl::ExtractIndices<pcl::PointXYZHSV> extract;
    extract.setInputCloud(hsv_cloud);

    std::cerr << 4 << std::endl;

    histogram_array.histograms.resize(input_indices->cluster_indices.size());
    histogram_array.header = input_indices->header;
    for (size_t i = 0; i < input_indices->cluster_indices.size(); ++i) {

      // organized pointcloud
      pcl_msgs::PointIndices::Ptr nonnan_indices (new pcl_msgs::PointIndices);
      for (auto index : input_indices->cluster_indices.at(i).indices) {
        pcl::PointXYZRGB p = rgb_cloud->points.at(index);
        if (!std::isnan(p.x) && !std::isnan(p.y) && !std::isnan(p.z)) {
          nonnan_indices->indices.push_back(index);
        }
      }


      std::cerr << 5 << std::endl;

      pcl::IndicesPtr indices(new std::vector<int>(nonnan_indices->indices));
      extract.setIndices(indices);

      std::cerr << 6 << std::endl;

      pcl::PointCloud<pcl::PointXYZHSV> segmented_cloud;
      extract.filter(segmented_cloud);

      std::cerr << 7 << std::endl;

      histogram_array.histograms.at(i).header = input_indices->header;
      if (histogram_policy_ == jsk_recognition_utils::HUE) {
        jsk_recognition_utils::computeColorHistogram1d(segmented_cloud,
                                                       histogram_array.histograms.at(i).histogram,
                                                       bin_size_,
                                                       white_threshold_,
                                                       black_threshold_);
      } else if (histogram_policy_ == jsk_recognition_utils::HUE_AND_SATURATION) {
        jsk_recognition_utils::computeColorHistogram2d(segmented_cloud,
                                                       histogram_array.histograms.at(i).histogram,
                                                       bin_size_,
                                                       white_threshold_,
                                                       black_threshold_);
      } else {
        ROS_WARN("Invalid histogram policy");
        return false;
      }
    }

    std::cerr << 8 << std::endl;

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


    jsk_recognition_msgs::ColorHistogramArray histogram_array;
    compute_color_histogram(current_cloud_, current_cluster_, histogram_array);


    res.success = true;
    return true;
  }

} // namespace neatness_estimator

#include <pluginlib/class_list_macros.h>
PLUGINLIB_EXPORT_CLASS(neatness_estimator::DifferenceReasoner, nodelet::Nodelet)
