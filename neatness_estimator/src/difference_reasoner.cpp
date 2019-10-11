#include "neatness_estimator/difference_reasoner.h"

namespace neatness_estimator
{

  void DifferenceReasoner::onInit()
  {
    nh_ = getNodeHandle();
    pnh_ = getPrivateNodeHandle();
    pnh_.getParam("prefix", prefix_);

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

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::fromROSMsg(*current_cloud_, *cloud);

    res.success = true;
    return true;
  }

} // namespace neatness_estimator

#include <pluginlib/class_list_macros.h>
PLUGINLIB_EXPORT_CLASS(neatness_estimator::DifferenceReasoner, nodelet::Nodelet)
