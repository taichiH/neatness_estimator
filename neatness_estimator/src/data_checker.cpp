#include "neatness_estimator/data_checker.h"

namespace neatness_estimator
{

  void DataChecker::onInit()
  {
    nh_ = getNodeHandle();
    pnh_ = getPrivateNodeHandle();

    pnh_.getParam("prefix", prefix_);

    server_ = pnh_.advertiseService("read", &DataChecker::service_callback, this);

    sub_point_cloud_.subscribe(pnh_, "input_cloud", 1);
    sub_image_.subscribe(pnh_, "input_image", 1);
    sub_boxes_.subscribe(pnh_, "input_boxes", 1);

    bool approximate_sync;
    pnh_.getParam("approximate_sync", approximate_sync);
    if (approximate_sync) {
      async_ = boost::make_shared<message_filters::Synchronizer<ApproximateSyncPolicy> >(1000);
      async_->connectInput(sub_boxes_, sub_point_cloud_, sub_image_);
      async_->registerCallback(boost::bind(&DataChecker::callback, this, _1, _2, _3));
    } else {
      sync_  = boost::make_shared<message_filters::Synchronizer<SyncPolicy> >(1000);
      sync_->connectInput(sub_boxes_, sub_point_cloud_, sub_image_);
      sync_->registerCallback(boost::bind(&DataChecker::callback, this, _1, _2, _3));
    }

  }


  bool DataChecker::get_read_dir(std::string& target_dir)
  {
    const boost::filesystem::path path(prefix_.c_str());

    std::vector<double> saved_dirs;
    for (const auto& e : boost::make_iterator_range(boost::filesystem::directory_iterator(path), {})) {
      saved_dirs.push_back(std::stod(e.path().filename().string()));
    }

    auto target = std::max_element(saved_dirs.begin(), saved_dirs.end());
    int idx = std::distance(saved_dirs.begin(), target);
    target_dir = std::to_string(static_cast<int>(saved_dirs.at(idx)));

    return true;
  }


  bool DataChecker::read_pcd(std::string path)
  {
    std::cerr << path << std::endl;

    prev_cloud_.reset(new pcl::PointCloud<pcl::PointXYZRGB>);
    if (pcl::io::loadPCDFile<pcl::PointXYZRGB> (path, *prev_cloud_) == -1) {
      PCL_ERROR ("Couldn't read file %s \n", path.c_str());
      return false;
    } else {
      ROS_INFO("success read pcd data from %s ", path.c_str());
    }

    return true;
  }

  bool DataChecker::read_image(std::string path)
  {
    std::cerr << path << std::endl;

    try {
      prev_image_ = cv::imread(path);
      ROS_INFO("success read image from %s ", path.c_str());
    } catch (cv::Exception& e) {
      ROS_ERROR("failed to load image \n", e.what());
      return false;
    }

    return true;
  }

  bool DataChecker::read_boxes(std::string path)
  {
    prev_boxes_.reset(new jsk_recognition_msgs::BoundingBoxArray);

    try {
      YAML::Node base = YAML::LoadFile(path);
      YAML::Node header = base["header"];

      prev_boxes_->header.frame_id = header["frame_id"].as<std::string>();
      std::string sec_str = header["stamp_sec"].as<std::string>();
      unsigned long sec = stoul(sec_str, nullptr, 0);
      std::string nsec_str = header["stamp_nsec"].as<std::string>();
      unsigned long nsec = stoul(nsec_str, nullptr, 0);
      prev_boxes_->header.stamp = ros::Time(sec, nsec);

      YAML::Node boxes = base["boxes"];
      for (std::size_t i=0; i<boxes.size(); ++i) {
        jsk_recognition_msgs::BoundingBox box;
        YAML::Node box_map = boxes[i]["box"];
        box.pose.position.x = box_map["x"].as<double>();
        box.pose.position.y = box_map["y"].as<double>();
        box.pose.position.z = box_map["z"].as<double>();
        box.pose.orientation.x = box_map["qx"].as<double>();
        box.pose.orientation.y = box_map["qy"].as<double>();
        box.pose.orientation.z = box_map["qz"].as<double>();
        box.pose.orientation.w = box_map["qw"].as<double>();
        box.dimensions.x = box_map["dimx"].as<double>();
        box.dimensions.y = box_map["dimy"].as<double>();
        box.dimensions.z = box_map["dimz"].as<double>();

        prev_boxes_->boxes.push_back(box);
      }

    } catch (YAML::ParserException& e) {
      ROS_ERROR("failed read boxes: %s", e.what());
      return false;
    }

    return true;
  }



  void DataChecker::read_files(std_srvs::SetBool::Response& res)
  {
    std::string target_dir;
    if ( !get_read_dir(target_dir) ) {
      res.success = false;
      return;
    }

    std::stringstream ss;
    ss << prefix_ << "/" << target_dir;

    std::stringstream pcd_read_path;
    pcd_read_path << ss.str() << "/" << target_dir << ".pcd";
    if (!read_pcd(pcd_read_path.str())) {
      res.success = false;
      return;
    }

    std::stringstream image_read_path;
    image_read_path << ss.str() << "/" << target_dir << ".jpg";
    if (!read_image(image_read_path.str())) {
      res.success = false;
      return;
    }

    std::stringstream boxes_read_path;
    boxes_read_path << ss.str() << "/" << target_dir << ".yaml";
    if (!read_boxes(boxes_read_path.str())) {
      res.success = false;
      return;
    }

  }


  void DataChecker::callback(const jsk_recognition_msgs::BoundingBoxArray::ConstPtr& boxes_msg,
                             const sensor_msgs::PointCloud2::ConstPtr& cloud_msg,
                             const sensor_msgs::Image::ConstPtr& image_msg)
  {
    boost::mutex::scoped_lock lock(mutex_);

    // current data

    header_ = cloud_msg->header;

    cloud_.reset(new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::fromROSMsg(*cloud_msg, *cloud_);

    try {
      cv_bridge::CvImagePtr cv_image = cv_bridge::toCvCopy(image_msg, "bgr8");
      image_ = cv_image->image;
    } catch (cv_bridge::Exception& e) {
      ROS_ERROR("Failed to convert sensor_msgs::Image to cv::Mat \n%s", e.what());
      cloud_->points.clear();
      return;
    }

    boxes_ = boxes_msg;

  }


  bool DataChecker::service_callback(std_srvs::SetBool::Request& req,
                                     std_srvs::SetBool::Response& res)
  {
    boost::mutex::scoped_lock lock(mutex_);
    read_files(res);

    return true;
  }

} // namespace neatness_estimator

#include <pluginlib/class_list_macros.h>
PLUGINLIB_EXPORT_CLASS(neatness_estimator::DataChecker, nodelet::Nodelet)
