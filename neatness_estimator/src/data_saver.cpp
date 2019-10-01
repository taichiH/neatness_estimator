#include "neatness_estimator/data_saver.h"

namespace neatness_estimator
{

  void DataSaver::onInit()
  {
    nh_ = getNodeHandle();
    pnh_ = getPrivateNodeHandle();

    pnh_.getParam("binary", binary_);
    pnh_.getParam("prefix", prefix_);
    pnh_.getParam("compressed", compressed_);

    server_ = pnh_.advertiseService("save", &DataSaver::service_callback, this);

    sub_point_cloud_.subscribe(pnh_, "input_cloud", 1);
    sub_image_.subscribe(pnh_, "input_image", 1);
    sub_boxes_.subscribe(pnh_, "input_boxes", 1);

    bool approximate_sync;
    pnh_.getParam("approximate_sync", approximate_sync);
    if (approximate_sync) {
      async_ = boost::make_shared<message_filters::Synchronizer<ApproximateSyncPolicy> >(1000);
      async_->connectInput(sub_boxes_, sub_point_cloud_, sub_image_);
      async_->registerCallback(boost::bind(&DataSaver::callback, this, _1, _2, _3));
    } else {
      sync_  = boost::make_shared<message_filters::Synchronizer<SyncPolicy> >(1000);
      sync_->connectInput(sub_boxes_, sub_point_cloud_, sub_image_);
      sync_->registerCallback(boost::bind(&DataSaver::callback, this, _1, _2, _3));
    }

  }


  bool DataSaver::create_save_dir(std::stringstream& ss,
                                  std::string dir_name)
  {
    const boost::filesystem::path path(prefix_.c_str());
    boost::system::error_code error;

    if (!boost::filesystem::exists(path)) {
      const bool ret = boost::filesystem::create_directories(path, error);
      if (!ret || error) {
        ROS_ERROR("failed create directories : \n%s", prefix_.c_str());
        return false;
      } else {
        ROS_INFO("create: %s", prefix_.c_str());
      }
    }

    ss << prefix_ << "/" << dir_name;
    const boost::filesystem::path save_dir(ss.str().c_str());
    const bool ret = boost::filesystem::create_directory(save_dir, error);
    if (!ret || error) {
      ROS_ERROR("failed create save_dir : \n%s", ss.str().c_str());
      return false;
    }
    return true;
  }


  bool DataSaver::save_pcd(std::string save_path)
  {
    ROS_INFO("save pcd path: \n%s", save_path.c_str());
    pcl::PCDWriter writer;
    try {
      if(binary_) {
        if(compressed_) {
          writer.writeBinaryCompressed(save_path, *cloud_);
        } else {
          writer.writeBinary(save_path, *cloud_);
        }
      } else {
        writer.writeASCII(save_path, *cloud_);
      }
    } catch (...) {
      return false;
    }
    return true;
  }

  bool DataSaver::save_image(std::string save_path)
  {
    ROS_INFO("save image path: \n%s", save_path.c_str());
    cv::imwrite(save_path, image_);
    return true;
  }


  bool DataSaver::save_boxes(std::string save_path)
  {
    ROS_INFO("save yaml path: \n%s", save_path.c_str());
    YAML::Emitter emitter;
    emitter.SetOutputCharset(YAML::EscapeNonAscii);
    emitter.SetIndent(2);
    emitter.SetMapFormat(YAML::Flow);

    // start all map
    emitter << YAML::BeginMap; // all

    // start header map
    emitter << YAML::Key << "header";
    emitter << YAML::Value; // header
    emitter << YAML::BeginSeq;
    emitter << YAML::BeginMap << YAML::Key << "frame_id" << YAML::Value << boxes_->header.frame_id << YAML::EndMap
            << YAML::BeginMap << YAML::Key << "stamp" << YAML::Key << std::to_string(boxes_->header.stamp.toSec()) << YAML::EndMap;
    emitter << YAML::EndSeq;
    // end header map

    // start boxes map
    emitter << YAML::Key << "boxes";
    emitter << YAML::Value;

    emitter << YAML::Flow; // list
    emitter << YAML::BeginSeq; // list begin
    for (auto box : boxes_->boxes){
      // start box map
      emitter << YAML::BeginMap;
      emitter << YAML::Key << "box";
      emitter << YAML::Value;
      emitter << YAML::BeginSeq
              << YAML::BeginMap << YAML::Key << "x" << YAML::Value << std::to_string(box.pose.position.x) << YAML::EndMap
              << YAML::BeginMap << YAML::Key << "y" << YAML::Value << std::to_string(box.pose.position.y) << YAML::EndMap
              << YAML::BeginMap << YAML::Key << "z" << YAML::Value << std::to_string(box.pose.position.z) << YAML::EndMap
              << YAML::BeginMap << YAML::Key << "qx" << YAML::Value << std::to_string(box.pose.orientation.x) << YAML::EndMap
              << YAML::BeginMap << YAML::Key << "qy" << YAML::Value << std::to_string(box.pose.orientation.y) << YAML::EndMap
              << YAML::BeginMap << YAML::Key << "qz" << YAML::Value << std::to_string(box.pose.orientation.z) << YAML::EndMap
              << YAML::BeginMap << YAML::Key << "qw" << YAML::Value << std::to_string(box.pose.orientation.w) << YAML::EndMap
              << YAML::BeginMap << YAML::Key << "dimx" << YAML::Value << std::to_string(box.dimensions.x) << YAML::EndMap
              << YAML::BeginMap << YAML::Key << "dimy" << YAML::Value << std::to_string(box.dimensions.y) << YAML::EndMap
              << YAML::BeginMap << YAML::Key << "dimz" << YAML::Value << std::to_string(box.dimensions.z) << YAML::EndMap
              << YAML::EndSeq;
      emitter << YAML::EndMap;
      // end box map
    }
    emitter << YAML::EndSeq; // list end
    // end boxes map

    emitter << YAML::EndMap; // all
    // end all map

    std::ofstream fo(save_path);
    fo << emitter.c_str();

    return true;
  }



  void DataSaver::callback(const jsk_recognition_msgs::BoundingBoxArray::ConstPtr& boxes_msg,
                           const sensor_msgs::PointCloud2::ConstPtr& cloud_msg,
                           const sensor_msgs::Image::ConstPtr& image_msg)
  {
    boost::mutex::scoped_lock lock(mutex_);

    std::cerr << "callback!!!" << std::endl;
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


  bool DataSaver::service_callback(std_srvs::SetBool::Request& req,
                                   std_srvs::SetBool::Response& res)
  {
    boost::mutex::scoped_lock lock(mutex_);

    std::cerr << "service_callback!!!" << std::endl;

    if ( cloud_->points.empty() ) {
      res.success = false;
      return false;
    }

    std::stringstream ss;
    std::string dir_name = std::to_string(cloud_->header.stamp);
    if ( !create_save_dir(ss, dir_name) ) {
      res.success = false;
    }

    std::stringstream pcd_save_path;
    pcd_save_path << ss.str() << "/" << cloud_->header.stamp << ".pcd";
    if ( !save_pcd(pcd_save_path.str()) ) {
      res.success = false;
      return false;
    }


    std::stringstream image_save_path;
    image_save_path << ss.str() << "/" << cloud_->header.stamp << ".jpg";
    if ( !save_image(image_save_path.str()) ) {
      res.success = false;
      return false;
    }

    std::stringstream boxes_save_path;
    boxes_save_path << ss.str() << "/" << cloud_->header.stamp << ".yaml";
    if ( !save_boxes(boxes_save_path.str()) ) {
      res.success = false;
      return false;
    }

    res.success = true;
    return true;
  }

} // namespace neatness_estimator

#include <pluginlib/class_list_macros.h>
PLUGINLIB_EXPORT_CLASS(neatness_estimator::DataSaver, nodelet::Nodelet)
