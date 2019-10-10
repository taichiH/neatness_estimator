#include "neatness_estimator/data_saver.h"

namespace neatness_estimator
{

  void DataSaver::onInit()
  {
    nh_ = getNodeHandle();
    pnh_ = getPrivateNodeHandle();
    pnh_.getParam("prefix", prefix_);

    server_ = pnh_.advertiseService("save", &DataSaver::service_callback, this);

    sub_point_cloud_.subscribe(pnh_, "input_cloud", 1);
    sub_image_.subscribe(pnh_, "input_image", 1);
    sub_cluster_.subscribe(pnh_, "input_cluster", 1);

    // topic name added to topic_manager when subscribe function is called
    ros::this_node::getSubscribedTopics(topics_);
    for (size_t i=0; i<topics_.size(); ++i) {
      std::cerr << i << ", " << topics_.at(i) << std::endl;
    }

    bool approximate_sync;
    pnh_.getParam("approximate_sync", approximate_sync);
    std::cerr << "approximate_sync: " << approximate_sync << std::endl;

    if (approximate_sync) {
      async_ = boost::make_shared<message_filters::Synchronizer<ApproximateSyncPolicy> >(1000);
      async_->connectInput(sub_point_cloud_, sub_image_, sub_cluster_);
      async_->registerCallback(boost::bind(&DataSaver::callback, this, _1, _2, _3));
    } else {

      sync_  = boost::make_shared<message_filters::Synchronizer<SyncPolicy> >(1000);
      sync_->connectInput(sub_point_cloud_, sub_image_, sub_cluster_);
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



  void DataSaver::callback(const sensor_msgs::PointCloud2::ConstPtr& cloud_msg,
                           const sensor_msgs::Image::ConstPtr& image_msg,
                           const jsk_recognition_msgs::ClusterPointIndices::ConstPtr& cluster_msg)
  {
    boost::mutex::scoped_lock lock(mutex_);

    std::cerr << "callback!!! " << std::endl;

    cloud_msg_ = cloud_msg;
    image_msg_ = image_msg;
    cluster_msg_ = cluster_msg;
  }


  bool DataSaver::service_callback(std_srvs::SetBool::Request& req,
                                   std_srvs::SetBool::Response& res)
  {
    boost::mutex::scoped_lock lock(mutex_);

    std::cerr << "service_callback!!!" << std::endl;

    if ( cloud_msg_->height * cloud_msg_->width == 0 ) {
      res.success = false;
      return false;
    }

    std::stringstream ss;
    std::string dir_name = std::to_string(cloud_msg_->header.stamp.sec);
    if ( !create_save_dir(ss, dir_name) ) {
      res.success = false;
    }

    std::stringstream bag_save_path;
    bag_save_path << ss.str() << "/" << cloud_msg_->header.stamp.sec << ".bag";
    rosbag::Bag bag;
    bag.open(bag_save_path.str(), rosbag::bagmode::Write);
    // topics : [clock, cloud, rgb, cluster]
    bag.write(topics_.at(1), cloud_msg_->header.stamp, *cloud_msg_);
    bag.write(topics_.at(2), image_msg_->header.stamp, *image_msg_);
    bag.write(topics_.at(3), cluster_msg_->header.stamp, *cluster_msg_);
    bag.close();


    res.success = true;
    return true;
  }

} // namespace neatness_estimator

#include <pluginlib/class_list_macros.h>
PLUGINLIB_EXPORT_CLASS(neatness_estimator::DataSaver, nodelet::Nodelet)
