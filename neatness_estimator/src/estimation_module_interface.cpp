#include "neatness_estimator/estimation_module_interface.h"

namespace neatness_estimator
{

  void EstimationModuleInterface::onInit()
  {
    nh_ = getNodeHandle();
    pnh_ = getPrivateNodeHandle();
    pnh_.getParam("prefix", prefix_);
    pnh_.getParam("fe_service_topic", fe_service_topic_);
    pnh_.getParam("de_service_topic", de_service_topic_);
    pnh_.getParam("fg_class_names", label_lst_);

    call_server_ =
      pnh_.advertiseService("call", &EstimationModuleInterface::service_callback, this);

    feature_client_ =
      pnh_.serviceClient<neatness_estimator_msgs::GetFeatures>(fe_service_topic_);
    difference_client_ =
      pnh_.serviceClient<neatness_estimator_msgs::GetDifference>(de_service_topic_);

    sub_point_cloud_.subscribe(pnh_, "input_cloud", 1);
    sub_image_.subscribe(pnh_, "input_image", 1);
    sub_cluster_.subscribe(pnh_, "input_cluster", 1);
    sub_labels_.subscribe(pnh_, "input_labels", 1);
    sub_instance_boxes_.subscribe(pnh_, "input_instance_boxes", 1);
    sub_cluster_boxes_.subscribe(pnh_, "input_cluster_boxes", 1);

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
      async_->connectInput(sub_point_cloud_,
                           sub_image_,
                           sub_cluster_,
                           sub_labels_,
                           sub_instance_boxes_,
                           sub_cluster_boxes_);
      async_->registerCallback(boost::bind(&EstimationModuleInterface::callback, this, _1, _2, _3, _4, _5, _6));
    } else {
      sync_  = boost::make_shared<message_filters::Synchronizer<SyncPolicy> >(1000);
      sync_->connectInput(sub_point_cloud_,
                          sub_image_,
                          sub_cluster_,
                          sub_labels_,
                          sub_instance_boxes_,
                          sub_cluster_boxes_);
      sync_->registerCallback(boost::bind(&EstimationModuleInterface::callback, this, _1, _2, _3, _4, _5, _6));
    }

  }


  bool EstimationModuleInterface::create_features_vec(const neatness_estimator_msgs::Features& features)
  {
    if (features_vec_.size() > 1) {
      features_vec_.erase(features_vec_.begin());
    }
    features_vec_.push_back(features);

    return features_vec_.size() == 2;
  }


  bool EstimationModuleInterface::create_save_dir(std::stringstream& ss,
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



  void EstimationModuleInterface::callback
  (const sensor_msgs::PointCloud2::ConstPtr& cloud_msg,
   const sensor_msgs::Image::ConstPtr& image_msg,
   const jsk_recognition_msgs::ClusterPointIndices::ConstPtr& cluster_msg,
   const jsk_recognition_msgs::LabelArray::ConstPtr& labels_msg,
   const jsk_recognition_msgs::BoundingBoxArray::ConstPtr& instance_boxes_msg,
   const jsk_recognition_msgs::BoundingBoxArray::ConstPtr& cluster_boxes_msg)
  {
    boost::mutex::scoped_lock lock(mutex_);

    std::cerr << "callback!!! " << std::endl;

    cloud_msg_ = cloud_msg;
    image_msg_ = image_msg;
    cluster_msg_ = cluster_msg;
    labels_msg_ = labels_msg;
    instance_boxes_msg_ = instance_boxes_msg;
    cluster_boxes_msg_ = cluster_boxes_msg;
  }

  bool EstimationModuleInterface::service_callback
  (neatness_estimator_msgs::GetDifference::Request& req,
   neatness_estimator_msgs::GetDifference::Response& res)
  {
    boost::mutex::scoped_lock lock(mutex_);

    std::cerr << "call_service_callback!!!" << std::endl;

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
    // topics : [cloud, rgb, cluster, labels]
    bag.write(topics_.at(0), cloud_msg_->header.stamp, *cloud_msg_);
    bag.write(topics_.at(1), image_msg_->header.stamp, *image_msg_);
    bag.write(topics_.at(2), cluster_msg_->header.stamp, *cluster_msg_);
    bag.write(topics_.at(3), labels_msg_->header.stamp, *labels_msg_);
    bag.write(topics_.at(4), instance_boxes_msg_->header.stamp, *instance_boxes_msg_);
    bag.write(topics_.at(5), cluster_boxes_msg_->header.stamp, *cluster_boxes_msg_);
    bag.close();
    ROS_INFO("rosbag file saved \n %s", bag_save_path.str().c_str());


    bool success = false;
    if (req.task == "two_scene") {
      success = get_two_scene_difference(res);
    } else if (req.task == "items") {
      success = get_items_difference
        (res, req.base_target_index, req.ref_target_indices, ss.str());
    }

    if (!success) {
      res.success = false;
      return false;
    }

    res.success = true;
    return true;
  }

  bool EstimationModuleInterface::get_two_scene_difference
  (neatness_estimator_msgs::GetDifference::Response& res)
  {
    // get feature service call
    neatness_estimator_msgs::GetFeatures feature_client_msg;
    feature_client_msg.request.cloud = *cloud_msg_;
    feature_client_msg.request.image = *image_msg_;
    feature_client_msg.request.cluster = *cluster_msg_;
    feature_client_msg.request.instance_boxes = *instance_boxes_msg_;
    feature_client_msg.request.cluster_boxes = *cluster_boxes_msg_;
    feature_client_.call(feature_client_msg);

    if (!feature_client_msg.response.success) {
      ROS_WARN("failed to call %s", fe_service_topic_.c_str());
      return false;
    }

    bool buffered = create_features_vec(feature_client_msg.response.features);
    if (buffered) {
      // compare histogram service call
      neatness_estimator_msgs::GetDifference difference_msg;
      difference_msg.request.features = features_vec_;
      difference_client_.call(difference_msg);

      if (!difference_msg.response.success) {
        ROS_WARN("failed to call %s", de_service_topic_.c_str());
        return false;
      } else {
        res.message = "success compare data";
        res.labels = difference_msg.response.labels;
        res.color_distance = difference_msg.response.color_distance;
        res.geometry_distance = difference_msg.response.geometry_distance;
        res.group_distance = difference_msg.response.group_distance;
      }
    } else {
      res.message = "need wait for features service call to compare datas";
    }

    return true;
  }

  bool EstimationModuleInterface::get_index_features
  (int index,
   neatness_estimator_msgs::Features& features)
  {
    // get base_target_index item feature
    jsk_recognition_msgs::ClusterPointIndices cluster_msg;
    cluster_msg.header = cluster_msg_->header;
    cluster_msg.cluster_indices.push_back(cluster_msg_->cluster_indices.at(index));
    jsk_recognition_msgs::BoundingBoxArray instance_boxes_msg;
    instance_boxes_msg.header = instance_boxes_msg_->header;
    instance_boxes_msg.boxes.push_back(instance_boxes_msg_->boxes.at(index));

    int cluster_boxes_index = 0;
    for (int i=0; i<cluster_boxes_msg_->boxes.size(); ++i) {
      if (cluster_boxes_msg_->boxes.at(i).label == instance_boxes_msg_->boxes.at(index).label) {
        ROS_INFO("found corresponding index");
        cluster_boxes_index = i;
        break;
      }
    }

    jsk_recognition_msgs::BoundingBoxArray cluster_boxes_msg;
    cluster_boxes_msg.header = cluster_boxes_msg_->header;
    cluster_boxes_msg.boxes.push_back
      (cluster_boxes_msg_->boxes.at(cluster_boxes_index));

    neatness_estimator_msgs::GetFeatures feature_client_msg;
    feature_client_msg.request.cloud = *cloud_msg_;
    feature_client_msg.request.image = *image_msg_;
    feature_client_msg.request.cluster = cluster_msg;
    feature_client_msg.request.instance_boxes = instance_boxes_msg;
    feature_client_msg.request.cluster_boxes = cluster_boxes_msg;
    feature_client_.call(feature_client_msg);

    if (!feature_client_msg.response.success) {
      ROS_WARN("failed to call %s", fe_service_topic_.c_str());
      return false;
    }

    features = feature_client_msg.response.features;

    return true;
  }

  int EstimationModuleInterface::get_confidence_color
  (std::vector<float> v)
  {
    double results = std::accumulate(v.begin(), v.end(), 0.0) / v.size();
    return static_cast<int>(results * 255);
  }

  bool EstimationModuleInterface::get_items_difference
  (neatness_estimator_msgs::GetDifference::Response& res,
   unsigned int base_target_index,
   std::vector<unsigned int> ref_target_indices,
   std::string save_prefix)
  {
    // TODO: support multi reference indices
    int ref_target_index = ref_target_indices.at(0);

    std::vector<neatness_estimator_msgs::Features> features_vec(2);
    neatness_estimator_msgs::Features features;
    get_index_features(base_target_index, features_vec.at(0));
    get_index_features(ref_target_index, features_vec.at(1));

    // compare histogram service call
    neatness_estimator_msgs::GetDifference difference_msg;
    difference_msg.request.features = features_vec;
    difference_client_.call(difference_msg);

    if (!difference_msg.response.success) {
      ROS_WARN("failed to call %s", de_service_topic_.c_str());
      return false;
    } else {
      res.message = "success compare data";
      res.labels = difference_msg.response.labels;
      res.color_distance = difference_msg.response.color_distance;
      res.geometry_distance = difference_msg.response.geometry_distance;
      res.group_distance = difference_msg.response.group_distance;
    }

    // in this function, always response array size assume 0
    std::vector<float> results;
    results.push_back(res.color_distance.at(0));
    results.push_back(res.geometry_distance.at(0));
    results.push_back(res.group_distance.at(0));
    int normalized_color = get_confidence_color(results);

    std::vector<int> target_indices{base_target_index, ref_target_index};
    cv::Mat color_mask
      (cv::Size(image_msg_->height, image_msg_->width), CV_8UC3, cv::Scalar(0,0,0));
    for (auto index : target_indices) {
      for (auto point_index : cluster_msg_->cluster_indices.at(index).indices) {
        size_t y = int(point_index / image_msg_->width);
        size_t x = int(point_index % image_msg_->width);
        color_mask.at<cv::Vec3b>(y,x)[2] = normalized_color;
      }
    }

    std::stringstream color_mask_save_path;
    color_mask_save_path << save_prefix << "/logs/color_mask.jpg";
    cv::imwrite(color_mask_save_path.str(), color_mask);

    return true;
  }

} // namespace neatness_estimator

#include <pluginlib/class_list_macros.h>
PLUGINLIB_EXPORT_CLASS(neatness_estimator::EstimationModuleInterface, nodelet::Nodelet)
