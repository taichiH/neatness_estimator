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
    pnh_.getParam("get_color_mask", get_color_mask_);
    pnh_.getParam("approximate_sync", approximate_sync_);
    pnh_.getParam("only_color_and_geometry", only_color_and_geometry_);

    std::cout << "approximate_sync: " << approximate_sync_ << std::endl;
    std::cout << "only_color_and_geometry: " << only_color_and_geometry_ << std::endl;

    call_server_ =
      pnh_.advertiseService("call", &EstimationModuleInterface::service_callback, this);
    feature_client_ =
      pnh_.serviceClient<neatness_estimator_msgs::GetFeatures>(fe_service_topic_);
    difference_client_ =
      pnh_.serviceClient<neatness_estimator_msgs::GetDifference>(de_service_topic_);

    if (only_color_and_geometry_) {
      sub_point_cloud_.subscribe(pnh_, "input_cloud", 1);
      sub_image_.subscribe(pnh_, "input_image", 1);
      sub_cluster_.subscribe(pnh_, "input_cluster", 1);
    } else {
      sub_point_cloud_.subscribe(pnh_, "input_cloud", 1);
      sub_image_.subscribe(pnh_, "input_image", 1);
      sub_cluster_.subscribe(pnh_, "input_cluster", 1);
      sub_labels_.subscribe(pnh_, "input_labels", 1);
      sub_instance_boxes_.subscribe(pnh_, "input_instance_boxes", 1);
      sub_cluster_boxes_.subscribe(pnh_, "input_cluster_boxes", 1);
    }

    // topic name added to topic_manager when subscribe function is called
    ros::this_node::getSubscribedTopics(topics_);
    for (size_t i=0; i<topics_.size(); ++i) {
      std::cout << i << ", " << topics_.at(i) << std::endl;
    }

    register_callback(only_color_and_geometry_);
  }

  void EstimationModuleInterface::register_callback(bool only_color_and_geometry)
  {
    if (approximate_sync_) {
      if (only_color_and_geometry) {
        async_color_and_geo_ =
          boost::make_shared<message_filters::Synchronizer<ApproximateSyncColorAndGeo> >(1000);
        async_color_and_geo_->connectInput(sub_point_cloud_, sub_image_, sub_cluster_);
        async_color_and_geo_->registerCallback
          (boost::bind(&EstimationModuleInterface::color_and_geometry_callback,this, _1, _2, _3));
      } else {
        async_ = boost::make_shared<message_filters::Synchronizer<ApproximateSync> >(1000);
        async_->connectInput(sub_point_cloud_, sub_image_, sub_cluster_,
                             sub_labels_, sub_instance_boxes_, sub_cluster_boxes_);
        async_->registerCallback
          (boost::bind(&EstimationModuleInterface::callback,this, _1, _2, _3, _4, _5, _6));
      }
    } else {
      if (only_color_and_geometry) {
        sync_color_and_geo_  = boost::make_shared<message_filters::Synchronizer<SyncColorAndGeo> >(1000);
        sync_color_and_geo_->connectInput(sub_point_cloud_, sub_image_, sub_cluster_);
        sync_color_and_geo_->registerCallback
          (boost::bind(&EstimationModuleInterface::color_and_geometry_callback, this, _1, _2, _3));
      } else {
        sync_  = boost::make_shared<message_filters::Synchronizer<Sync> >(1000);
        sync_->connectInput(sub_point_cloud_, sub_image_, sub_cluster_,
                            sub_labels_, sub_instance_boxes_, sub_cluster_boxes_);
        sync_->registerCallback
          (boost::bind(&EstimationModuleInterface::callback,this, _1, _2, _3, _4, _5, _6));
      }
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

    if ( !is_called_ ) {
      std::cerr << __func__ << " callback function called" << std::endl;
      is_called_ = true;
    }

    cloud_msg_ = cloud_msg;
    image_msg_ = image_msg;
    cluster_msg_ = cluster_msg;
    labels_msg_ = labels_msg;
    instance_boxes_msg_ = instance_boxes_msg;
    cluster_boxes_msg_ = cluster_boxes_msg;
  }

  void EstimationModuleInterface::color_and_geometry_callback
  (const sensor_msgs::PointCloud2::ConstPtr& cloud_msg,
   const sensor_msgs::Image::ConstPtr& image_msg,
   const jsk_recognition_msgs::ClusterPointIndices::ConstPtr& cluster_msg)
  {
    boost::mutex::scoped_lock lock(mutex_);

    if ( !is_called_ ) {
      std::cerr << __func__ << " callback function called" << std::endl;
      is_called_ = true;
    }

    jsk_recognition_msgs::ClusterPointIndices dummy_cluster;
    dummy_cluster.cluster_indices.push_back(cluster_msg->cluster_indices.at(0));

    jsk_recognition_msgs::LabelArray dummy_labels;
    jsk_recognition_msgs::BoundingBoxArray dummy_instance_boxes;
    jsk_recognition_msgs::BoundingBoxArray dummy_cluster_boxes;
    for (int i=0; i<dummy_cluster.cluster_indices.size(); ++i) {
      jsk_recognition_msgs::Label dummy_label;
      jsk_recognition_msgs::BoundingBox dummy_box;
      dummy_labels.labels.push_back(dummy_label);
      dummy_instance_boxes.boxes.push_back(dummy_box);
      dummy_cluster_boxes.boxes.push_back(dummy_box);
    }
    dummy_cluster.header = cloud_msg->header;
    dummy_labels.header = cloud_msg->header;
    dummy_instance_boxes.header = cloud_msg->header;
    dummy_cluster_boxes.header = cloud_msg->header;

    cloud_msg_ = cloud_msg;
    image_msg_ = image_msg;
    cluster_msg_ = boost::make_shared<jsk_recognition_msgs::ClusterPointIndices>(dummy_cluster);
    labels_msg_ = boost::make_shared<jsk_recognition_msgs::LabelArray>(dummy_labels);
    instance_boxes_msg_ = boost::make_shared<jsk_recognition_msgs::BoundingBoxArray>(dummy_instance_boxes);
    cluster_boxes_msg_ = boost::make_shared<jsk_recognition_msgs::BoundingBoxArray>(dummy_cluster_boxes);
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

    if ( !only_color_and_geometry_) {
      bag.write(topics_.at(0), cloud_msg_->header.stamp, *cloud_msg_);
      bag.write(topics_.at(1), image_msg_->header.stamp, *image_msg_);
      bag.write(topics_.at(2), cluster_msg_->header.stamp, *cluster_msg_);
      bag.write(topics_.at(3), labels_msg_->header.stamp, *labels_msg_);
      bag.write(topics_.at(4), instance_boxes_msg_->header.stamp, *instance_boxes_msg_);
      bag.write(topics_.at(5), cluster_boxes_msg_->header.stamp, *cluster_boxes_msg_);
    }

    bag.close();
    ROS_INFO("rosbag file saved \n %s", bag_save_path.str().c_str());

    bool success = false;
    if (req.task == "two_scene") {
      ROS_INFO("get two scene difference");
      success = get_two_scene_difference(res);
    } else if (req.task == "items") {
      ROS_INFO("get items difference");
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
    feature_client_msg.request.task = "two_scene";
    feature_client_.call(feature_client_msg);

    if (!feature_client_msg.response.success) {
      ROS_WARN("failed to call %s", fe_service_topic_.c_str());
      return false;
    }

    bool buffered = create_features_vec(feature_client_msg.response.features);
    if (buffered) {
      ROS_WARN("buffered");
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
      ROS_WARN("need wait for features service call to compare datas");
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

    for (int i=0; i<instance_boxes_msg_->boxes.size(); ++i) {
      if (instance_boxes_msg_->boxes.at(i).label == instance_boxes_msg_->boxes.at(index).label) {
        instance_boxes_msg.boxes.push_back(instance_boxes_msg_->boxes.at(i));
      }
    }

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
    feature_client_msg.request.index = cluster_boxes_index;
    feature_client_msg.request.task = "items";
    feature_client_.call(feature_client_msg);

    if (!feature_client_msg.response.success) {
      ROS_WARN("failed to call %s", fe_service_topic_.c_str());
      ROS_WARN("cloud info: %d, %d",
               feature_client_msg.request.cloud.width,
               feature_client_msg.request.cloud.height);
      ROS_WARN("image info: %d, %d",
               feature_client_msg.request.image.width,
               feature_client_msg.request.image.height);
      ROS_WARN("cluster info: %d", feature_client_msg.request.cluster.cluster_indices.size());
      ROS_WARN("instance_boxes info: %d", feature_client_msg.request.instance_boxes.boxes.size());
      ROS_WARN("cluster_boxes info: %d", feature_client_msg.request.cluster_boxes.boxes.size());
      return false;
    }

    features = feature_client_msg.response.features;

    return true;
  }

  int EstimationModuleInterface::get_confidence_color
  (std::vector<float> v)
  {
    double results = std::accumulate(v.begin(), v.end(), 0.0) / v.size();
    return static_cast<int>((1 - results) * 255);
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
    std::vector<unsigned int> targets{base_target_index, ref_target_index};
    bool success = true;
    for (int i=0; i<targets.size(); ++i) {
      success = get_index_features(targets.at(i), features_vec.at(i));
      if ( !success ) {
        ROS_WARN("failed to get_index_features");
        ROS_WARN("i: %d, target_index: %d", i, targets.at(i));
        break;
      }
    }
    if ( !success ) return false;

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

    // debug mask log
    if (get_color_mask_) {
      cv::Mat debug_image;
      try {
        cv_bridge::CvImagePtr cv_image = cv_bridge::toCvCopy
          (image_msg_, "bgr8");
        debug_image = cv_image->image;
      } catch (cv_bridge::Exception& e) {
        ROS_ERROR("Failed to convert sensor_msgs::Image to cv::Mat \n%s", e.what());
        return false;
      }

      std::vector<int> target_indices{base_target_index, ref_target_index};
      for (int i=0; i<target_indices.size(); ++i) {
        int index = target_indices.at(i);
        cv::Mat tmp_mask = cv::Mat::zeros(debug_image.rows, debug_image.cols, CV_8UC1);
        for (auto point_index : cluster_msg_->cluster_indices.at(index).indices) {
          int y = int(point_index / image_msg_->width);
          int x = int(point_index % image_msg_->width);
          tmp_mask.at<unsigned char>(y,x) = 255;
        }

        std::vector<std::vector<cv::Point> > contours;
        std::vector<cv::Vec4i> hierarchy;
        try {
          cv::findContours(tmp_mask, contours, hierarchy,
                           CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, cv::Point(0, 0));
          cv::Point min_pt(debug_image.cols * debug_image.cols,
                           debug_image.rows * debug_image.rows);
          for (auto contour : contours) {
            for (size_t i=0; i<contour.size(); ++i) {
              cv::Point pt1;
              auto pt2 = contour.at(i);
              if (i == 0) {
                pt1 = contour.at(contour.size() - 1);
              } else {
                pt1 = contour.at(i-1);
              }
              if (pt1.y < min_pt.y) {
                min_pt.x = pt1.x;
                min_pt.y = pt1.y;
              }
              cv::line(debug_image, pt1, pt2, cv::Scalar(0,0,255), 3);
            }
          }

          // color, geo, group
          std::vector<std::string> feature_labels(results.size());
          feature_labels[0] = "color: "; feature_labels[1] = "geo: "; feature_labels[2] = "group: ";
          int txt_offset = -10;
          for (int i=0; i<results.size(); ++i) {
            std::string txt = feature_labels[i] + std::to_string(results[i]);
            cv::putText(debug_image, txt,
                        cv::Point(min_pt.x, min_pt.y + txt_offset),
                        cv::FONT_HERSHEY_SIMPLEX,
                        0.3, cv::Scalar(0,0,0), 1);
            txt_offset -= 10;
          }
        } catch (cv::Exception& e) {
          ROS_WARN("failed create debug image: \n%s", e.what());
        }

        std::stringstream image_save_path;
        image_save_path << save_prefix << "/image.jpg";
        ROS_INFO("save debug image to: %s", image_save_path.str().c_str());
        cv::imwrite(image_save_path.str(), debug_image);
      }

    }

    return true;
  }

} // namespace neatness_estimator

#include <pluginlib/class_list_macros.h>
PLUGINLIB_EXPORT_CLASS(neatness_estimator::EstimationModuleInterface, nodelet::Nodelet)
