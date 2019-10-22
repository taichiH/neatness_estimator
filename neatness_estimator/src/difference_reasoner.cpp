#include "neatness_estimator/difference_reasoner.h"

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
    pnh_.getParam("instance_boxes_topic", instance_boxes_topic_);
    pnh_.getParam("cluster_boxes_topic", cluster_boxes_topic_);

    server_ = pnh_.advertiseService("read", &DifferenceReasoner::service_callback, this);
    display_feature_client_ = pnh_.serviceClient<neatness_estimator_msgs::GetDisplayFeature>
      ("service_topic");

  }

  bool DifferenceReasoner::get_read_dirs()
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
    current_dir_ = current_ss.str();

    std::stringstream prev_ss;
    prev_ss << prefix_ << "/"
            << std::to_string(static_cast<int>(saved_dirs.at(1))) << "/"
            << std::to_string(static_cast<int>(saved_dirs.at(1))) << ".bag";
    prev_dir_ = prev_ss.str();


    boost::system::error_code error;
    current_log_dir_ =
      prefix_ + "/" + std::to_string(static_cast<int>(saved_dirs.at(0))) + "/logs/";
    const boost::filesystem::path log_dir_path(current_log_dir_.c_str());
    if (!boost::filesystem::exists(log_dir_path)) {
      if (!boost::filesystem::create_directory(current_log_dir_, error) || error) {
        ROS_ERROR("failed create current logs dir : \n%s", current_log_dir_.c_str());
        return false;
      }
    }

    prev_log_dir_ =
      prefix_ + "/" + std::to_string(static_cast<int>(saved_dirs.at(1))) + "/logs/";
    if (!boost::filesystem::exists(boost::filesystem::path(prev_log_dir_.c_str()))) {
      if (!boost::filesystem::create_directory(prev_log_dir_, error) || error) {
        ROS_ERROR("failed create prev logs dir : \n%s", prev_log_dir_.c_str());
        return false;
      }
    }

    current_save_data_dir_ =
      prefix_ + "/" + std::to_string(static_cast<int>(saved_dirs.at(0))) + "/data/";
    const boost::filesystem::path data_dir_path(current_save_data_dir_.c_str());
    if (!boost::filesystem::exists(data_dir_path)) {
      if (!boost::filesystem::create_directory(current_save_data_dir_, error) || error) {
        ROS_ERROR("failed create data dir : \n%s", current_save_data_dir_.c_str());
        return false;
      }
    }

    prev_save_data_dir_ =
      prefix_ + "/" + std::to_string(static_cast<int>(saved_dirs.at(1))) + "/data/";
    if (!boost::filesystem::exists(boost::filesystem::path(prev_save_data_dir_.c_str()))) {
      if (!boost::filesystem::create_directory(prev_save_data_dir_, error) || error) {
        ROS_ERROR("failed create data dir : \n%s", current_save_data_dir_.c_str());
        return false;
      }
    }

    return true;
  }


  bool DifferenceReasoner::read_data()
  {
    current_cluster_.reset(new jsk_recognition_msgs::ClusterPointIndices);
    current_cloud_.reset(new sensor_msgs::PointCloud2);
    current_image_.reset(new sensor_msgs::Image);
    current_instance_boxes_.reset(new jsk_recognition_msgs::BoundingBoxArray);
    current_cluster_boxes_.reset(new jsk_recognition_msgs::BoundingBoxArray);

    prev_cluster_.reset(new jsk_recognition_msgs::ClusterPointIndices);
    prev_cloud_.reset(new sensor_msgs::PointCloud2);
    prev_image_.reset(new sensor_msgs::Image);
    prev_instance_boxes_.reset(new jsk_recognition_msgs::BoundingBoxArray);
    prev_cluster_boxes_.reset(new jsk_recognition_msgs::BoundingBoxArray);


    rosbag::Bag bag;
    try {
      bag.open(current_dir_);
      for (rosbag::MessageInstance const m : rosbag::View(bag)) {
        if (m.getTopic() == cluster_topic_)
          current_cluster_ = m.instantiate<jsk_recognition_msgs::ClusterPointIndices>();
        if (m.getTopic() == cloud_topic_)
          current_cloud_ = m.instantiate<sensor_msgs::PointCloud2>();
        if (m.getTopic() == image_topic_)
          current_image_ = m.instantiate<sensor_msgs::Image>();
        if (m.getTopic() == instance_boxes_topic_)
          current_instance_boxes_ = m.instantiate<jsk_recognition_msgs::BoundingBoxArray>();
        if (m.getTopic() == cluster_boxes_topic_)
          current_cluster_boxes_ = m.instantiate<jsk_recognition_msgs::BoundingBoxArray>();
      }
      bag.close();

      bag.open(prev_dir_);
      for (rosbag::MessageInstance const m : rosbag::View(bag)) {
        if (m.getTopic() == cluster_topic_)
          prev_cluster_ = m.instantiate<jsk_recognition_msgs::ClusterPointIndices>();
        if (m.getTopic() == cloud_topic_)
          prev_cloud_ = m.instantiate<sensor_msgs::PointCloud2>();
        if (m.getTopic() == image_topic_)
          prev_image_ = m.instantiate<sensor_msgs::Image>();
        if (m.getTopic() == instance_boxes_topic_)
          prev_instance_boxes_ = m.instantiate<jsk_recognition_msgs::BoundingBoxArray>();
        if (m.getTopic() == cluster_boxes_topic_)
          prev_cluster_boxes_ = m.instantiate<jsk_recognition_msgs::BoundingBoxArray>();
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

  bool DifferenceReasoner::save_pcd(std::string save_path,
                                    const pcl::PointCloud<pcl::PointXYZRGB>& cloud)
  {
    ROS_INFO("save pcd path: \n%s", save_path.c_str());
    pcl::PCDWriter writer;
    try {
      writer.writeBinary<pcl::PointXYZRGB>(save_path, cloud);
    } catch (...) {
      ROS_ERROR("failed save pcd");
      return false;
    }
    return true;
  }

  bool DifferenceReasoner::save_image(std::string save_path,
                                      const cv::Mat& image,
                                      const cv::Mat& mask_image,
                                      const cv::Mat& debug_image)
  {
    ROS_INFO("save image path: \n%s", save_path.c_str());
    cv::imwrite(save_path + "log_image.jpg", image);
    cv::imwrite(save_path + "log_mask_image.jpg", mask_image);
    cv::imwrite(save_path + "log_debug_image.jpg", debug_image);

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

    return true;
  }

  bool DifferenceReasoner::compute_histograms
  (const pcl::PointCloud<pcl::PointXYZRGB>::Ptr& rgb_cloud,
   jsk_recognition_msgs::ClusterPointIndices::ConstPtr& input_indices,
   jsk_recognition_msgs::ColorHistogramArray& color_histogram_array,
   std::vector<jsk_recognition_msgs::Histogram>& geometry_histogram_array,
   cv::Mat& mask_image,
   cv::Mat& debug_image)
  {

    for (size_t i = 0; i < input_indices->cluster_indices.size(); ++i) {
      index_ = sorted_indices_.at(i);

      cv::Mat tmp_mask = cv::Mat::zeros
        (mask_image.cols, mask_image.rows, CV_8UC1);

      pcl::PointIndices::Ptr nonnan_indices (new pcl::PointIndices);
      for (auto point_index : input_indices->cluster_indices.at(index_).indices) {
        size_t y = int(point_index / current_image_->width);
        size_t x = int(point_index % current_image_->width);
        mask_image.at<unsigned char>(y,x) = 255;
        tmp_mask.at<unsigned char>(y,x) = 255;
        pcl::PointXYZRGB p = rgb_cloud->points.at(point_index);
        if (!std::isnan(p.x) && !std::isnan(p.y) && !std::isnan(p.z)) {
          nonnan_indices->indices.push_back(point_index);
        }
      }

      std::vector<std::vector<cv::Point> > contours;
      std::vector<cv::Vec4i> hierarchy;
      cv::findContours(tmp_mask, contours, hierarchy,
                       CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, cv::Point(0, 0));

      cv::Point base_point;
      for (auto contour : contours) {
        for (size_t i=0; i<contour.size(); ++i) {
          cv::Point pt1;
          auto pt2 = contour.at(i);
          if (i == 0) {
            pt1 = contour.at(contour.size() - 1);
            base_point = contour.at(i);
          } else {
            pt1 = contour.at(i-1);
          }
          cv::line(debug_image, pt1, pt2, cv::Scalar(0,0,255), 3);
        }
      }

      std::string txt1 = "label: " + std::to_string(labels_.at(index_));
      std::string txt2 = "index: " + std::to_string(i);
      cv::putText(debug_image, txt1,
                  cv::Point(base_point.x, base_point.y - 15),
                  cv::FONT_HERSHEY_SIMPLEX,
                  0.3, cv::Scalar(0,0,0), 1);
      cv::putText(debug_image, txt2,
                  cv::Point(base_point.x, base_point.y - 5),
                  cv::FONT_HERSHEY_SIMPLEX,
                  0.3, cv::Scalar(0,0,0), 1);

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

  bool DifferenceReasoner::load_image(const sensor_msgs::Image::ConstPtr& input_msg,
                                      cv::Mat& input_image)
  {
    try {
      cv_bridge::CvImagePtr cv_image = cv_bridge::toCvCopy
        (input_msg, "bgr8");
      input_image = cv_image->image;
    } catch (cv_bridge::Exception& e) {
      ROS_ERROR("Failed to convert sensor_msgs::Image to cv::Mat \n%s", e.what());
      return false;
    }

    return true;
  }

  bool DifferenceReasoner::save_color_histogram
  (std::string save_dir,
   const jsk_recognition_msgs::ColorHistogramArray& color_histogram_array)
  {
    std::ofstream f;
    try {
      f.open(save_dir + "color_histograms.csv");
      for (size_t i=0; i<color_histogram_array.histograms.size(); ++i) {
        size_t index = sorted_indices_.at(i);
        f << std::to_string(labels_.at(index)) + ", ";
        for (auto v : color_histogram_array.histograms.at(index).histogram) {
          f << std::to_string(v) + ",";
        }
        f << "\n";
      }
      f.close();
    } catch (...){
      ROS_ERROR("failed save histogram");
      return false;
    }
    return true;
  }

  bool DifferenceReasoner::save_geometry_histogram
  (std::string save_dir,
   const std::vector<jsk_recognition_msgs::Histogram>& geometry_histogram_array)
  {
    try {
      std::ofstream f;
      f.open(save_dir + "geometry_histograms.csv");
      for (size_t i=0; i<geometry_histogram_array.size(); ++i) {
        size_t index = sorted_indices_.at(i);
        f << std::to_string(labels_.at(index)) + ", ";
        for (auto v : geometry_histogram_array.at(index).histogram) {
          f << std::to_string(v) + ",";
        }
        f << "\n";
      }
      f.close();
    } catch (...){
      ROS_ERROR("failed save histogram");
      return false;
    }

    return true;
  }

  bool DifferenceReasoner::create_sorted_indices
  (const std::vector<jsk_recognition_msgs::BoundingBox> input_boxes,
   std::vector<size_t>& sorted_indices)
  {
    sorted_indices.clear();
    sorted_indices.resize(input_boxes.size());
    std::iota(sorted_indices.begin(), sorted_indices.end(), 0);

    std::sort(sorted_indices.begin(), sorted_indices.end(),
              [&input_boxes](size_t l, size_t r)
              {return input_boxes.at(l).pose.position.y >
                  input_boxes.at(r).pose.position.y;});

    std::cerr << "sorted indices: " << std::endl;
    for(size_t i=0; i<sorted_indices.size(); ++i) {
      auto v = sorted_indices.at(i);
      std::cerr << "{";
      std::cerr << v << ", ";
      std::cerr << input_boxes.at(v).pose.position.y << ", ";
      std::cerr << "label: " << input_boxes.at(v).label << ", ";
      std::cerr << "}";
    }
    std::cerr << std::endl;

    return true;
  }

  bool DifferenceReasoner::run_prev()
  {
    cv::Mat image;
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr rgb_cloud
      (new pcl::PointCloud<pcl::PointXYZRGB>);
    jsk_recognition_msgs::ColorHistogramArray color_histogram_array;
    std::vector<jsk_recognition_msgs::Histogram> geometry_histogram_array;

    load_image(prev_image_, image);

    create_sorted_indices(prev_instance_boxes_->boxes,
                          sorted_indices_);

    labels_.resize(prev_instance_boxes_->boxes.size());
    for (size_t i = 0; i < prev_instance_boxes_->boxes.size(); ++i) {
      labels_.at(i) = prev_instance_boxes_->boxes.at(i).label;
    }

    pcl::fromROSMsg(*prev_cloud_, *rgb_cloud);
    save_pcd(prev_log_dir_ + "log_pcd.pcd", *rgb_cloud);


    cv::Mat prev_mask_image = cv::Mat::zeros
      (prev_image_->width, prev_image_->height, CV_8UC1);

    cv::Mat debug_image = image.clone();
    compute_histograms(rgb_cloud,
                       prev_cluster_,
                       color_histogram_array,
                       geometry_histogram_array,
                       prev_mask_image,
                       debug_image);

    save_image(prev_log_dir_, image, prev_mask_image, debug_image);
    save_color_histogram(prev_save_data_dir_, color_histogram_array);
    save_geometry_histogram(prev_save_data_dir_, geometry_histogram_array);

    neatness_estimator_msgs::GetDisplayFeature client_msg;
    client_msg.request.save_dir = prev_save_data_dir_;
    client_msg.request.instance_boxes = *prev_instance_boxes_;
    client_msg.request.cluster_boxes = *prev_cluster_boxes_;
    display_feature_client_.call(client_msg);

    return true;
  }

  bool DifferenceReasoner::run_current()
  {
    cv::Mat image;
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr rgb_cloud
      (new pcl::PointCloud<pcl::PointXYZRGB>);
    jsk_recognition_msgs::ColorHistogramArray color_histogram_array;
    std::vector<jsk_recognition_msgs::Histogram> geometry_histogram_array;

    load_image(current_image_, image);


    create_sorted_indices(current_instance_boxes_->boxes,
                          sorted_indices_);

    labels_.resize(current_instance_boxes_->boxes.size());
    for (size_t i = 0; i < current_instance_boxes_->boxes.size(); ++i) {
      labels_.at(i) = current_instance_boxes_->boxes.at(i).label;
    }

    pcl::fromROSMsg(*current_cloud_, *rgb_cloud);
    save_pcd(current_log_dir_ + "log_pcd.pcd", *rgb_cloud);

    cv::Mat current_mask_image = cv::Mat::zeros
      (current_image_->width, current_image_->height, CV_8UC1);

    cv::Mat debug_image = image.clone();
    compute_histograms(rgb_cloud,
                       current_cluster_,
                       color_histogram_array,
                       geometry_histogram_array,
                       current_mask_image,
                       debug_image);

    save_image(current_log_dir_, image, current_mask_image, debug_image);
    save_color_histogram(current_save_data_dir_, color_histogram_array);
    save_geometry_histogram(current_save_data_dir_, geometry_histogram_array);

    neatness_estimator_msgs::GetDisplayFeature client_msg;
    client_msg.request.save_dir = current_save_data_dir_;
    client_msg.request.instance_boxes = *current_instance_boxes_;
    client_msg.request.cluster_boxes = *current_cluster_boxes_;
    display_feature_client_.call(client_msg);

    return true;
  }

  bool DifferenceReasoner::service_callback(std_srvs::SetBool::Request& req,
                                            std_srvs::SetBool::Response& res)
  {
    boost::mutex::scoped_lock lock(mutex_);
    if ( !get_read_dirs() ) {
      res.success = false;
      return false;
    }

    if ( !read_data() ) {
      res.success = false;
      return false;
    }

    run_current();
    run_prev();

    res.success = true;
    return true;
  }

} // namespace neatness_estimator

#include <pluginlib/class_list_macros.h>
PLUGINLIB_EXPORT_CLASS(neatness_estimator::DifferenceReasoner, nodelet::Nodelet)
