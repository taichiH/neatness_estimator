#include "neatness_estimator/estimation_module_interface.h"

namespace neatness_estimator
{

  void EstimationModuleInterface::onInit()
  {
    nh_ = getNodeHandle();
    pnh_ = getPrivateNodeHandle();

    pnh_.getParam("approximate_sync", approximate_sync_);

    int target_idx;
    int ref_idx;
    pnh_.getParam("target_idx", target_idx);
    pnh_.getParam("ref_idx", ref_idx);
    pair_ = std::make_pair(target_idx, ref_idx);

    service_server_ =
      pnh_.advertiseService("call", &EstimationModuleInterface::service_callback, this);

    sub_point_cloud_.subscribe(pnh_, "input_cloud", 1);
    sub_image_.subscribe(pnh_, "input_image", 1);
    sub_cluster_.subscribe(pnh_, "input_cluster", 1);
    register_callback();
  }

  void EstimationModuleInterface::register_callback()
  {
    if (approximate_sync_) {
      async_ = boost::make_shared<message_filters::Synchronizer<ApproximateSync> >(1000);
      async_->connectInput(sub_point_cloud_, sub_image_, sub_cluster_);
      async_->registerCallback
        (boost::bind(&EstimationModuleInterface::callback,this, _1, _2, _3));
    } else {
      sync_  = boost::make_shared<message_filters::Synchronizer<Sync> >(1000);
      sync_->connectInput(sub_point_cloud_, sub_image_, sub_cluster_);
      sync_->registerCallback
        (boost::bind(&EstimationModuleInterface::callback,this, _1, _2, _3));
    }
  }

  void EstimationModuleInterface::callback
  (const sensor_msgs::PointCloud2::ConstPtr& cloud_msg,
   const sensor_msgs::Image::ConstPtr& image_msg,
   const jsk_recognition_msgs::ClusterPointIndices::ConstPtr& cluster_msg)
  {
    boost::mutex::scoped_lock lock(mutex_);

    cloud_msg_ = cloud_msg;
    image_msg_ = image_msg;
    cluster_msg_ = cluster_msg;

    run();
  }

  bool EstimationModuleInterface::service_callback
  (neatness_estimator_msgs::GetDifference::Request& req,
   neatness_estimator_msgs::GetDifference::Response& res)
  {
    boost::mutex::scoped_lock lock(mutex_);

    pair_ = std::make_pair(req.target_idx, req.ref_idx);

    run();

    res.success = true;
    return true;
  }

  bool EstimationModuleInterface::run()
  {

    int idx = pair_.first;
    ObjectFeature feature;
    compute_object_feature(idx, feature);

    // compare histogram


    return true;
  }

  bool EstimationModuleInterface::compute_object_feature
  (int idx,
   ObjectFeature& feature)
  {


    pcl::PointCloud<pcl::PointXYZRGB>::Ptr rgb_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::fromROSMsg(*cloud_msg_, *rgb_cloud);
    cv::Mat image;
    load_image(image_msg_, image);

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr clustered_cloud(new pcl::PointCloud<pcl::PointXYZRGB>());
    cv::Mat mask_image;
    get_clustered_cloud(rgb_cloud, cluster_msg_, idx, mask_image, clustered_cloud);

    neatness_estimator_msgs::Histogram color_histogram;
    if ( !compute_color_histogram(image, mask_image, color_histogram) ) {
      return false;
    }

    neatness_estimator_msgs::Histogram geometry_histogram;
    if ( !compute_geometry_histogram(clustered_cloud, geometry_histogram) ) {
      return false;
    }

    feature.color = color_histogram;
    feature.geometry = geometry_histogram;
    feature.size = clustered_cloud->points.size();

    return true;
  }

  bool EstimationModuleInterface::compute_color_histogram
  (const cv::Mat& image,
   const cv::Mat& mask,
   neatness_estimator_msgs::Histogram& color_histogram)
  {
    cv::cvtColor(image, image, CV_BGR2GRAY);

    const int ch_width = 260;
    const int sch = image.channels();

    std::vector<cv::MatND> hist(3);
    const int hist_size = 256;
    const int hdims[] = {hist_size};
    const float hranges[] = {0,256};
    const float* ranges[] = {hranges};

    for(int i=0; i<sch; ++i) {
      cv::calcHist(&image, 1, &i, mask, hist[i], 1, hdims, ranges, true, false);
    }

    for (int i=0; i<hist_size; ++i) {
      float val = hist[0].at<float>(i);
      color_histogram.histogram.push_back(val);
    }
    return true;
  }

  bool EstimationModuleInterface::compute_geometry_histogram
  (const pcl::PointCloud<pcl::PointXYZRGB>::Ptr& rgb_cloud,
   neatness_estimator_msgs::Histogram& geometry_histogram)
  {
    pcl::NormalEstimationOMP<pcl::PointXYZRGB, pcl::Normal> normal_estimation;
    pcl::PointCloud<pcl::Normal>::Ptr cloud_normals(new pcl::PointCloud<pcl::Normal>());
    pcl::search::KdTree<pcl::PointXYZRGB>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZRGB>());
    normal_estimation.setSearchMethod (tree);
    normal_estimation.setRadiusSearch(0.01);
    normal_estimation.setInputCloud(rgb_cloud);
    normal_estimation.compute(*cloud_normals);

    cv::Mat histogram;
    pcl::CVFHEstimation<pcl::PointXYZRGB, pcl::Normal, pcl::VFHSignature308> cvfh;
    cvfh.setInputCloud(rgb_cloud);
    cvfh.setInputNormals(cloud_normals);
    cvfh.setSearchMethod(tree);
    cvfh.setEPSAngleThreshold(5.0f / 180.0f * M_PI);
    cvfh.setCurvatureThreshold(0.025f);
    cvfh.setClusterTolerance(0.015f);
    cvfh.setNormalizeBins(false);

    if (cloud_normals->size() <= 0) {
      ROS_WARN("cloud_normals size: %d", cloud_normals->size());
      return false;
    }

    pcl::PointCloud<pcl::VFHSignature308>::Ptr cvfhs
      (new pcl::PointCloud<pcl::VFHSignature308>());
    cvfh.compute(*cvfhs);
    int feature_size = sizeof(pcl::VFHSignature308) / sizeof(cvfhs->points[0].histogram[0]);
    if (feature_size <= 0) {
      ROS_WARN("feature_size: %d", feature_size);
      return false;
    }
    histogram = cv::Mat(sizeof(char), feature_size, CV_32F);
    for (int i = 0; i < histogram.cols; i++) {
      histogram.at<float>(0, i) = cvfhs->points[0].histogram[i];
    }
    cv::normalize(histogram, histogram, 0, 1, cv::NORM_MINMAX, -1, cv::Mat());

    for (int i = 0; i < histogram.cols; i++) {
      geometry_histogram.histogram.push_back(histogram.at<float>(0, i));
    }

    return true;
  }

  bool EstimationModuleInterface::get_clustered_cloud
  (const pcl::PointCloud<pcl::PointXYZRGB>::Ptr& rgb_cloud,
   jsk_recognition_msgs::ClusterPointIndices::ConstPtr& cluster_indices,
   int index,
   cv::Mat& mask_image,
   pcl::PointCloud<pcl::PointXYZRGB>::Ptr& clustered_cloud)
  {
    pcl::PointIndices::Ptr nonnan_indices (new pcl::PointIndices);
    for (auto point_index : cluster_indices->cluster_indices.at(index).indices) {
      size_t y = int(point_index / image_msg_->width);
      size_t x = int(point_index % image_msg_->width);
      mask_image.at<unsigned char>(y,x) = 255;
      pcl::PointXYZRGB p = rgb_cloud->points.at(point_index);
      if (!std::isnan(p.x) && !std::isnan(p.y) && !std::isnan(p.z)) {
        nonnan_indices->indices.push_back(point_index);
      }
    }

    pcl::ExtractIndices<pcl::PointXYZRGB> extract;
    extract.setInputCloud(rgb_cloud);
    extract.setIndices(nonnan_indices);
    extract.filter(*clustered_cloud);

    return true;
  }

  bool EstimationModuleInterface::load_image
  (const sensor_msgs::Image::ConstPtr& input_msg,
   cv::Mat& input_image)
  {
    try {
      ROS_INFO("image encoding: %s", input_msg->encoding.c_str());
      cv_bridge::CvImagePtr cv_image = cv_bridge::toCvCopy
        (input_msg, input_msg->encoding);
      input_image = cv_image->image;
    } catch (cv_bridge::Exception& e) {
      ROS_ERROR("Failed to convert sensor_msgs::Image to cv::Mat \n%s", e.what());
      return false;
    }

    return true;
  }

} // namespace neatness_estimator

#include <pluginlib/class_list_macros.h>
PLUGINLIB_EXPORT_CLASS(neatness_estimator::EstimationModuleInterface, nodelet::Nodelet)
