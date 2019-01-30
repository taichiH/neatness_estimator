#include <ros/ros.h>
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <sensor_msgs/PointCloud2.h>
#include <jsk_recognition_msgs/ClusterPointIndices.h>
#include <jsk_recognition_msgs/BoundingBoxArray.h>
#include <jsk_recognition_msgs/LabelArray.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl_msgs/PointIndices.h>
#include <tf/transform_listener.h>
#include <tf/transform_broadcaster.h>

#include <pcl/point_types.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/filter.h>
#include <pcl/kdtree/kdtree.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl_ros/transforms.h>

static pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
static pcl::PointCloud<pcl::PointXYZ>::Ptr clustered_cloud(new pcl::PointCloud<pcl::PointXYZ>);
static pcl::PointCloud<pcl::PointXYZ>::Ptr filtered_cloud(new pcl::PointCloud<pcl::PointXYZ>);
static ros::Publisher output_cluster_indices_pub;
static ros::Publisher output_box_pub;

tf::TransformListener *listener_;

struct Points {
  Points() {
    x = 0.0;
    y = 0.0;
    z = 0.0;
  };
  Points(float _x, float _y, float _z) {
    x = _x;
    y = _y;
    z = _z;
  };
  float x, y, z;
};

void callback(const jsk_recognition_msgs::ClusterPointIndices::ConstPtr& cluster_indices,
              const sensor_msgs::PointCloud2::ConstPtr& point_cloud,
              const jsk_recognition_msgs::LabelArray::ConstPtr& labels)
{
  cloud->clear();
  pcl::fromROSMsg(*point_cloud, *cloud);
  // std::string target_frame_ = "map";
  std::string target_frame_ = cluster_indices->header.frame_id;

  jsk_recognition_msgs::ClusterPointIndices output_cluster_indices;
  jsk_recognition_msgs::BoundingBoxArray output_box_msg;

  if(cloud->points.size() > 0){
    int label_index = 0;
    for(auto point_indices : cluster_indices->cluster_indices){
      pcl::PointIndices::Ptr inliers (new pcl::PointIndices);
      inliers->indices = point_indices.indices;

      pcl::ExtractIndices<pcl::PointXYZ> extract;
      extract.setInputCloud (cloud);
      extract.setIndices (inliers);
      extract.setNegative (false);
      extract.filter (*clustered_cloud);

      pcl::VoxelGrid<pcl::PointXYZ> vg;
      pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered (new pcl::PointCloud<pcl::PointXYZ>);
      vg.setInputCloud (clustered_cloud);
      vg.setLeafSize (0.01f, 0.01f, 0.01f);
      vg.filter (*cloud_filtered);
      clustered_cloud->clear();

      std::vector<int> tmp_indices;
      pcl::removeNaNFromPointCloud(*cloud_filtered, *filtered_cloud, tmp_indices);
      cloud_filtered->clear();

      pcl::search::KdTree<pcl::PointXYZ>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZ>);
      tree->setInputCloud(filtered_cloud);
      std::vector<pcl::PointIndices> output_indices;
      pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
      ec.setClusterTolerance (0.01);
      ec.setMinClusterSize (10);
      ec.setMaxClusterSize (5000);
      ec.setSearchMethod (tree);
      // ec.setIndices(inliers);
      ec.setInputCloud (filtered_cloud);
      ec.extract (output_indices);

      pcl::PointCloud<pcl::PointXYZ>::Ptr transformed_cloud (new pcl::PointCloud<pcl::PointXYZ>);

      tf::StampedTransform transform;
      try {
        listener_->lookupTransform (target_frame_,
                                    cluster_indices->header.frame_id,
                                    ros::Time(0),
                                    transform);
      } catch (tf::TransformException ex){
        ROS_ERROR("%s",ex.what());
        return;
      }

      pcl_ros::transformPointCloud(*filtered_cloud, *transformed_cloud, transform);
      filtered_cloud->clear();
      jsk_recognition_msgs::BoundingBox box;
      pcl_msgs::PointIndices point_indices_msg;

      int size, index = 0;
      if(output_indices.size() > 0){
        for(int i=0; i < output_indices.size(); i++){
          if(output_indices[i].indices.size() > size){
            size = output_indices[i].indices.size();
            index = i;
          }
        }

        Points max_p(0.0, 0.0, 0.0);
        Points min_p(std::pow(2, 24), std::pow(2, 24), std::pow(2, 24));

        for(int i=0; i < output_indices.at(index).indices.size(); i++){
          point_indices_msg.indices.push_back(output_indices.at(index).indices[i]);
          int indices = output_indices.at(index).indices[i];
          float x = transformed_cloud->points.at(indices).x;
          float y = transformed_cloud->points.at(indices).y;
          float z = transformed_cloud->points.at(indices).z;
          max_p.x = x > max_p.x ? x : max_p.x;
          max_p.y = y > max_p.y ? y : max_p.y;
          max_p.z = z > max_p.z ? z : max_p.z;
          min_p.x = x < min_p.x ? x : min_p.x;
          min_p.y = y < min_p.y ? y : min_p.y;
          min_p.z = z < min_p.z ? z : min_p.z;
        }

        Points new_max_p(0, 0, 0);
        Points new_min_p(std::pow(2, 24), std::pow(2, 24), std::pow(2, 24));
        std::vector<float> vec = {max_p.x, max_p.y, max_p.z, min_p.x, min_p.y, min_p.z};

        box.dimensions.x = std::fabs(max_p.x - min_p.x);
        box.dimensions.y = std::fabs(max_p.y - min_p.y);
        box.dimensions.z = std::fabs(max_p.z - min_p.z);
        box.pose.position.x = std::min(min_p.x, max_p.x) + box.dimensions.x * 0.5;
        box.pose.position.y = std::min(min_p.y, max_p.y) + box.dimensions.y * 0.5;
        box.pose.position.z = std::min(min_p.z, max_p.z) + box.dimensions.z * 0.5;
        box.pose.orientation.x = 1.0;
        box.pose.orientation.y = 0.0;
        box.pose.orientation.z = 0.0;
        box.pose.orientation.w = 0.0;
        box.label = labels->labels[label_index].id;
        box.header = cluster_indices->header;
        box.header.frame_id = target_frame_;
      } else {
        box.dimensions.x = 0;
        box.dimensions.y = 0;
        box.dimensions.z = 0;
        box.pose.position.x = 0;
        box.pose.position.y = 0;
        box.pose.position.z = 0;
        box.pose.orientation.x = 1.0;
        box.pose.orientation.y = 0.0;
        box.pose.orientation.z = 0.0;
        box.pose.orientation.w = 0.0;
        box.label = labels->labels[label_index].id;
        box.header = cluster_indices->header;
        box.header.frame_id = target_frame_;
        output_box_msg.boxes.push_back(box);

        ROS_ERROR("output_indices.size() == 0");
      }
      output_box_msg.boxes.push_back(box);

      point_indices_msg.header = cluster_indices->header;
      output_cluster_indices.cluster_indices.push_back(point_indices_msg);
      label_index++;
    }
    output_box_msg.header = cluster_indices->header;
    output_box_msg.header.frame_id = target_frame_;
    output_box_pub.publish(output_box_msg);
    output_cluster_indices.header = cluster_indices->header;
    output_cluster_indices_pub.publish(output_cluster_indices);
  }
}

typedef message_filters::sync_policies::ApproximateTime<
  jsk_recognition_msgs::ClusterPointIndices,
  sensor_msgs::PointCloud2,
  jsk_recognition_msgs::LabelArray
  > SyncPolicy;

int main(int argc, char** argv)
{
  ros::init(argc, argv, "multi_euclidean_clustering");

  ros::NodeHandle nh("~");

  tf::TransformListener lis;
  listener_ = &lis;

  output_cluster_indices_pub = nh.advertise<jsk_recognition_msgs::ClusterPointIndices>("output_indices",1);
  output_box_pub = nh.advertise<jsk_recognition_msgs::BoundingBoxArray>("output_box",1);

  message_filters::Subscriber<jsk_recognition_msgs::ClusterPointIndices> sub_cluster_indices;
  message_filters::Subscriber<sensor_msgs::PointCloud2> sub_point_cloud;
  message_filters::Subscriber<jsk_recognition_msgs::LabelArray> sub_labels;
  sub_cluster_indices.subscribe(nh, "input_cluster_indices", 1);
  sub_point_cloud.subscribe(nh, "input_point_cloud", 1);
  sub_labels.subscribe(nh, "input_instance_labels", 1);
  boost::shared_ptr<message_filters::Synchronizer<SyncPolicy> >sync_
    = boost::make_shared<message_filters::Synchronizer<SyncPolicy> >(1000);
  sync_->connectInput(sub_cluster_indices, sub_point_cloud, sub_labels);
  sync_->registerCallback(callback);

  ros::spin();
  return 0;
}
