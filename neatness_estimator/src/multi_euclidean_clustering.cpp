#include <ros/ros.h>
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <sensor_msgs/PointCloud2.h>
#include <jsk_recognition_msgs/ClusterPointIndices.h>
#include <jsk_recognition_msgs/BoundingBoxArray.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl_msgs/PointIndices.h>

#include <pcl/ModelCoefficients.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/filter.h>
#include <pcl/features/normal_3d.h>
#include <pcl/kdtree/kdtree.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/segmentation/extract_clusters.h>

static pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
static pcl::PointCloud<pcl::PointXYZ>::Ptr clustered_cloud(new pcl::PointCloud<pcl::PointXYZ>);
static pcl::PointCloud<pcl::PointXYZ>::Ptr filtered_cloud(new pcl::PointCloud<pcl::PointXYZ>);
static ros::Publisher output_cluster_indices_pub;
static ros::Publisher output_box_pub;

void callback(const jsk_recognition_msgs::ClusterPointIndices::ConstPtr& cluster_indices,
              const sensor_msgs::PointCloud2::ConstPtr& point_cloud)
{
  cloud->clear();
  pcl::fromROSMsg(*point_cloud, *cloud);

  jsk_recognition_msgs::ClusterPointIndices output_cluster_indices;
  jsk_recognition_msgs::BoundingBoxArray output_box_msg;
  if(cloud->points.size() > 0){
    for(auto point_indices : cluster_indices->cluster_indices){
      // pcl::PointCloud<pcl::PointXYZ>::Ptr input_cloud(new pcl::PointCloud<pcl::PointXYZ>);

      pcl::PointIndices::Ptr inliers (new pcl::PointIndices);
      inliers->indices = point_indices.indices;
      
      // input_cloud->width = cloud->width;
      // input_cloud->height = cloud->height;
      // input_cloud->is_dense = cloud->is_dense;
      // input_cloud->points.resize(cloud->width * cloud->height);

      // for(int i=0; i < point_indices.indices.size(); i++){
      //   input_cloud->points.at(point_indices.indices[i])
      //     = cloud->points.at(point_indices.indices[i]);
      // }

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
      ec.setClusterTolerance (0.05); // 2cm
      ec.setMinClusterSize (10);
      ec.setMaxClusterSize (5000);
      ec.setSearchMethod (tree);
      // ec.setIndices(inliers);
      ec.setInputCloud (filtered_cloud);
      ec.extract (output_indices);

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

        float max_x = 0.0;
        float max_y = 0.0;
        float max_z = 0.0;
        float min_x = std::pow(2, 24);;
        float min_y = std::pow(2, 24);;
        float min_z = std::pow(2, 24);

        for(int i=0; i < output_indices.at(index).indices.size(); i++){
          point_indices_msg.indices.push_back(output_indices.at(index).indices[i]);
          int indices = output_indices.at(index).indices[i];
          float x = filtered_cloud->points.at(indices).x;
          float y = filtered_cloud->points.at(indices).y;
          float z = filtered_cloud->points.at(indices).z;
          max_x = x > max_x ? x : max_x;
          max_y = y > max_y ? y : max_y;
          max_z = z > max_z ? z : max_z;
          min_x = x < min_x ? x : min_x;
          min_y = y < min_y ? y : min_y;
          min_z = z < min_z ? z : min_z;
        }
        filtered_cloud->clear();
        
        box.dimensions.x = max_x - min_x;
        box.dimensions.y = max_y - min_y;
        box.dimensions.z = max_z - min_z;
        box.pose.position.x = min_x + (max_x - min_x) * 0.5;
        box.pose.position.y = min_y + (max_y - min_y) * 0.5;
        box.pose.position.z = min_z + (max_z - min_z) * 0.5;
        box.pose.orientation.x = 1.0;
        box.pose.orientation.y = 0.0;
        box.pose.orientation.z = 0.0;
        box.pose.orientation.w = 0.0;
        box.header = cluster_indices->header;
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
        output_box_msg.boxes.push_back(box);

        ROS_ERROR("output_indices.size() == 0");
      }
      output_box_msg.boxes.push_back(box);

      point_indices_msg.header = cluster_indices->header;
      output_cluster_indices.cluster_indices.push_back(point_indices_msg);

    }
    output_box_msg.header = cluster_indices->header;
    output_box_pub.publish(output_box_msg);
    output_cluster_indices.header = cluster_indices->header;
    output_cluster_indices_pub.publish(output_cluster_indices);
  }
}

typedef message_filters::sync_policies::ApproximateTime<
  jsk_recognition_msgs::ClusterPointIndices,
  sensor_msgs::PointCloud2
  > SyncPolicy;

int main(int argc, char** argv)
{
  ros::init(argc, argv, "multi_euclidean_clustering");

  ros::NodeHandle nh("~");
  output_cluster_indices_pub = nh.advertise<jsk_recognition_msgs::ClusterPointIndices>("output_indices",1);
  output_box_pub = nh.advertise<jsk_recognition_msgs::BoundingBoxArray>("output_box",1);

  message_filters::Subscriber<jsk_recognition_msgs::ClusterPointIndices> sub_cluster_indices;
  message_filters::Subscriber<sensor_msgs::PointCloud2> sub_point_cloud;
  sub_cluster_indices.subscribe(nh, "input_cluster_indices", 1);
  sub_point_cloud.subscribe(nh, "input_point_cloud", 1);
  boost::shared_ptr<message_filters::Synchronizer<SyncPolicy> >sync_
    = boost::make_shared<message_filters::Synchronizer<SyncPolicy> >(1000);
  sync_->connectInput(sub_cluster_indices, sub_point_cloud);
  sync_->registerCallback(callback);

  ros::spin();
  return 0;
}
