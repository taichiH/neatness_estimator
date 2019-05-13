#include <ros/ros.h>
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <sensor_msgs/PointCloud2.h>
#include <jsk_recognition_msgs/ClusterPointIndices.h>
#include <jsk_recognition_msgs/LabelArray.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl_msgs/PointIndices.h>

#include <pcl/point_types.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/filter.h>
#include <pcl/kdtree/kdtree.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/segmentation/extract_clusters.h>

static pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
static pcl::PointCloud<pcl::PointXYZ>::Ptr clustered_cloud(new pcl::PointCloud<pcl::PointXYZ>);
static pcl::PointCloud<pcl::PointXYZ>::Ptr filtered_cloud(new pcl::PointCloud<pcl::PointXYZ>);
static ros::Publisher output_cluster_indices_pub;

float voxel_thresh_ = 0.01;
float cluster_tolerance_ = 0.01;
float min_cluster_ = 10;
float max_cluster_ = 5000;

void callback(const jsk_recognition_msgs::ClusterPointIndices::ConstPtr& cluster_indices,
              const sensor_msgs::PointCloud2::ConstPtr& point_cloud,
              const jsk_recognition_msgs::LabelArray::ConstPtr& labels)
{
  jsk_recognition_msgs::ClusterPointIndices output_cluster_indices;
  cloud->clear();
  pcl::fromROSMsg(*point_cloud, *cloud);

  if(cloud->points.size() > 0){
    for(auto point_indices : cluster_indices->cluster_indices){

      // organized pointcloud
      pcl::PointIndices::Ptr nonnan_indices (new pcl::PointIndices);
      for (auto index : point_indices.indices) {
        pcl::PointXYZ p = cloud->points[index];
        if (!std::isnan(p.x) && !std::isnan(p.y) && !std::isnan(p.z)) {
          nonnan_indices->indices.push_back(index);
        }
      }

      pcl::search::KdTree<pcl::PointXYZ>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZ>);
      std::vector<pcl::PointIndices> output_indices;
      pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
      tree->setInputCloud(cloud);
      ec.setClusterTolerance (cluster_tolerance_);
      ec.setMinClusterSize (min_cluster_);
      ec.setMaxClusterSize (max_cluster_);
      ec.setSearchMethod (tree);
      ec.setIndices(nonnan_indices);
      ec.setInputCloud (cloud);
      ec.extract (output_indices);

      pcl_msgs::PointIndices point_indices_msg;
      int size, index = 0;
      if(output_indices.size() > 0){
        for(int i=0; i < output_indices.size(); i++){
          if(output_indices[i].indices.size() > size){
            size = output_indices[i].indices.size();
            index = i;
          }
        }

        // set extracted indices to ros msg
        point_indices_msg.indices = output_indices.at(index).indices;
      }
      point_indices_msg.header = cluster_indices->header;
      output_cluster_indices.cluster_indices.push_back(point_indices_msg);
    }
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
  nh.getParam("voxel_thresh", voxel_thresh_);
  nh.getParam("cluster_tolerance", cluster_tolerance_);
  nh.getParam("min_cluster", min_cluster_);
  nh.getParam("max_cluster", max_cluster_);

  output_cluster_indices_pub = nh.advertise<jsk_recognition_msgs::ClusterPointIndices>("output_indices",1);

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
