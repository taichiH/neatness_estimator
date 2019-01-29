#include <ros/ros.h>
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <sensor_msgs/PointCloud2.h>
#include <jsk_recognition_msgs/ClusterPointIndices.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl_msgs/PointIndices.h>

#include <pcl/ModelCoefficients.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/features/normal_3d.h>
#include <pcl/kdtree/kdtree.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/segmentation/extract_clusters.h>

static pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
static ros::Publisher output_cluster_indices_pub;

void callback(const jsk_recognition_msgs::ClusterPointIndices::ConstPtr& cluster_indices,
              const sensor_msgs::PointCloud2::ConstPtr& point_cloud)
{
  cloud->clear();
  pcl::fromROSMsg(*point_cloud, *cloud);

  jsk_recognition_msgs::ClusterPointIndices output_cluster_indices;

  if(cloud->points.size() > 0){
    for(auto point_indices : cluster_indices->cluster_indices){
      pcl::PointCloud<pcl::PointXYZ>::Ptr clustered_cloud(new pcl::PointCloud<pcl::PointXYZ>);
      for(int i=0; i < point_indices.indices.size(); i++){
        clustered_cloud->points.push_back(cloud->points.at(point_indices.indices[i]));
      }
      pcl::search::KdTree<pcl::PointXYZ>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZ>);
      tree->setInputCloud(clustered_cloud);
      std::vector<pcl::PointIndices> output_indices;
      pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
      ec.setClusterTolerance (0.05); // 2cm
      ec.setMinClusterSize (10);
      ec.setMaxClusterSize (5000);
      ec.setSearchMethod (tree);
      ec.setInputCloud (clustered_cloud);
      ec.extract (output_indices);
      clustered_cloud->clear();

      int size, index = 0;
      for(int i=0; i < output_indices.size(); i++){
        if(output_indices[i].indices.size() > size){
          size = output_indices[i].indices.size();
          index = i;
        }
      }

      pcl_msgs::PointIndices point_indices_msg;
      for(int i=0; i < output_indices[index].indices.size(); i++){
        point_indices_msg.indices.push_back(output_indices[index].indices[i]);
      }
      point_indices_msg.header = cluster_indices->header;
      output_cluster_indices.cluster_indices.push_back(point_indices);
    }
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
  message_filters::Subscriber<jsk_recognition_msgs::ClusterPointIndices> sub_cluster_indices;
  message_filters::Subscriber<sensor_msgs::PointCloud2> sub_point_cloud;
  sub_cluster_indices.subscribe(nh, "~input_cluster_indices", 1);
  sub_point_cloud.subscribe(nh, "~input_point_cloud", 1);
  boost::shared_ptr<message_filters::Synchronizer<SyncPolicy> >sync_
    = boost::make_shared<message_filters::Synchronizer<SyncPolicy> >(1000);
  sync_->connectInput(sub_cluster_indices, sub_point_cloud);
  sync_->registerCallback(callback);

  ros::spin();
  return 0;
}
