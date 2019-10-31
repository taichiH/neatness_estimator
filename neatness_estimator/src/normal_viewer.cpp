#include <iostream>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/visualization/pcl_visualizer.h>

pcl::visualization::PCLVisualizer::Ptr normalsVis
(pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr cloud,
 pcl::PointCloud<pcl::Normal>::ConstPtr normals)
{
  pcl::visualization::PCLVisualizer::Ptr viewer (new pcl::visualization::PCLVisualizer ("3D Viewer"));
  viewer->setBackgroundColor (0, 0, 0);
  pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb(cloud);
  viewer->addPointCloud<pcl::PointXYZRGB>(cloud, rgb, "sample cloud");
  viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "sample cloud");
  viewer->addPointCloudNormals<pcl::PointXYZRGB, pcl::Normal>(cloud, normals, 10, 0.02, "normals");
  viewer->addCoordinateSystem(0.1);
  viewer->setCameraPosition(0,0,0,0,0,0,0,0,0);
  viewer->initCameraParameters();
  return (viewer);
}

int main(int argc, char** argv) {
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZRGB>);
  if (pcl::io::loadPCDFile<pcl::PointXYZRGB> (argv[1], *cloud) == -1) //* load the file
    {
      PCL_ERROR ("Couldn't read file %s \n", argv[1]);
      return (-1);
    }

    pcl::NormalEstimationOMP<pcl::PointXYZRGB, pcl::Normal> normal_estimation;
    pcl::PointCloud<pcl::Normal>::Ptr cloud_normals(new pcl::PointCloud<pcl::Normal>());
    pcl::search::KdTree<pcl::PointXYZRGB>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZRGB>());
    normal_estimation.setSearchMethod (tree);
    normal_estimation.setRadiusSearch(0.01);
    normal_estimation.setInputCloud(cloud);
    normal_estimation.compute(*cloud_normals);
    pcl::visualization::PCLVisualizer::Ptr viewer;
    viewer = normalsVis(cloud, cloud_normals);
    viewer->spin();

  return 0;
}
