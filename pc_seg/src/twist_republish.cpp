#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <geometry_msgs/Twist.h>
#include <geometry_msgs/TwistStamped.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/common/common.h>
#include <pcl_ros/point_cloud.h>
// #include <tf/transform_listener.h>

typedef pcl::PointXYZI PointType;

void cloudCallback(const sensor_msgs::PointCloud2::ConstPtr cloud_msg);
void twistCallback(const geometry_msgs::Twist::ConstPtr twist_msg);


ros::Publisher twist_pub;
ros::Publisher pointcloud_pub;
int main(int argc, char** argv) {
    ros::init(argc, argv, "twist_republish");
    ros::NodeHandle nh;
    twist_pub = nh.advertise<geometry_msgs::TwistStamped>("/cmd_vel_stamped", 5);
    // pointcloud_pub = nh.advertise<sensor_msgs::PointCloud2>("/velodyne_points_stamped", 5);
    ros::Subscriber twist_sub = nh.subscribe<geometry_msgs::Twist>("/husky_velocity_controller/cmd_vel", 5, twistCallback);
    // ros::Subscriber pointcloud_sub = nh.subscribe<sensor_msgs::PointCloud2>("/velodyne_points", 5, cloudCallback);
    ros::spin();
    return 0;
}

void twistCallback(const geometry_msgs::Twist::ConstPtr twist_msg) {
    geometry_msgs::TwistStamped twist_stamped;
    twist_stamped.header.frame_id = "/base_link";
    twist_stamped.header.stamp = ros::Time::now();
    twist_stamped.twist.angular = twist_msg->angular;
    twist_stamped.twist.linear = twist_msg->linear; 
    twist_pub.publish(twist_stamped);
}
/*
void cloudCallback(const sensor_msgs::PointCloud2::ConstPtr cloud_msg) {
    pcl::PointCloud<PointType>::Ptr cloud_in(new pcl::PointCloud<PointType>);
    pcl::fromROSMsg(*cloud_msg, *cloud_in);
    // maybe adding some preprocess here
    cloud_in->header.frame_id = "velodyne";
    pcl_conversions::toPCL(ros::Time::now(), cloud_in->header.stamp);
    pointcloud_pub.publish(cloud_in);
}
*/