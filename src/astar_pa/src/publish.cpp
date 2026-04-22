//
// Created by sjtu on 2022/3/29.
//
#include <ros/ros.h>
#include <iostream>
#include <geometry_msgs/PoseWithCovarianceStamped.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/Quaternion.h>
#include <tf/tf.h>

using namespace std;

int main(int argc, char **argv) {
    ros::init(argc, argv, "fixed_source_target_publisher");
    ros::NodeHandle nh;
    ros::Publisher pub_source = nh.advertise<geometry_msgs::PoseWithCovarianceStamped>("initialpose", 10);
    ros::Publisher pub_target = nh.advertise<geometry_msgs::PoseStamped>("move_base_simple/goal", 10);

    double s_x, s_y, s_z,s_yaw,s_roll,s_pitch;
    double t_x, t_y, t_z,t_yaw,t_roll,t_pitch;
    if (!nh.getParam("source_x", s_x))s_x = -8.804;
    if (!nh.getParam("source_y", s_y))s_y = 1.949;
    if (!nh.getParam("source_z", s_z))s_z = 0.0;
    if (!nh.getParam("source_oy", s_yaw))s_yaw = 0.0;
    if (!nh.getParam("source_or", s_roll))s_roll = 0.0;
    if (!nh.getParam("source_op", s_pitch))s_pitch = 0.0;

    if (!nh.getParam("target_x", t_x))t_x = 1.386;
    if (!nh.getParam("target_y", t_y))t_y = -0.946;
    if (!nh.getParam("target_z", t_z))t_z = 0.0;
    if (!nh.getParam("target_oy", t_yaw))t_yaw = 0.0;
    if (!nh.getParam("target_or", t_roll))t_roll = 0.0;
    if (!nh.getParam("target_op", t_pitch))t_pitch = 0.0;

    geometry_msgs::PoseWithCovarianceStamped msg_source;
    geometry_msgs::PoseStamped msg_target;

    geometry_msgs::Quaternion s_q=tf::createQuaternionMsgFromRollPitchYaw(s_roll,s_pitch,s_yaw);
    geometry_msgs::Quaternion t_q=tf::createQuaternionMsgFromRollPitchYaw(t_roll,t_pitch,t_yaw);


    msg_source.header.frame_id = "map";
    msg_source.pose.pose.position.x =s_x;
    msg_source.pose.pose.position.y =s_y;
    msg_source.pose.pose.position.z = s_z;
    msg_source.pose.pose.orientation = s_q;

    msg_target.header.frame_id = "map";
    msg_target.pose.position.x = t_x;
    msg_target.pose.position.y = t_y;
    msg_target.pose.position.z = t_z;
    msg_target.pose.orientation = t_q;


    ros::Rate r(0.2);
    while(ros::ok())
    {
        ROS_INFO("pub");
        pub_source.publish(msg_source);
        //pub_target.publish(msg_target);
        r.sleep();
        ros::spinOnce();
    }


}
