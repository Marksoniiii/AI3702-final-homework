#include <iostream>
#include <ros/ros.h>
#include <nav_msgs/OccupancyGrid.h>
#include <nav_msgs/Path.h>
#include <geometry_msgs/PoseWithCovarianceStamped.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/Quaternion.h>
#include "Astar.h"
#include <chrono>
#include <thread>
using namespace cv;
using namespace std;

/* parameters */
MapParamNode MapParam;
#define EPISILON 1.
/* global variable */
bool solve_flag = false;
bool map_flag = false;
double dilation_distance = 0.15;
mutex lock_flag;
Mat Maptest;
GridAstar solver(&MapParam, EPISILON);
/********************/
bool map_build_flag=false;
void World2MapGrid(MapParamNode &MapParam, Point2d &src_point, Point &dst_point) {
    Mat P_src = Mat(Vec2d(src_point.x, src_point.y), CV_64FC1);
    Mat P_dst = MapParam.Rotation.inv() * (P_src - MapParam.Translation);

    dst_point.x = round(P_dst.at<double>(0, 0));
    dst_point.y = MapParam.height - 1 - round(P_dst.at<double>(1, 0));
}

void MapGrid2world(MapParamNode &MapParam, Point &src_point, Point2d &dst_point) {
    Mat P_src = Mat(Vec2d(src_point.x, MapParam.height - 1 - src_point.y), CV_64FC1);

    Mat P_dst = MapParam.Rotation * P_src + MapParam.Translation;

    dst_point.x = P_dst.at<double>(0, 0);
    dst_point.y = P_dst.at<double>(1, 0);
}

void MapCallback(const nav_msgs::OccupancyGrid &msg) {
    while(!map_build_flag){
        std::this_thread::sleep_for(chrono::milliseconds(100));
    }
    lock_flag.lock();
    map_flag = false;
    lock_flag.unlock();
    // Get the parameters of map
    MapParam.resolution = msg.info.resolution;
    MapParam.height = msg.info.height;
    MapParam.width = msg.info.width;
    // The origin of the MapGrid is on the bottom left corner of the map
    MapParam.x = msg.info.origin.position.x;
    MapParam.y = msg.info.origin.position.y;

    int pad = round(dilation_distance/MapParam.resolution);

    // Calculate the pose of map with respect to the world of rviz
    double roll, pitch, yaw;
    geometry_msgs::Quaternion q = msg.info.origin.orientation;
    tf::Quaternion quat(q.x, q.y, q.z, q.w); // x, y, z, w
    tf::Matrix3x3(quat).getRPY(roll, pitch, yaw);
    double theta = yaw;

    //从rviz上所给定的起点和终点坐标是真实世界坐标系下的位置，需要转化为地图坐标下的表示
    //MapParam.Rotation MapParam.Translation 用于该变换
    MapParam.Rotation = Mat::zeros(2, 2, CV_64FC1);
    MapParam.Rotation.at<double>(0, 0) = MapParam.resolution * cos(theta);
    MapParam.Rotation.at<double>(0, 1) = MapParam.resolution * sin(-theta);
    MapParam.Rotation.at<double>(1, 0) = MapParam.resolution * sin(theta);
    MapParam.Rotation.at<double>(1, 1) = MapParam.resolution * cos(theta);
    MapParam.Translation = Mat(Vec2d(MapParam.x, MapParam.y), CV_64FC1);

    cout << "Map:" << endl;
    cout << "MapParam.height:" << MapParam.height << endl;
    cout << "MapParam.width:" << MapParam.width << endl;
    ROS_INFO("Building map");
    auto _curTimePoint = std::chrono::steady_clock::now();
    Maptest = Mat(MapParam.height, MapParam.width, CV_8UC1);
    int GridFlag;
    for (int i = 0; i < MapParam.height; i++) {
        for (int j = 0; j < MapParam.width; j++) {
            GridFlag = msg.data[i * MapParam.width + j];
            GridFlag = (GridFlag < 0) ? 100 : GridFlag; // set Unknown to 0
            Maptest.at<uchar>(j, MapParam.height - i - 1) = 255 - round(GridFlag * 255.0 / 100.0);
        }
    }
    solver.reset(&Maptest,pad);
    auto curTime = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(curTime -
                                                                          _curTimePoint);
    ROS_INFO("Build map finish... Cost %ld ms",duration.count());

    lock_flag.lock();
    map_flag = true;
    lock_flag.unlock();
}

void StartPointCallback(const geometry_msgs::PoseWithCovarianceStamped &msg) {
    Point2d src_point = Point2d(msg.pose.pose.position.x, msg.pose.pose.position.y);
    World2MapGrid(MapParam, src_point, MapParam.StartPoint);
    cout << "StartPoint:" << MapParam.StartPoint << endl;
    lock_flag.lock();
    solve_flag = true;
    lock_flag.unlock();
}

void TargetPointtCallback(const geometry_msgs::PoseStamped &msg) {
    Point2d src_point = Point2d(msg.pose.position.x, msg.pose.position.y);
    World2MapGrid(MapParam, src_point, MapParam.TargetPoint);
    int p = Maptest.at<uchar>(MapParam.TargetPoint.x, MapParam.TargetPoint.y);
    cout << "flag:" << p << endl;
    MapGrid2world(MapParam, MapParam.TargetPoint, src_point);
    cout << "TargetPoint world:" << src_point << endl;
    cout << "TargetPoint:" << MapParam.TargetPoint << endl;
    lock_flag.lock();
    solve_flag = true;
    lock_flag.unlock();
}

void PathGrid2world(MapParamNode &MapParam, vector<Point> &PathList, nav_msgs::Path &plan_path) {
    for (int i = 0; i < PathList.size(); i++) {
        Point2d dst_point;
        MapGrid2world(MapParam, PathList[i], dst_point);
        if (i == 0 || i == PathList.size() - 1)
            ROS_DEBUG("(%d,%d) -> %.3f %.3f", PathList[i].x, PathList[i].y, dst_point.x, dst_point.y);
        geometry_msgs::PoseStamped pose_stamped;
        pose_stamped.header.stamp = ros::Time::now();
        pose_stamped.header.frame_id = "map";
        pose_stamped.pose.position.x = dst_point.x;
        pose_stamped.pose.position.y = dst_point.y;
        pose_stamped.pose.position.z = 0;
        pose_stamped.pose.orientation.w = 1.0;
        plan_path.poses.push_back(pose_stamped);
    }
}

int main(int argc, char *argv[]) {
    ros::init(argc, argv, "astar");
    ros::NodeHandle n;


    geometry_msgs::PointStamped astar_step;

    if (!n.getParam("dilation_distance", dilation_distance));
    ROS_INFO("set dilation distance %.3lf\n",dilation_distance);
    map_build_flag = true;

    // Subscribe topics
    ros::Subscriber Map_sub = n.subscribe("map", 10, MapCallback);
    //ros::Subscriber StarPoint_sub = n.subscribe("initialpose", 10, StartPointCallback);
    //ros::Subscriber TargetPoint_sub = n.subscribe("move_base_simple/goal", 10, TargetPointtCallback);
    ros::Subscriber StarPoint_sub = n.subscribe("move_base/NavfnROS/Astar/initialpose", 10, StartPointCallback);
    ros::Subscriber TargetPoint_sub = n.subscribe("move_base/NavfnROS/Astar/target", 10, TargetPointtCallback);
    // Publisher topics
    ros::Publisher path_pub = n.advertise<nav_msgs::Path>("move_base/NavfnROS/nav_path", 10);

    ros::Rate loop_rate(20);


    vector<Point> &PathList = solver.result;

    while (ros::ok()) {
        lock_flag.lock();
        if (solve_flag and map_flag) {
            /* do once search*/
            ROS_INFO("start solving");
            auto _curTimePoint = std::chrono::steady_clock::now();
            solver.solve();
            auto curTime = std::chrono::steady_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(curTime -
                                                                                  _curTimePoint);

            ROS_INFO("Solving finish... Cost %ld ms",duration.count());
            nav_msgs::Path plan_path;
            plan_path.header.frame_id = "map";
            if (not PathList.empty()) {
                ROS_INFO("find_path cost %d", (int) PathList.size()-1);/*length=N-1*/
                PathGrid2world(MapParam, PathList, plan_path);
                path_pub.publish(plan_path);
            }
            else
            {
                ROS_INFO("not find");
            }
            solve_flag = false;
        }
        lock_flag.unlock();
        loop_rate.sleep();
        ros::spinOnce();
    }

    return 0;
}
