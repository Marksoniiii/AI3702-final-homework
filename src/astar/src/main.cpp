#include <ros/ros.h>

#include <algorithm>
#include <cmath>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/PoseWithCovarianceStamped.h>
#include <nav_msgs/OccupancyGrid.h>
#include <nav_msgs/Path.h>
#include <opencv2/opencv.hpp>

#include "Astar.h"
#include "OccMapTransform.h"

using cv::Mat;
using cv::Point;
using cv::Point2d;

namespace {

ros::Subscriber map_sub;
ros::Subscriber start_point_sub;
ros::Subscriber target_point_sub;
ros::Publisher mask_pub;
ros::Publisher path_pub;

nav_msgs::OccupancyGrid inflated_mask_msg;
nav_msgs::Path path_msg;
pathplanning::AstarConfig planner_config;
pathplanning::Astar astar;
OccupancyGridParam occ_grid_param;

Point start_point;
Point target_point;

bool map_ready = false;
bool start_ready = false;
bool target_ready = false;
bool planning_requested = false;
int loop_rate_hz = 10;
double inflate_radius_meter = 0.15;

bool IsPointInsideMap(const Point& point) {
    return point.x >= 0 && point.x < occ_grid_param.width && point.y >= 0 &&
           point.y < occ_grid_param.height;
}

void PublishMaskFromMat(const Mat& mask, const nav_msgs::OccupancyGrid& map_msg) {
    inflated_mask_msg.header.stamp = ros::Time::now();
    inflated_mask_msg.header.frame_id = "map";
    inflated_mask_msg.info = map_msg.info;
    inflated_mask_msg.data.assign(map_msg.info.width * map_msg.info.height, 0);

    for (int row = 0; row < occ_grid_param.height; ++row) {
        for (int col = 0; col < occ_grid_param.width; ++col) {
            const int map_index = (occ_grid_param.height - 1 - row) * occ_grid_param.width + col;
            inflated_mask_msg.data[map_index] = mask.at<unsigned char>(row, col) > 0 ? 100 : 0;
        }
    }
}

void MapCallback(const nav_msgs::OccupancyGrid& msg) {
    occ_grid_param.GetOccupancyGridParam(msg);

    Mat free_map(occ_grid_param.height, occ_grid_param.width, CV_8UC1, cv::Scalar(0));
    for (int row = 0; row < occ_grid_param.height; ++row) {
        for (int col = 0; col < occ_grid_param.width; ++col) {
            const int map_index = (occ_grid_param.height - 1 - row) * occ_grid_param.width + col;
            const int occ_value = msg.data[map_index];
            free_map.at<unsigned char>(row, col) = (occ_value >= 0 && occ_value < 50) ? 255 : 0;
        }
    }

    planner_config.inflate_radius = std::max(
        0, static_cast<int>(std::round(inflate_radius_meter / occ_grid_param.resolution)));
    Mat inflated_mask;
    astar.InitAstar(free_map, inflated_mask, planner_config);
    PublishMaskFromMat(inflated_mask, msg);

    map_ready = true;
    planning_requested = start_ready && target_ready;
}

void StartPointCallback(const geometry_msgs::PoseWithCovarianceStamped& msg) {
    if (!map_ready) {
        ROS_WARN("Map has not been received yet, ignore start point.");
        return;
    }

    Point2d world_point(msg.pose.pose.position.x, msg.pose.pose.position.y);
    occ_grid_param.Map2ImageTransform(world_point, start_point);
    start_ready = IsPointInsideMap(start_point);
    planning_requested = map_ready && start_ready && target_ready;

    if (!start_ready) {
        ROS_WARN("Start point is outside the map boundary.");
    }
}

void TargetPointCallback(const geometry_msgs::PoseStamped& msg) {
    if (!map_ready) {
        ROS_WARN("Map has not been received yet, ignore target point.");
        return;
    }

    Point2d world_point(msg.pose.position.x, msg.pose.position.y);
    occ_grid_param.Map2ImageTransform(world_point, target_point);
    target_ready = IsPointInsideMap(target_point);
    planning_requested = map_ready && start_ready && target_ready;

    if (!target_ready) {
        ROS_WARN("Target point is outside the map boundary.");
    }
}

void PublishPath(const std::vector<Point>& path_points) {
    path_msg.header.stamp = ros::Time::now();
    path_msg.header.frame_id = "map";
    path_msg.poses.clear();

    for (const Point& path_point : path_points) {
        Point2d world_point;
        Point point_copy = path_point;
        occ_grid_param.Image2MapTransform(point_copy, world_point);

        geometry_msgs::PoseStamped pose_stamped;
        pose_stamped.header = path_msg.header;
        pose_stamped.pose.position.x = world_point.x;
        pose_stamped.pose.position.y = world_point.y;
        pose_stamped.pose.position.z = 0.0;
        pose_stamped.pose.orientation.w = 1.0;
        path_msg.poses.push_back(pose_stamped);
    }

    path_pub.publish(path_msg);
}

}  // namespace

int main(int argc, char* argv[]) {
    ros::init(argc, argv, "astar");
    ros::NodeHandle nh;
    ros::NodeHandle nh_priv("~");

    nh_priv.param<bool>("AllowDiagonal", planner_config.allow_diagonal, true);
    nh_priv.param<bool>("UseChebyshev", planner_config.use_chebyshev, true);
    nh_priv.param<int>("OccupyThresh", planner_config.occupy_thresh, 50);
    nh_priv.param<double>("InflateRadius", inflate_radius_meter, 0.15);
    nh_priv.param<int>("rate", loop_rate_hz, 10);

    map_sub = nh.subscribe("map", 1, MapCallback);
    start_point_sub = nh.subscribe("move_base/NavfnROS/Astar/initialpose", 1, StartPointCallback);
    target_point_sub = nh.subscribe("move_base/NavfnROS/Astar/target", 1, TargetPointCallback);

    mask_pub = nh.advertise<nav_msgs::OccupancyGrid>("mask", 1, true);
    path_pub = nh.advertise<nav_msgs::Path>("move_base/NavfnROS/nav_path", 1, true);

    ros::Rate loop_rate(loop_rate_hz);
    while (ros::ok()) {
        if (planning_requested && map_ready) {
            std::vector<Point> planned_path;
            const double start_time = ros::Time::now().toSec();
            const bool success = astar.PathPlanning(start_point, target_point, planned_path);

            if (success) {
                PublishPath(planned_path);
                const int steps = astar.GetLastGridSteps();
                const double end_time = ros::Time::now().toSec();
                ROS_INFO("Path found. Planning time: %.6f s, steps: %d, path points: %zu",
                         end_time - start_time, steps, planned_path.size());
            } else {
                path_msg.header.stamp = ros::Time::now();
                path_msg.header.frame_id = "map";
                path_msg.poses.clear();
                path_pub.publish(path_msg);
                ROS_WARN("No valid path found between the selected start and target.");
            }

            planning_requested = false;
        }

        if (map_ready) {
            inflated_mask_msg.header.stamp = ros::Time::now();
            mask_pub.publish(inflated_mask_msg);
        }

        ros::spinOnce();
        loop_rate.sleep();
    }

    return 0;
}
