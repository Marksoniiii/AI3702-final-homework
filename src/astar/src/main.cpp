#include <cmath>
#include <string>

#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/PoseWithCovarianceStamped.h>
#include <nav_msgs/OccupancyGrid.h>
#include <nav_msgs/Path.h>
#include <ros/ros.h>
#include <tf/transform_datatypes.h>
#include <tf/transform_listener.h>

#include "Astar.h"

using namespace std;
using namespace cv;

namespace {

MapParamNode g_map_param;
Mat g_map;

bool g_map_ready = false;
bool g_need_plan = false;
bool g_use_tf_start = false;
bool g_manual_start_fallback = true;
bool g_treat_unknown_as_obstacle = true;

double g_dilation_distance = 0.25;
double g_inflated_cell_penalty = 8.0;
double g_preferred_clearance = 0.26;
double g_clearance_penalty_weight = 1.4;
int g_occupied_threshold = 65;
string g_global_frame = "map";
string g_robot_frame = "base_link";
string g_path_topic = "move_base/NavfnROS/nav_path";
string g_start_topic = "/initialpose";
string g_goal_topic = "/move_base_simple/goal";

bool WorldToGrid(double wx, double wy, Point &grid_point) {
    if (!g_map_ready || g_map_param.resolution <= 0.0) {
        return false;
    }

    double dx = wx - g_map_param.origin_x;
    double dy = wy - g_map_param.origin_y;
    double cos_yaw = cos(g_map_param.origin_yaw);
    double sin_yaw = sin(g_map_param.origin_yaw);

    double mx = (cos_yaw * dx + sin_yaw * dy) / g_map_param.resolution;
    double my = (-sin_yaw * dx + cos_yaw * dy) / g_map_param.resolution;

    grid_point.x = static_cast<int>(floor(mx));
    grid_point.y = static_cast<int>(floor(my));
    return grid_point.x >= 0 && grid_point.x < g_map_param.width &&
           grid_point.y >= 0 && grid_point.y < g_map_param.height;
}

Point2d GridToWorld(const Point &grid_point) {
    double map_x = (static_cast<double>(grid_point.x) + 0.5) * g_map_param.resolution;
    double map_y = (static_cast<double>(grid_point.y) + 0.5) * g_map_param.resolution;
    double cos_yaw = cos(g_map_param.origin_yaw);
    double sin_yaw = sin(g_map_param.origin_yaw);

    Point2d world_point;
    world_point.x = g_map_param.origin_x + cos_yaw * map_x - sin_yaw * map_y;
    world_point.y = g_map_param.origin_y + sin_yaw * map_x + cos_yaw * map_y;
    return world_point;
}

void BuildBinaryMap(const nav_msgs::OccupancyGrid::ConstPtr &msg) {
    g_map_param.resolution = msg->info.resolution;
    g_map_param.width = static_cast<int>(msg->info.width);
    g_map_param.height = static_cast<int>(msg->info.height);
    g_map_param.origin_x = msg->info.origin.position.x;
    g_map_param.origin_y = msg->info.origin.position.y;

    double roll = 0.0;
    double pitch = 0.0;
    tf::Quaternion quat;
    tf::quaternionMsgToTF(msg->info.origin.orientation, quat);
    tf::Matrix3x3(quat).getRPY(roll, pitch, g_map_param.origin_yaw);

    g_map = Mat::zeros(g_map_param.height, g_map_param.width, CV_8UC1);
    for (int y = 0; y < g_map_param.height; ++y) {
        for (int x = 0; x < g_map_param.width; ++x) {
            int value = msg->data[y * g_map_param.width + x];
            bool occupied = false;

            if (value < 0) {
                occupied = g_treat_unknown_as_obstacle;
            } else {
                occupied = value >= g_occupied_threshold;
            }

            g_map.at<uchar>(y, x) = occupied ? 0 : 255;
        }
    }

    g_map_ready = true;
    g_need_plan = true;
    ROS_INFO("Map received: %d x %d, resolution %.3f, origin(%.3f, %.3f), yaw %.3f",
             g_map_param.width, g_map_param.height, g_map_param.resolution,
             g_map_param.origin_x, g_map_param.origin_y, g_map_param.origin_yaw);
}

bool UpdateStartPointFromPose(double wx, double wy, const char *source_name) {
    Point grid_point;
    if (!WorldToGrid(wx, wy, grid_point)) {
        ROS_WARN("%s start point is out of map bounds: world(%.3f, %.3f)", source_name, wx, wy);
        return false;
    }

    bool changed = (!g_map_param.has_start || g_map_param.StartPoint != grid_point);
    g_map_param.StartPoint = grid_point;
    g_map_param.has_start = true;
    g_need_plan = true;
    if (changed) {
        ROS_INFO("%s start point: world(%.3f, %.3f) -> grid(%d, %d)",
                 source_name, wx, wy, grid_point.x, grid_point.y);
    }
    return true;
}

bool UpdateTargetPointFromPose(double wx, double wy, const char *source_name) {
    Point grid_point;
    if (!WorldToGrid(wx, wy, grid_point)) {
        ROS_WARN("%s target point is out of map bounds: world(%.3f, %.3f)", source_name, wx, wy);
        return false;
    }

    bool changed = (!g_map_param.has_target || g_map_param.TargetPoint != grid_point);
    g_map_param.TargetPoint = grid_point;
    g_map_param.has_target = true;
    g_need_plan = true;
    if (changed) {
        ROS_INFO("%s target point: world(%.3f, %.3f) -> grid(%d, %d)",
                 source_name, wx, wy, grid_point.x, grid_point.y);
    }
    return true;
}

bool TryUpdateStartFromTf(tf::TransformListener &tf_listener) {
    tf::StampedTransform transform;
    try {
        tf_listener.lookupTransform(g_global_frame, g_robot_frame, ros::Time(0), transform);
    } catch (const tf::TransformException &ex) {
        ROS_WARN_THROTTLE(2.0, "Failed to get robot pose from TF (%s -> %s): %s",
                          g_global_frame.c_str(), g_robot_frame.c_str(), ex.what());
        return false;
    }

    return UpdateStartPointFromPose(transform.getOrigin().x(),
                                    transform.getOrigin().y(),
                                    "TF");
}

void MapCallback(const nav_msgs::OccupancyGrid::ConstPtr &msg) {
    BuildBinaryMap(msg);
}

void StartPointCallback(const geometry_msgs::PoseWithCovarianceStamped::ConstPtr &msg) {
    if (!g_map_ready) {
        ROS_WARN_THROTTLE(1.0, "Start point received before map is ready.");
        return;
    }

    UpdateStartPointFromPose(msg->pose.pose.position.x, msg->pose.pose.position.y, "Manual");
}

void TargetPointCallback(const geometry_msgs::PoseStamped::ConstPtr &msg) {
    if (!g_map_ready) {
        ROS_WARN_THROTTLE(1.0, "Target point received before map is ready.");
        return;
    }

    UpdateTargetPointFromPose(msg->pose.position.x, msg->pose.position.y, "Manual");
}

void PathGridToWorld(const vector<Point> &path_list, nav_msgs::Path &plan_path) {
    plan_path.poses.clear();
    plan_path.header.frame_id = g_global_frame;
    plan_path.header.stamp = ros::Time::now();

    for (size_t i = 0; i < path_list.size(); ++i) {
        const Point &grid_point = path_list[i];
        Point2d world_point = GridToWorld(grid_point);

        geometry_msgs::PoseStamped pose;
        pose.header = plan_path.header;
        pose.pose.position.x = world_point.x;
        pose.pose.position.y = world_point.y;
        pose.pose.position.z = 0.0;

        double yaw = 0.0;
        if (i + 1 < path_list.size()) {
            Point2d next_world_point = GridToWorld(path_list[i + 1]);
            yaw = atan2(next_world_point.y - world_point.y,
                        next_world_point.x - world_point.x);
        } else if (i > 0) {
            Point2d prev_world_point = GridToWorld(path_list[i - 1]);
            yaw = atan2(world_point.y - prev_world_point.y,
                        world_point.x - prev_world_point.x);
        }

        tf::Quaternion quat = tf::createQuaternionFromYaw(yaw);
        tf::quaternionTFToMsg(quat, pose.pose.orientation);
        plan_path.poses.push_back(pose);
    }
}

}  // namespace

int main(int argc, char **argv) {
    ros::init(argc, argv, "astar");
    ros::NodeHandle nh;
    ros::NodeHandle pnh("~");

    pnh.param("dilation_distance", g_dilation_distance, 0.25);
    pnh.param("inflated_cell_penalty", g_inflated_cell_penalty, 8.0);
    pnh.param("preferred_clearance", g_preferred_clearance, 0.26);
    pnh.param("clearance_penalty_weight", g_clearance_penalty_weight, 1.4);
    pnh.param("occupied_threshold", g_occupied_threshold, 65);
    pnh.param("treat_unknown_as_obstacle", g_treat_unknown_as_obstacle, true);
    pnh.param("use_tf_start", g_use_tf_start, false);
    pnh.param("manual_start_fallback", g_manual_start_fallback, true);
    pnh.param("global_frame", g_global_frame, string("map"));
    pnh.param("robot_frame", g_robot_frame, string("base_link"));
    pnh.param("path_topic", g_path_topic, string("move_base/NavfnROS/nav_path"));
    pnh.param("start_topic", g_start_topic, string("/initialpose"));
    pnh.param("goal_topic", g_goal_topic, string("/move_base_simple/goal"));

    ROS_INFO("A* planner config: use_tf_start=%s, dilation_distance=%.3f, preferred_clearance=%.3f, inflated_cell_penalty=%.2f, clearance_penalty_weight=%.2f, occupied_threshold=%d",
             g_use_tf_start ? "true" : "false", g_dilation_distance, g_preferred_clearance,
             g_inflated_cell_penalty, g_clearance_penalty_weight, g_occupied_threshold);

    tf::TransformListener tf_listener;
    GridAstar astar_solver(&g_map_param, 1.0);
    nav_msgs::Path plan_path;

    ros::Subscriber map_sub = nh.subscribe("map", 1, MapCallback);
    ros::Subscriber start_sub = nh.subscribe(g_start_topic, 10, StartPointCallback);
    ros::Subscriber goal_sub = nh.subscribe(g_goal_topic, 10, TargetPointCallback);
    ros::Publisher path_pub = nh.advertise<nav_msgs::Path>(g_path_topic, 10, true);

    ros::Rate loop_rate(20);
    while (ros::ok()) {
        ros::spinOnce();

        if (g_map_ready && g_use_tf_start) {
            bool tf_ok = TryUpdateStartFromTf(tf_listener);
            if (!tf_ok && !g_manual_start_fallback) {
                g_map_param.has_start = false;
            }
        }

        if (g_map_ready && g_map_param.has_start && g_map_param.has_target && g_need_plan) {
            int padding_cells = 0;
            if (g_map_param.resolution > 0.0) {
                padding_cells = static_cast<int>(ceil(g_dilation_distance / g_map_param.resolution));
            }

            Mat planning_map = g_map.clone();
            astar_solver.setInflatedCellPenalty(g_inflated_cell_penalty);
            double preferred_clearance_cells = 0.0;
            if (g_map_param.resolution > 0.0) {
                preferred_clearance_cells = g_preferred_clearance / g_map_param.resolution;
            }
            astar_solver.setClearancePreference(preferred_clearance_cells, g_clearance_penalty_weight);
            astar_solver.reset(&planning_map, padding_cells);

            ROS_INFO("Planning from grid(%d, %d) to grid(%d, %d)...",
                     g_map_param.StartPoint.x, g_map_param.StartPoint.y,
                     g_map_param.TargetPoint.x, g_map_param.TargetPoint.y);
            astar_solver.solve();

            if (!astar_solver.result.empty()) {
                PathGridToWorld(astar_solver.result, plan_path);
                path_pub.publish(plan_path);
                ROS_INFO("Path found with %d poses.", static_cast<int>(plan_path.poses.size()));
            } else {
                plan_path.poses.clear();
                plan_path.header.frame_id = g_global_frame;
                plan_path.header.stamp = ros::Time::now();
                path_pub.publish(plan_path);
                ROS_WARN("Path not found.");
            }

            g_need_plan = false;
        }

        loop_rate.sleep();
    }

    return 0;
}
