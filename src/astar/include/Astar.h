#ifndef ASTAR_H
#define ASTAR_H

#include <cmath>
#include <iostream>
#include <limits>
#include <opencv2/opencv.hpp>
#include <ros/ros.h>
#include "astar_algo.h"

using namespace std;
using namespace cv;

struct MapParamNode {
    Point StartPoint, TargetPoint;
    double resolution;
    int height;
    int width;
    double origin_x;
    double origin_y;
    double origin_yaw;
    bool has_start;
    bool has_target;

    MapParamNode()
        : StartPoint(-1, -1),
          TargetPoint(-1, -1),
          resolution(0.0),
          height(0),
          width(0),
          origin_x(0.0),
          origin_y(0.0),
          origin_yaw(0.0),
          has_start(false),
          has_target(false) {}
};

class GridAstar : public Astar<int8_t> {
private:
    MapParamNode *mapParam;
    Mat *Map;
    Mat clearanceMap;
    double inflatedCellPenalty;
    double preferredClearanceCells;
    double clearancePenaltyWeight;

    bool _is_in_bounds(int x, int y) const;
    bool _is_free(int x, int y) const;
    bool _is_hard_free(int x, int y) const;
    bool _is_lethal(int x, int y) const;
    bool _is_soft_inflated(int x, int y) const;
    bool _find_nearest_hard_free(const Point &seed, Point &result) const;

protected:
    void _find_neighbor(int u, vector<int> &neighbor) override;
    double _compute_dist(int v1, int v2) override;
    double _compute_step_cost(int v1, int v2) override;
    void get_result(vector<Point> &p);

public:
    vector<Point> result;

    GridAstar(MapParamNode *m, double epi)
        : Astar<int8_t>(epi),
          mapParam(m),
          Map(nullptr),
          clearanceMap(),
          inflatedCellPenalty(8.0),
          preferredClearanceCells(4.0),
          clearancePenaltyWeight(1.2),
          result() {}

    void reset(Mat *map, int padding = 0);
    void solve();
    void setInflatedCellPenalty(double penalty) { inflatedCellPenalty = max(0.0, penalty); }
    void setClearancePreference(double preferred_clearance_cells, double penalty_weight) {
        preferredClearanceCells = max(0.0, preferred_clearance_cells);
        clearancePenaltyWeight = max(0.0, penalty_weight);
    }
};

inline bool GridAstar::_is_in_bounds(int x, int y) const {
    return x >= 0 && x < mapParam->width && y >= 0 && y < mapParam->height;
}

inline bool GridAstar::_is_free(int x, int y) const {
    return _is_in_bounds(x, y) && Map && Map->at<uchar>(y, x) > 0;
}

inline bool GridAstar::_is_hard_free(int x, int y) const {
    return _is_in_bounds(x, y) && Map && Map->at<uchar>(y, x) == 255;
}

inline bool GridAstar::_is_lethal(int x, int y) const {
    return !_is_in_bounds(x, y) || !Map || Map->at<uchar>(y, x) == 0;
}

inline bool GridAstar::_is_soft_inflated(int x, int y) const {
    return _is_in_bounds(x, y) && Map && Map->at<uchar>(y, x) == 128;
}

inline bool GridAstar::_find_nearest_hard_free(const Point &seed, Point &result) const {
    if (!_is_in_bounds(seed.x, seed.y) || !Map) {
        return false;
    }

    if (_is_hard_free(seed.x, seed.y)) {
        result = seed;
        return true;
    }

    int max_radius = max(mapParam->width, mapParam->height);
    double best_dist_sq = numeric_limits<double>::max();
    bool found = false;

    for (int radius = 1; radius < max_radius; ++radius) {
        bool found_on_this_ring = false;
        for (int dy = -radius; dy <= radius; ++dy) {
            for (int dx = -radius; dx <= radius; ++dx) {
                if (abs(dx) != radius && abs(dy) != radius) {
                    continue;
                }

                int x = seed.x + dx;
                int y = seed.y + dy;
                if (!_is_hard_free(x, y)) {
                    continue;
                }

                double dist_sq = static_cast<double>(dx * dx + dy * dy);
                if (dist_sq < best_dist_sq) {
                    best_dist_sq = dist_sq;
                    result = Point(x, y);
                    found = true;
                    found_on_this_ring = true;
                }
            }
        }

        if (found_on_this_ring) {
            return true;
        }
    }

    return found;
}

inline void GridAstar::reset(Mat *map, int padding) {
    vertices.clear();
    vertices.reserve(mapParam->height * mapParam->width);
    for (int i = 0; i < mapParam->height * mapParam->width; ++i) {
        vertices.emplace_back(0);
    }

    Map = map;
    if (!Map) {
        return;
    }

    if (padding > 0) {
        Mat obstacle_mask = (*Map == 0);
        Mat dilated_obstacles;
        Mat kernel = getStructuringElement(MORPH_ELLIPSE, Size(padding * 2 + 1, padding * 2 + 1));
        dilate(obstacle_mask, dilated_obstacles, kernel);

        for (int y = 0; y < Map->rows; ++y) {
            for (int x = 0; x < Map->cols; ++x) {
                if (obstacle_mask.at<uchar>(y, x) != 0) {
                    Map->at<uchar>(y, x) = 0;
                } else if (dilated_obstacles.at<uchar>(y, x) != 0) {
                    Map->at<uchar>(y, x) = 128;
                } else {
                    Map->at<uchar>(y, x) = 255;
                }
            }
        }
    }

    Mat traversable_mask;
    compare(*Map, 0, traversable_mask, CMP_GT);
    traversable_mask.convertTo(traversable_mask, CV_8UC1);
    distanceTransform(traversable_mask, clearanceMap, DIST_L2, 3);
}

inline void GridAstar::solve() {
    result.clear();
    path.clear();

    if (!Map || !mapParam->has_start || !mapParam->has_target) {
        return;
    }

    Point start_point = mapParam->StartPoint;
    Point target_point = mapParam->TargetPoint;

    if (!_is_in_bounds(start_point.x, start_point.y)) {
        ROS_WARN("Start point is out of map bounds.");
        return;
    }
    if (!_is_in_bounds(target_point.x, target_point.y)) {
        ROS_WARN("Target point is out of map bounds.");
        return;
    }
    if (!_is_free(start_point.x, start_point.y)) {
        ROS_WARN("Start point is not reachable.");
        return;
    }
    if (!_is_free(target_point.x, target_point.y)) {
        ROS_WARN("Target point is not reachable.");
        return;
    }

    if (!_is_hard_free(target_point.x, target_point.y)) {
        Point adjusted_target;
        if (_find_nearest_hard_free(target_point, adjusted_target)) {
            ROS_WARN("Target point is too close to obstacles, adjusted from (%d, %d) to (%d, %d).",
                     target_point.x, target_point.y, adjusted_target.x, adjusted_target.y);
            target_point = adjusted_target;
        } else {
            ROS_WARN("Failed to find a safe target point away from obstacles.");
            return;
        }
    }

    int start = start_point.y * mapParam->width + start_point.x;
    int end = target_point.y * mapParam->width + target_point.x;

    _solve(start, end);
    get_result(result);
}

inline void GridAstar::_find_neighbor(int u, vector<int> &neighbor) {
    neighbor.clear();

    int y = u / mapParam->width;
    int x = u % mapParam->width;

    for (int dy = -1; dy <= 1; ++dy) {
        for (int dx = -1; dx <= 1; ++dx) {
            if (dx == 0 && dy == 0) {
                continue;
            }

            int nx = x + dx;
            int ny = y + dy;
            if (!_is_free(nx, ny)) {
                continue;
            }

            // Prevent the robot from slipping diagonally through obstacle corners.
            if (dx != 0 && dy != 0) {
                if (_is_lethal(x + dx, y) || _is_lethal(x, y + dy)) {
                    continue;
                }
            }

            neighbor.emplace_back(ny * mapParam->width + nx);
        }
    }
}

inline double GridAstar::_compute_dist(int v1, int v2) {
    int y1 = v1 / mapParam->width;
    int x1 = v1 % mapParam->width;
    int y2 = v2 / mapParam->width;
    int x2 = v2 % mapParam->width;

    double dx = static_cast<double>(x1 - x2);
    double dy = static_cast<double>(y1 - y2);
    return sqrt(dx * dx + dy * dy);
}

inline double GridAstar::_compute_step_cost(int v1, int v2) {
    double base_cost = _compute_dist(v1, v2);

    int y2 = v2 / mapParam->width;
    int x2 = v2 % mapParam->width;
    if (_is_soft_inflated(x2, y2)) {
        base_cost += inflatedCellPenalty;
    }

    if (!clearanceMap.empty()) {
        double clearance_cells = static_cast<double>(clearanceMap.at<float>(y2, x2));
        if (clearance_cells < preferredClearanceCells) {
            base_cost += (preferredClearanceCells - clearance_cells) * clearancePenaltyWeight;
        }
    }

    return base_cost;
}

inline void GridAstar::get_result(vector<Point> &p) {
    p.clear();
    for (const auto &pt : path) {
        p.emplace_back(pt % mapParam->width, pt / mapParam->width);
    }
}

#endif
