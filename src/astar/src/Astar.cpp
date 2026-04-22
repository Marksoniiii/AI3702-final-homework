#include "Astar.h"

#include <algorithm>
#include <cmath>

namespace pathplanning {

namespace {

bool LexicographicallyBetter(int lhs_step, int lhs_geometry, int lhs_turn, int rhs_step,
                             int rhs_geometry, int rhs_turn) {
    if (lhs_step != rhs_step) {
        return lhs_step < rhs_step;
    }
    if (lhs_geometry != rhs_geometry) {
        return lhs_geometry < rhs_geometry;
    }
    return lhs_turn < rhs_turn;
}

}  // namespace

bool Astar::QueueNode::operator>(const QueueNode& other) const {
    if (f_step_cost != other.f_step_cost) {
        return f_step_cost > other.f_step_cost;
    }
    if (h_step_cost != other.h_step_cost) {
        return h_step_cost > other.h_step_cost;
    }
    if (geometry_cost != other.geometry_cost) {
        return geometry_cost > other.geometry_cost;
    }
    if (turn_cost != other.turn_cost) {
        return turn_cost > other.turn_cost;
    }
    if (line_bias != other.line_bias) {
        return line_bias > other.line_bias;
    }
    return index > other.index;
}

void Astar::InitAstar(const cv::Mat& map, AstarConfig config) {
    cv::Mat mask;
    InitAstar(map, mask, config);
}

void Astar::InitAstar(const cv::Mat& map, cv::Mat& mask, AstarConfig config) {
    raw_map_ = map.clone();
    config_ = config;

    neighbors_.clear();
    neighbors_.push_back(cv::Point(1, 0));
    neighbors_.push_back(cv::Point(0, 1));
    neighbors_.push_back(cv::Point(0, -1));
    neighbors_.push_back(cv::Point(-1, 0));
    if (config_.allow_diagonal) {
        neighbors_.push_back(cv::Point(1, 1));
        neighbors_.push_back(cv::Point(1, -1));
        neighbors_.push_back(cv::Point(-1, 1));
        neighbors_.push_back(cv::Point(-1, -1));
    }

    MapProcess(mask);
}

bool Astar::PathPlanning(const cv::Point& start_point, const cv::Point& target_point,
                         std::vector<cv::Point>& path) {
    path.clear();
    last_grid_steps_ = 0;
    start_point_ = start_point;
    target_point_ = target_point;

    if (free_map_.empty() || !IsInside(start_point_) || !IsInside(target_point_) ||
        !IsFree(start_point_) || !IsFree(target_point_)) {
        return false;
    }

    search_nodes_.assign(free_map_.rows * free_map_.cols, SearchNode());
    std::priority_queue<QueueNode, std::vector<QueueNode>, std::greater<QueueNode>> open_list;

    const int start_index = PointToIndex(start_point_);
    SearchNode& start_node = search_nodes_[start_index];
    start_node.step_cost = 0;
    start_node.geometry_cost = 0;
    start_node.turn_cost = 0;

    open_list.push({StepHeuristic(start_point_), StepHeuristic(start_point_), 0, 0,
                    LineBias(start_point_), start_index});

    while (!open_list.empty()) {
        const QueueNode current_queue_node = open_list.top();
        open_list.pop();

        SearchNode& current_node = search_nodes_[current_queue_node.index];
        const cv::Point current_point = IndexToPoint(current_queue_node.index);
        const int expected_f_step = current_node.step_cost + StepHeuristic(current_point);
        if (current_node.closed || current_queue_node.f_step_cost != expected_f_step) {
            continue;
        }
        current_node.closed = true;

        if (current_point == target_point_) {
            const bool ok = ReconstructPath(current_queue_node.index, path);
            if (!ok) {
                return false;
            }
            last_grid_steps_ = std::max(0, static_cast<int>(path.size()) - 1);
            if (config_.enable_path_simplify) {
                SimplifyPath(path);
            }
            return true;
        }

        for (const cv::Point& offset : neighbors_) {
            const cv::Point next_point = current_point + offset;
            if (!IsInside(next_point) || !IsFree(next_point)) {
                continue;
            }

            const bool diagonal_move = (offset.x != 0 && offset.y != 0);
            if (diagonal_move) {
                const cv::Point side_point_x(current_point.x + offset.x, current_point.y);
                const cv::Point side_point_y(current_point.x, current_point.y + offset.y);
                if (!IsInside(side_point_x) || !IsInside(side_point_y) || !IsFree(side_point_x) ||
                    !IsFree(side_point_y)) {
                    continue;
                }
            }

            const int next_index = PointToIndex(next_point);
            SearchNode& next_node = search_nodes_[next_index];
            if (next_node.closed) {
                continue;
            }

            const int tentative_step_cost = current_node.step_cost + 1;
            const int tentative_geometry_cost =
                current_node.geometry_cost + GeometryMoveCost(current_point, next_point);
            const int tentative_turn_cost =
                current_node.turn_cost +
                TurnPenalty(current_node.parent, current_point, next_point);

            if (!LexicographicallyBetter(tentative_step_cost, tentative_geometry_cost,
                                         tentative_turn_cost, next_node.step_cost,
                                         next_node.geometry_cost, next_node.turn_cost)) {
                continue;
            }

            next_node.step_cost = tentative_step_cost;
            next_node.geometry_cost = tentative_geometry_cost;
            next_node.turn_cost = tentative_turn_cost;
            next_node.parent = current_queue_node.index;

            const int next_h_step = StepHeuristic(next_point);
            open_list.push({tentative_step_cost + next_h_step, next_h_step,
                            tentative_geometry_cost, tentative_turn_cost,
                            LineBias(next_point), next_index});
        }
    }

    return false;
}

int Astar::GetLastGridSteps() const {
    return last_grid_steps_;
}

void Astar::DrawPath(cv::Mat& map, const std::vector<cv::Point>& path, cv::InputArray mask,
                     cv::Scalar color, int thickness, cv::Scalar mask_color) {
    if (path.empty()) {
        return;
    }
    if (!mask.empty()) {
        map.setTo(mask_color, mask);
    }
    for (const cv::Point& point : path) {
        cv::rectangle(map, point, point, color, thickness);
    }
}

void Astar::MapProcess(cv::Mat& mask) {
    cv::Mat gray_map;
    if (raw_map_.channels() == 3) {
        cv::cvtColor(raw_map_, gray_map, cv::COLOR_BGR2GRAY);
    } else {
        gray_map = raw_map_.clone();
    }

    if (config_.occupy_thresh < 0) {
        cv::threshold(gray_map, free_map_, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
    } else {
        cv::threshold(gray_map, free_map_, config_.occupy_thresh, 255, cv::THRESH_BINARY);
    }

    cv::Mat before_inflate = free_map_.clone();
    if (config_.inflate_radius > 0) {
        const int kernel_size = 2 * config_.inflate_radius + 1;
        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE,
                                                   cv::Size(kernel_size, kernel_size));
        cv::erode(before_inflate, free_map_, kernel);
    }

    cv::bitwise_xor(before_inflate, free_map_, mask);
}

bool Astar::IsInside(const cv::Point& point) const {
    return point.x >= 0 && point.x < free_map_.cols && point.y >= 0 && point.y < free_map_.rows;
}

bool Astar::IsFree(const cv::Point& point) const {
    return free_map_.at<unsigned char>(point.y, point.x) > 0;
}

int Astar::PointToIndex(const cv::Point& point) const {
    return point.y * free_map_.cols + point.x;
}

cv::Point Astar::IndexToPoint(int index) const {
    return cv::Point(index % free_map_.cols, index / free_map_.cols);
}

int Astar::StepHeuristic(const cv::Point& point) const {
    const int dx = std::abs(point.x - target_point_.x);
    const int dy = std::abs(point.y - target_point_.y);
    if (config_.use_chebyshev && config_.allow_diagonal) {
        return std::max(dx, dy);
    }
    return dx + dy;
}

int Astar::LineBias(const cv::Point& point) const {
    const int line_dx = target_point_.x - start_point_.x;
    const int line_dy = target_point_.y - start_point_.y;
    const int px = point.x - start_point_.x;
    const int py = point.y - start_point_.y;
    return std::abs(line_dx * py - line_dy * px);
}

int Astar::GeometryMoveCost(const cv::Point& from, const cv::Point& to) const {
    const int dx = std::abs(from.x - to.x);
    const int dy = std::abs(from.y - to.y);
    return (dx == 1 && dy == 1) ? 14 : 10;
}

int Astar::TurnPenalty(int parent_index, const cv::Point& current_point,
                       const cv::Point& next_point) const {
    if (parent_index < 0) {
        return 0;
    }

    const cv::Point parent_point = IndexToPoint(parent_index);
    const cv::Point previous_direction = current_point - parent_point;
    const cv::Point next_direction = next_point - current_point;
    return previous_direction == next_direction ? 0 : 1;
}

bool Astar::ReconstructPath(int target_index, std::vector<cv::Point>& path) const {
    path.clear();
    int current_index = target_index;
    while (current_index != -1) {
        path.push_back(IndexToPoint(current_index));
        current_index = search_nodes_[current_index].parent;
    }
    std::reverse(path.begin(), path.end());
    return !path.empty() && path.front() == start_point_ && path.back() == target_point_;
}

bool Astar::HasLineOfSight(const cv::Point& start_point, const cv::Point& end_point) const {
    cv::LineIterator iterator(free_map_, start_point, end_point, 8);
    for (int i = 0; i < iterator.count; ++i, ++iterator) {
        const cv::Point point = iterator.pos();
        if (!IsInside(point) || !IsFree(point)) {
            return false;
        }
    }
    return true;
}

void Astar::SimplifyPath(std::vector<cv::Point>& path) const {
    if (path.size() < 3) {
        return;
    }

    std::vector<cv::Point> simplified_path;
    simplified_path.push_back(path.front());

    int anchor_index = 0;
    while (anchor_index < static_cast<int>(path.size()) - 1) {
        int furthest_index = anchor_index + 1;
        for (int candidate_index = anchor_index + 2; candidate_index < static_cast<int>(path.size());
             ++candidate_index) {
            if (!HasLineOfSight(path[anchor_index], path[candidate_index])) {
                break;
            }
            furthest_index = candidate_index;
        }
        simplified_path.push_back(path[furthest_index]);
        anchor_index = furthest_index;
    }

    path.swap(simplified_path);
}

}  // namespace pathplanning
