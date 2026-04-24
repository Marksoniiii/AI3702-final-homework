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

    if (traversable_map_.empty() || !IsInside(start_point_) || !IsInside(target_point_) ||
        !IsFree(start_point_) || !IsFree(target_point_)) {
        return false;
    }

    search_nodes_.assign(traversable_map_.rows * traversable_map_.cols, SearchNode());
    std::priority_queue<QueueNode, std::vector<QueueNode>, std::greater<QueueNode>> open_list;

    const int start_index = PointToIndex(start_point_);
    SearchNode& start_node = search_nodes_[start_index];
    start_node.step_cost = 0;
    start_node.geometry_cost = 0;
    start_node.turn_cost = 0;

    const int start_h_step = SearchHeuristic(start_point_);
    open_list.push(
        {start_h_step, start_h_step, 0, 0, LineBias(start_point_), start_index});

    while (!open_list.empty()) {
        const QueueNode current_queue_node = open_list.top();
        open_list.pop();

        SearchNode& current_node = search_nodes_[current_queue_node.index];
        const cv::Point current_point = IndexToPoint(current_queue_node.index);
        const int expected_f_step = current_node.step_cost + SearchHeuristic(current_point);
        if (current_node.closed || current_queue_node.f_step_cost != expected_f_step) {
            continue;
        }
        current_node.closed = true;

        if (current_point == target_point_) {
            std::vector<cv::Point> raw_path;
            const bool ok = ReconstructPath(current_queue_node.index, raw_path);
            if (!ok) {
                return false;
            }

            last_grid_steps_ = std::max(0, static_cast<int>(raw_path.size()) - 1);
            if (config_.use_corner_smoothing) {
                SmoothPathCorners(raw_path, path);
            } else {
                path = raw_path;
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

            int tentative_step_cost;
            int tentative_geometry_cost;
            int tentative_turn_cost;
            int tentative_parent_index = current_queue_node.index;

            if (config_.use_theta_star && current_node.parent >= 0) {
                const cv::Point parent_point = IndexToPoint(current_node.parent);
                if (HasLineOfSight(parent_point, next_point)) {
                    const SearchNode& parent_node = search_nodes_[current_node.parent];
                    tentative_step_cost =
                        parent_node.step_cost + LineTraversalCost(parent_point, next_point);
                    tentative_geometry_cost = tentative_step_cost;
                    tentative_turn_cost =
                        parent_node.turn_cost +
                        TurnPenalty(parent_node.parent, parent_point, next_point);
                    tentative_parent_index = current_node.parent;
                } else {
                    tentative_step_cost =
                        current_node.step_cost + LineTraversalCost(current_point, next_point);
                    tentative_geometry_cost = tentative_step_cost;
                    tentative_turn_cost =
                        current_node.turn_cost +
                        TurnPenalty(current_node.parent, current_point, next_point);
                }
            } else {
                tentative_step_cost =
                    current_node.step_cost + GeometryMoveCost(current_point, next_point);
                tentative_geometry_cost =
                    current_node.geometry_cost + GeometryMoveCost(current_point, next_point) +
                    PointPenalty(next_point);
                tentative_turn_cost =
                    current_node.turn_cost +
                    TurnPenalty(current_node.parent, current_point, next_point);
            }

            if (!LexicographicallyBetter(tentative_step_cost, tentative_geometry_cost,
                                         tentative_turn_cost, next_node.step_cost,
                                         next_node.geometry_cost, next_node.turn_cost)) {
                continue;
            }

            next_node.step_cost = tentative_step_cost;
            next_node.geometry_cost = tentative_geometry_cost;
            next_node.turn_cost = tentative_turn_cost;
            next_node.parent = tentative_parent_index;

            const int next_h_step = SearchHeuristic(next_point);
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

    cv::Mat free_map_before_inflate;
    if (config_.occupy_thresh < 0) {
        cv::threshold(gray_map, free_map_before_inflate, 0, 255,
                      cv::THRESH_BINARY | cv::THRESH_OTSU);
    } else {
        cv::threshold(gray_map, free_map_before_inflate, config_.occupy_thresh, 255,
                      cv::THRESH_BINARY);
    }

    cv::Mat hard_safe_map = free_map_before_inflate.clone();
    cv::Mat total_safe_map = free_map_before_inflate.clone();

    const int total_inflate_radius = std::max(0, config_.inflate_radius);
    const int soft_band_cells = std::max(0, config_.soft_inflation_band_cells);
    const int hard_inflate_radius =
        config_.use_soft_inflation_cost ? std::max(0, total_inflate_radius - soft_band_cells)
                                        : total_inflate_radius;

    if (hard_inflate_radius > 0) {
        const int kernel_size = 2 * hard_inflate_radius + 1;
        const cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE,
                                                         cv::Size(kernel_size, kernel_size));
        cv::erode(free_map_before_inflate, hard_safe_map, kernel);
    }

    if (total_inflate_radius > 0) {
        const int kernel_size = 2 * total_inflate_radius + 1;
        const cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE,
                                                         cv::Size(kernel_size, kernel_size));
        cv::erode(free_map_before_inflate, total_safe_map, kernel);
    }

    cv::bitwise_xor(free_map_before_inflate, total_safe_map, mask);

    soft_inflation_mask_ = cv::Mat::zeros(mask.size(), CV_8UC1);
    if (config_.use_soft_inflation_cost && total_inflate_radius > hard_inflate_radius) {
        cv::Mat band_mask;
        cv::bitwise_xor(hard_safe_map, total_safe_map, band_mask);
        cv::bitwise_and(hard_safe_map, band_mask, soft_inflation_mask_);
    }

    clearance_distance_map_ = cv::Mat::zeros(free_map_before_inflate.size(), CV_32FC1);
    if (config_.use_clearance_cost || config_.use_inflation_potential_cost) {
        cv::distanceTransform(free_map_before_inflate, clearance_distance_map_, cv::DIST_L2, 3);
    }

    search_cost_map_ = cv::Mat();
    if (config_.use_theta_star && !clearance_distance_map_.empty() &&
        config_.map_resolution > 0.0) {
        search_cost_map_ =
            cv::Mat(free_map_before_inflate.size(), CV_8UC1, cv::Scalar(255));

        const double lethal_radius_cells =
            static_cast<double>(std::max(hard_inflate_radius,
                                         hard_inflate_radius +
                                             std::max(0, config_.theta_hard_margin_cells)));
        const double inflation_radius_cells =
            std::max(lethal_radius_cells + 1.0,
                     config_.inflation_potential_radius / config_.map_resolution);
        const double inflation_band =
            std::max(1e-6, inflation_radius_cells - lethal_radius_cells);
        const double scaling = std::max(1e-6, config_.inflation_potential_scaling_factor);
        const double exp_denominator = std::exp(scaling) - 1.0;

        for (int row = 0; row < search_cost_map_.rows; ++row) {
            for (int col = 0; col < search_cost_map_.cols; ++col) {
                if (free_map_before_inflate.at<unsigned char>(row, col) == 0) {
                    search_cost_map_.at<unsigned char>(row, col) = 0;
                    continue;
                }

                const double clearance = clearance_distance_map_.at<float>(row, col);
                if (clearance <= lethal_radius_cells) {
                    search_cost_map_.at<unsigned char>(row, col) = 0;
                    continue;
                }
                if (clearance >= inflation_radius_cells) {
                    search_cost_map_.at<unsigned char>(row, col) = 255;
                    continue;
                }

                const double normalized = std::max(
                    0.0, std::min(1.0, (clearance - lethal_radius_cells) / inflation_band));
                const double shaped = (std::exp(scaling * normalized) - 1.0) / exp_denominator;
                const int traversability =
                    std::max(1, std::min(254, static_cast<int>(std::round(254.0 * shaped))));
                search_cost_map_.at<unsigned char>(row, col) =
                    static_cast<unsigned char>(traversability);
            }
        }
    }

    if (!search_cost_map_.empty()) {
        traversable_map_ = search_cost_map_.clone();
    } else if (config_.use_soft_inflation_cost) {
        traversable_map_ = hard_safe_map.clone();
    } else {
        traversable_map_ = total_safe_map.clone();
    }
}

bool Astar::IsInside(const cv::Point& point) const {
    return point.x >= 0 && point.x < traversable_map_.cols && point.y >= 0 &&
           point.y < traversable_map_.rows;
}

bool Astar::IsFree(const cv::Point& point) const {
    const int value = traversable_map_.at<unsigned char>(point.y, point.x);
    return value > 0;
}

bool Astar::IsSoftInflationCell(const cv::Point& point) const {
    return !soft_inflation_mask_.empty() &&
           soft_inflation_mask_.at<unsigned char>(point.y, point.x) > 0;
}

int Astar::ClearancePenalty(const cv::Point& point) const {
    if (!config_.use_clearance_cost || clearance_distance_map_.empty()) {
        return 0;
    }

    const float clearance = clearance_distance_map_.at<float>(point.y, point.x);
    const int radius = std::max(0, config_.clearance_cost_radius_cells);
    if (radius <= 0 || clearance >= static_cast<float>(radius)) {
        return 0;
    }

    const float shortage = static_cast<float>(radius) - clearance;
    return static_cast<int>(std::round(shortage * config_.clearance_cost_weight));
}

int Astar::InflationPotentialPenalty(const cv::Point& point) const {
    if (!config_.use_inflation_potential_cost || clearance_distance_map_.empty() ||
        config_.inflation_potential_radius <= 0.0 || config_.map_resolution <= 0.0) {
        return 0;
    }

    const double clearance_m =
        static_cast<double>(clearance_distance_map_.at<float>(point.y, point.x)) *
        config_.map_resolution;
    if (clearance_m >= config_.inflation_potential_radius) {
        return 0;
    }

    const double normalized =
        std::exp(-config_.inflation_potential_scaling_factor * clearance_m);
    return static_cast<int>(
        std::round(config_.inflation_potential_weight * normalized));
}

int Astar::PointPenalty(const cv::Point& point) const {
    if (!search_cost_map_.empty()) {
        const int traversability = search_cost_map_.at<unsigned char>(point.y, point.x);
        return ((255 - traversability) * 8) / 16;
    }

    return (config_.use_soft_inflation_cost && IsSoftInflationCell(point)
                ? config_.soft_inflation_penalty
                : 0) +
           ClearancePenalty(point) + InflationPotentialPenalty(point);
}

bool Astar::HasLineOfSight(const cv::Point& from, const cv::Point& to) const {
    cv::LineIterator iterator(traversable_map_, from, to, 8);
    const float theta_clearance_threshold =
        static_cast<float>(std::max(0, config_.theta_line_check_radius_cells));
    for (int i = 0; i < iterator.count; ++i, ++iterator) {
        const cv::Point point = iterator.pos();
        if (!IsInside(point) || !IsFree(point)) {
            return false;
        }
        if (config_.use_theta_star && !clearance_distance_map_.empty() &&
            clearance_distance_map_.at<float>(point.y, point.x) <= theta_clearance_threshold) {
            return false;
        }
        if (!search_cost_map_.empty() &&
            search_cost_map_.at<unsigned char>(point.y, point.x) <
                std::max(1, config_.theta_min_traversability)) {
            return false;
        }
    }
    return true;
}

int Astar::LineTraversalCost(const cv::Point& from, const cv::Point& to) const {
    const int dx = from.x - to.x;
    const int dy = from.y - to.y;
    const int base_cost =
        static_cast<int>(std::round(std::sqrt(dx * dx + dy * dy) * 10.0));

    cv::LineIterator iterator(traversable_map_, from, to, 8);
    int penalty_sum = 0;
    int samples = 0;
    for (int i = 0; i < iterator.count; ++i, ++iterator) {
        const cv::Point point = iterator.pos();
        penalty_sum += PointPenalty(point);
        ++samples;
    }

    const int average_penalty = samples > 0 ? penalty_sum / samples : 0;
    return base_cost + average_penalty;
}

int Astar::SearchHeuristic(const cv::Point& point) const {
    if (config_.use_theta_star) {
        const int dx = point.x - target_point_.x;
        const int dy = point.y - target_point_.y;
        return static_cast<int>(std::round(std::sqrt(dx * dx + dy * dy) * 10.0));
    }
    return StepHeuristic(point);
}

bool Astar::SegmentIsFree(const cv::Point& from, const cv::Point& to) const {
    return HasLineOfSight(from, to);
}

bool Astar::BuildRoundedCorner(const cv::Point& previous, const cv::Point& corner,
                               const cv::Point& next,
                               std::vector<cv::Point>& corner_points) const {
    corner_points.clear();

    const cv::Point dir_in = corner - previous;
    const cv::Point dir_out = next - corner;
    if (dir_in == dir_out) {
        return false;
    }

    if ((dir_in.x != 0 && dir_in.y != 0) || (dir_out.x != 0 && dir_out.y != 0)) {
        return false;
    }

    const int smooth_cells = std::max(1, config_.corner_smoothing_cells);
    cv::Point best_late_turn = corner;
    cv::Point best_exit = corner;
    int best_keep_straight = 0;
    int best_drop = 0;

    for (int keep_straight = smooth_cells; keep_straight >= 1; --keep_straight) {
        const cv::Point late_turn = corner + dir_in * keep_straight;
        if (!IsInside(late_turn) || !IsFree(late_turn) || !SegmentIsFree(corner, late_turn)) {
            continue;
        }

        for (int drop = smooth_cells; drop >= 1; --drop) {
            const cv::Point exit = corner + dir_out * drop;
            if (!IsInside(exit) || !IsFree(exit)) {
                continue;
            }
            if (!SegmentIsFree(late_turn, exit)) {
                continue;
            }

            if (keep_straight > best_keep_straight ||
                (keep_straight == best_keep_straight && drop > best_drop)) {
                best_keep_straight = keep_straight;
                best_drop = drop;
                best_late_turn = late_turn;
                best_exit = exit;
            }
        }
    }

    if (best_keep_straight == 0 || best_drop == 0) {
        return false;
    }

    std::vector<cv::Point> samples;
    samples.push_back(corner);
    if (samples.back() != best_late_turn) {
        samples.push_back(best_late_turn);
    }

    cv::LineIterator iterator(traversable_map_, best_late_turn, best_exit, 8);
    for (int i = 1; i < iterator.count; ++i) {
        ++iterator;
        const cv::Point point = iterator.pos();
        if (samples.back() != point) {
            samples.push_back(point);
        }
    }
    if (samples.back() != best_exit) {
        samples.push_back(best_exit);
    }

    for (const cv::Point& point : samples) {
        if (!IsInside(point) || !IsFree(point)) {
            return false;
        }
    }
    for (size_t i = 1; i < samples.size(); ++i) {
        if (!SegmentIsFree(samples[i - 1], samples[i])) {
            return false;
        }
    }

    corner_points = samples;
    return corner_points.size() >= 2;
}

void Astar::SmoothPathCorners(const std::vector<cv::Point>& raw_path,
                              std::vector<cv::Point>& smoothed_path) const {
    smoothed_path.clear();
    if (raw_path.size() <= 2) {
        smoothed_path = raw_path;
        return;
    }

    smoothed_path.push_back(raw_path.front());
    for (size_t i = 1; i + 1 < raw_path.size(); ++i) {
        const cv::Point& previous = raw_path[i - 1];
        const cv::Point& corner = raw_path[i];
        const cv::Point& next = raw_path[i + 1];

        std::vector<cv::Point> rounded_corner;
        if (BuildRoundedCorner(previous, corner, next, rounded_corner)) {
            if (smoothed_path.back() == rounded_corner.front()) {
                rounded_corner.erase(rounded_corner.begin());
            }
            smoothed_path.insert(smoothed_path.end(), rounded_corner.begin(),
                                 rounded_corner.end());
        } else if (smoothed_path.back() != corner) {
            smoothed_path.push_back(corner);
        }
    }

    if (smoothed_path.back() != raw_path.back()) {
        smoothed_path.push_back(raw_path.back());
    }
}

int Astar::PointToIndex(const cv::Point& point) const {
    return point.y * traversable_map_.cols + point.x;
}

cv::Point Astar::IndexToPoint(int index) const {
    return cv::Point(index % traversable_map_.cols, index / traversable_map_.cols);
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
    if (previous_direction == next_direction) {
        return 0;
    }

    int penalty = 1;
    if (config_.use_turn_clearance_penalty && !clearance_distance_map_.empty()) {
        const float clearance = clearance_distance_map_.at<float>(current_point.y, current_point.x);
        const int radius = std::max(0, config_.turn_clearance_radius_cells);
        if (radius > 0 && clearance < static_cast<float>(radius)) {
            const float shortage = static_cast<float>(radius) - clearance;
            penalty += static_cast<int>(
                std::round(shortage * config_.turn_clearance_weight));
        }
    }

    if (config_.use_deferred_turn_penalty) {
        const int lookahead = std::max(0, config_.deferred_turn_lookahead_cells);
        int free_ahead = 0;
        for (int step = 1; step <= lookahead; ++step) {
            const cv::Point candidate = current_point + previous_direction * step;
            if (!IsInside(candidate) || !IsFree(candidate)) {
                break;
            }
            ++free_ahead;
        }
        penalty += free_ahead * std::max(0, config_.deferred_turn_weight);
    }
    return penalty;
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

}  // namespace pathplanning
