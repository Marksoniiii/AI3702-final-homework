#ifndef ASTAR_H
#define ASTAR_H

#include <opencv2/opencv.hpp>
#include <queue>
#include <utility>
#include <vector>

namespace pathplanning {

struct AstarConfig {
    bool allow_diagonal;
    bool use_chebyshev;
    int occupy_thresh;
    int inflate_radius;
    bool use_soft_inflation_cost;
    int soft_inflation_penalty;
    int soft_inflation_band_cells;
    bool use_clearance_cost;
    int clearance_cost_radius_cells;
    int clearance_cost_weight;
    bool use_inflation_potential_cost;
    double inflation_potential_radius;
    double inflation_potential_scaling_factor;
    int inflation_potential_weight;
    double map_resolution;
    bool use_corner_smoothing;
    int corner_smoothing_cells;
    bool use_turn_clearance_penalty;
    int turn_clearance_radius_cells;
    int turn_clearance_weight;
    bool use_deferred_turn_penalty;
    int deferred_turn_lookahead_cells;
    int deferred_turn_weight;
    bool use_theta_star;
    int theta_hard_margin_cells;
    int theta_line_check_radius_cells;
    int theta_min_traversability;

    AstarConfig(bool allow_diagonal_ = true, bool use_chebyshev_ = true,
                int occupy_thresh_ = 50, int inflate_radius_ = 0,
                bool use_soft_inflation_cost_ = false,
                int soft_inflation_penalty_ = 200,
                int soft_inflation_band_cells_ = 1,
                bool use_clearance_cost_ = false,
                int clearance_cost_radius_cells_ = 4,
                int clearance_cost_weight_ = 20,
                bool use_inflation_potential_cost_ = false,
                double inflation_potential_radius_ = 1.0,
                double inflation_potential_scaling_factor_ = 3.0,
                int inflation_potential_weight_ = 120,
                double map_resolution_ = 0.05,
                bool use_corner_smoothing_ = false,
                int corner_smoothing_cells_ = 2,
                bool use_turn_clearance_penalty_ = false,
                int turn_clearance_radius_cells_ = 6,
                int turn_clearance_weight_ = 40,
                bool use_deferred_turn_penalty_ = false,
                int deferred_turn_lookahead_cells_ = 6,
                int deferred_turn_weight_ = 6,
                bool use_theta_star_ = false,
                int theta_hard_margin_cells_ = 1,
                int theta_line_check_radius_cells_ = 3,
                int theta_min_traversability_ = 120)
        : allow_diagonal(allow_diagonal_),
          use_chebyshev(use_chebyshev_),
          occupy_thresh(occupy_thresh_),
          inflate_radius(inflate_radius_),
          use_soft_inflation_cost(use_soft_inflation_cost_),
          soft_inflation_penalty(soft_inflation_penalty_),
          soft_inflation_band_cells(soft_inflation_band_cells_),
          use_clearance_cost(use_clearance_cost_),
          clearance_cost_radius_cells(clearance_cost_radius_cells_),
          clearance_cost_weight(clearance_cost_weight_),
          use_inflation_potential_cost(use_inflation_potential_cost_),
          inflation_potential_radius(inflation_potential_radius_),
          inflation_potential_scaling_factor(inflation_potential_scaling_factor_),
          inflation_potential_weight(inflation_potential_weight_),
          map_resolution(map_resolution_),
          use_corner_smoothing(use_corner_smoothing_),
          corner_smoothing_cells(corner_smoothing_cells_),
          use_turn_clearance_penalty(use_turn_clearance_penalty_),
          turn_clearance_radius_cells(turn_clearance_radius_cells_),
          turn_clearance_weight(turn_clearance_weight_),
          use_deferred_turn_penalty(use_deferred_turn_penalty_),
          deferred_turn_lookahead_cells(deferred_turn_lookahead_cells_),
          deferred_turn_weight(deferred_turn_weight_),
          use_theta_star(use_theta_star_),
          theta_hard_margin_cells(theta_hard_margin_cells_),
          theta_line_check_radius_cells(theta_line_check_radius_cells_),
          theta_min_traversability(theta_min_traversability_) {}
};

class Astar {
public:
    void InitAstar(const cv::Mat& map, AstarConfig config = AstarConfig());
    void InitAstar(const cv::Mat& map, cv::Mat& mask, AstarConfig config = AstarConfig());
    bool PathPlanning(const cv::Point& start_point, const cv::Point& target_point,
                      std::vector<cv::Point>& path);
    int GetLastGridSteps() const;
    void DrawPath(cv::Mat& map, const std::vector<cv::Point>& path,
                  cv::InputArray mask = cv::noArray(),
                  cv::Scalar color = cv::Scalar(0, 0, 255), int thickness = 1,
                  cv::Scalar mask_color = cv::Scalar(255, 255, 255));

private:
    static constexpr int kInfCost = 1000000000;

    struct SearchNode {
        int step_cost;
        int geometry_cost;
        int turn_cost;
        int parent;
        bool closed;

        SearchNode()
            : step_cost(kInfCost),
              geometry_cost(kInfCost),
              turn_cost(kInfCost),
              parent(-1),
              closed(false) {}
    };

    struct QueueNode {
        int f_step_cost;
        int h_step_cost;
        int geometry_cost;
        int turn_cost;
        int line_bias;
        int index;

        bool operator>(const QueueNode& other) const;
    };

    void MapProcess(cv::Mat& mask);
    bool IsInside(const cv::Point& point) const;
    bool IsFree(const cv::Point& point) const;
    bool IsSoftInflationCell(const cv::Point& point) const;
    int ClearancePenalty(const cv::Point& point) const;
    int InflationPotentialPenalty(const cv::Point& point) const;
    int PointPenalty(const cv::Point& point) const;
    bool HasLineOfSight(const cv::Point& from, const cv::Point& to) const;
    int LineTraversalCost(const cv::Point& from, const cv::Point& to) const;
    int SearchHeuristic(const cv::Point& point) const;
    bool SegmentIsFree(const cv::Point& from, const cv::Point& to) const;
    bool BuildRoundedCorner(const cv::Point& previous, const cv::Point& corner,
                            const cv::Point& next, std::vector<cv::Point>& corner_points) const;
    void SmoothPathCorners(const std::vector<cv::Point>& raw_path,
                           std::vector<cv::Point>& smoothed_path) const;
    int PointToIndex(const cv::Point& point) const;
    cv::Point IndexToPoint(int index) const;
    int StepHeuristic(const cv::Point& point) const;
    int LineBias(const cv::Point& point) const;
    int GeometryMoveCost(const cv::Point& from, const cv::Point& to) const;
    int TurnPenalty(int parent_index, const cv::Point& current_point,
                    const cv::Point& next_point) const;
    bool ReconstructPath(int target_index, std::vector<cv::Point>& path) const;

    cv::Mat raw_map_;
    cv::Mat traversable_map_;
    cv::Mat search_cost_map_;
    cv::Mat soft_inflation_mask_;
    cv::Mat clearance_distance_map_;
    cv::Point start_point_;
    cv::Point target_point_;
    std::vector<cv::Point> neighbors_;
    std::vector<SearchNode> search_nodes_;
    AstarConfig config_;
    int last_grid_steps_ = 0;
};

}  // namespace pathplanning

#endif  // ASTAR_H
