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
    bool enable_path_simplify;

    AstarConfig(bool allow_diagonal_ = true, bool use_chebyshev_ = true,
                int occupy_thresh_ = 50, int inflate_radius_ = 0,
                bool enable_path_simplify_ = true)
        : allow_diagonal(allow_diagonal_),
          use_chebyshev(use_chebyshev_),
          occupy_thresh(occupy_thresh_),
          inflate_radius(inflate_radius_),
          enable_path_simplify(enable_path_simplify_) {}
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
    int PointToIndex(const cv::Point& point) const;
    cv::Point IndexToPoint(int index) const;
    int StepHeuristic(const cv::Point& point) const;
    int LineBias(const cv::Point& point) const;
    int GeometryMoveCost(const cv::Point& from, const cv::Point& to) const;
    int TurnPenalty(int parent_index, const cv::Point& current_point,
                    const cv::Point& next_point) const;
    bool ReconstructPath(int target_index, std::vector<cv::Point>& path) const;
    bool HasLineOfSight(const cv::Point& start_point, const cv::Point& end_point) const;
    void SimplifyPath(std::vector<cv::Point>& path) const;

    cv::Mat raw_map_;
    cv::Mat free_map_;
    cv::Point start_point_;
    cv::Point target_point_;
    std::vector<cv::Point> neighbors_;
    std::vector<SearchNode> search_nodes_;
    AstarConfig config_;
    int last_grid_steps_ = 0;
};

}  // namespace pathplanning

#endif  // ASTAR_H
