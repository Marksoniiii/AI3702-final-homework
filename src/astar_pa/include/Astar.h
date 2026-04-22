#ifndef ASTAR_H
#define ASTAR_H

#include <iostream>
#include <opencv2/opencv.hpp>
#include <cmath>
#include <nav_msgs/OccupancyGrid.h>
#include <tf/tf.h>
#include "astar_algo.h"
#include <mutex>

using namespace std;
using namespace cv;



struct MapParamNode {
    Point StartPoint, TargetPoint;
    Mat Rotation;
    Mat Translation;
    double resolution;
    int height;
    int width;
    int x;
    int y;

};

class GridAstar : public Astar<int8_t> {
private:
    MapParamNode *mapParam;
    Mat *Map;
protected:
    void _find_neighbor(int u, vector<int> &neighbor) override;

    double _compute_dist(int v1, int v2) override;

    void get_result(vector<Point> &p);

public:
    vector<Point> result;

    GridAstar(MapParamNode *m, double epi) : Astar<int8_t>(epi), mapParam(m),
                                             result(), Map(nullptr) {}

    void reset(Mat *map,int padding=3);

    void solve();
};


void GridAstar::reset(Mat *map, int padding) {
    if (not vertices.empty())vertices.clear();
    for (int i = 0; i < mapParam->height; i++) {
        for (int j = 0; j < mapParam->width; j++) {
            vertices.emplace_back(0);
        }
    }
    Map = map;

    Mat tmp = map->clone();
    Mat kernel = getStructuringElement(MORPH_ELLIPSE, Size(padding * 2, padding * 2));
    erode(tmp, *map, kernel);
}

void GridAstar::solve() {
    if (Map) {
        int start = mapParam->StartPoint.y * mapParam->width + mapParam->StartPoint.x;
        int end = mapParam->TargetPoint.y * mapParam->width + mapParam->TargetPoint.x;
        if (Map->at<uchar>(mapParam->StartPoint.x, mapParam->StartPoint.y) == 0) {
            ROS_INFO("source is not reachable");
            path.clear();
            result.clear();
            return;
        }
        if (Map->at<uchar>(mapParam->TargetPoint.x, mapParam->TargetPoint.y) == 0) {
            ROS_INFO("target is not reachable");
            path.clear();
            result.clear();
            return;
        }
        _solve(start, end);
        get_result(result);
    }
}

void GridAstar::_find_neighbor(int u, vector<int> &neighbor) {
    int h = u / mapParam->width;
    int w = u % mapParam->width;
    for (int i = -1; i <= 1; ++i) {
        for (int j = -1; j <= 1; ++j) {
            if ((i != 0 or j != 0) and h + i >= 0 and h + i < mapParam->height and w + j >= 0 and
                w + j < mapParam->width
                and Map->at<uchar>(w + j, h + i) == 255) {
                neighbor.emplace_back((h + i) * mapParam->width + w + j);
            }
        }
    }
}

double GridAstar::_compute_dist(int v1, int v2) {
    /* l2 distance as h(v1,v2)*/
    int h1 = v1 / mapParam->width;
    int w1 = v1 % mapParam->width;
    int h2 = v2 / mapParam->width;
    int w2 = v2 % mapParam->width;
    double dh = (double) (h1 - h2);
    double dw = (double) (w1 - w2);
    return sqrt(dh * dh + dw * dw);
}

void GridAstar::get_result(vector<Point> &p) {
    p.clear();
    for (const auto &pt:path) {
        p.emplace_back(pt % mapParam->width, pt / mapParam->width);
    }
}


#endif
