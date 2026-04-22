#ifndef ASTAR_ASTAR_ALGO_H
#define ASTAR_ASTAR_ALGO_H

#include <iostream>
#include <queue>
#include <vector>

using namespace std;

template<class T>
/* T is a mappoint class*/
class Astar {
protected:

    struct vertex {
        T node;
        int pre;
        int g;
        bool flag;
        /* pre -1 means null and g -1 means inf*/
        vertex(T n) : node(move(n)), pre(-1), g(-1),flag(false) {}
    };

    double epilode;
    vector<vertex> vertices;
    vector<int> path;

    /* not implemented*/
    virtual double _compute_dist(int v1, int v2) { return 0; }

    virtual void _find_neighbor(int u, vector<int> &neighbor) {}

    void search_path(int start,int end);
    void _solve(int start, int end);
    void _reset();
public:
    Astar(double epi) : vertices(), epilode(epi) {}
    virtual void solve(int start,int end){_solve(start,end);}
    /* reset should not reset the path*/
    virtual void reset() {/* push the vertices*/}
    virtual void get_result(vector<int>& p){p.clear();for(const auto& pt:path)p.push_back(pt);}
};

template<class T>
void Astar<T>::_solve(int start, int end) {

    priority_queue<pair<double, int>,vector<pair<double,int>>,greater<pair<double,int>>> q;
    vertex &s = vertices[start];
    double f = _compute_dist(start, end);
    s.g = 0;
    q.push(pair<double, int>(0. + epilode * f, start));
    while (not q.empty()) {
        pair<double, int> v = q.top();
        q.pop();
        vertices[v.second].flag = true;/* mark close*/
        /* if reach the target , terminate */
        if (v.second == end)break;
        vector<int> eight_successors;
        _find_neighbor(v.second, eight_successors);
        for (const int &u:eight_successors) {
            int gu = vertices[v.second].g + 1;
            double fu = (double) gu + epilode * _compute_dist(u, end);
            if (!vertices[u].flag &&(gu < vertices[u].g || vertices[u].g == -1)) {
                /* -1 is not initialized,
                 * not flag is open
                 * when g>=0 and not flag, it's in the openlist*/
                vertices[u].g = gu;
                q.push(pair<double, int>(fu, u));
                vertices[u].pre = v.second;
            }
        }
    }
    search_path(start,end);
    _reset();
}

template<class T>
void Astar<T>::search_path(int start,int end) {
    path.clear();
    int pre = end;
    vector<int> inv_path;
    while (pre != -1) {
        inv_path.push_back(pre);
        pre = vertices[pre].pre;
    }
    if(inv_path[inv_path.size()-1]==start)
    {
        for (int i = (int) inv_path.size() - 1; i >= 0; --i)path.push_back(inv_path[i]);
    }
}

template<class T>
void Astar<T>::_reset() {
    for(vertex& p:vertices)
    {
        p.flag = false;
        p.g = -1;
        p.pre = -1;
    }
}

#endif //ASTAR_ASTAR_ALGO_H
