#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <iomanip>
#include <cmath>
#include <numeric>
#include <matplotlibcpp.h>
#include <Eigen/Dense>

namespace plt = matplotlibcpp;
using namespace std;
using namespace Eigen;

const int NUM_COLUMNS = 16;
const int FIT_WINDOW = 20;
const int OVERLAP = 10;

// 检查是否为跳变值（与前后点差异大）
bool isSpike(double prev, double curr, double next, double threshold = 5.0) {
    return (std::abs(curr - prev) > threshold && std::abs(curr - next) > threshold);
}

// 三阶拟合函数
Vector4d polyfit(const vector<double>& x, const vector<double>& y) {
    int N = x.size();
    MatrixXd A(N, 4);
    VectorXd Y(N);

    for (int i = 0; i < N; ++i) {
        A(i, 0) = pow(x[i], 3);
        A(i, 1) = pow(x[i], 2);
        A(i, 2) = x[i];
        A(i, 3) = 1.0;
        Y(i) = y[i];
    }

    Vector4d coeffs = A.colPivHouseholderQr().solve(Y);
    return coeffs;
}

// 使用拟合系数生成拟合值
vector<double> polyval(const Vector4d& coeffs, const vector<double>& x_vals) {
    vector<double> y_vals;
    for (double x : x_vals) {
        double y = coeffs[0]*pow(x,3) + coeffs[1]*pow(x,2) + coeffs[2]*x + coeffs[3];
        y_vals.push_back(y);
    }
    return y_vals;
}

int main() {
    ifstream file("distance_matrix_from_bag.txt");
    if (!file.is_open()) {
        cerr << "无法打开文件！" << endl;
        return -1;
    }

    vector<double> timestamps;
    vector<vector<double>> columns(NUM_COLUMNS);

    string line;
    while (getline(file, line)) {
        stringstream ss(line);
        double timestamp;
        ss >> timestamp;
        timestamps.push_back(timestamp);

        for (int i = 0; i < NUM_COLUMNS; ++i) {
            double value;
            if (ss >> value) {
                columns[i].push_back(value);
            } else {
                columns[i].push_back(0.0); // 不足补0
            }
        }
    }
    file.close();

    // 修复跳变值（若当前值比上一个值变化超过50%则修复）
    for (size_t i = 1; i < timestamps.size(); ++i) {
        for (int j = 0; j < NUM_COLUMNS; ++j) {
            double prev = columns[j][i - 1];
            double curr = columns[j][i];

            if (std::abs(prev) > 1e-6) { // 避免除以0
                double ratio = std::abs((curr - prev) / prev);
                if (ratio > 0.1) {
                    columns[j][i] = prev; // 替换为前一个值
                }
            }
        }
    }


    // 对每列做拟合+平滑
    vector<vector<double>> smoothed_columns(NUM_COLUMNS, vector<double>(timestamps.size(), 0.0));

    for (int j = 0; j < NUM_COLUMNS; ++j) {
        for (size_t i = 0; i + FIT_WINDOW <= timestamps.size(); i += FIT_WINDOW) {
            size_t start = (i >= OVERLAP) ? i - OVERLAP : 0;
            size_t end = min(start + FIT_WINDOW + OVERLAP, timestamps.size());

            vector<double> time_segment(timestamps.begin() + start, timestamps.begin() + end);
            vector<double> data_segment(columns[j].begin() + start, columns[j].begin() + end);

            Vector4d coeffs = polyfit(time_segment, data_segment);

            // 计算当前窗口的20个点
            vector<double> time_fit(timestamps.begin() + i, timestamps.begin() + i + FIT_WINDOW);
            vector<double> fitted = polyval(coeffs, time_fit);

            for (size_t k = 0; k < FIT_WINDOW; ++k) {
                smoothed_columns[j][i + k] = fitted[k];
            }
        }
    }

    // 画图
    plt::figure_size(1600, 800);
    for (int i = 0; i < NUM_COLUMNS; ++i) {
        plt::named_plot("smoothed_dim_" + to_string(i), timestamps, smoothed_columns[i]);
    }

    plt::xlabel("Time (s)");
    plt::ylabel("Distance");
    plt::legend();
    plt::title("Smoothed Distance Matrix Over Time");
    plt::grid(true);
    plt::show();

    return 0;
}