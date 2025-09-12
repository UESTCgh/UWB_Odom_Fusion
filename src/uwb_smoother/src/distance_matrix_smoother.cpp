#include <ros/ros.h>
#include <std_msgs/Float32MultiArray.h>
#include <Eigen/Dense>
#include <deque>
#include <vector>
#include <cmath>
#include <algorithm>

using namespace std;
using namespace Eigen;

static const int NUM_COLUMNS = 16;   // 每帧数据长度
static const int FIT_WINDOW  = 20;   // 拟合窗口长度
static const int OVERLAP     = 10;   // 窗口重叠
static const double JUMP_RATIO = 0.10; // 跳变判据(10%)

struct Smoother {
  // 环形缓冲
  deque<double> ts;
  vector< deque<double> > cols;

  // 输出（最近一帧）
  vector<double> last_smoothed;

  Smoother(): cols(NUM_COLUMNS), last_smoothed(NUM_COLUMNS, 0.0) {}

  // 三阶拟合
  Vector4d polyfit(const vector<double>& x, const vector<double>& y){
    int N = (int)x.size();
    MatrixXd A(N,4);
    VectorXd Y(N);
    for (int i=0;i<N;++i){
      A(i,0)=pow(x[i],3); A(i,1)=pow(x[i],2); A(i,2)=x[i]; A(i,3)=1.0;
      Y(i)=y[i];
    }
    return A.colPivHouseholderQr().solve(Y);
  }

  vector<double> polyval(const Vector4d& c, const vector<double>& xs){
    vector<double> ys; ys.reserve(xs.size());
    for(double x: xs) ys.push_back(c[0]*x*x*x + c[1]*x*x + c[2]*x + c[3]);
    return ys;
  }

  // 进一帧原始数据（带简单跳变抑制）
  void push(double t, const vector<double>& data){
    if ((int)data.size() < NUM_COLUMNS) return;

    ts.push_back(t);
    for (int j=0;j<NUM_COLUMNS;++j){
      double v = data[j];
      // 跳变修复：与上一帧相差超过10%则回填上一值
      if (!cols[j].empty()){
        double prev = cols[j].back();
        if (fabs(prev) > 1e-6){
          double ratio = fabs((v - prev)/prev);
          if (ratio > JUMP_RATIO) v = prev;
        }
      }
      cols[j].push_back(v);
    }

    // 只保留必要长度（窗口 + 重叠 + 若干冗余）
    const size_t MAX_KEEP = FIT_WINDOW + OVERLAP + 10;
    if (ts.size() > MAX_KEEP){
      ts.pop_front();
      for (auto& q: cols) q.pop_front();
    }
  }

  // 计算“最近 FIT_WINDOW 段”的平滑输出
  bool computeSmooth(){
    if (ts.size() < (size_t)FIT_WINDOW) return false;

    size_t end = ts.size();
    size_t start = (end >= (size_t)(FIT_WINDOW+OVERLAP)) ? end - (FIT_WINDOW+OVERLAP) : 0;
    vector<double> tseg(ts.begin()+start, ts.end());

    // 我们只关心“最后 FIT_WINDOW 个点”的拟合输出
    size_t fit_start = (end >= (size_t)FIT_WINDOW) ? end - FIT_WINDOW : 0;
    vector<double> tfit(ts.begin()+fit_start, ts.end());

    for (int j=0;j<NUM_COLUMNS;++j){
      vector<double> yseg(cols[j].begin()+start, cols[j].end());
      Vector4d c = polyfit(tseg, yseg);
      auto yfit = polyval(c, tfit);
      last_smoothed[j] = yfit.back(); // 取“当前时刻”的平滑值
    }
    return true;
  }
};

int main(int argc, char** argv){
  ros::init(argc, argv, "distance_matrix_smoother");
  ros::NodeHandle nh("~");

  string sub_topic = "/uwb1/distance_matrix";
  string pub_topic = "/uwb1/distance_matrix_smoothed";
  nh.param("sub_topic", sub_topic, sub_topic);
  nh.param("pub_topic", pub_topic, pub_topic);

  Smoother smoother;
  ros::Publisher pub = nh.advertise<std_msgs::Float32MultiArray>(pub_topic, 10);

  auto cb = [&](const std_msgs::Float32MultiArray::ConstPtr& msg){
    // 时间戳：用系统时间或消息 header（若有 header）
    double t = ros::Time::now().toSec();

    // 拆帧：前 NUM_COLUMNS 个值
    vector<double> data;
    data.reserve(NUM_COLUMNS);
    for (int i=0;i<NUM_COLUMNS && i<(int)msg->data.size(); ++i)
      data.push_back(msg->data[i]);

    smoother.push(t, data);

    if (smoother.computeSmooth()){
      std_msgs::Float32MultiArray out;
      out.data.assign(smoother.last_smoothed.begin(), smoother.last_smoothed.end());
      pub.publish(out);
    }
  };

  ros::Subscriber sub = nh.subscribe<std_msgs::Float32MultiArray>(sub_topic, 50, cb);

  ROS_INFO_STREAM("Subscribing: " << sub_topic << "  Publishing: " << pub_topic);
  ros::spin();
  return 0;
}
