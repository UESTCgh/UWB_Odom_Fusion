#include <ros/ros.h>
#include <nlink_parser/LinktrackNodeframe2.h>
#include <nlink_parser/LinktrackNodeframe0.h>
#include <std_msgs/String.h>
#include <std_msgs/Float32MultiArray.h>
#include <nav_msgs/Odometry.h>
#include <geometry_msgs/PointStamped.h>
#include <tf/transform_datatypes.h>
#include <vector>
#include <set>
#include <sstream>
#include <mutex>
#include <map>
#include <cmath>
#include <deque>

//二进制传输
#include <std_msgs/UInt8MultiArray.h>

/*
话题发送实例：

1.发送goal点
rostopic pub -r 180 /uwb2/target_position geometry_msgs/PoseStamped "header:
  frame_id: 'map'
  stamp:
    secs: 0
    nsecs: 0
pose:
  position:
    x: 1.0
    y: -0.8
    z: 4.0
  orientation:
    x: 0.0
    y: 0.0
    z: 0.707
    w: 0.707"

订阅话题：rostopic echo /uwb3/pose_matrix 

2.发送odom点
rostopic pub -r 50 /odom2 nav_msgs/Odometry \
'{header: {stamp: now, frame_id: "map"}, child_frame_id: "base_link", pose: {pose: {position: {x: 1111111.0, y: 2.44, z: 0.0}, orientation: {x: 0.0, y: 0.0, z: 0.707, w: 0.707}}}, twist: {twist: {linear: {x: 0.0, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.0}}}}'

订阅话题：rostopic echo /uwb3/target_matrix 

3.发送坐标矩阵
rostopic pub -r 50 /uwb3/custom_matrix std_msgs/Float32MultiArray "layout:
  dim:
    - label: 'rows'
      size: 4
      stride: 8
    - label: 'cols'
      size: 2
      stride: 2
  data_offset: 0
data: [1.1, 1.2, 2.1, 2.2, 3.1, 3.2, 4.1, 422.2]"

订阅话题：rostopic echo /uwb2/matrix_from_uwb3
*/

class UWBNode {
private:
    ros::NodeHandle nh_;
    ros::NodeHandle private_nh_;

    int node_id_;
    int total_nodes_;
    int required_nodes_;

    ros::Publisher data_trans_pub_;
    ros::Publisher matrix_pub_;
    ros::Publisher pose_matrix_pub_;
    ros::Publisher target_matrix_pub_;

    ros::Subscriber frame2_sub_;
    ros::Subscriber nodeframe0_sub_;
    ros::Subscriber odom_sub_;
    ros::Subscriber target_pos_sub_;
    ros::Subscriber custom_matrix_sub_;

    ros::Timer cleanup_timer_;

    std::map<int, ros::Publisher> matrix_from_others_pub_;
    std::map<std::pair<int, int>, std::pair<float, ros::Time>> last_valid_dist_map_;

    std::vector<std::vector<float>> distance_matrix_;
    std::vector<std::vector<float>> pose_matrix_;
    std::vector<std::vector<float>> target_matrix_;  // 行：节点ID，列：(x, y, z)
    std::vector<std::vector<float>> custom_matrix_;  // 4行2列

    struct TimedMatrix {
        std::vector<std::vector<float>> matrix;
        ros::Time timestamp;
    };
    std::map<int, TimedMatrix> custom_matrix_map_;

    // 距离数据
    struct DistanceEntry {
        int source_id;
        int target_id;
        float distance;
        ros::Time timestamp;
    };

    // 目标点
    struct Target {
        int target_id;
        float x;
        float y;
        float z;
    };

    // Pose 数据
    struct PoseEntry {
        int id;
        float x;
        float y;
        float z;
        float yaw;
        ros::Time timestamp;
    };

    struct PoseInfo {
        float x = 0.0f, y = 0.0f, z = 0.0f, yaw = 0.0f;
    };
    PoseInfo self_pose_;
    PoseInfo target_pose_;


    struct FrameData {
        std::map<int, PoseInfo> poses;
        std::map<int, PoseInfo> targets;
        std::map<int, std::vector<std::vector<float>>> custom_matrices;
        std::vector<DistanceEntry> distances;
        std::set<int> received_nodes;
    };
    std::map<double, FrameData> frame_data_map_;
    std::mutex frame_data_mutex_;

    //存储UWB，用于时间同步
    struct CachedUWB {
        double timestamp;
        FrameData frame;
    };
    std::deque<CachedUWB> uwb_buffer_;
    size_t max_buffer_size_ = 100; // 缓存上限
    size_t avg_window = 4; 
    
    ros::Timer publish_timer_;
    ros::Timer print_timer_;
    double print_rate_;
    double matrix_publish_rate_;
    double publish_period_;
    double distance_diff_threshold_;

    int uwb_frame_count_ = 0;
    int uwb_publish_interval_ = 10;  // 每10帧发布一次，可设为参数

    std::mutex data_mutex_;
    std::string odom_topic_;
    std::string target_topic_;

public:
    UWBNode() : private_nh_("~") {
        // 参数
        private_nh_.param<int>("node_id", node_id_, 0);
        private_nh_.param<int>("total_nodes", total_nodes_, 6);
        private_nh_.param<int>("required_nodes", required_nodes_, 5);
        private_nh_.param<double>("matrix_print_rate", print_rate_, 1.0);
        private_nh_.param<double>("matrix_publish_rate", matrix_publish_rate_, 100.0);
        private_nh_.param<double>("distance_diff_threshold", distance_diff_threshold_, 0.5);
        private_nh_.param<std::string>("odom_topic", odom_topic_, "/odom");
        private_nh_.param<std::string>("target_topic", target_topic_, "/target_position");
        private_nh_.param<int>("uwb_publish_interval", uwb_publish_interval_, 10);

        publish_period_ = 1.0 / matrix_publish_rate_;

        distance_matrix_.resize(total_nodes_, std::vector<float>(total_nodes_, -1.0));
        pose_matrix_.resize(total_nodes_, std::vector<float>(4, 0.0f));
        target_matrix_.resize(total_nodes_, std::vector<float>(4, 0.0f));
        custom_matrix_.resize(total_nodes_, std::vector<float>(2, 0.0f));//4*2初始化


        cleanup_timer_ = nh_.createTimer(ros::Duration(2.0), &UWBNode::cleanupStaleData, this);

        //发布话题
        data_trans_pub_ = nh_.advertise<std_msgs::String>("/uwb" + std::to_string(node_id_) + "/data_transmission", 10);
        matrix_pub_ = nh_.advertise<std_msgs::Float32MultiArray>("/uwb" + std::to_string(node_id_) + "/distance_matrix", 10);
        pose_matrix_pub_ = nh_.advertise<std_msgs::Float32MultiArray>("/uwb" + std::to_string(node_id_) + "/pose_matrix", 10);
        target_matrix_pub_ = nh_.advertise<std_msgs::Float32MultiArray>("/uwb" + std::to_string(node_id_) + "/target_matrix", 10);

        for (int i = 0; i < total_nodes_; ++i) {
            if (i == node_id_) continue;  // 不给自己发布
            std::string topic_name = "/uwb" + std::to_string(node_id_) + "/matrix_from_uwb" + std::to_string(i);
            matrix_from_others_pub_[i] = nh_.advertise<std_msgs::Float32MultiArray>(topic_name, 10);
        }        

        //订阅话题
        custom_matrix_sub_ = nh_.subscribe<std_msgs::Float32MultiArray>(
            "/uwb" + std::to_string(node_id_) + "/custom_matrix", 10, &UWBNode::customMatrixCallback, this);        

        frame2_sub_ = nh_.subscribe<nlink_parser::LinktrackNodeframe2>(
            "/uwb" + std::to_string(node_id_) + "/nodeframe2", 10, &UWBNode::frame2Callback, this);

         //传感器 Tatget
        nodeframe0_sub_ = nh_.subscribe<nlink_parser::LinktrackNodeframe0>(
            "/uwb" + std::to_string(node_id_) + "/nodeframe0", 10, &UWBNode::parseNodeframe0, this);

        odom_sub_ = nh_.subscribe<nav_msgs::Odometry>(odom_topic_, 10, &UWBNode::odomCallback, this);
        target_pos_sub_ = nh_.subscribe<geometry_msgs::PoseStamped>(target_topic_, 10, &UWBNode::targetCallback, this);
        
        //回调发布
        print_timer_ = nh_.createTimer(ros::Duration(1.0 / print_rate_), &UWBNode::printMatrixCallback, this);
        publish_timer_ = nh_.createTimer(ros::Duration(publish_period_), &UWBNode::publishMatrixCallback, this);
    }

    void odomCallback(const nav_msgs::Odometry::ConstPtr& msg) {
        std::lock_guard<std::mutex> lock(data_mutex_);
        self_pose_.x = msg->pose.pose.position.x;
        self_pose_.y = msg->pose.pose.position.y;
        self_pose_.z = msg->pose.pose.position.z;

        tf::Quaternion q(
            msg->pose.pose.orientation.x,
            msg->pose.pose.orientation.y,
            msg->pose.pose.orientation.z,
            msg->pose.pose.orientation.w);
        double roll, pitch, yaw;
        tf::Matrix3x3(q).getRPY(roll, pitch, yaw);
        self_pose_.yaw = yaw;

        if (node_id_ < total_nodes_) {
            pose_matrix_[node_id_][0] = self_pose_.x;
            pose_matrix_[node_id_][1] = self_pose_.y;
            pose_matrix_[node_id_][2] = self_pose_.z;
            pose_matrix_[node_id_][3] = self_pose_.yaw;
        }
    }

    void targetCallback(const geometry_msgs::PoseStamped::ConstPtr& msg) {
        std::lock_guard<std::mutex> lock(data_mutex_);
        target_pose_.x = msg->pose.position.x;
        target_pose_.y = msg->pose.position.y;
        target_pose_.z = msg->pose.position.z;
        //获取yaw
        tf::Quaternion q(
            msg->pose.orientation.x,
            msg->pose.orientation.y,
            msg->pose.orientation.z,
            msg->pose.orientation.w);
        double roll, pitch, yaw;
        tf::Matrix3x3(q).getRPY(roll, pitch, yaw);
        target_pose_.yaw = yaw;
    
        if (node_id_ < total_nodes_) {
            target_matrix_[node_id_][0] = target_pose_.x;
            target_matrix_[node_id_][1] = target_pose_.y;
            target_matrix_[node_id_][2] = target_pose_.z;
            //取0
             target_matrix_[node_id_][3] = 0; 
            // target_matrix_[node_id_][3] = target_pose_.yaw;  // yaw
        }
    }

    void customMatrixCallback(const std_msgs::Float32MultiArray::ConstPtr& msg) {
        std::lock_guard<std::mutex> lock(data_mutex_);
    
        if (msg->data.size() != total_nodes_ * 2) {
            ROS_WARN("Received custom_matrix size %lu != 8", msg->data.size());
            return;
        }
    
        for (int i = 0; i < total_nodes_; ++i) {
            for (int j = 0; j < 2; ++j) {
                custom_matrix_[i][j] = msg->data[i * 2 + j];
            }
        }
    
        // ROS_INFO("Updated custom_matrix for node %d", node_id_);
    }
    
    //拼装
    void frame2Callback(const nlink_parser::LinktrackNodeframe2::ConstPtr& msg) {
        std::lock_guard<std::mutex> lock(data_mutex_);
        if (msg->nodes.empty()) return;

        // 获取 UWB system_time，并四舍五入到 10ms 精度
        ros::Time uwb_timestamp;
        uwb_timestamp.sec = msg->system_time / 1000;
        uwb_timestamp.nsec = (msg->system_time % 1000) * 1000000;
        double uwb_time_sec = uwb_timestamp.toSec();
        double rounded_time = std::round(uwb_time_sec * 100.0) / 100.0; // 0.01s 精度

        // 加锁 frame_data_map_
        {
            std::lock_guard<std::mutex> frame_lock(frame_data_mutex_);
            auto& frame = frame_data_map_[rounded_time];

            frame.poses[node_id_] = self_pose_;
            frame.targets[node_id_] = target_pose_;
            frame.custom_matrices[node_id_] = custom_matrix_;
            frame.received_nodes.insert(node_id_);
        }

        // 检查目标点数据有效性
        if (std::isnan(target_pose_.x) || std::isnan(target_pose_.y) || 
            std::isnan(target_pose_.z) || std::isnan(target_pose_.yaw)) {
            ROS_WARN("[frame2Callback] Invalid target pose data: NaN values detected.");
            return;
        }

        // 构建二进制数据 buffer
        std::string buffer;
        buffer.reserve(84); // 固定长度

        // source_id (1字节)
        buffer.push_back(static_cast<char>(node_id_));

        // timestamp (4字节 float, 已四舍五入)
        float ts = static_cast<float>(rounded_time);
        buffer.append(reinterpret_cast<const char*>(&ts), 4);

        // target_pose (4 float)
        float target_data[4] = {target_pose_.x, target_pose_.y, target_pose_.z, target_pose_.yaw};
        for (int i = 0; i < 4; ++i)
            buffer.append(reinterpret_cast<const char*>(&target_data[i]), 4);

        // self_pose (4 float)
        float self_data[4] = {self_pose_.x, self_pose_.y, self_pose_.z, self_pose_.yaw};
        for (int i = 0; i < 4; ++i)
            buffer.append(reinterpret_cast<const char*>(&self_data[i]), 4);

        // custom_matrix (4x2 float)
        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 2; ++j)
                buffer.append(reinterpret_cast<const char*>(&custom_matrix_[i][j]), 4);
        }

        // distance: 最多写入3项 (每项: 1字节id + 4字节float)
        int written = 0;
        {
            std::lock_guard<std::mutex> frame_lock(frame_data_mutex_);
            auto& frame = frame_data_map_[rounded_time];
            for (const auto& node : msg->nodes) {
                if (node.id < 256 && node.dis > 0.001f) {
                    buffer.push_back(static_cast<uint8_t>(node.id));
                    float d = node.dis;
                    buffer.append(reinterpret_cast<const char*>(&d), 4);
                    frame.distances.push_back({node_id_, node.id, d, uwb_timestamp});
                    if (++written == 3) break;
                }
            }
        }
        
        // 不足3项补齐 (ID=255, dist=-1.0f)
        while (written < 3) {
            uint8_t invalid_id = 255;
            float invalid_dist = -1.0f;
            buffer.push_back(invalid_id);
            buffer.append(reinterpret_cast<const char*>(&invalid_dist), 4);
            ++written;
        }

        // 检查是否收集到足够节点数据
        {
            std::lock_guard<std::mutex> frame_lock(frame_data_mutex_);
            auto& frame = frame_data_map_[rounded_time];

            // 覆盖成最新 pose / target
            frame.poses[node_id_] = self_pose_;
            frame.targets[node_id_] = target_pose_;
            frame.custom_matrices[node_id_] = custom_matrix_;
            frame.received_nodes.insert(node_id_);

            //加入缓冲
            uwb_buffer_.push_back({rounded_time, frame});
            if (uwb_buffer_.size() > max_buffer_size_) {
                uwb_buffer_.pop_front();
            }


            // if (frame.received_nodes.size() >= required_nodes_) {
            //     processFrameData(rounded_time, frame);
            //     frame_data_map_.erase(rounded_time);
            // }
        }

        // 发布二进制数据
        std_msgs::String data_trans_msg;
        data_trans_msg.data = buffer;
        // data_trans_pub_.publish(data_trans_msg);
        if (++uwb_frame_count_ >= uwb_publish_interval_) {
            data_trans_pub_.publish(data_trans_msg);
            uwb_frame_count_ = 0;  // 重置计数
        }
    }

    void cleanupStaleData(const ros::TimerEvent&) {
        ros::Time now = ros::Time::now();
        double dist_timeout = 5.0;
        double frame_timeout = 1.0;
        double custom_timeout = 5.0;
        {
            std::lock_guard<std::mutex> lock(frame_data_mutex_);
            for (auto it = frame_data_map_.begin(); it != frame_data_map_.end(); ) {
                if ((now.toSec() - it->first) > frame_timeout) {
                    it = frame_data_map_.erase(it);
                } else {
                    ++it;
                }
            }
        }

        // 清理 last_valid_dist_map_ 和 custom_matrix_map_
        {
            std::lock_guard<std::mutex> lock(data_mutex_);

            // 1. 清理 last_valid_dist_map_
            for (auto it = last_valid_dist_map_.begin(); it != last_valid_dist_map_.end(); ) {
                if ((now - it->second.second).toSec() > dist_timeout) {
                    it = last_valid_dist_map_.erase(it);
                } else {
                    ++it;
                }
            }

            // 2. 清理 custom_matrix_map_
            for (auto it = custom_matrix_map_.begin(); it != custom_matrix_map_.end(); ) {
                if ((now - it->second.timestamp).toSec() > custom_timeout) {
                    ROS_WARN("Removing stale custom_matrix from node %d", it->first);
                    it = custom_matrix_map_.erase(it);
                } else {
                    ++it;
                }
            }
        }
    }

    void processFrameData(double timestamp, FrameData& frame) {
        for (const auto& entry : frame.distances) {
            int sid = entry.source_id;
            int tid = entry.target_id;

            // ROS_INFO("[processFrameData] sid=%d, tid=%d, dist=%.3f", sid, tid, entry.distance);
            // ROS_INFO("[processFrameData] pose[sid]=%.3f,%.3f,%.3f  pose[tid]=%.3f,%.3f,%.3f",
            //         pose_matrix_[sid][0], pose_matrix_[sid][1], pose_matrix_[sid][2],
            //         pose_matrix_[tid][0], pose_matrix_[tid][1], pose_matrix_[tid][2]);
        
            if (sid < total_nodes_ && tid < total_nodes_) {
                float dz = pose_matrix_[sid][2] - pose_matrix_[tid][2];  // Z值差
                float d3d_squared = entry.distance * entry.distance;
                float dz_squared = dz * dz;

                if (d3d_squared < dz_squared) continue;
                float d2d = std::sqrt(std::max(0.0f, d3d_squared - dz_squared));  // 避免负值
                
                distance_matrix_[sid][tid] = d2d;
            }
        }
    
        for (const auto& kv : frame.poses) {
            int id = kv.first;
            pose_matrix_[id][0] = kv.second.x;
            pose_matrix_[id][1] = kv.second.y;
            pose_matrix_[id][2] = kv.second.z;
            pose_matrix_[id][3] = kv.second.yaw;
        }
    
        for (const auto& kv : frame.targets) {
            int id = kv.first;
            target_matrix_[id][0] = kv.second.x;
            target_matrix_[id][1] = kv.second.y;
            target_matrix_[id][2] = kv.second.z;
            target_matrix_[id][3] = kv.second.yaw;
        }
    
        for (const auto& kv : frame.custom_matrices) {
            int id = kv.first;
            custom_matrix_map_[id] = TimedMatrix{kv.second, ros::Time::now()};
        }
    
        // ROS_INFO("Frame at time %.3f processed and fused", timestamp);
    }    
    
    //解析函数
    void parseNodeframe0(const nlink_parser::LinktrackNodeframe0::ConstPtr& msg) {
        std::lock_guard<std::mutex> lock(data_mutex_);
        for (const auto& node : msg->nodes) {
            try {

                const std::vector<uint8_t>& data = node.data;

                if (data.size() != 84) {
                    ROS_ERROR("[parseNodeframe0] Invalid buffer size: %zu", data.size());
                    continue;
                }

                size_t offset = 0;

                auto read_float = [&](float& out) {
                    if (offset + 4 > data.size()) throw std::runtime_error("Buffer overflow");
                    std::memcpy(&out, &data[offset], 4);
                    offset += 4;
                };

                uint8_t source_id = data[offset++];

                float timestamp;
                read_float(timestamp);

                float tx, ty, tz, tyaw;
                read_float(tx); read_float(ty); read_float(tz); read_float(tyaw);

                float sx, sy, sz, syaw;
                read_float(sx); read_float(sy); read_float(sz); read_float(syaw);

                std::vector<std::vector<float>> custom_matrix(4, std::vector<float>(2));
                for (int i = 0; i < 4; ++i)
                    for (int j = 0; j < 2; ++j)
                        read_float(custom_matrix[i][j]);

                ros::Time parsed_time(timestamp);

                std::lock_guard<std::mutex> frame_lock(frame_data_mutex_);
                auto& frame = frame_data_map_[timestamp];
                frame.poses[source_id] = {sx, sy, sz, syaw};
                frame.targets[source_id] = {tx, ty, tz, tyaw};
                frame.custom_matrices[source_id] = custom_matrix;
                frame.received_nodes.insert(source_id);

                for (int i = 0; i < 3; ++i) {
                    uint8_t tid = data[offset++];
                    float dist;
                    read_float(dist);

                    if (tid != 255 && dist > 0.001f && tid < total_nodes_ && std::isfinite(dist)) {
                        frame.distances.push_back({source_id, tid, dist, parsed_time});
                        last_valid_dist_map_[{source_id, tid}] = std::make_pair(dist, ros::Time::now());
                    } else {
                        // 尝试补偿
                        for (int possible_id = 0; possible_id < total_nodes_; ++possible_id) {
                            auto key = std::make_pair(source_id, possible_id);
                            if (last_valid_dist_map_.count(key)) {
                                float recovered_dist = last_valid_dist_map_[key].first;
                                frame.distances.push_back({source_id, possible_id, recovered_dist, parsed_time});
                                break;
                            }
                        }
                    }
                }

                if (frame.received_nodes.size() >= required_nodes_) {
                    processFrameData(timestamp, frame);
                    frame_data_map_.erase(timestamp);
                }
            } catch (const std::exception& e) {
                ROS_WARN("[parseNodeframe0] Failed to parse: %s", e.what());
            }
        }
    }



    // 距离阵
    void publishMatrix() {
        std::lock_guard<std::mutex> lock(data_mutex_);
        std_msgs::Float32MultiArray matrix_msg;
        matrix_msg.layout.dim.resize(2);
        matrix_msg.layout.dim[0].label = "rows";
        matrix_msg.layout.dim[0].size = total_nodes_;
        matrix_msg.layout.dim[0].stride = total_nodes_ * total_nodes_;
        matrix_msg.layout.dim[1].label = "cols";
        matrix_msg.layout.dim[1].size = total_nodes_;
        matrix_msg.layout.dim[1].stride = total_nodes_;

        std::vector<std::vector<float>> averaged_matrix = distance_matrix_;
        for (int i = 0; i < total_nodes_; i++) averaged_matrix[i][i] = 0.0f;

        for (int i = 0; i < total_nodes_; i++) {
            for (int j = i + 1; j < total_nodes_; j++) {
                float val_i_j = distance_matrix_[i][j];
                float val_j_i = distance_matrix_[j][i];
                if (val_i_j > 0 && val_j_i > 0) {
                    float diff = std::fabs(val_i_j - val_j_i);
                    if (diff <= distance_diff_threshold_) {
                        float avg = (val_i_j + val_j_i) / 2.0f;
                        averaged_matrix[i][j] = avg;
                        averaged_matrix[j][i] = avg;
                    } else {
                        averaged_matrix[i][j] = averaged_matrix[j][i] = -1.0f;
                    }
                } else if (val_i_j > 0) averaged_matrix[j][i] = val_i_j;
                else if (val_j_i > 0) averaged_matrix[i][j] = val_j_i;
            }
        }

        matrix_msg.data.resize(total_nodes_ * total_nodes_);
        for (int i = 0; i < total_nodes_; i++) {
            for (int j = 0; j < total_nodes_; j++) {
                matrix_msg.data[i * total_nodes_ + j] = averaged_matrix[i][j];
            }
        }
        matrix_pub_.publish(matrix_msg);
    }
    //odom
    void publishPoseMatrix() {
        std::lock_guard<std::mutex> lock(data_mutex_);
        std_msgs::Float32MultiArray matrix_msg;
        matrix_msg.layout.dim.resize(2);
        matrix_msg.layout.dim[0].label = "rows";
        matrix_msg.layout.dim[0].size = total_nodes_;
        matrix_msg.layout.dim[0].stride = total_nodes_ * 4;
        matrix_msg.layout.dim[1].label = "cols";
        matrix_msg.layout.dim[1].size = 4;
        matrix_msg.layout.dim[1].stride = 4;
    
        matrix_msg.data.resize(total_nodes_ * 4);
        for (int i = 0; i < total_nodes_; i++) {
            for (int j = 0; j < 4; j++) {
                matrix_msg.data[i * 4 + j] = pose_matrix_[i][j];
            }
        }
        pose_matrix_pub_.publish(matrix_msg);
    }
    //目标点
    void publishTargetMatrix() {
        std::lock_guard<std::mutex> lock(data_mutex_);
        std_msgs::Float32MultiArray matrix_msg;
        matrix_msg.layout.dim.resize(2);
        matrix_msg.layout.dim[0].label = "rows";
        matrix_msg.layout.dim[0].size = total_nodes_;
        matrix_msg.layout.dim[0].stride = total_nodes_ * 4;
        matrix_msg.layout.dim[1].label = "cols";
        matrix_msg.layout.dim[1].size = 4;
        matrix_msg.layout.dim[1].stride = 4;
    
        matrix_msg.data.resize(total_nodes_ * 4);
        for (int i = 0; i < total_nodes_; i++) {
            for (int j = 0; j < 4; j++) {
                matrix_msg.data[i * 4 + j] = target_matrix_[i][j];
            }
        }
        target_matrix_pub_.publish(matrix_msg);
    }
    //解算出的矩阵
    void publishCustomMatrices() {
        std::lock_guard<std::mutex> lock(data_mutex_); 
        for (const auto& pair : custom_matrix_map_) {
            int source_id = pair.first;
            const auto& mat = pair.second.matrix;
            // 发布
            if (matrix_from_others_pub_.count(source_id)) {
                std_msgs::Float32MultiArray msg;
                msg.layout.dim.resize(2);
                msg.layout.dim[0].label = "rows";
                msg.layout.dim[0].size = total_nodes_;
                msg.layout.dim[0].stride = total_nodes_*2;
                msg.layout.dim[1].label = "cols";
                msg.layout.dim[1].size = 2;
                msg.layout.dim[1].stride = 2;
                
                msg.data.resize(total_nodes_*2);
                for (int i = 0; i < total_nodes_; ++i) {
                    for (int j = 0; j < 2; ++j) {
                        msg.data[i * 2 + j] = mat[i][j];
                    }
                }
    
                matrix_from_others_pub_[source_id].publish(msg);
            }
        }
    } 

    void publishMatrixCallback(const ros::TimerEvent&) {
        // TODO: 时间同步
        // 缓冲区清理，防止溢出
        {
            std::lock_guard<std::mutex> lock(frame_data_mutex_);
            while (uwb_buffer_.size() > max_buffer_size_) {
                uwb_buffer_.pop_front();
            }
        }

        if (uwb_buffer_.size() < avg_window) {
            return;
        }

        FrameData avg_frame;
        std::map<std::pair<int,int>, std::pair<float,int>> dist_sum; // (sid, tid) -> (sum, count)

        {
            std::lock_guard<std::mutex> lock(frame_data_mutex_);

            for (size_t i = 0; i < avg_window; ++i) {
                CachedUWB cuwb = uwb_buffer_.front();
                uwb_buffer_.pop_front();

                for (const auto& d : cuwb.frame.distances) {
                    auto key = std::make_pair(d.source_id, d.target_id);
                    dist_sum[key].first  += d.distance;
                    dist_sum[key].second += 1;
                }
            }
        }
        ros::Time now_time = ros::Time::now();
        for (auto& kv : dist_sum) {
            float avg_dist = kv.second.first / kv.second.second;
            avg_frame.distances.push_back({
                kv.first.first,   // sid
                kv.first.second,  // tid
                avg_dist,         // 平均距离
                now_time
            });
        }
        processFrameData(now_time.toSec(), avg_frame);
        
        //发布
        publishMatrix();
        publishPoseMatrix();
        publishTargetMatrix();
        publishCustomMatrices();
    }

    // 定时打印距离矩阵和时间戳
    void printMatrixCallback(const ros::TimerEvent& event) {
        std::lock_guard<std::mutex> lock(data_mutex_);

        // 距离矩阵
        ROS_INFO("[Node %d] Distance Buffer Entries:", node_id_);

        std::string header = "ID |";
        for (int j = 0; j < total_nodes_; ++j) {
            header += "  " + std::to_string(j) + "  |";
        }
        ROS_INFO("%s", header.c_str());
        for (int i = 0; i < total_nodes_; ++i) {
            std::string row = std::to_string(i) + " |";
            
            for (int j = 0; j < total_nodes_; ++j) {
                if (distance_matrix_[i][j] < 0) {
                    row += "   -  |"; // 未知距离
                } else {
                    char buffer[32];
                    sprintf(buffer, "%5.2f |", distance_matrix_[i][j]);
                    row += buffer;
                }
            }
            ROS_INFO("%s", row.c_str());
        }

        ROS_INFO("frame_data_map_ size: %lu", frame_data_map_.size());
        ROS_INFO("last_valid_dist_map_ size: %lu", last_valid_dist_map_.size());
        ROS_INFO("custom_matrix_map_ size: %lu", custom_matrix_map_.size()); 

        // // Pose
        // ROS_INFO("[Node %d] Pose Matrix (x, y, z, yaw):", node_id_);
        // for (int i = 0; i < total_nodes_; ++i) {
        //     std::string row = "ID " + std::to_string(i) + ": ";
        //     for (int j = 0; j < 4; ++j) {
        //         char buffer[16];
        //         sprintf(buffer, "%7.3f ", pose_matrix_[i][j]);
        //         row += buffer;
        //     }
        //     ROS_INFO("%s", row.c_str());
        // }      

        // // Goal
        // ROS_INFO("[Node %d] Goal Matrix (x, y, z, yaw):", node_id_);
        // for (int i = 0; i < total_nodes_; ++i) {
        //     std::string row = "ID " + std::to_string(i) + ": ";
        //     for (int j = 0; j < 4; ++j) {
        //         char buffer[16];
        //         sprintf(buffer, "%7.3f ", target_matrix_[i][j]);
        //         row += buffer;
        //     }
        //     ROS_INFO("%s", row.c_str());
        // }      

        // // 打印接收到的 custom_matrix
        // ROS_INFO("[Node %d] Custom Matrices from other nodes:", node_id_);
        // for (const auto& pair : custom_matrix_map_) {
        //     int source_id = pair.first;
        //     const auto& mat = pair.second;

        //     ROS_INFO("  From node %d:", source_id);
        //     for (int i = 0; i < total_nodes_; ++i) {
        //         ROS_INFO("    Row %d: [%.2f, %.2f]", i, mat[i][0], mat[i][1]);
        //     }
        // }
    }
};

int main(int argc, char** argv) {
    ros::init(argc, argv, "uwb_node");
    UWBNode node;
    ros::spin();
    return 0;
}