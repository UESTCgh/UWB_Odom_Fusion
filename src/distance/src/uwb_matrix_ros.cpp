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

/*
话题发送实例：

1.发送goal点
rostopic pub -r 180 /uwb1/target_position geometry_msgs/PoseStamped "header:
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
rostopic pub -r 50 /uwb2/custom_matrix std_msgs/Float32MultiArray "layout:
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

    std::map<int, std::vector<std::vector<float>>> custom_matrix_map_;
    std::map<int, ros::Publisher> matrix_from_others_pub_;

    std::vector<std::vector<float>> distance_matrix_;
    std::vector<std::vector<float>> pose_matrix_;
    std::vector<std::vector<float>> target_matrix_;  // 行：节点ID，列：(x, y, z)
    std::vector<std::vector<float>> custom_matrix_;  // 4行2列

    // std::set<int> received_nodes_;
    // bool self_data_collected_ = false;

    // ros::Time self_timestamp_;

    // 距离数据
    struct DistanceEntry {
        int source_id;
        int target_id;
        float distance;
        ros::Time timestamp;
    };
    // std::vector<DistanceEntry> distance_buffer_;

    // 目标点
    struct Target {
        int target_id;
        float x;
        float y;
        float z;
    };
    // std::vector<Target> target_buffer_;

    // Pose 数据
    struct PoseEntry {
        int id;
        float x;
        float y;
        float z;
        float yaw;
        ros::Time timestamp;
    };
    // std::vector<PoseEntry> pose_buffer_;

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
    

    ros::Time last_publish_time_;
    ros::Timer publish_timer_;
    ros::Timer print_timer_;
    double matrix_print_rate_;
    double matrix_publish_rate_;
    double publish_period_;
    double distance_diff_threshold_;

    std::mutex data_mutex_;
    std::string odom_topic_;
    std::string target_topic_;

    const double TIME_TOLERANCE = 0.01; // 10ms

public:
    UWBNode() : private_nh_("~") {
        // 参数
        private_nh_.param<int>("node_id", node_id_, 0);
        private_nh_.param<int>("total_nodes", total_nodes_, 6);
        private_nh_.param<int>("required_nodes", required_nodes_, 5);
        private_nh_.param<double>("matrix_print_rate", matrix_print_rate_, 1.0);
        private_nh_.param<double>("matrix_publish_rate", matrix_publish_rate_, 100.0);
        private_nh_.param<double>("distance_diff_threshold", distance_diff_threshold_, 0.5);
        private_nh_.param<std::string>("odom_topic", odom_topic_, "/odom");
        private_nh_.param<std::string>("target_topic", target_topic_, "/target_position");

        publish_period_ = 1.0 / matrix_publish_rate_;

        distance_matrix_.resize(total_nodes_, std::vector<float>(total_nodes_, -1.0));
        pose_matrix_.resize(total_nodes_, std::vector<float>(4, 0.0f));
        target_matrix_.resize(total_nodes_, std::vector<float>(4, 0.0f));
        custom_matrix_.resize(total_nodes_, std::vector<float>(2, 0.0f));//4*2初始化

        last_publish_time_ = ros::Time::now();

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

        nodeframe0_sub_ = nh_.subscribe<nlink_parser::LinktrackNodeframe0>(
            "/uwb" + std::to_string(node_id_) + "/nodeframe0", 10, &UWBNode::parseNodeframe0, this);

        odom_sub_ = nh_.subscribe<nav_msgs::Odometry>(odom_topic_, 10, &UWBNode::odomCallback, this);
        target_pos_sub_ = nh_.subscribe<geometry_msgs::PoseStamped>(target_topic_, 10, &UWBNode::targetCallback, this);
        
        //回调发布
        print_timer_ = nh_.createTimer(ros::Duration(1.0 / matrix_print_rate_), &UWBNode::printMatrixCallback, this);
        publish_timer_ = nh_.createTimer(ros::Duration(publish_period_), &UWBNode::publishMatrixCallback, this);
    }

    bool isSameFrame(double t1, double t2) {
        return fabs(t1 - t2) < TIME_TOLERANCE;
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
            target_matrix_[node_id_][3] = target_pose_.yaw;  // yaw
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

        ros::Time uwb_timestamp;
        uwb_timestamp.sec = msg->system_time / 1000;
        uwb_timestamp.nsec = (msg->system_time % 1000) * 1000000;
        // self_timestamp_ = uwb_timestamp;
        double uwb_time_sec = msg->system_time / 1000.0;

        auto& frame = frame_data_map_[uwb_time_sec];  // 这一行非常关键

        std_msgs::String data_trans_msg;
        std::stringstream ss;

        // 拼接id time target pose
        ss << node_id_ << "|"
           << std::fixed << std::setprecision(2) << uwb_timestamp.toSec() << "|"
           << target_pose_.x << "," << target_pose_.y << "," << target_pose_.z << "," << target_pose_.yaw<< "|"
           << self_pose_.x << "," << self_pose_.y << "," << self_pose_.z << "," << self_pose_.yaw << "|";

        // 拼接 4x2 矩阵
        for (int i = 0; i < total_nodes_; ++i) {
            for (int j = 0; j < 2; ++j) {
                if (i != 0 || j != 0) ss << ",";  // 第一个元素不加逗号
                ss << std::fixed << std::setprecision(2) << custom_matrix_[i][j];
            }
        }
        ss << "|";  // 矩阵后面加分隔符

        //拼接distance
        for (size_t i = 0; i < msg->nodes.size(); ++i) {
            const auto& node = msg->nodes[i];
            if (i > 0) ss << ",";
            ss << (int)node.id << ":" << std::fixed << std::setprecision(2) << node.dis;
            if (node.id < total_nodes_ && node.dis > 0.001) {
                frame.distances.push_back({node_id_, node.id, node.dis, uwb_timestamp});
            }
        }

        {
            std::lock_guard<std::mutex> frame_lock(frame_data_mutex_);
            auto& frame = frame_data_map_[uwb_time_sec];
            frame.poses[node_id_] = self_pose_;
            frame.targets[node_id_] = target_pose_;
            frame.custom_matrices[node_id_] = custom_matrix_;
            frame.received_nodes.insert(node_id_);

            for (size_t i = 0; i < msg->nodes.size(); ++i) {
                const auto& node = msg->nodes[i];
                if (node.id < total_nodes_ && node.dis > 0.001) {
                    frame.distances.push_back({node_id_, node.id, node.dis, ros::Time(uwb_time_sec)});
                }
            }
            if (frame.received_nodes.size() >= 1) {
                processFrameData(uwb_time_sec, frame);
                frame_data_map_.erase(uwb_time_sec);
            }
        }

        // self_data_collected_ = true;
        data_trans_msg.data = ss.str();
        data_trans_pub_.publish(data_trans_msg);
    }

    void processFrameData(double timestamp, FrameData& frame) {
        for (const auto& entry : frame.distances) {
            int sid = entry.source_id;
            int tid = entry.target_id;
        
            if (sid < total_nodes_ && tid < total_nodes_) {
                float dz = pose_matrix_[sid][2] - pose_matrix_[tid][2];  // Z值差
                float d3d_squared = entry.distance * entry.distance;
                float dz_squared = dz * dz;
        
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
            custom_matrix_map_[id] = kv.second;
        }
    
        // ROS_INFO("Frame at time %.3f processed and fused", timestamp);
    }    
    
    //解析函数
    void parseNodeframe0(const nlink_parser::LinktrackNodeframe0::ConstPtr& msg) {
        std::lock_guard<std::mutex> lock(data_mutex_);
        for (const auto& node : msg->nodes) {
            try {
                std::string data_str(node.data.begin(), node.data.end());
    
                // 分割字段
                std::vector<std::string> segments;
                size_t start = 0, end = 0;
                while ((end = data_str.find('|', start)) != std::string::npos) {
                    segments.push_back(data_str.substr(start, end - start));
                    start = end + 1;
                }
                segments.push_back(data_str.substr(start));
    
                if (segments.size() < 5) {
                    ROS_WARN("Malformed data: too few segments (%lu): %s", segments.size(), data_str.c_str());
                    continue;
                }
    
                int source_id = std::stoi(segments[0]);
                double timestamp = std::stod(segments[1]);
                ros::Time parsed_time(timestamp);
    
                std::lock_guard<std::mutex> frame_lock(frame_data_mutex_);
                auto& frame = frame_data_map_[timestamp];
                frame.received_nodes.insert(source_id);
    
                // 1. 目标点
                {
                    std::stringstream ss(segments[2]);
                    float x, y, z, yaw;
                    char delim;
                    if (ss >> x >> delim >> y >> delim >> z >> delim >> yaw) {
                        if (source_id < total_nodes_ && source_id != node_id_) {
                            target_matrix_[source_id][0] = x;
                            target_matrix_[source_id][1] = y;
                            target_matrix_[source_id][2] = z;
                            target_matrix_[source_id][3] = yaw;
                        }
                    } else {
                        ROS_WARN("Failed to parse target for node %d: %s", source_id, segments[2].c_str());
                    }
                }
    
                // 2. odom
                {
                    std::stringstream ss(segments[3]);
                    float x, y, z, yaw;
                    char delim;
                    if (ss >> x >> delim >> y >> delim >> z >> delim >> yaw) {
                        if (source_id < total_nodes_) {
                            pose_matrix_[source_id][0] = x;
                            pose_matrix_[source_id][1] = y;
                            pose_matrix_[source_id][2] = z;
                            pose_matrix_[source_id][3] = yaw;
                        }
                    } else {
                        ROS_WARN("Failed to parse odom for node %d: %s", source_id, segments[3].c_str());
                    }
                }
    
                // 3. custom_matrix
                {
                    std::stringstream ss(segments[4]);
                    std::string val;
                    std::vector<float> values;
                    while (getline(ss, val, ',')) {
                        try {
                            values.push_back(std::stof(val));
                        } catch (...) {
                            ROS_WARN("Invalid float in custom matrix from node %d: %s", source_id, val.c_str());
                        }
                    }
                    if (values.size() == total_nodes_ * 2) {
                        std::vector<std::vector<float>> mat(total_nodes_, std::vector<float>(2, 0.0f));
                        for (int i = 0; i < total_nodes_; ++i)
                            for (int j = 0; j < 2; ++j)
                                mat[i][j] = values[i * 2 + j];
                        custom_matrix_map_[source_id] = mat;
                    } else {
                        ROS_WARN("Matrix size mismatch from node %d: expected %d, got %lu", source_id, total_nodes_ * 2, values.size());
                    }
                }
    
                // 4. distances（第5段可能不存在，需检查）
                if (segments.size() >= 6) {
                    std::stringstream ss(segments[5]);
                    std::string pair;
                    while (getline(ss, pair, ',')) {
                        size_t colon = pair.find(':');
                        if (colon != std::string::npos) {
                            try {
                                int tid = std::stoi(pair.substr(0, colon));
                                float dist = std::stof(pair.substr(colon + 1));
                                if (tid < total_nodes_ && dist > 0.001) {
                                    frame.distances.push_back({source_id, tid, dist, parsed_time});
                                }
                            } catch (...) {
                                ROS_WARN("Bad distance entry from node %d: %s", source_id, pair.c_str());
                            }
                        }
                    }
                }
    
                if (frame.received_nodes.size() >= required_nodes_) {
                    processFrameData(timestamp, frame);
                    frame_data_map_.erase(timestamp);
                }
    
            } catch (const std::exception& e) {
                ROS_ERROR("Exception in parseNodeframe0: %s", e.what());
                continue;  // 跳过当前节点
            } catch (...) {
                ROS_ERROR("Unknown error in parseNodeframe0.");
                continue;
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
        for (const auto& pair : custom_matrix_map_) {
            int source_id = pair.first;
            const auto& mat = pair.second;
    
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

    void cleanExpiredFrames() {
        std::lock_guard<std::mutex> lock(frame_data_mutex_);
        ros::Time now = ros::Time::now();
        double expire_time = now.toSec() - 1.0; // 设置为 1 秒前的数据

        for (auto it = frame_data_map_.begin(); it != frame_data_map_.end();) {
                if (it->first < expire_time) {
                    it = frame_data_map_.erase(it);
                } else {
                    ++it;
                }
            }
    }


    void publishMatrixCallback(const ros::TimerEvent&) {
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
                    char buffer[8];
                    sprintf(buffer, "%5.2f |", distance_matrix_[i][j]);
                    row += buffer;
                }
            }
            ROS_INFO("%s", row.c_str());
        }

        ROS_INFO("frame_data_map_ size: %lu", frame_data_map_.size());

        // Pose
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
        cleanExpiredFrames();
    }
};

int main(int argc, char** argv) {
    ros::init(argc, argv, "uwb_node");
    UWBNode node;
    ros::spin();
    return 0;
}


