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
rostopic pub /uwb1/target_position geometry_msgs/PoseStamped "header:
  frame_id: 'map'
  stamp:
    secs: 0
    nsecs: 0
pose:
  position:
    x: 1.0
    y: 2.0
    z: 4.0
  orientation:
    x: 0.0
    y: 0.0
    z: 0.707
    w: 0.707"

订阅话题：rostopic echo /uwb3/pose_matrix 

2.发送odom点
rostopic pub /odom2 nav_msgs/Odometry \
'{header: {stamp: now, frame_id: "map"}, child_frame_id: "base_link", pose: {pose: {position: {x: 1111111.0, y: 2.44, z: 111}, orientation: {x: 0.0, y: 0.0, z: 0.707, w: 0.707}}}, twist: {twist: {linear: {x: 0.0, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.0}}}}'

订阅话题：rostopic echo /uwb3/target_matrix 

3.发送坐标矩阵
rostopic pub /uwb2/custom_matrix std_msgs/Float32MultiArray "layout:
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

    std::set<int> received_nodes_;
    bool self_data_collected_ = false;

    ros::Time self_timestamp_;

    // 距离数据
    struct DistanceEntry {
        int source_id;
        int target_id;
        float distance;
        ros::Time timestamp;
    };
    std::vector<DistanceEntry> distance_buffer_;

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
    
        if (msg->data.size() != 8) {
            ROS_WARN("Received custom_matrix size %lu != 8", msg->data.size());
            return;
        }
    
        for (int i = 0; i < total_nodes_; ++i) {
            for (int j = 0; j < 2; ++j) {
                custom_matrix_[i][j] = msg->data[i * 2 + j];
            }
        }
    
        ROS_INFO("Updated custom_matrix for node %d", node_id_);
    }
    
    //拼装
    void frame2Callback(const nlink_parser::LinktrackNodeframe2::ConstPtr& msg) {
        std::lock_guard<std::mutex> lock(data_mutex_);

        ros::Time uwb_timestamp;
        uwb_timestamp.sec = msg->system_time / 1000;
        uwb_timestamp.nsec = (msg->system_time % 1000) * 1000000;
        self_timestamp_ = uwb_timestamp;

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
                distance_buffer_.push_back({node_id_, node.id, node.dis, uwb_timestamp});
            }
        }

        self_data_collected_ = true;
        data_trans_msg.data = ss.str();
        data_trans_pub_.publish(data_trans_msg);
    }
    
    //解析函数
    void parseNodeframe0(const nlink_parser::LinktrackNodeframe0::ConstPtr& msg) {
        std::lock_guard<std::mutex> lock(data_mutex_);
        for (const auto& node : msg->nodes) {
            std::string data_str(node.data.begin(), node.data.end());
            size_t id_pos = 0;
            size_t time_pos = data_str.find('|');
            if (time_pos == std::string::npos) continue;
            int source_id = std::stoi(data_str.substr(id_pos, time_pos));
            received_nodes_.insert(source_id);

            //time
            size_t second_pos = data_str.find('|', time_pos + 1);
            double timestamp = std::stod(data_str.substr(time_pos + 1, second_pos - time_pos - 1));
            ros::Time parsed_time(timestamp);

            //target
            size_t target_pos = data_str.find('|', second_pos + 1);
            std::string target = data_str.substr(second_pos + 1, target_pos - second_pos - 1);
            std::stringstream ss_target(target);
            float x, y, z,yaw;
            char delimiter;
            if (ss_target >> x >> delimiter >> y >> delimiter >> z>> delimiter >>yaw) {
                // ROS_INFO("Parsed target position: ID=%d, x=%.2f, y=%.2f, z=%.2f", source_id, x, y, z);
                if (source_id < total_nodes_) {
                    if (source_id != node_id_) {  //关键区别
                        target_matrix_[source_id][0] = x;
                        target_matrix_[source_id][1] = y;
                        target_matrix_[source_id][2] = z;
                        target_matrix_[source_id][3] = yaw;
                    }
                }
            } else {
                ROS_WARN("Failed to parse target position from data: %s", target.c_str());
                // target_buffer_.push_back({source_id, 0.0f, 0.0f, 0.0f}); // Default to (0, 0, 0) if parsing fails
            }

            // odom
            size_t odom_pos = data_str.find('|', target_pos + 1);
            if (odom_pos != std::string::npos) {
                std::string odom_data = data_str.substr(target_pos + 1, odom_pos - target_pos - 1);
                std::stringstream ss_odom(odom_data);

                float x, y, z, yaw;
                char delimiter;

                if (ss_odom >> x >> delimiter >> y >> delimiter >> z >> delimiter >> yaw) {
                    // Store x, y, z, yaw, and timestamp for different odometry data
                    // pose_buffer_.push_back({source_id, x, y, z, yaw, parsed_time});
                    // ROS_INFO("Parsed odometry data: ID=%d, x=%.2f, y=%.2f, z=%.2f, yaw=%.2f, timestamp=%.2f", source_id, x, y, z, yaw, parsed_time.toSec());
                    // 更新矩阵
                    if (source_id < total_nodes_) {
                        pose_matrix_[source_id][0] = x;
                        pose_matrix_[source_id][1] = y;
                        pose_matrix_[source_id][2] = z;
                        pose_matrix_[source_id][3] = yaw;
                    }
                } else {
                    ROS_WARN("Failed to parse odometry data from: %s", odom_data.c_str());
                }
            } else {
                ROS_WARN("Odometry data not found in string: %s", data_str.c_str());
            }


            // 接收 custom_matrix
            size_t matrix_pos = data_str.find('|', odom_pos + 1);
            if (matrix_pos != std::string::npos) {
                std::string matrix_data = data_str.substr(odom_pos + 1, matrix_pos - odom_pos - 1);
                std::stringstream ss(matrix_data);
                std::string val_str;
                std::vector<float> values;

                while (std::getline(ss, val_str, ',')) {
                    try {
                        values.push_back(std::stof(val_str));
                    } catch (...) {
                        ROS_WARN("Invalid float in custom matrix from node %d: %s", source_id, val_str.c_str());
                    }
                }

                if (values.size() == total_nodes_*2) {
                    std::vector<std::vector<float>> parsed_matrix(total_nodes_, std::vector<float>(2, 0.0f));
                    for (int i = 0; i < total_nodes_; ++i) {
                        for (int j = 0; j < 2; ++j) {
                            parsed_matrix[i][j] = values[i * 2 + j];
                        }
                    }
                    //存储
                    custom_matrix_map_[source_id] = parsed_matrix;
                } else {
                    ROS_WARN("Matrix size mismatch from node %d: expected 8 values, got %lu", source_id, values.size());
                }
            }

            // distances
            size_t dist_pos = data_str.rfind('|');
            if (dist_pos == std::string::npos || dist_pos <= second_pos) continue;
            std::string distances = data_str.substr(dist_pos + 1);

            std::stringstream ss(distances);
            std::string pair;
            while (getline(ss, pair, ',')) {
                size_t colon_pos = pair.find(':');
                if (colon_pos != std::string::npos) {
                    int target_id = std::stoi(pair.substr(0, colon_pos));
                    float distance = std::stof(pair.substr(colon_pos + 1));
                    if (target_id < total_nodes_ && source_id < total_nodes_ && distance > 0.001) {
                        distance_buffer_.push_back({source_id, target_id, distance, parsed_time});
                    }
                }
            }
        }
    }

    //对齐uwb
    void checkAndUpdateMatrix() {
        if (self_data_collected_ && (received_nodes_.size() + 1 >= required_nodes_)) {
            std::map<std::pair<int, int>, DistanceEntry> best_entry_map;
            for (const auto& entry : distance_buffer_) {
                auto key = std::make_pair(entry.source_id, entry.target_id);
                if (best_entry_map.find(key) == best_entry_map.end()) {
                    best_entry_map[key] = entry;
                } else {
                    double old_diff = std::abs((best_entry_map[key].timestamp - self_timestamp_).toSec());
                    double new_diff = std::abs((entry.timestamp - self_timestamp_).toSec());
                    if (new_diff < old_diff) {
                        best_entry_map[key] = entry;
                    }
                }
            }
            for (const auto& item : best_entry_map) {
                distance_matrix_[item.second.source_id][item.second.target_id] = item.second.distance;
            }
            distance_buffer_.clear();
            received_nodes_.clear();
            self_data_collected_ = false;
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

    void publishMatrixCallback(const ros::TimerEvent&) {
        checkAndUpdateMatrix();
        //发布
        publishMatrix();
        publishPoseMatrix();
        publishTargetMatrix();
        publishCustomMatrices();
    }

    // 定时打印距离矩阵和时间戳
    void printMatrixCallback(const ros::TimerEvent& event) {
        std::lock_guard<std::mutex> lock(data_mutex_);

        // 打印距离矩阵
        ROS_INFO("[Node %d] Distance Buffer Entries (with timestamps):", node_id_);
        std::set<int> printed_sources;
        for (const auto& entry : distance_buffer_) {
            if (printed_sources.insert(entry.source_id).second) {
                ROS_INFO("  [UWB%d] time = %.3f", entry.source_id, entry.timestamp.toSec());
            }
        }

        std::string header = "ID |";
        for (int j = 0; j < total_nodes_; ++j) {
            header += "  " + std::to_string(j) + "  |";
        }
        ROS_INFO("%s", header.c_str());

        std::string separator = "---|";
        for (int j = 0; j < total_nodes_; ++j) {
            separator += "------|";
        }
        ROS_INFO("%s", separator.c_str());

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
        
        ROS_INFO("%s", separator.c_str());

        // Pose
        ROS_INFO("[Node %d] Pose Matrix (x, y, z, yaw):", node_id_);
        for (int i = 0; i < total_nodes_; ++i) {
            std::string row = "ID " + std::to_string(i) + ": ";
            for (int j = 0; j < 4; ++j) {
                char buffer[16];
                sprintf(buffer, "%7.3f ", pose_matrix_[i][j]);
                row += buffer;
            }
            ROS_INFO("%s", row.c_str());
        }      

        // Goal
        ROS_INFO("[Node %d] Goal Matrix (x, y, z, yaw):", node_id_);
        for (int i = 0; i < total_nodes_; ++i) {
            std::string row = "ID " + std::to_string(i) + ": ";
            for (int j = 0; j < 4; ++j) {
                char buffer[16];
                sprintf(buffer, "%7.3f ", target_matrix_[i][j]);
                row += buffer;
            }
            ROS_INFO("%s", row.c_str());
        }      

        // 打印接收到的 custom_matrix
        ROS_INFO("[Node %d] Custom Matrices from other nodes:", node_id_);
        for (const auto& pair : custom_matrix_map_) {
            int source_id = pair.first;
            const auto& mat = pair.second;

            ROS_INFO("  From node %d:", source_id);
            for (int i = 0; i < total_nodes_; ++i) {
                ROS_INFO("    Row %d: [%.2f, %.2f]", i, mat[i][0], mat[i][1]);
            }
        }
    }
};

int main(int argc, char** argv) {
    ros::init(argc, argv, "uwb_node");
    UWBNode node;
    ros::spin();
    return 0;
}