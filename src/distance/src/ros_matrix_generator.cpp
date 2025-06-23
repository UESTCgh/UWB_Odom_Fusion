#include <ros/ros.h>
#include <vector>
#include <iostream>
#include <cmath>
#include <map>
#include <set>
#include <std_msgs/Float32MultiArray.h>
#include <std_msgs/MultiArrayDimension.h>
#include <nlink_parser/LinktrackNodeframe2.h>
#include <nlink_parser/LinktrackNode2.h>

class UWBDistanceMatrixGenerator {
private:
    ros::NodeHandle nh_;
    ros::NodeHandle private_nh_;
    
    // 发布者 - 现在只需要一个
    ros::Publisher combined_pub_;        // 发布合并的数据（时间戳+节点ID+距离矩阵）
    
    // 订阅者
    std::vector<ros::Subscriber> nodeframe2_subs_;  // 订阅多个UWB的nodeframe2话题
    
    // 参数
    int num_nodes_;                      // UWB节点数量
    int num_uwb_devices_;                // UWB设备数量
    double distance_timeout_;            // 距离数据过期时间（秒）
    double publish_rate_;                // 发布频率（秒）
    
    // 数据存储
    std::vector<std::vector<float>> distance_matrix_;       // 距离矩阵
    std::vector<std::vector<ros::Time>> timestamp_matrix_;  // 时间戳矩阵
    
    // 设备状态跟踪
    std::set<int> updated_devices_;      // 已收到数据的设备ID集合
    ros::Time last_device_update_time_;  // 最后一次设备更新时间
    ros::Time full_matrix_time_;         // 完整矩阵的构建时间
    
    // 计时器
    ros::Timer publish_timer_;           // 定时发布器
    
    // 用于记录最近一次发布的时间
    ros::Time last_matrix_pub_time_;

public:
    UWBDistanceMatrixGenerator() : private_nh_("~") {
        // 从参数服务器获取配置参数
        private_nh_.param<int>("num_nodes", num_nodes_, 10); // 默认10个节点
        private_nh_.param<int>("num_uwb_devices", num_uwb_devices_, 5); // 默认5个UWB设备
        private_nh_.param<double>("distance_timeout", distance_timeout_, 5.0); // 默认5秒超时
        private_nh_.param<double>("publish_rate", publish_rate_, 1.0); // 默认1秒发布一次
        
        // 初始化距离矩阵和时间戳矩阵
        distance_matrix_.resize(num_nodes_, std::vector<float>(num_nodes_, -1.0)); // -1表示无效距离
        timestamp_matrix_.resize(num_nodes_, std::vector<ros::Time>(num_nodes_));
        
        // 创建单个发布者 - 合并所有数据
        combined_pub_ = nh_.advertise<std_msgs::Float32MultiArray>("/uwb/distance_data", 10);
        
        // 订阅多个UWB节点的nodeframe2话题
        for (int i = 0; i < num_uwb_devices_; ++i) {
            std::string topic = "/uwb" + std::to_string(i) + "/nodeframe2";
            ros::Subscriber sub = nh_.subscribe(topic, 10, &UWBDistanceMatrixGenerator::nodeframeCallback, this);
            nodeframe2_subs_.push_back(sub);
            ROS_INFO("Subscribed to UWB topic: %s", topic.c_str());
        }
        
        // 初始化时间戳
        last_matrix_pub_time_ = ros::Time::now();
        last_device_update_time_ = ros::Time::now();
        full_matrix_time_ = ros::Time::now();
        
        // 启动定时发布器
        publish_timer_ = nh_.createTimer(ros::Duration(publish_rate_), 
                                         &UWBDistanceMatrixGenerator::timerCallback, this);
        
        ROS_INFO("UWB Distance Matrix Generator initialized with %d nodes, waiting for all %d devices",
                 num_nodes_, num_uwb_devices_);
        ROS_INFO("Distance timeout set to %.1f seconds", distance_timeout_);
    }
    
    void nodeframeCallback(const nlink_parser::LinktrackNodeframe2::ConstPtr& msg) {
        // 获取当前时间戳
        ros::Time current_time = ros::Time::now();
        
        // 提取发送者信息
        uint8_t anchor_id = msg->id;
        
        // 跳过无效ID
        if (anchor_id >= num_nodes_) {
            ROS_WARN("Received message from anchor ID %d which is out of range (max: %d)", 
                     anchor_id, num_nodes_-1);
            return;
        }
        
        // 处理节点数据
        for (const auto& node2 : msg->nodes) {
            // 跳过无效ID的节点
            if (node2.id >= num_nodes_) {
                continue;
            }
            
            // 从node2中提取距离信息
            float distance = node2.dis;
            
            // 更新距离矩阵和时间戳矩阵
            distance_matrix_[anchor_id][node2.id] = distance;
            timestamp_matrix_[anchor_id][node2.id] = current_time;
            
            // 矩阵对称性（如果适用）
            distance_matrix_[node2.id][anchor_id] = distance;
            timestamp_matrix_[node2.id][anchor_id] = current_time;
        }
        
        // 记录这个设备ID已经上报数据
        updated_devices_.insert(anchor_id);
        last_device_update_time_ = current_time;
        
        // 检查是否所有设备都已上报数据
        if (updated_devices_.size() >= num_uwb_devices_) {
            ROS_INFO("All %d devices have reported data, matrix complete", num_uwb_devices_);
            full_matrix_time_ = current_time;
            
            // 清空更新状态，准备下一轮数据收集
            updated_devices_.clear();
            
            // 发布完整矩阵
            publishCombinedData(current_time, true);
        }
    }
    
    void timerCallback(const ros::TimerEvent& event) {
        ros::Time current_time = ros::Time::now();
        
        // 如果超过一定时间没有收到所有设备的数据，也发布当前的矩阵
        ros::Duration since_last_update = current_time - last_device_update_time_;
        if (since_last_update.toSec() > distance_timeout_) {
            ROS_WARN("Timeout waiting for all devices. Publishing with %zu/%d devices reporting", 
                     updated_devices_.size(), num_uwb_devices_);
            updated_devices_.clear();
            publishCombinedData(current_time, false);
        }
    }
    
    void publishCombinedData(const ros::Time& current_time, bool is_complete) {
        // 清理过期的距离数据
        cleanExpiredDistances(current_time);
        
        // 找出有效的节点（至少有一条与其他节点的连接）
        std::set<int> active_nodes;
        for (int i = 0; i < num_nodes_; ++i) {
            bool is_active = false;
            for (int j = 0; j < num_nodes_; ++j) {
                if (i != j && distance_matrix_[i][j] > 0) {
                    is_active = true;
                    break;
                }
            }
            if (is_active) {
                active_nodes.insert(i);
            }
        }
        
        // 如果没有有效节点，不发布
        if (active_nodes.empty()) {
            ROS_WARN("No active nodes detected, skipping matrix publication");
            return;
        }
        
        // 将活跃节点转换为向量
        std::vector<int> active_node_ids(active_nodes.begin(), active_nodes.end());
        
        // 创建合并后的消息
        std_msgs::Float32MultiArray combined_msg;
        
        // 设置布局 - 我们将使用3D布局
        // dim[0] = 1行（单个消息）
        // dim[1] = 3列（时间戳 + 节点ID列表 + 距离矩阵）
        // dim[2] = 每列中的元素数
        combined_msg.layout.dim.resize(3);
        
        // 第一维：消息
        combined_msg.layout.dim[0].label = "message";
        combined_msg.layout.dim[0].size = 1;
        combined_msg.layout.dim[0].stride = 1 + active_node_ids.size() + active_node_ids.size() * active_node_ids.size();
        
        // 第二维：数据类型（时间戳、节点ID、距离矩阵）
        combined_msg.layout.dim[1].label = "data_type";
        combined_msg.layout.dim[1].size = 3;
        combined_msg.layout.dim[1].stride = 0;  // 将在下面计算
        
        // 第三维：每种数据类型的元素数量
        combined_msg.layout.dim[2].label = "elements";
        combined_msg.layout.dim[2].size = 0;  // 动态大小，在每个部分中指定
        
        // 计算每种数据类型的偏移量和大小
        int timestamp_offset = 0;
        int node_ids_offset = 1;  // 时间戳之后
        int matrix_offset = node_ids_offset + active_node_ids.size();
        
        // 设置步长
        combined_msg.layout.dim[1].stride = matrix_offset + active_node_ids.size() * active_node_ids.size();
        
        // 填充数据
        combined_msg.data.clear();
        
        // 1. 添加时间戳（秒）
        combined_msg.data.push_back(full_matrix_time_.toSec());
        
        // 2. 添加节点ID列表
        for (int id : active_node_ids) {
            combined_msg.data.push_back(static_cast<float>(id));
        }
        
        // 3. 添加距离矩阵（按行顺序）
        for (int i : active_node_ids) {
            for (int j : active_node_ids) {
                combined_msg.data.push_back(distance_matrix_[i][j]);
            }
        }
        
        // 发布合并的消息
        combined_pub_.publish(combined_msg);
        last_matrix_pub_time_ = current_time;
        
        // 打印完整状态和当前精简的距离矩阵
        if (is_complete) {
            ROS_INFO("Publishing COMPLETE combined data with timestamp: %.3f", full_matrix_time_.toSec());
        } else {
            ROS_INFO("Publishing PARTIAL combined data with timestamp: %.3f", full_matrix_time_.toSec());
        }
        
        printCompactDistanceMatrix(active_node_ids);
    }
    
    void cleanExpiredDistances(const ros::Time& current_time) {
        // 检查并清理过期的距离数据
        for (int i = 0; i < num_nodes_; ++i) {
            for (int j = 0; j < num_nodes_; ++j) {
                if (i != j) {  // 跳过自身距离
                    ros::Duration elapsed = current_time - timestamp_matrix_[i][j];
                    if (elapsed.toSec() > distance_timeout_ && distance_matrix_[i][j] != -1.0) {
                        ROS_INFO("Distance from node %d to node %d expired (%.1f seconds old)", 
                                i, j, elapsed.toSec());
                        distance_matrix_[i][j] = -1.0;  // 标记为无效
                    }
                }
            }
        }
    }
    
    void printCompactDistanceMatrix(const std::vector<int>& active_node_ids) {
        ROS_INFO("Current Distance Matrix (Active Nodes Only):");
        
        // 打印列标题
        std::string header = "ID  |";
        for (int j : active_node_ids) {
            char buffer[8];
            sprintf(buffer, "  %2d  |", j);
            header += buffer;
        }
        ROS_INFO("%s", header.c_str());
        
        // 打印分隔线
        std::string separator = "----|";
        for (size_t j = 0; j < active_node_ids.size(); ++j) {
            separator += "------|";
        }
        ROS_INFO("%s", separator.c_str());
        
        // 打印矩阵行
        for (size_t i = 0; i < active_node_ids.size(); ++i) {
            int row_id = active_node_ids[i];
            std::string row = "";
            char row_header[8];
            sprintf(row_header, " %2d |", row_id);
            row += row_header;
            
            for (size_t j = 0; j < active_node_ids.size(); ++j) {
                int col_id = active_node_ids[j];
                char buffer[8];
                if (distance_matrix_[row_id][col_id] < 0) {
                    sprintf(buffer, "   -  |");
                } else {
                    sprintf(buffer, "%5.2f |", distance_matrix_[row_id][col_id]);
                }
                row += buffer;
            }
            ROS_INFO("%s", row.c_str());
        }
        
        // 打印时间戳信息
        ROS_INFO("Matrix timestamp: %.3f", full_matrix_time_.toSec());
        ROS_INFO("%s", separator.c_str());
        
        // 解释如何解析合并的消息
        ROS_INFO("Combined message format: [timestamp, node_id_1, ..., node_id_n, matrix_values...]");
    }
};

int main(int argc, char** argv) {
    ros::init(argc, argv, "uwb_distance_matrix_generator");
    
    UWBDistanceMatrixGenerator generator;
    
    ros::spin();
    
    return 0;
}
