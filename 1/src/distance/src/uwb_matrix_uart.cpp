#include <ros/ros.h>
#include <nlink_parser/LinktrackNodeframe2.h>
#include <nlink_parser/LinktrackNodeframe0.h>
#include <std_msgs/String.h>
#include <std_msgs/Float32MultiArray.h>
#include <vector>
#include <set>
#include <sstream>
#include <mutex>
#include <fcntl.h>
#include <termios.h>
#include <unistd.h>
#include <iomanip>

class UWBNode {
private:
    ros::NodeHandle nh_;
    ros::NodeHandle private_nh_;

    int node_id_;
    int total_nodes_;
    int required_nodes_;

    ros::Publisher matrix_pub_;
    ros::Subscriber frame2_sub_;
    ros::Subscriber nodeframe0_sub_;

    std::vector<std::vector<float>> distance_matrix_;
    std::set<int> received_nodes_;
    std::vector<ros::Time> node_timestamps_;
    bool self_data_collected_;

    struct DistanceEntry {
        int source_id;
        int target_id;
        float distance;
    };
    std::vector<DistanceEntry> distance_buffer_;

    ros::Time last_publish_time_;
    ros::Timer publish_timer_;
    ros::Timer print_timer_;
    double matrix_print_rate_;
    double matrix_publish_rate_;
    double publish_period_;
    double distance_diff_threshold_;

    std::mutex data_mutex_;

    int serial_fd_ = -1;
    std::string serial_port_;
    int baud_rate_;

public:
    UWBNode() : private_nh_("~"), self_data_collected_(false) {
        private_nh_.param<int>("node_id", node_id_, 0);
        private_nh_.param<int>("total_nodes", total_nodes_, 6);
        private_nh_.param<int>("required_nodes", required_nodes_, 5);
        private_nh_.param<double>("matrix_print_rate", matrix_print_rate_, 1.0);
        private_nh_.param<double>("matrix_publish_rate", matrix_publish_rate_, 100.0);
        private_nh_.param<double>("distance_diff_threshold", distance_diff_threshold_, 0.5);
        private_nh_.param<std::string>("serial_port", serial_port_, "/dev/ttyUSB1");
        private_nh_.param<int>("baud_rate", baud_rate_, 921600);

        publish_period_ = 1.0 / matrix_publish_rate_;

        ROS_INFO("Starting UWB node %d (of %d total nodes)", node_id_, total_nodes_);
        ROS_INFO("Will update matrix after receiving data from %d nodes", required_nodes_);

        distance_matrix_.resize(total_nodes_, std::vector<float>(total_nodes_, -1.0));
        node_timestamps_.resize(total_nodes_, ros::Time(0));
        last_publish_time_ = ros::Time::now();

        matrix_pub_ = nh_.advertise<std_msgs::Float32MultiArray> ("/uwb" + std::to_string(node_id_) + "/distance_matrix", 10);

        frame2_sub_ = nh_.subscribe<nlink_parser::LinktrackNodeframe2> ("/uwb" + std::to_string(node_id_) + "/nodeframe2", 10, boost::bind(&UWBNode::frame2Callback, this, _1));
        nodeframe0_sub_ = nh_.subscribe<nlink_parser::LinktrackNodeframe0> ("/uwb" + std::to_string(node_id_) + "/nodeframe0", 10, boost::bind(&UWBNode::parseNodeframe0, this, _1));

        print_timer_ = nh_.createTimer(ros::Duration(1.0 / matrix_print_rate_), &UWBNode::printMatrixCallback, this);
        publish_timer_ = nh_.createTimer(ros::Duration(publish_period_), &UWBNode::publishMatrixCallback, this);

        initSerial();
    }

    ~UWBNode() {
        if (serial_fd_ >= 0) close(serial_fd_);
    }

    void initSerial() {
        serial_fd_ = open(serial_port_.c_str(), O_RDWR | O_NOCTTY | O_NDELAY);
        if (serial_fd_ == -1) {
            ROS_ERROR("Failed to open serial port %s", serial_port_.c_str());
            return;
        }
        fcntl(serial_fd_, F_SETFL, 0);
        struct termios options;
        tcgetattr(serial_fd_, &options);
        cfsetispeed(&options, baud_rate_);
        cfsetospeed(&options, baud_rate_);
        options.c_cflag |= (CLOCAL | CREAD);
        options.c_cflag &= ~CSIZE;
        options.c_cflag |= CS8;
        options.c_cflag &= ~PARENB;
        options.c_cflag &= ~CSTOPB;
        tcsetattr(serial_fd_, TCSANOW, &options);
    }

    void frame2Callback(const nlink_parser::LinktrackNodeframe2::ConstPtr& msg) {
        std::lock_guard<std::mutex> lock(data_mutex_);

        uint32_t uwb_ms_time = msg->system_time;
        ros::Time uwb_timestamp;
        uwb_timestamp.sec = uwb_ms_time / 1000;
        uwb_timestamp.nsec = (uwb_ms_time % 1000) * 1000000;
        node_timestamps_[node_id_] = uwb_timestamp;

        std::stringstream ss;
        ss << "ID:" << (int)msg->id << ";TIME:" << std::fixed << std::setprecision(6)
           << uwb_timestamp.toSec() << ";LOCAL_TIME:" << msg->local_time << ";DISTANCES:";

        for (size_t i = 0; i < msg->nodes.size(); ++i) {
            const auto& node = msg->nodes[i];
            if (i > 0) ss << ",";
            ss << (int)node.id << ":" << node.dis;

            int target_id = node.id;
            float distance = node.dis;
            if (target_id < total_nodes_ && distance > 0.001) {
                distance_buffer_.push_back({node_id_, target_id, distance});
            }
        }

        self_data_collected_ = true;

        std::string out = ss.str();
        if (serial_fd_ >= 0) {
            write(serial_fd_, out.c_str(), out.length());
            write(serial_fd_, "\n", 1);
        }

        checkAndUpdateMatrix();
    }

    
    // 解析接收到的nodeframe0消息
    void parseNodeframe0(const nlink_parser::LinktrackNodeframe0::ConstPtr& msg) {
        std::lock_guard<std::mutex> lock(data_mutex_);
        
        // 遍历节点数据
        for (const auto& node : msg->nodes) {
            // 转换data为字符串
            std::string data_str(node.data.begin(), node.data.end());
            
            // 找到并解析源ID
            size_t id_pos = data_str.find("ID:");
            size_t time_pos = data_str.find(";TIME:");
            
            if (id_pos == std::string::npos || time_pos == std::string::npos) {
                continue;
            }
            
            std::string id_str = data_str.substr(id_pos + 3, time_pos - id_pos - 3);
            int source_id;
            try {
                source_id = std::stoi(id_str);
            } catch (...) {
                continue;
            }
            
            // 将源节点ID添加到已接收集合
            received_nodes_.insert(source_id);
            
            // 解析时间戳
            size_t local_time_pos = data_str.find(";LOCAL_TIME:");
            size_t dist_pos = data_str.find(";DISTANCES:");
            
            if (dist_pos == std::string::npos) {
                continue;
            }
            
            // 确定TIME字段的结束位置
            size_t time_end_pos = local_time_pos != std::string::npos ? local_time_pos : dist_pos;
            
            std::string time_str = data_str.substr(time_pos + 6, time_end_pos - time_pos - 6);
            try {
                double timestamp = std::stod(time_str);
                node_timestamps_[source_id] = ros::Time(timestamp);
            } catch (...) {
                node_timestamps_[source_id] = ros::Time::now();
            }
            
            // 解析距离数据
            std::string distances = data_str.substr(dist_pos + 11); // ";DISTANCES:"长度为11
            
            // 解析每一对"ID:距离"
            std::stringstream ss(distances);
            std::string pair;
            
            while (getline(ss, pair, ',')) {
                size_t colon_pos = pair.find(':');
                if (colon_pos != std::string::npos) {
                    try {
                        int target_id = std::stoi(pair.substr(0, colon_pos));
                        float distance = std::stof(pair.substr(colon_pos + 1));
                        
                        // 将有效的距离数据添加到缓冲区
                        if (target_id < total_nodes_ && source_id < total_nodes_ && distance > 0.001) {
                            distance_buffer_.push_back({source_id, target_id, distance});
                        }
                    } catch (...) {}
                }
            }
        }
        
        // 检查是否可以更新矩阵
        checkAndUpdateMatrix();
    }
    
    // 检查并更新矩阵
    void checkAndUpdateMatrix() {
        // 检查是否满足更新条件: 
        // 1. 自己的数据已收集
        // 2. 已收到至少required_nodes_个节点的数据(包括自己)
        
        if (self_data_collected_ && (received_nodes_.size() + 1 >= required_nodes_)) {
            ROS_DEBUG("Updating matrix with data from %zu nodes", received_nodes_.size() + 1);
            
            // 将缓冲的距离数据应用到距离矩阵
            for (const auto& entry : distance_buffer_) {
                distance_matrix_[entry.source_id][entry.target_id] = entry.distance;
            }
            
            // 清空缓冲区和状态，为下一轮数据收集做准备
            distance_buffer_.clear();
            received_nodes_.clear();
            self_data_collected_ = false;
        }
    }
    
    // 发布矩阵数据
    void publishMatrix() {
        std::lock_guard<std::mutex> lock(data_mutex_);
        
        std_msgs::Float32MultiArray matrix_msg;
        
        // 设置维度信息
        matrix_msg.layout.dim.resize(2);
        matrix_msg.layout.dim[0].label = "rows";
        matrix_msg.layout.dim[0].size = total_nodes_;
        matrix_msg.layout.dim[0].stride = total_nodes_ * total_nodes_;
        matrix_msg.layout.dim[1].label = "cols";
        matrix_msg.layout.dim[1].size = total_nodes_;
        matrix_msg.layout.dim[1].stride = total_nodes_;
        
        // 创建临时矩阵用于计算平均值，避免修改原始数据
        std::vector<std::vector<float>> averaged_matrix = distance_matrix_;
        
        // 将对角线元素都设置为0
        for (int i = 0; i < total_nodes_; i++) {
            averaged_matrix[i][i] = 0.0f;
        }
        
        // 计算对称元素的平均值（加入过滤机制）
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
                        ROS_WARN("Dropping asymmetric pair (%d, %d): %.2f vs %.2f (diff %.2f)",
                                i, j, val_i_j, val_j_i, diff);
                        averaged_matrix[i][j] = -1.0f;
                        averaged_matrix[j][i] = -1.0f;
                    }
                }
                else if (val_i_j > 0) {
                    averaged_matrix[j][i] = val_i_j;
                }
                else if (val_j_i > 0) {
                    averaged_matrix[i][j] = val_j_i;
                }
            }
        }

        
        // 填充数据
        matrix_msg.data.resize(total_nodes_ * total_nodes_);
        for (int i = 0; i < total_nodes_; i++) {
            for (int j = 0; j < total_nodes_; j++) {
                matrix_msg.data[i * total_nodes_ + j] = averaged_matrix[i][j];
            }
        }
        
        // 发布消息
        matrix_pub_.publish(matrix_msg);
    }
    
    // 定时发布距离矩阵
    void publishMatrixCallback(const ros::TimerEvent& event) {
        publishMatrix();
    }
    
    // 定时打印距离矩阵和时间戳
    void printMatrixCallback(const ros::TimerEvent& event) {
        std::lock_guard<std::mutex> lock(data_mutex_);
        
        ros::Time current_time = ros::Time::now();
        
        // 打印ID和UWB时间戳信息
        ROS_INFO("[Node %d] UWB Timestamps:", node_id_);
        for (int i = 0; i < total_nodes_; i++) {
            if (node_timestamps_[i].isZero()) {
                ROS_INFO("  Node %d: No data received", i);
            } else {
                ROS_INFO("  Node %d: UWB time: %.3f seconds", i, node_timestamps_[i].toSec());
            }
        }
        
        ROS_INFO("[Node %d] Current Distance Matrix (collected from %zu nodes):", 
                node_id_, received_nodes_.size() + (self_data_collected_ ? 1 : 0));
        
        // 打印列标题
        std::string header = "ID |";
        for (int j = 0; j < total_nodes_; ++j) {
            header += "  " + std::to_string(j) + "  |";
        }
        ROS_INFO("%s", header.c_str());
        
        // 打印分隔线
        std::string separator = "---|";
        for (int j = 0; j < total_nodes_; ++j) {
            separator += "------|";
        }
        ROS_INFO("%s", separator.c_str());
        
        // 打印矩阵行
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
    }
};

int main(int argc, char** argv) {
    ros::init(argc, argv, "uwb_node");
    UWBNode node;
    ros::spin();
    return 0;
}
