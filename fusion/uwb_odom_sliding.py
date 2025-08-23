#########################################################
# Organization:     UESTC (Shenzhen)                    #
# Author:           Hao, Guo                            #
# Email:            1417150646@qq.com                   #
# Github:           https://github.com/UESTCgh          #
#########################################################

import time
import numpy as np
# import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from collections import deque

# Apply ROS
import rospy
# from std_msgs.msg import Float32MultiArray
from std_msgs.msg import Float32MultiArray
from geometry_msgs.msg import PoseArray, Pose ,Point

from visualization_msgs.msg import Marker, MarkerArray

np.set_printoptions(precision=5, suppress=True)


class UWBFusion:

    def __init__(self, N: int, DIM: int,
                 pose_topic: str, dist_topic: str, pub_topic: str,
                 iteration=8, buffer=128,uwb_filter_threshold=0.05, frames_window=30):
        # dimension will be 1, 2 or 3
        assert DIM in [1, 2, 3]

        self.N = N
        self.DIM = DIM
        self.left_fixed_point = 0
        self.right_fixed_point = 0
        self.tstamp = None
        self.num_frames = 0
        self.est = deque(maxlen=buffer)

        # temp variable
        self.is_initialized = False
        self.pose_ready = False
        self.dist_ready = False

        self.odom_last_frame = None
        self.odom_curr_frame = None
        self.last_frame = None
        self.curr_frame = None
        self.smooth_frame = None
        self.uwb_distance_matrix = None
        self.point_distance_matrix = None
        self.constraint_distance_matrix = None
        self.odom_distance_matrix = None
        self.drop_current_frame = False
        # uwb flter threshold
        self.uwb_filter_threshold = uwb_filter_threshold
        # sliding window
        self.frames_window = frames_window

        self.window_frames = deque(maxlen=self.frames_window)  # 保存最近10帧 (N, DIM)
        self.window_odoms = deque(maxlen=self.frames_window)   # 每帧对应的 odom 数据

        self.colors = ['r', 'g', 'm', 'c', 'y', 'k']

        # optimization setting
        self.iteration = iteration

        # Apply ROS
        rospy.Subscriber(pose_topic, Float32MultiArray, self.pose_callback)
        rospy.Subscriber(dist_topic, Float32MultiArray, self.dist_callback)
        self.pose_publisher = rospy.Publisher(pub_topic, Float32MultiArray, queue_size=10)
        self.marker_publisher = rospy.Publisher("/uwb1/marker_array", MarkerArray, queue_size=10)

        self.rate = rospy.Rate(200)

    def pose_callback(self, msg):
        # Odometry poses (N, DIM)
        self.odom_curr_frame = np.array(msg.data).reshape(self.N, -1)[:, :self.DIM]
        self.pose_ready = True

    # input 
    def dist_callback(self, msg):
        new_dist = np.array(msg.data).reshape(-1, 1)

        if not hasattr(self, 'kf_states'):
            self.kf_states = {}
            self.kf_params = {'A': 1.0, 'H': 1.0, 'Q': 1e-2, 'R': 0.2}

        if not hasattr(self, 'last_valid_distance'):
            self.last_valid_distance = new_dist.copy()

            # 初始化滤波器状态
            for idx in range(len(new_dist)):
                i, j = divmod(idx, self.N)
                self.kf_states[(i, j)] = [new_dist[idx, 0], 1.0]

            self.uwb_distance_matrix = new_dist
            self.dist_ready = True
            self.drop_current_frame = False
            return

        N = self.N

        # lost found
        if np.any(new_dist == -1):
            coords = np.argwhere(new_dist == -1)
            for idx_arr in coords:
                idx = idx_arr[0]
                i, j = divmod(idx, N)
                rospy.logwarn(f"[DROP] Invalid value at Node {i}–{j}: -1 detected → Frame dropped.")
                self.drop_current_frame = True
            return

        # 卡尔曼滤波处理
        filtered = []
        for idx in range(len(new_dist)):
            i, j = divmod(idx, N)
            value = new_dist[idx, 0]
            key = (i, j)

            # 如果状态不存在，初始化
            if key not in self.kf_states:
                self.kf_states[key] = [value, 1.0]

            x, P = self.kf_states[key]
            A = self.kf_params['A']
            H = self.kf_params['H']
            Q = self.kf_params['Q']
            R = self.kf_params['R']

            # 预测
            x_pred = A * x
            P_pred = A * P * A + Q

            # 更新（合法观测值）
            K = P_pred * H / (H * P_pred * H + R)
            x_new = x_pred + K * (value - H * x_pred)
            P_new = (1 - K * H) * P_pred

            # 存储状态
            self.kf_states[key] = [x_new, P_new]
            filtered.append(x_new)

        self.last_valid_distance = new_dist.copy()
        alpha = 0.5
        if not hasattr(self, 'uwb_distance_matrix'):
            self.uwb_distance_matrix = np.array(filtered).reshape(-1, 1)
        else:
            self.uwb_distance_matrix = alpha * self.uwb_distance_matrix + (1 - alpha) * np.array(filtered).reshape(-1, 1)
        self.dist_ready = True
        self.drop_current_frame = False  


    def update_constraint_distance(self):
        distances = pdist(self.curr_frame, metric='euclidean')
        self.point_distance_matrix = squareform(distances).reshape(-1, 1)
        self.constraint_distance_matrix = \
            np.linalg.norm(self.curr_frame - self.last_frame, axis=1)[..., None]
        self.odom_distance_matrix = \
            np.linalg.norm(self.odom_curr_frame - self.odom_last_frame, axis=1)[..., None]

    def cal_jacobian_residual(self):
        """
        @brief: calculate Jacobian & residual

        @output:
        1.distance constraint
        J1  (N^2, (N-fixed)*2)  Jacobian
        r1  (N^2, 1)            residual

        2.inter-frame constraint
        J2  (N, (N-fixed)*2)    Jacobian
        r2  (N, 1)              residual

        3.sliding-windows constraint
        J2  (N, (N-fixed)*2)    Jacobian
        r2  (N, 1)              residual
        """
        J1 = np.zeros((self.N ** 2, self.N * self.DIM))
        for row in range(J1.shape[0]):
            row_i, row_j = row // self.N, row % self.N
            # none residual
            if row_i == row_j:
                continue
            for col in range(J1.shape[1]):
                col_i, col_j = col // self.DIM, col % self.DIM
                # fixed point
                if (col_i < self.left_fixed_point) or \
                    (col_i > self.N - self.right_fixed_point - 1):
                    continue
                elif row_i == col_i:
                    J1[row, col] = 2 * (self.curr_frame[col_i, col_j] - self.curr_frame[row_j, col_j])
                elif row_j == col_i:
                    J1[row, col] = 2 * (self.curr_frame[col_i, col_j] - self.curr_frame[row_i, col_j])
        J1 = J1[:, (self.left_fixed_point * self.DIM):((self.N - self.right_fixed_point) * self.DIM)]
        r1 = self.point_distance_matrix**2 - self.uwb_distance_matrix**2

        J2 = np.zeros((self.N, self.N * self.DIM))
        for row in range(J2.shape[0]):
            for col in range(J2.shape[1]):
                col_i, col_j = col // self.DIM, col % self.DIM
                # fixed point
                if (col_i < self.left_fixed_point) or \
                    (col_i > self.N - self.right_fixed_point - 1):
                    continue
                elif row == col_i:
                    J2[row, col] = 2 * (self.curr_frame[col_i, col_j] - self.last_frame[col_i, col_j])
        J2 = J2[:, (self.left_fixed_point * self.DIM):((self.N - self.right_fixed_point) * self.DIM)]
        r2 = self.constraint_distance_matrix**2 - self.odom_distance_matrix**2

        window_size = min(len(self.window_frames), self.frames_window)
        if window_size >= 1:
            J3_list = []
            r3_list = []
            for w in range(1, window_size + 1):
                prev_frame = self.window_frames[-w]
                prev_odom = self.window_odoms[-w]

                J3w = np.zeros((self.N, self.N * self.DIM))
                for row in range(J3w.shape[0]):
                    for col in range(J3w.shape[1]):
                        col_i, col_j = col // self.DIM, col % self.DIM
                        if (col_i < self.left_fixed_point) or \
                           (col_i > self.N - self.right_fixed_point - 1):
                            continue
                        elif row == col_i:
                            J3w[row, col] = 2 * (self.curr_frame[col_i, col_j] - prev_frame[col_i, col_j])
                J3w = J3w[:, (self.left_fixed_point * self.DIM):((self.N - self.right_fixed_point) * self.DIM)]

                r3w = np.linalg.norm(self.curr_frame - prev_frame, axis=1, keepdims=True) ** 2 - \
                      np.linalg.norm(self.odom_curr_frame - prev_odom, axis=1, keepdims=True) ** 2

                weight = 1.0 / (w**0.5) 
                J3_list.append(weight*J3w)
                r3_list.append(weight*r3w)

            J3 = np.vstack(J3_list)  # (window_size * N, var_dim)
            r3 = np.vstack(r3_list)  # (window_size * N, 1)
        else:
            J3 = np.zeros_like(J2)
            r3 = np.zeros_like(r2)

        np.set_printoptions(precision=4, suppress=True, linewidth=200)
        # rospy.loginfo(f"slidingwindow {self.frames_window}")
        # rospy.loginfo(f"[SlidingConstraint] Constructed J3 and r3 with {window_size} history frames.")
        # rospy.loginfo(f"J3 shape: {J3.shape}, r3 shape: {r3.shape}")
        # rospy.loginfo(f"J3 sample (first 3 rows):\n{J3[:3]}")
        # rospy.loginfo(f"r3 sample (first 10 values):\n{r3[:10].flatten()}")

        return J1, r1, J2, r2 ,J3 ,r3

    # Gaussian-Newton optimization
    def gauss_newton_optimization(self, J1, r1, J2, r2, J3 ,r3 ,
                                  info_matrix1=None, info_matrix2=None,info_matrix3=None,
                                  alpha=1., beta=1.,gama=1.,
                                  max_iter=1, thresh=1.e-5, step=0.05,
                                  ep=1.0, lm=1.e-4):
        if info_matrix1 is None:
            info_matrix1 = np.eye(J1.shape[0])
        if info_matrix2 is None:
            info_matrix2 = np.eye(J2.shape[0])
        if info_matrix3 is None:
            info_matrix3 = np.eye(J3.shape[0])
        info_matrix1 *= alpha
        info_matrix2 *= beta
        info_matrix3 *= gama

        for i in range(max_iter):
            H = J1.T @ info_matrix1 @ J1 + J2.T @ info_matrix2 @ J2 + J3.T @ info_matrix3 @ J3
            v = -J1.T @ info_matrix1 @ r1 - J2.T @ info_matrix2 @ r2 - J3.T @ info_matrix3 @ r3 
            # damping factor
            H = H + (ep + lm * H) * np.eye(H.shape[0])
            delta = np.linalg.solve(H, v)

            self.curr_frame[self.left_fixed_point:(self.N - self.right_fixed_point)] = \
                (self.curr_frame[self.left_fixed_point:(self.N - self.right_fixed_point)].reshape(-1, 1) + step * delta).reshape(-1, self.DIM)

            if np.linalg.norm(step * delta) < thresh:
                return

    def update_position(self):
        self.last_frame = self.curr_frame.copy()

        for iter in range(self.iteration):

            if (self.left_fixed_point >= self.N) or \
                (self.right_fixed_point >= self.N) or \
                ((self.left_fixed_point + self.right_fixed_point) >= self.N):
                rospy.logfatal('Fixed all point, please check!!!')

            self.update_constraint_distance()
            J1, r1, J2, r2 , J3 , r3 = self.cal_jacobian_residual()
            self.gauss_newton_optimization(J1, r1, J2, r2 , J3 , r3 )

    def system_init(self):
        # The initialization strategy is defined by user
        self.curr_frame = np.zeros((self.N, self.DIM))
        self.curr_frame[0, 0] = self.uwb_distance_matrix[1]
        self.curr_frame[2, 1] = self.uwb_distance_matrix[6]
        self.curr_frame[3, 0] = self.uwb_distance_matrix[1]
        self.curr_frame[3, 1] = self.uwb_distance_matrix[6]
        self.last_frame = self.curr_frame.copy()
        self.odom_last_frame = self.odom_curr_frame.copy()
        self.is_initialized = True

    def publish_markers(self):
        marker_array = MarkerArray()
        
        for i in range(self.N):
            # ----------------------------
            # 1. 节点位置 Marker (球体)
            # ----------------------------
            marker = Marker()
            marker.header.frame_id = "uwb"
            marker.header.stamp = rospy.Time.now()
            marker.ns = "current_positions"
            marker.id = i
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.pose.position.x = self.smooth_frame[i, 0]
            marker.pose.position.y = self.smooth_frame[i, 1]
            marker.pose.position.z = 0.0 if self.DIM < 3 else self.smooth_frame[i, 2]
            marker.pose.orientation.w = 1.0
            marker.scale.x = 0.15
            marker.scale.y = 0.15
            marker.scale.z = 0.15

            # 设置颜色
            color = self.colors[i % len(self.colors)]
            if color == 'r':
                marker.color.r = 1.0
            elif color == 'g':
                marker.color.g = 1.0
            elif color == 'b':
                marker.color.b = 1.0
            elif color == 'm':
                marker.color.r = 1.0
                marker.color.b = 1.0
            elif color == 'c':
                marker.color.g = 1.0
                marker.color.b = 1.0
            elif color == 'y':
                marker.color.r = 1.0
                marker.color.g = 1.0
            else:
                marker.color.r = marker.color.g = marker.color.b = 0.5  

            marker.color.a = 1.0
            marker.lifetime = rospy.Duration(0.1)

            marker_array.markers.append(marker)

            # ----------------------------
            # 2. 节点 ID Marker
            # ----------------------------
            text_marker = Marker()
            text_marker.header.frame_id = "uwb"
            text_marker.header.stamp = rospy.Time.now()
            text_marker.ns = "labels"
            text_marker.id = 1000 + i 
            text_marker.type = Marker.TEXT_VIEW_FACING
            text_marker.action = Marker.ADD

            text_marker.pose.position.x = self.smooth_frame[i, 0]
            text_marker.pose.position.y = self.smooth_frame[i, 1]
            text_marker.pose.position.z = 0.25  
            text_marker.pose.orientation.w = 1.0

            text_marker.scale.z = 0.15  
            text_marker.color.r = 1.0
            text_marker.color.g = 1.0
            text_marker.color.b = 1.0
            text_marker.color.a = 1.0

            text_marker.text = str(i)  
            text_marker.lifetime = rospy.Duration(0.1)

            marker_array.markers.append(text_marker)

        # 发布所有 Marker
        self.marker_publisher.publish(marker_array)

    # sliding filter
    def apply_sliding_window_filter(self, method='mean'):
        """
        滑动窗口滤波：对 self.curr_frame 进行位置平滑
        method: 'mean' 平均滤波, 'median' 中值滤波
        """
        if len(self.est) < self.frames_window:
            self.smooth_frame = self.curr_frame.copy()
            return  

        window_data = np.stack(list(self.est)[-self.frames_window:], axis=0)  # (window, N, DIM)

        if method == 'median':
            smoothed = np.median(window_data, axis=0)
            # rospy.loginfo(f"[Sliding Filter] Median filter applied ({self.frames_window} frames).")
        else:
            smoothed = np.mean(window_data, axis=0)
            # rospy.loginfo(f"[Sliding Filter] Mean filter applied ({self.frames_window} frames).")

        self.smooth_frame = smoothed.copy()

    def __call__(self):

        # System init
        if not self.is_initialized:
            self.system_init()

        self.update_position()

        # sliding window
        self.est.append(self.curr_frame.copy())

        self.window_frames.append(self.curr_frame.copy())
        self.window_odoms.append(self.odom_curr_frame.copy())

        self.apply_sliding_window_filter(method = 'median')


        # Publish uwb pose
        pose_array_msg = PoseArray()
        pose_array_msg.header.stamp = rospy.Time.now()
        pose_array_msg.header.frame_id = "uwb"


        flat_pose = Float32MultiArray()
        flat_pose.data = self.curr_frame[:, :self.DIM].flatten().tolist()
        self.pose_publisher.publish(flat_pose)

        # log
        # if self.num_frames % 100 == 0:
        #     rospy.loginfo(f"[UWB Fusion Localization] Frame {self.num_frames}")
        #     for i in range(self.N):
        #         coord = self.curr_frame[i]
        #         coord_str = ", ".join([f"{x:.3f}" for x in coord])
        #         rospy.loginfo(f"Node {i}: ({coord_str})")

        if self.num_frames % 5 == 0:
            self.publish_markers()
            # self.publish_trajectory_markers()

        self.pose_ready = False
        self.dist_ready = False
        self.num_frames += 1

def get_param():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--N', type=int, default=4)
    parser.add_argument('--DIM', type=int, default=2)
    parser.add_argument('--pose_topic', type=str, default='/uwb1/pose_matrix')
    parser.add_argument('--dist_topic', type=str, default='/uwb1/distance_matrix')
    parser.add_argument('--pub_topic', type=str, default='/uwb1/custom_matrix')
    parser.add_argument('--iteration', type=int, default=4)
    parser.add_argument('--uwb_filter_threshold', type=float, default=0.05)
    parser.add_argument('--frames_window', type=int, default=100)

    args = parser.parse_args()
    return args


def run():
    # get parameter
    args = get_param()
    rospy.init_node('UWB_Fusion_node')

    # UWB-Fusion
    uwb_system = UWBFusion(args.N, args.DIM,
                           args.pose_topic, args.dist_topic, args.pub_topic,
                           args.iteration)
    uwb_system.left_fixed_point = 0
    uwb_system.right_fixed_point = 0

    while not rospy.is_shutdown():
        if uwb_system.pose_ready and uwb_system.dist_ready:
            if uwb_system.drop_current_frame:
                rospy.logwarn("[DROP] Frame skipped due to invalid UWB or jump.")
            else:
                uwb_system()
        uwb_system.rate.sleep()


if __name__ == '__main__':
    run()
