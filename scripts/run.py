#########################################################
# Organization:     UESTC (Shenzhen)                    #
# Author:           Chuxian, Li                         #
# Email:            lichuxian6666@gmail.com             #
# Github:           https://github.com/Lchuxian         #
#########################################################

import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from collections import deque

# Apply ROS
import rospy
from std_msgs.msg import Float32MultiArray
from geometry_msgs.msg import PoseArray, Pose

np.set_printoptions(precision=5, suppress=True)


class UWBFusion:

    def __init__(self, N: int, DIM: int,
                 pose_topic: str, dist_topic: str, pub_topic: str,
                 iteration=8, buffer=128):
        # dimension will be 1, 2 or 3
        assert DIM in [1, 2, 3]

        self.N = N
        self.DIM = DIM
        self.left_fixed_point = 0
        self.right_fixed_point = 0
        self.tstamp = None
        self.num_frames = 0
        self.est = deque(maxlen=buffer)
        self.gt = deque(maxlen=buffer)

        # temp variable
        self.is_initialized = False
        self.pose_ready = False
        self.dist_ready = False

        self.odom_last_frame = None
        self.odom_curr_frame = None
        self.last_frame = None
        self.curr_frame = None
        self.uwb_distance_matrix = None
        self.point_distance_matrix = None
        self.constraint_distance_matrix = None
        self.odom_distance_matrix = None

        # optimization setting
        self.iteration = iteration

        # Apply ROS
        rospy.Subscriber(pose_topic, Float32MultiArray, self.pose_callback)
        rospy.Subscriber(dist_topic, Float32MultiArray, self.dist_callback)
        self.pose_publisher = rospy.Publisher(pub_topic, PoseArray, queue_size=10)

        self.fig, self.ax = None, None
        self.init_plot()
        self.rate = rospy.Rate(10)

        # TODO: sliding window
        self.frames_window = 8
        self.t0 = 0
        self.t1 = 0

    def init_plot(self):
        self.colors = ['r', 'g', 'm', 'c', 'y', 'k']
        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(6, 6))
        self.ax.set_xlim(-5, 5)
        self.ax.set_ylim(-5, 5)
        self.ax.set_title("Real-Time UWB Fusion Localization")
        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        self.ax.grid(True)

    def realtime_visualization(self):
        self.ax.clear()
        # self.ax.set_xlim(-5, 5)
        # self.ax.set_ylim(-5, 5)
        # self.ax.set_title("Real-Time UWB Fusion Localization")
        # self.ax.set_xlabel("X")
        # self.ax.set_ylabel("Y")
        # self.ax.grid(True)

        est = np.stack(list(self.est), axis=0)
        # gt = np.stack(list(self.gt), axis=0)

        for i in range(self.N):
            color = self.colors[i % len(self.colors)]
            self.ax.plot(est[:, i, 0], est[:, i, 1], color=color, linewidth=2)
            self.ax.text(est[-1, i, 0], est[-1, i, 1],
                         f"{i}", color=color, fontsize=10, fontweight='bold',
                         bbox=dict(facecolor='white', edgecolor='none', alpha=0.6))
            # self.ax.plot(gt[:, i, 0], gt[:, i, 1], linestyle='--', color='gray', linewidth=1)

        plt.draw()
        plt.pause(0.0001)

    def pose_callback(self, msg):
        # Odometry poses (N, DIM)
        self.odom_curr_frame = np.array(msg.data).reshape(self.N, -1)[:, :self.DIM]
        self.pose_ready = True

    def dist_callback(self, msg):
        # UWB distances matrix (N^2, 1)
        self.uwb_distance_matrix = np.array(msg.data).reshape(-1, 1)
        self.dist_ready = True

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
        r1  (N, 1)              residual
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
        return J1, r1, J2, r2

    # Gaussian-Newton optimization
    def gauss_newton_optimization(self, J1, r1, J2, r2,
                                  info_matrix1=None, info_matrix2=None,
                                  alpha=1., beta=1.,
                                  max_iter=1, thresh=1.e-5, step=0.05,
                                  ep=1.0, lm=1.e-4):
        if info_matrix1 is None:
            info_matrix1 = np.eye(J1.shape[0])
        if info_matrix2 is None:
            info_matrix2 = np.eye(J2.shape[0])
        info_matrix1 *= alpha
        info_matrix2 *= beta

        for i in range(max_iter):
            H = J1.T @ info_matrix1 @ J1 + J2.T @ info_matrix2 @ J2
            v = -J1.T @ info_matrix1 @ r1 - J2.T @ info_matrix2 @ r2
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
            J1, r1, J2, r2 = self.cal_jacobian_residual()
            self.gauss_newton_optimization(J1, r1, J2, r2)

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

    def __call__(self, t):
        self.tstamp = t
        # self.gt.append(self.odom_curr_frame.copy())

        t_start = time.time()

        # System init
        if not self.is_initialized:
            self.system_init()

        self.update_position()
        rospy.loginfo(f'Localization time cost: {time.time() - t_start}')

        self.est.append(self.curr_frame.copy())
        self.realtime_visualization()

        # Publish uwb pose
        pose_array_msg = PoseArray()
        pose_array_msg.header.stamp = rospy.Time.now()
        pose_array_msg.header.frame_id = "uwb"

        for i in range(self.N):
            pose = Pose()
            pose.position.x = self.curr_frame[i, 0]
            pose.position.y = self.curr_frame[i, 1]
            pose.position.z = 0.0 if self.DIM < 3 else self.curr_frame[i, 2]
            pose_array_msg.poses.append(pose)

        self.pose_publisher.publish(pose_array_msg)

        # log
        rospy.loginfo(f'Total time cost: {time.time() - t_start}')
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
    parser.add_argument('--pub_topic', type=str, default='/uwb1/uwb_locator')
    parser.add_argument('--iteration', type=int, default=4)
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
    uwb_system.left_fixed_point = 2
    uwb_system.right_fixed_point = 0

    tstamp = 0
    while not rospy.is_shutdown():
        if uwb_system.pose_ready and uwb_system.dist_ready:
            uwb_system(tstamp)


if __name__ == '__main__':
    run()
