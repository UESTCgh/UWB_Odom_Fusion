#########################################################
# Organization:     UESTC (Shenzhen)                    #
# Author:           Chuxian, Li                         #
# Email:            lichuxian6666@gmail.com             #
# Github:           https://github.com/Lchuxian         #
#########################################################

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
import rospy
from std_msgs.msg import Float32MultiArray
from geometry_msgs.msg import PoseArray, Pose
import pandas as pd

np.set_printoptions(precision=5, suppress=True)


class UWBFusion:
    def __init__(self, N, DIM, left_fixed_point=2, noise=0.1, iteration=8, num=1000000):
        assert DIM in [1, 2, 3]
        self.mapping_step = False
        self.N = N
        self.DIM = DIM
        self.left_fixed_point = left_fixed_point
        self.noise = noise
        self.num = num
        self.iteration = iteration
        self.num_frames = 0

        self.est = np.zeros((num, N, DIM))
        self.gt = np.zeros((num, N, DIM))
        self.tstamp = np.zeros((num, 1))
        self.prev_frame = self.last_frame = self.curr_frame = None

        self.initial_d01_list = []
        self.initial_d01 = None  # 记录平均0-1距离

        rospy.init_node('uwb_fusion_node', anonymous=True)
        self.pose_ready = self.dist_ready = False
        self.pose_input = self.uwb_distance_input = None

        rospy.Subscriber("/uwb1/pose_matrix", Float32MultiArray, self.pose_callback)
        rospy.Subscriber("/uwb1/distance_matrix", Float32MultiArray, self.dist_callback)
        self.pose_pub = rospy.Publisher('/uwb1/est_pose_array', PoseArray, queue_size=10)

        self.fig, self.ax = None, None
        self.init_plot()
        self.rate = rospy.Rate(10)

    @staticmethod
    def generate_range_Gaussian_noise(size, dev=0.1):
        return np.random.normal(0, scale=dev, size=size)

    def pose_callback(self, msg):
        self.pose_input = np.array(msg.data).reshape(self.N, 4)[:, :self.DIM]
        self.pose_ready = True

    def dist_callback(self, msg):
        self.uwb_distance_input = np.array(msg.data).reshape(self.N, self.N).reshape(-1, 1)
        self.dist_ready = True

    def init_plot(self):
        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(6, 6))
        self.ax.set_xlim(-20, 20)
        self.ax.set_ylim(-20, 20)
        self.ax.set_title("Real-Time UWB Fusion Localization")
        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        self.ax.grid(True)

    def realtime_visualization(self):
        self.ax.clear()
        self.ax.set_xlim(-20, 20)
        self.ax.set_ylim(-20, 20)
        self.ax.set_title("Real-Time UWB Fusion Localization")
        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        self.ax.grid(True)

        colors = ['r', 'g', 'm', 'c', 'y', 'k']
        for i in range(self.N):
            color = colors[i % len(colors)]
            self.ax.plot(self.gt[:self.num_frames, i, 0],
                         self.gt[:self.num_frames, i, 1], linestyle='--', color='gray', linewidth=1)
            self.ax.plot(self.est[:self.num_frames, i, 0],
                         self.est[:self.num_frames, i, 1], color=color, linewidth=2)
            self.ax.text(self.est[self.num_frames - 1, i, 0],
                         self.est[self.num_frames - 1, i, 1],
                         f"{i}", color=color, fontsize=10, fontweight='bold',
                         bbox=dict(facecolor='white', edgecolor='none', alpha=0.6))
        plt.draw()
        plt.pause(0.01)

    def update_uwb_distance(self):
        self.uwb_distance_matrix = self.uwb_distance_input.copy()

    def update_constraint_distance(self):
        distances = pdist(self.curr_frame, metric='euclidean')
        self.point_distance_matrix = squareform(distances).reshape(-1, 1)
        self.constraint_distance_matrix = np.linalg.norm(self.curr_frame - self.last_frame, axis=1)[..., None]
        if self.num_frames != 0:
            self.odom_distance_matrix = np.linalg.norm(self.gt[self.num_frames] - self.gt[self.num_frames - 1], axis=1)[..., None] + \
                                        self.generate_range_Gaussian_noise((self.N, 1), dev=0.001)
        else:
            self.odom_distance_matrix = self.generate_range_Gaussian_noise((self.N, 1), dev=0.001)

    def cal_jacobian_residual_fixed_left(self):
        if self.enable_d01_constraint and self.initial_d01 is not None:
            J1 = np.zeros((self.N ** 2, self.N * self.DIM))
            for row in range(J1.shape[0]):
                row_i, row_j = row // self.N, row % self.N
                if row_i == row_j:
                    continue
                for col in range(J1.shape[1]):
                    col_i, col_j = col // self.DIM, col % self.DIM
                    if col_i < self.left_fixed_point:
                        continue
                    elif row_i == col_i:
                        J1[row, col] = 2 * (self.curr_frame[col_i, col_j] - self.curr_frame[row_j, col_j])
                    elif row_j == col_i:
                        J1[row, col] = 2 * (self.curr_frame[col_i, col_j] - self.curr_frame[row_i, col_j])
            J1 = J1[:, self.left_fixed_point * self.DIM:]
            r1 = self.point_distance_matrix**2 - self.uwb_distance_matrix**2

            J2 = np.zeros((self.N, self.N * self.DIM))
            for row in range(J2.shape[0]):
                for col in range(J2.shape[1]):
                    col_i, col_j = col // self.DIM, col % self.DIM
                    if col_i < self.left_fixed_point:
                        continue
                    elif row == col_i:
                        J2[row, col] = 2 * (self.curr_frame[col_i, col_j] - self.last_frame[col_i, col_j])
            J2 = J2[:, self.left_fixed_point * self.DIM:]
            r2 = self.constraint_distance_matrix**2 - self.odom_distance_matrix**2
            return J1, r1, J2, r2

    def gauss_newton_optimization(self, J1, r1, J2, r2,
                                  alpha=1., beta=0.5,
                                  max_iter=1, thresh=1e-5, step=0.1,
                                  ep=1.0, lm=1e-4):
        info_matrix1 = np.eye(J1.shape[0]) * alpha
        info_matrix2 = np.eye(J2.shape[0]) * beta
        for _ in range(max_iter):
            H = J1.T @ info_matrix1 @ J1 + J2.T @ info_matrix2 @ J2
            v = -J1.T @ info_matrix1 @ r1 - J2.T @ info_matrix2 @ r2
            H += (ep + lm * H) * np.eye(H.shape[0])
            delta = np.linalg.solve(H, v)
            self.curr_frame[self.left_fixed_point:] = (self.curr_frame[self.left_fixed_point:].reshape(-1, 1) + step * delta).reshape(-1, self.DIM)
            if np.linalg.norm(step * delta) < thresh:
                break

    def update_position(self, points):
        self.prev_frame = self.last_frame.copy()
        self.last_frame = self.curr_frame.copy()
        self.update_uwb_distance()

        for _ in range(self.iteration):
            self.update_constraint_distance()
            J1, r1, J2, r2 = self.cal_jacobian_residual_fixed_left()
            self.gauss_newton_optimization(J1, r1, J2, r2)

        # 手动修正 agent 1 的位置
        if self.enable_d01_constraint and self.initial_d01 is not None:
            self.curr_frame[1] = self.curr_frame[0] - np.array([-self.initial_d01, 0.0])

        self.est[self.num_frames] = self.curr_frame.copy()

        if self.num_frames < 1:
            self.last_frame = self.curr_frame.copy()

    def publish_estimation(self):
        msg = PoseArray()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = "map"
        for i in range(self.N):
            p = Pose()
            p.position.x = self.curr_frame[i][0]
            p.position.y = self.curr_frame[i][1]
            msg.poses.append(p)
        self.pose_pub.publish(msg)

    def save_results(self):
        est_flat = self.est[:self.num_frames].reshape(self.num_frames, -1)
        gt_flat = self.gt[:self.num_frames].reshape(self.num_frames, -1)
        df = pd.DataFrame(np.hstack((self.tstamp[:self.num_frames], est_flat, gt_flat)))
        df.to_csv("uwb_fusion_result.csv", header=False, index=False)

    def spin(self):
        while not rospy.is_shutdown():
            if self.pose_ready and self.dist_ready:
                self.pose_ready = self.dist_ready = False
                t = rospy.get_time()
                points = self.pose_input.copy()
                self.tstamp[self.num_frames] = t
                self.gt[self.num_frames] = points

                if self.num_frames < 50:
                    # 从距离矩阵中读取 agent 0-1 的距离
                    idx_01 = self.N * 0 + 1
                    d01 = self.uwb_distance_input[idx_01][0]
                    self.initial_d01_list.append(d01)
                    rospy.loginfo(f"[Frame {self.num_frames}] Recording d01 (from UWB) = {d01:.4f}")

                    if self.num_frames == 0:
                        self.curr_frame = self.pose_input.copy()
                        self.last_frame = self.pose_input.copy()
                    self.est[self.num_frames] = self.curr_frame.copy()
                else:
                    if self.num_frames == 50:
                        self.initial_d01 = np.mean(self.initial_d01_list)
                        self.enable_d01_constraint = True
                        rospy.loginfo(f"Initialized fixed d01 = {self.initial_d01:.4f}")
                        self.curr_frame = points.copy()
                        self.last_frame = points.copy()

                    self.update_position(points)
                    self.est[self.num_frames] = self.curr_frame.copy()

                self.num_frames += 1
                self.realtime_visualization()
                self.publish_estimation()
            self.rate.sleep()
        self.save_results()



def get_param():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--N', type=int, default=4)
    parser.add_argument('--DIM', type=int, default=2)
    parser.add_argument('--noise', type=float, default=0.01)
    parser.add_argument('--iteration', type=int, default=4)
    return parser.parse_args()


def run():
    args = get_param()
    uwb_system = UWBFusion(args.N, args.DIM, left_fixed_point=2, noise=args.noise, iteration=args.iteration)
    uwb_system.spin()


if __name__ == '__main__':
    run()
