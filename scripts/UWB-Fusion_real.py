#########################################################
# Organization:     UESTC (Shenzhen)                    #
# Author:           Chuxian, Li                         #
# Email:            lichuxian6666@gmail.com             #
# Github:           https://github.com/Lchuxian         #
#########################################################

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
np.set_printoptions(precision=5, suppress=True)


class TrajectoryGenerator:

    def __init__(self, N, DIM, init_min, init_max,
                 acceleration_max=0.1,
                 acceleration_change_rate=0.05,
                 velocity_max=0.5):
        self.N = N
        self.DIM = DIM
        self.is_initialized = False
        # initial point (N, DIM)
        self.points = np.random.uniform(init_min, init_max, size=(N, DIM))
        self.acceleration = np.zeros((N, DIM))
        self.velocity = np.zeros((N, DIM))
        self.acceleration_max = acceleration_max
        self.acceleration_change_rate = acceleration_change_rate
        self.velocity_max = velocity_max

    def __call__(self, mapping_step=True):
        if self.is_initialized:
            self.acceleration += np.random.uniform(-self.acceleration_change_rate, self.acceleration_change_rate, size=(self.N, self.DIM))
            self.acceleration = np.clip(self.acceleration, -self.acceleration_max, self.acceleration_max)
            self.velocity += self.acceleration
            self.velocity = np.clip(self.velocity, -self.velocity_max, self.velocity_max)
            if mapping_step:
                self.acceleration[1:] = 0.
                self.velocity[1:] = 0.
            else:
                self.acceleration[0] = 0.
                self.velocity[0] = 0.
            self.points += self.velocity
            return self.points
        else:
            self.is_initialized = True
            return self.points


class UWBFusion:

    def __init__(self, N, DIM, left_fixed_point=1, right_fixed_point=3, noise=0.1, iteration=8, buffer=4096):
        # dimension will be 1, 2 or 3
        assert DIM in [1, 2, 3]

        self.mapping_step = True

        self.N = N
        self.DIM = DIM
        self.left_fixed_point = left_fixed_point
        self.right_fixed_point = right_fixed_point
        self.noise = noise
        self.buffer = buffer
        self.num_frames = 0
        self.est = np.zeros((buffer, N, DIM))
        self.gt = np.zeros((buffer, N, DIM))
        self.tstamp = np.zeros((buffer, 1))

        # temp variable
        self.prev_frame = None
        self.last_frame = None
        self.curr_frame = None
        self.uwb_distance_matrix = None
        self.point_distance_matrix = None
        self.constraint_distance_matrix = None
        self.odom_distance_matrix = None

        # optimization setting
        self.iteration = iteration

        # TODO: sliding window
        self.frames_window = 8
        self.t0 = 0
        self.t1 = 0

    @staticmethod
    def generate_range_Gaussian_noise(size, dev=0.1):
        return np.random.normal(0, scale=dev, size=size)

    def visualization(self):
        plt.figure(figsize=(10, 10))
        for i in range(self.N):
        # i = 3
            plt.plot(self.gt[:self.num_frames, i, 0],
                     self.gt[:self.num_frames, i, 1], color='blue')
            plt.plot(self.est[:self.num_frames, i, 0],
                     self.est[:self.num_frames, i, 1], color='red')
        plt.title('UWB-Fusion Localization')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.grid(True)
        plt.show()

    def update_uwb_distance(self):
        # UWB distance applied Gaussian noise (N^2, 1)
        distances = pdist(self.gt[self.num_frames], metric='euclidean')
        self.uwb_distance_matrix = squareform(distances).reshape(-1, 1)
        for row in range(self.N**2):
            if (row // self.N) != (row % self.N):
                self.uwb_distance_matrix[row] += \
                    self.generate_range_Gaussian_noise(size=1, dev=self.noise)

    def update_constraint_distance(self):
        distances = pdist(self.curr_frame, metric='euclidean')
        self.point_distance_matrix = squareform(distances).reshape(-1, 1)
        self.constraint_distance_matrix = \
            np.linalg.norm(self.curr_frame - self.last_frame, axis=1)[..., None]
        if self.num_frames != 0:
            self.odom_distance_matrix = \
                np.linalg.norm(self.gt[self.num_frames] - self.gt[self.num_frames - 1], axis=1)[..., None] + \
                self.generate_range_Gaussian_noise(size=(self.N, 1), dev=0.001)
        else:
            self.odom_distance_matrix = \
                self.generate_range_Gaussian_noise(size=(self.N, 1), dev=0.001)

    def cal_jacobian_residual_fixed_left(self):
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
                if col_i < self.left_fixed_point:
                    continue
                elif row_i == col_i:
                    J1[row, col] = 2 * (self.curr_frame[col_i, col_j] - self.curr_frame[row_j, col_j])
                elif row_j == col_i:
                    J1[row, col] = 2 * (self.curr_frame[col_i, col_j] - self.curr_frame[row_i, col_j])
        J1 = J1[:, (self.left_fixed_point * self.DIM):]
        r1 = self.point_distance_matrix**2 - self.uwb_distance_matrix**2

        J2 = np.zeros((self.N, self.N * self.DIM))
        for row in range(J2.shape[0]):
            for col in range(J2.shape[1]):
                col_i, col_j = col // self.DIM, col % self.DIM
                # fixed point
                if col_i < self.left_fixed_point:
                    continue
                elif row == col_i:
                    J2[row, col] = 2 * (self.curr_frame[col_i, col_j] - self.last_frame[col_i, col_j])
        J2 = J2[:, self.left_fixed_point * self.DIM:]
        r2 = self.constraint_distance_matrix**2 - self.odom_distance_matrix**2
        return J1, r1, J2, r2

    def cal_jacobian_residual_fixed_right(self):
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
                if col_i > self.N - self.right_fixed_point - 1:
                    continue
                elif row_i == col_i:
                    J1[row, col] = 2 * (self.curr_frame[col_i, col_j] - self.curr_frame[row_j, col_j])
                elif row_j == col_i:
                    J1[row, col] = 2 * (self.curr_frame[col_i, col_j] - self.curr_frame[row_i, col_j])
        J1 = J1[:, :((self.N - self.right_fixed_point) * self.DIM)]
        r1 = self.point_distance_matrix**2 - self.uwb_distance_matrix**2

        J2 = np.zeros((self.N, self.N * self.DIM))
        for row in range(J2.shape[0]):
            for col in range(J2.shape[1]):
                col_i, col_j = col // self.DIM, col % self.DIM
                # fixed point
                if col_i > self.N - self.right_fixed_point - 1:
                    continue
                elif row == col_i:
                    J2[row, col] = 2 * (self.curr_frame[col_i, col_j] - self.last_frame[col_i, col_j])
        J2 = J2[:, :((self.N - self.right_fixed_point) * self.DIM)]
        r2 = self.constraint_distance_matrix**2 - self.odom_distance_matrix**2
        return J1, r1, J2, r2

    # Gaussian-Newton optimization
    def gauss_newton_optimization(self, J1, r1, J2, r2,
                                  info_matrix1=None, info_matrix2=None,
                                  alpha=1., beta=0.5,
                                  max_iter=1, thresh=1.e-5, step=0.1,
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
            if self.mapping_step:
                self.curr_frame[:(self.N - self.right_fixed_point)] = \
                    (self.curr_frame[:(self.N - self.right_fixed_point)].reshape(-1, 1) + step * delta).reshape(-1, self.DIM)
            else:
                self.curr_frame[self.left_fixed_point:] = \
                    (self.curr_frame[self.left_fixed_point:].reshape(-1, 1) + step * delta).reshape(-1, self.DIM)

            if np.linalg.norm(step * delta) < thresh:
                return

    def update_position(self, points):
        self.prev_frame = self.last_frame.copy()
        self.last_frame = self.curr_frame.copy()

        self.update_uwb_distance()
        for iter in range(self.iteration):
            self.update_constraint_distance()
            if self.mapping_step:
                J1, r1, J2, r2 = self.cal_jacobian_residual_fixed_right()
            else:
                J1, r1, J2, r2 = self.cal_jacobian_residual_fixed_left()
            self.gauss_newton_optimization(J1, r1, J2, r2)
        self.est[self.num_frames] = self.curr_frame.copy()

        if self.num_frames < 1:
            self.last_frame = self.curr_frame.copy()

    def __call__(self, t, points):
        self.tstamp[self.num_frames] = t
        self.gt[self.num_frames] = points.copy()

        # For simulation, ground-truth are used, for real environment, prior are used
        if self.num_frames == 0:
            self.last_frame = points.copy()
            self.curr_frame = points.copy()

        self.update_position(points)
        self.num_frames += 1


def get_param():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--N', type=int, default=4)
    parser.add_argument('--DIM', type=int, default=2)
    parser.add_argument('--noise', type=float, default=0.01)
    parser.add_argument('--init_min', type=float, default=-10.)
    parser.add_argument('--init_max', type=float, default=10.)
    parser.add_argument('--acceleration_max', type=float, default=0.005)
    parser.add_argument('--acceleration_change_rate', type=float, default=0.0002)
    parser.add_argument('--velocity_max', type=float, default=0.01)
    parser.add_argument('--frames', type=int, default=4096)
    parser.add_argument('--buffer', type=int, default=4096)

    parser.add_argument('--left_fixed_point', type=int, default=1)
    parser.add_argument('--right_fixed_point', type=int, default=3)
    parser.add_argument('--iteration', type=int, default=4)
    args = parser.parse_args()
    return args


def run():
    # get parameter
    args = get_param()

    # Trajectory Generator
    trajectory_generator = \
        TrajectoryGenerator(args.N, args.DIM, args.init_min, args.init_max,
                            args.acceleration_max, args.acceleration_change_rate, args.velocity_max)

    # UWB-Fusion
    uwb_system = UWBFusion(args.N, args.DIM, args.left_fixed_point, args.right_fixed_point,
                           args.noise, args.iteration, args.buffer)

    mapping_step = True
    for frame in range(args.frames):
        if frame == args.frames // 2:
            mapping_step = False
            uwb_system.mapping_step = False
            uwb_system.visualization()
        uwb_system(frame, trajectory_generator(mapping_step))

    uwb_system.visualization()

if __name__ == '__main__':
    run()
