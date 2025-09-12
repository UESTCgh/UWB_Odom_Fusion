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

        self.origin_prior_weight = 0.3
        self.odom_origin_noise = 0.1
        self.origin = np.zeros((1, DIM))            # 全局原点
        self._eps_norm = 1e-9                       # 防 0 的 ε

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
            
        # ==== 新增：到原点的距离（估计端） ====
        self.origin_distance_est = np.linalg.norm(self.curr_frame - self.origin, axis=1, keepdims=True)

        # ==== 新增：到原点的距离（里程计端，可用 gt+噪声来模拟里程计） ====
        self.odom_origin_distance = np.linalg.norm(
            self.gt[self.num_frames] - self.origin, axis=1, keepdims=True
        ) + self.generate_range_Gaussian_noise(size=(self.N, 1), dev=self.odom_origin_noise)

    
    def cal_jacobian_residual_origin_distance(self):
        """
        r3: (N_var, 1)    = ||x_i|| - d_i^{odom->origin}
        J3: (N_var, N_var*DIM)
        注：仅对参与优化的点产生梯度；固定块的点不在未知量里。
        """
        # 变量块索引与列偏移（和你现有规则一致）
        if self.mapping_step:
            var_ids = np.arange(0, self.N - self.right_fixed_point)    # 右侧固定 => 左边是变量
            col0 = 0
        else:
            var_ids = np.arange(self.left_fixed_point, self.N)         # 左侧固定 => 右边是变量
            col0 = self.left_fixed_point * self.DIM

        # 残差：估计到原点的距离 - 里程计到原点的距离
        r3 = (self.origin_distance_est[var_ids] - self.odom_origin_distance[var_ids])  # (N_var, 1)

        # 构造 J3：每个点 i 的行对应该点坐标的导数 x_i / ||x_i||
        Nvar = len(var_ids)
        J3 = np.zeros((Nvar, Nvar * self.DIM))
        for k, i in enumerate(var_ids):
            xi = self.curr_frame[i]  # (DIM,)
            norm_xi = np.linalg.norm(xi) + self._eps_norm
            # 对该点自身坐标的导数：x_i / ||x_i||
            grad = xi / norm_xi      # shape (DIM,)
            # 写入 J3 在该点对应的列块
            J3[k, k*self.DIM:(k+1)*self.DIM] = grad

        return J3, r3


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

        3.init-frame constraint
        J3  (N, (N-fixed)*2)    Jacobian
        r3  (N, 1)              residual
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
                              J3=None, r3=None, info_matrix3=None,   # 新增第三项
                              alpha=1., beta=0.5, zeta=None,         # 第三项权重 ζ
                              max_iter=1, thresh=1.e-5, step=0.1,
                              ep=1.0, lm=1.e-4):
        if info_matrix1 is None:
            info_matrix1 = np.eye(J1.shape[0])
        if info_matrix2 is None:
            info_matrix2 = np.eye(J2.shape[0])

        use_ori = (J3 is not None) and (r3 is not None)
        if use_ori:
            if info_matrix3 is None:
                info_matrix3 = np.eye(J3.shape[0])
            if zeta is None:
                zeta = self.origin_prior_weight

        info_matrix1 *= alpha
        info_matrix2 *= beta
        if use_ori:
            info_matrix3 *= zeta

        for _ in range(max_iter):
            H = J1.T @ info_matrix1 @ J1 + J2.T @ info_matrix2 @ J2
            v = -J1.T @ info_matrix1 @ r1 - J2.T @ info_matrix2 @ r2

            if use_ori:
                H += J3.T @ info_matrix3 @ J3
                v += -J3.T @ info_matrix3 @ r3

            H = H + (ep + lm * H) * np.eye(H.shape[0])  # damping
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
            
            J3, r3 = self.cal_jacobian_residual_origin_distance()
            self.gauss_newton_optimization(J1, r1, J2, r2,
                                       J3=J3, r3=r3,
                                       zeta=self.origin_prior_weight)
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
    parser.add_argument('--noise', type=float, default=0.1)
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
