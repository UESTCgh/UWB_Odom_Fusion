#!/usr/bin/env python3
# encoding: utf-8
import numpy as np
import rospy
from std_msgs.msg import Float32MultiArray
from geometry_msgs.msg import PoseArray, Pose, Point
from visualization_msgs.msg import Marker, MarkerArray
from sklearn.manifold import MDS
from collections import deque

# 全局变量，用于维持状态
last_coords = None          # 存储上一帧的坐标，用于稳定性比较
initial_orientation = None  # 初始坐标系方向
reference_frame = None      # 稳定后的参考坐标系

# ---------- 滤波器 ----------
class AverageSmoother:
    """
    滑动平均滤波器，通过对历史数据取平均值来平滑数据
    """
    def __init__(self, window_size=5):
        self.history = deque(maxlen=window_size)  # 存储最近的window_size个值

    def smooth(self, new_value):
        self.history.append(new_value)  # 添加新值到历史记录
        return np.mean(np.array(self.history), axis=0)  # 返回平均值

class KalmanFilter1D:
    """
    一维卡尔曼滤波器实现，用于降低噪声
    基于噪声测量值维持对真实状态的估计
    """
    def __init__(self, process_var=1e-4, measurement_var=1e-2):
        self.x = 0  # 状态估计
        self.P = 1  # 误差协方差
        self.Q = process_var     # 过程噪声方差
        self.R = measurement_var # 测量噪声方差

    def update(self, z):
        # 预测步骤
        self.P += self.Q
        
        # 更新步骤
        K = self.P / (self.P + self.R)  # 卡尔曼增益
        self.x += K * (z - self.x)      # 更新状态估计
        self.P = (1 - K) * self.P       # 更新误差协方差
        
        return self.x  # 返回滤波后的估计值

# ---------- MDS包装器 ----------
class FixedMDS(MDS):
    """
    多维缩放(MDS)的自定义实现
    实现了一个固定算法，将距离矩阵转换为坐标，而不依赖随机初始化
    """
    def fit(self, X, y=None, init=None):
        X = np.asarray(X)
        n = X.shape[0]
        if init is None:
            # 如果没有提供初始坐标，则使用随机坐标
            init = self.random_state_.randn(n, self.n_components)

        # 使用应力最小化对坐标进行迭代优化
        for _ in range(10):
            # 计算当前坐标之间的成对距离
            dis = np.linalg.norm(init[:, None, :] - init[None, :, :], axis=-1)
            
            # 计算用于更新坐标的梯度(B矩阵)
            B = np.zeros_like(dis)
            mask = dis != 0
            B[mask] = -X[mask] / dis[mask]  # 力与差异成正比
            B[np.diag_indices_from(B)] = -B.sum(axis=1)  # 确保零和
            
            # 使用梯度下降步骤更新坐标
            init -= 0.01 * B @ init

        return init  # 返回优化后的坐标

# ---------- 坐标处理 ----------
def align_with_x_axis(coords, id0=1, id1=0):
    """
    将坐标对齐，使ID0成为原点，ID1位于正x轴上
    这会创建一个一致的坐标系，以ID0为参考点
    
    参数:
        coords: 输入坐标数组
        id0: 锚点ID(将成为原点)
        id1: 用于与正x轴对齐的点ID
        
    返回:
        具有一致方向的变换坐标
    """
    aligned_coords = coords.copy()
    
    # 将ID0移至原点
    origin = aligned_coords[id0]
    aligned_coords = aligned_coords - origin
    
    # 计算旋转，使ID1与正x轴对齐
    x_vec = aligned_coords[id1]
    angle = -np.arctan2(x_vec[1], x_vec[0])
    
    # 创建旋转矩阵
    rot = np.array([
        [np.cos(angle), -np.sin(angle)],
        [np.sin(angle),  np.cos(angle)],
    ])
    
    # 对所有点应用旋转
    return aligned_coords @ rot.T

def detect_and_fix_flips(current_coords, previous_coords):
    """
    检测并修复坐标翻转，使用多种策略
    MDS可能产生相同解的镜像反射，此函数确保帧间的一致性
    
    参数:
        current_coords: 当前帧的坐标
        previous_coords: 上一帧的坐标
        
    返回:
        修正翻转后的坐标
    """
    # 复制以避免修改原始数据
    fixed_coords = current_coords.copy()
    
    # 计算两个帧中相对于ID0的向量方向
    prev_vectors = previous_coords - previous_coords[0]
    curr_vectors = fixed_coords - fixed_coords[0]
    
    # 计算X和Y方向的翻转
    x_flip_votes = y_flip_votes = 0
    for i in range(1, len(prev_vectors)):
        # 检查显著向量以避免噪声引起的误检测
        if abs(prev_vectors[i, 0]) > 0.1 and abs(curr_vectors[i, 0]) > 0.1:
            if np.sign(prev_vectors[i, 0]) != np.sign(curr_vectors[i, 0]):
                x_flip_votes += 1
        if abs(prev_vectors[i, 1]) > 0.1 and abs(curr_vectors[i, 1]) > 0.1:
            if np.sign(prev_vectors[i, 1]) != np.sign(curr_vectors[i, 1]):
                y_flip_votes += 1
    
    # 计算所有可能翻转的成对距离，找到最佳匹配
    def compute_pairwise_distances(coords):
        n = len(coords)
        distances = np.zeros((n, n))
        for i in range(n):
            for j in range(i+1, n):
                distances[i, j] = distances[j, i] = np.linalg.norm(coords[i] - coords[j])
        return distances
    
    # 计算原始和所有可能翻转版本的距离
    prev_dists = compute_pairwise_distances(previous_coords)
    curr_dists = compute_pairwise_distances(fixed_coords)
    flip_x_dists = compute_pairwise_distances(fixed_coords * np.array([-1, 1]))
    flip_y_dists = compute_pairwise_distances(fixed_coords * np.array([1, -1]))
    flip_both_dists = compute_pairwise_distances(fixed_coords * np.array([-1, -1]))
    
    # 计算距离差异以找到最佳配置
    orig_diff = np.sum(np.abs(curr_dists - prev_dists))
    flip_x_diff = np.sum(np.abs(flip_x_dists - prev_dists))
    flip_y_diff = np.sum(np.abs(flip_y_dists - prev_dists))
    flip_both_diff = np.sum(np.abs(flip_both_dists - prev_dists))
    
    # 根据投票和距离指标确定要应用的翻转
    flip_threshold = max(1, (len(prev_vectors) - 1) // 2)
    applied_strategy = None
    
    # 基于投票和距离差异应用翻转
    if (x_flip_votes >= flip_threshold and y_flip_votes >= flip_threshold) or \
       (flip_both_diff < orig_diff * 0.8 and flip_both_diff == min(orig_diff, flip_x_diff, flip_y_diff, flip_both_diff)):
        fixed_coords = fixed_coords * np.array([-1, -1])
        applied_strategy = "整体翻转(xy)"
    elif (x_flip_votes >= flip_threshold and x_flip_votes > y_flip_votes) or \
         (flip_x_diff < orig_diff * 0.8 and flip_x_diff == min(orig_diff, flip_x_diff, flip_y_diff)):
        fixed_coords[:, 0] = -fixed_coords[:, 0]
        applied_strategy = "X轴翻转"
    elif y_flip_votes >= flip_threshold or \
         (flip_y_diff < orig_diff * 0.8 and flip_y_diff == min(orig_diff, flip_y_diff)):
        fixed_coords[:, 1] = -fixed_coords[:, 1]
        applied_strategy = "Y轴翻转"
    
    if applied_strategy:
        rospy.logwarn(f"应用{applied_strategy}修正")
    
    return fixed_coords

def enforce_consistent_orientation(coords):
    """
    确保坐标与初始帧保持一致的方向
    此函数强制实施规范方向，其中:
    - ID1位于ID0的正X轴上
    - ID2位于ID0的正Y轴上
    
    参数:
        coords: 当前坐标
        
    返回:
        方向校正后的坐标
    """
    global initial_orientation, reference_frame
    
    fixed_coords = coords.copy()
    
    # 如果需要，初始化参考框架
    if reference_frame is None and initial_orientation is not None:
        reference_frame = initial_orientation.copy()
        return fixed_coords
    
    if reference_frame is None:
        initial_orientation = coords.copy()
        return fixed_coords
    
    # 强制ID1在正X轴上
    if fixed_coords[1, 0] < 0:
        fixed_coords[:, 0] = -fixed_coords[:, 0]
        rospy.logwarn("参考框架检测: ID1不在ID0右侧，应用X轴翻转")
    
    # 强制ID2在正Y轴上
    if fixed_coords[2, 1] < 0:
        fixed_coords[:, 1] = -fixed_coords[:, 1]
        rospy.logwarn("参考框架检测: ID2不在ID0上方，应用Y轴翻转")
    
    # 通过比较ID0和ID1之间的角度检查180度翻转
    ref_angle = np.arctan2(reference_frame[1, 1] - reference_frame[0, 1], 
                          reference_frame[1, 0] - reference_frame[0, 0])
    curr_angle = np.arctan2(fixed_coords[1, 1] - fixed_coords[0, 1], 
                           fixed_coords[1, 0] - fixed_coords[0, 0])
    
    # 计算角度差并归一化到[-π, π]
    angle_diff = (curr_angle - ref_angle) % (2*np.pi)
    if angle_diff > np.pi:
        angle_diff -= 2*np.pi
    
    # 检测并修复180度旋转
    if abs(abs(angle_diff) - np.pi) < 0.5:
        rospy.logwarn(f"参考框架检测到反向 (角度差: {angle_diff:.2f}弧度)，应用整体翻转")
        fixed_coords = fixed_coords * np.array([-1, -1])
    
    return fixed_coords

def detect_and_fix_flips(current_coords, previous_coords, angle_threshold=np.pi / 2):
    """
    根据向量夹角判断当前坐标是否翻转。若与上一帧方向相差过大，则整体翻转。

    参数:
        current_coords: 当前帧坐标
        previous_coords: 上一帧坐标
        angle_threshold: 允许最大角度偏差（默认90度）

    返回:
        修正后的坐标
    """
    fixed_coords = current_coords.copy()

    # 用 ID0->ID1 的向量方向作为参考
    prev_vec = previous_coords[1] - previous_coords[0]
    curr_vec = fixed_coords[1] - fixed_coords[0]

    # 归一化向量
    prev_unit = prev_vec / np.linalg.norm(prev_vec)
    curr_unit = curr_vec / np.linalg.norm(curr_vec)

    # 计算夹角余弦
    dot_product = np.dot(prev_unit, curr_unit)
    angle = np.arccos(np.clip(dot_product, -1.0, 1.0))

    # 若夹角接近 180°，说明发生了整体翻转
    if abs(angle - np.pi) < angle_threshold:
        rospy.logwarn(f"检测到方向反转 (夹角 {np.degrees(angle):.1f}°)，执行整体翻转")
        fixed_coords *= np.array([-1, -1])

    return fixed_coords


# ---------- 距离矩阵滤波 ----------
def apply_kalman_filter(raw_matrix):
    """
    对距离矩阵应用卡尔曼滤波
    单独过滤每个距离测量的噪声
    
    参数:
        raw_matrix: 来自UWB传感器的原始距离矩阵
        
    返回:
        滤波后的距离矩阵
    """
    filt = np.zeros((5, 5))
    for i in range(5):
        for j in range(5):
            # 跳过对角线(自身的距离始终为0)
            filt[i][j] = 0.0 if i == j else kalman_filters[i][j].update(raw_matrix[i][j])
    return filt

# ---------- ROS回调 ----------
def callback(msg):
    """
    处理传入距离矩阵消息的ROS回调函数
    将距离转换为坐标并应用多种校正策略
    
    参数:
        msg: 包含距离矩阵的ROS消息
    """
    global last_coords, initial_orientation, reference_frame

    # 使用滤波器处理距离矩阵
    raw_matrix = np.array(msg.data).reshape(5, 5)  # 扁平阵重塑为5x5矩阵
    filtered = apply_kalman_filter(raw_matrix)     # 应用卡尔曼滤波
    smoothed = dist_smoother.smooth(filtered)      # 应用时间平滑

    # 使用MDS计算坐标
    coords = mds.fit_transform(smoothed)
    
    # 对齐坐标(ID0在原点，ID1在X轴上)
    coords = align_with_x_axis(coords)
    
    # 首帧特殊处理
    if last_coords is None:
        # 确保ID2在ID0上方
        if coords[2, 1] < 0:
            coords[:, 1] = -coords[:, 1]
            
        last_coords = coords.copy()
        initial_orientation = coords.copy()
        rospy.loginfo(f"初始化坐标系完成: ID1在ID0右侧, ID2在ID0上方")
    else:
        # 应用多阶段校正管道
        coords = detect_and_fix_flips(coords, last_coords)
        coords = enforce_consistent_orientation(coords)
        # coords = correct_single_point_anomalies(coords, last_coords)
    
    # 应用最终坐标平滑
    coords = coord_smoother.smooth(coords)
    
    # 更新上一帧
    last_coords = coords.copy()
    
    # 如果需要，更新参考框架
    if reference_frame is None and initial_orientation is not None:
        frame_stabilizer_count = getattr(callback, 'frame_count', 0) + 1
        setattr(callback, 'frame_count', frame_stabilizer_count)
        
        # 10帧稳定后，锁定参考框架
        if frame_stabilizer_count >= 10:
            reference_frame = initial_orientation.copy()
            rospy.loginfo("参考框架已稳定确立")

    # 记录当前位置(节流以减少控制台输出)
    rospy.loginfo_throttle(0.5, "\n" + "\n".join(
        [f"ID {i}: ({coords[i][0]:.2f}, {coords[i][1]:.2f})" for i in range(5)]
    ))

    # 将结果发布到ROS主题
    publish_poses(coords)
    publish_ids(coords)
    publish_axes(coords)

# ---------- ROS发布器 ----------
def publish_poses(coords):
    """
    将计算的坐标作为ROS PoseArray消息发布
    
    参数:
        coords: 2D坐标数组
    """
    pose_array = PoseArray()
    pose_array.header.frame_id = "map"
    pose_array.header.stamp = rospy.Time.now()
    
    for i in range(5):
        pose = Pose()
        pose.position.x = coords[i][0]
        pose.position.y = coords[i][1]
        pose.position.z = 0.0           # 将Z设为0(2D定位)
        pose.orientation.w = 1.0        # 单位四元数(无旋转)
        pose_array.poses.append(pose)
        
    pose_array_pub.publish(pose_array)

def publish_ids(coords):
    """
    发布ID标记以在RViz中可视化
    在每个点上方显示ID号
    
    参数:
        coords: 2D坐标数组
    """
    marker_array = MarkerArray()
    
    for i in range(5):
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = rospy.Time.now()
        marker.ns = "uwb_ids"
        marker.id = i
        marker.type = Marker.TEXT_VIEW_FACING  # 面向相机的文本
        marker.action = Marker.ADD
        marker.pose.position.x = coords[i][0]
        marker.pose.position.y = coords[i][1]
        marker.pose.position.z = 0.3           # 文本悬浮在点上方
        marker.scale.z = 0.2                   # 文本大小
        marker.color.a = 1.0                   # 完全不透明
        marker.color.r = marker.color.g = marker.color.b = 1.0  # 白色文本
        marker.text = f"{i}"                   # 显示ID号
        
        marker_array.markers.append(marker)
        
    id_pub.publish(marker_array)

def publish_axes(coords):
    """
    发布坐标轴以在RViz中可视化
    显示以ID0为中心的X+/-和Y+/-轴，以及从ID0到ID1的主轴
    
    参数:
        coords: 2D坐标数组
    """
    marker_array = MarkerArray()
    
    # 为不同轴定义颜色
    colors = [
        (1.0, 0.0, 0.0),  # X+ (红色)
        (0.7, 0.0, 0.0),  # X- (深红色)
        (0.0, 1.0, 0.0),  # Y+ (绿色)
        (0.0, 0.7, 0.0),  # Y- (深绿色)
        (1.0, 0.5, 0.0)   # 主轴 (橙色)
    ]
    
    # 定义轴方向
    directions = [
        (coords[0,0], coords[0,1], coords[0,0]+0.5, coords[0,1]),      # X+
        (coords[0,0], coords[0,1], coords[0,0]-0.5, coords[0,1]),      # X-
        (coords[0,0], coords[0,1], coords[0,0], coords[0,1]+0.5),      # Y+
        (coords[0,0], coords[0,1], coords[0,0], coords[0,1]-0.5),      # Y-
        (coords[0,0], coords[0,1], coords[1,0], coords[1,1])           # 主轴 (ID0->ID1)
    ]
    
    # 为每个轴创建并配置标记
    for i, ((r, g, b), (x1, y1, x2, y2)) in enumerate(zip(colors, directions)):
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = rospy.Time.now()
        marker.ns = "axes"
        marker.id = i + 1
        marker.type = Marker.ARROW
        marker.action = Marker.ADD
        
        # 主轴箭头较细
        marker.scale.x = 0.5 if i != 4 else 0.2
        marker.scale.y = marker.scale.z = 0.05 if i != 4 else 0.03
        
        # 设置颜色
        marker.color.a = 1.0
        marker.color.r, marker.color.g, marker.color.b = r, g, b
        
        # 设置起点和终点
        marker.points = [Point(x=x1, y=y1, z=0.0), Point(x=x2, y=y2, z=0.0)]
        
        marker_array.markers.append(marker)
    
    axes_pub.publish(marker_array)

# ---------- 主函数 ----------
if __name__ == "__main__":
    rospy.init_node('mds_pose_solver')

    # 初始化ROS发布器
    pose_array_pub = rospy.Publisher('/pose_array', PoseArray, queue_size=10)
    id_pub = rospy.Publisher('/pose_id_marker', MarkerArray, queue_size=10)
    axes_pub = rospy.Publisher('/coordinate_axes', MarkerArray, queue_size=10)

    # 订阅UWB距离矩阵
    rospy.Subscriber('/uwb2/distance_matrix', Float32MultiArray, callback)

    # 初始化滤波器
    kalman_filters = [[KalmanFilter1D() for _ in range(5)] for _ in range(5)]
    dist_smoother = AverageSmoother(window_size=5)
    coord_smoother = AverageSmoother(window_size=5)
    mds = FixedMDS(n_components=2, dissimilarity='precomputed', random_state=42)

    rospy.spin()  # 保持节点运行直到关闭
