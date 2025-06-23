#!/usr/bin/env python3
# encoding: utf-8
import numpy as np
import rospy
from std_msgs.msg import Float32MultiArray
from geometry_msgs.msg import PoseArray, Pose, Point
from visualization_msgs.msg import Marker, MarkerArray
from sklearn.manifold import MDS
from collections import deque

# Global variables
last_coords = None
initial_orientation = None
reference_frame = None
NUM_NODES = 4  # 修改为4个节点

# ---------- Filters ----------
class AverageSmoother:
    def __init__(self, window_size=5):
        self.history = deque(maxlen=window_size)

    def smooth(self, new_value):
        self.history.append(new_value)
        return np.mean(np.array(self.history), axis=0)

class KalmanFilter1D:
    def __init__(self, process_var=1e-4, measurement_var=1e-2):
        self.x, self.P = 0, 1
        self.Q, self.R = process_var, measurement_var

    def update(self, z):
        self.P += self.Q
        K = self.P / (self.P + self.R)
        self.x += K * (z - self.x)
        self.P = (1 - K) * self.P
        return self.x

# ---------- MDS Wrapper ----------
class FixedMDS(MDS):
    def fit(self, X, y=None, init=None):
        X = np.asarray(X)
        n = X.shape[0]
        if init is None:
            init = self.random_state_.randn(n, self.n_components)

        for _ in range(10):
            dis = np.linalg.norm(init[:, None, :] - init[None, :, :], axis=-1)
            B = np.zeros_like(dis)
            mask = dis != 0
            B[mask] = -X[mask] / dis[mask]
            B[np.diag_indices_from(B)] = -B.sum(axis=1)
            init -= 0.01 * B @ init

        return init

# ---------- Coordinates Processing ----------
def align_with_x_axis(coords, id0=1, id1=0):
    """Aligns coordinates to make ID0 the origin and ID1 on positive x-axis"""
    aligned_coords = coords.copy()
    origin = aligned_coords[id0]
    aligned_coords = aligned_coords - origin
    x_vec = aligned_coords[id1]
    angle = -np.arctan2(x_vec[1], x_vec[0])
    rot = np.array([
        [np.cos(angle), -np.sin(angle)],
        [np.sin(angle),  np.cos(angle)],
    ])
    return aligned_coords @ rot.T

def detect_and_fix_flips(current_coords, previous_coords):
    """Detects and fixes coordinate flips using multiple strategies"""
    fixed_coords = current_coords.copy()
    
    prev_vectors = previous_coords - previous_coords[0]
    curr_vectors = fixed_coords - fixed_coords[0]
    
    x_flip_votes = y_flip_votes = 0
    for i in range(1, len(prev_vectors)):
        if abs(prev_vectors[i, 0]) > 0.1 and abs(curr_vectors[i, 0]) > 0.1:
            if np.sign(prev_vectors[i, 0]) != np.sign(curr_vectors[i, 0]):
                x_flip_votes += 1
        if abs(prev_vectors[i, 1]) > 0.1 and abs(curr_vectors[i, 1]) > 0.1:
            if np.sign(prev_vectors[i, 1]) != np.sign(curr_vectors[i, 1]):
                y_flip_votes += 1
    
    def compute_pairwise_distances(coords):
        n = len(coords)
        distances = np.zeros((n, n))
        for i in range(n):
            for j in range(i+1, n):
                distances[i, j] = distances[j, i] = np.linalg.norm(coords[i] - coords[j])
        return distances
    
    prev_dists = compute_pairwise_distances(previous_coords)
    curr_dists = compute_pairwise_distances(fixed_coords)
    flip_x_diff = np.sum(np.abs(compute_pairwise_distances(fixed_coords * np.array([-1, 1])) - prev_dists))
    flip_y_diff = np.sum(np.abs(compute_pairwise_distances(fixed_coords * np.array([1, -1])) - prev_dists))
    flip_both_diff = np.sum(np.abs(compute_pairwise_distances(fixed_coords * np.array([-1, -1])) - prev_dists))
    
    orig_diff = np.sum(np.abs(curr_dists - prev_dists))
    flip_threshold = max(1, (len(prev_vectors) - 1) // 2)
    applied_strategy = None
    
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
    global initial_orientation, reference_frame
    
    fixed_coords = coords.copy()
    
    if reference_frame is None and initial_orientation is not None:
        reference_frame = initial_orientation.copy()
        return fixed_coords
    
    if reference_frame is None:
        initial_orientation = coords.copy()
        return fixed_coords
    
    if fixed_coords[1, 0] < 0:
        fixed_coords[:, 0] = -fixed_coords[:, 0]
        rospy.logwarn("参考框架检测: ID1不在ID0右侧，应用X轴翻转")
    
    ref_angle = np.arctan2(reference_frame[1, 1] - reference_frame[0, 1], 
                          reference_frame[1, 0] - reference_frame[0, 0])
    curr_angle = np.arctan2(fixed_coords[1, 1] - fixed_coords[0, 1], 
                           fixed_coords[1, 0] - fixed_coords[0, 0])
    
    angle_diff = (curr_angle - ref_angle) % (2*np.pi)
    if angle_diff > np.pi:
        angle_diff -= 2*np.pi
    
    if abs(abs(angle_diff) - np.pi) < 0.5:
        rospy.logwarn(f"参考框架检测到反向 (角度差: {angle_diff:.2f}弧度)，应用整体翻转")
        fixed_coords = fixed_coords * np.array([-1, -1])
    
    return fixed_coords

def correct_single_point_anomalies(coords, previous_coords):
    fixed_coords = coords.copy()
    n_points = len(coords)
    
    movements = np.linalg.norm(coords - previous_coords, axis=1)
    avg_movement = np.mean(movements)
    std_movement = np.std(movements)
    
    anomaly_threshold = max(0.2, avg_movement + 3 * std_movement)
    
    for i in range(2, n_points):  # 仍然跳过ID0和ID1
        if movements[i] < anomaly_threshold:
            continue
            
        original_dist = movements[i]
        x_mirror = np.array([-fixed_coords[i,0], fixed_coords[i,1]])
        y_mirror = np.array([fixed_coords[i,0], -fixed_coords[i,1]])
        both_mirror = np.array([-fixed_coords[i,0], -fixed_coords[i,1]])
        
        x_mirror_dist = np.linalg.norm(x_mirror - previous_coords[i])
        y_mirror_dist = np.linalg.norm(y_mirror - previous_coords[i])
        both_mirror_dist = np.linalg.norm(both_mirror - previous_coords[i])
        
        options = [
            ("原始", original_dist, fixed_coords[i]),
            ("X轴镜像", x_mirror_dist, x_mirror),
            ("Y轴镜像", y_mirror_dist, y_mirror),
            ("整体镜像", both_mirror_dist, both_mirror)
        ]
        
        best = min(options, key=lambda x: x[1])
        
        if best[0] != "原始" and best[1] < original_dist * 0.5:
            fixed_coords[i] = best[2]
            rospy.logwarn(f"应用{best[0]}修正ID{i} (距离改善: {original_dist-best[1]:.2f}m)")
    
    return fixed_coords

# ---------- Distance Matrix Filtering ----------
def apply_kalman_filter(raw_matrix):
    filt = np.zeros((NUM_NODES, NUM_NODES))  # 改为4x4
    for i in range(NUM_NODES):
        for j in range(NUM_NODES):
            filt[i][j] = 0.0 if i == j else kalman_filters[i][j].update(raw_matrix[i][j])
    return filt

# ---------- ROS Callback ----------
def callback(msg):
    global last_coords, initial_orientation, reference_frame

    raw_matrix = np.array(msg.data).reshape(NUM_NODES, NUM_NODES)  # 改为4x4
    filtered = apply_kalman_filter(raw_matrix)
    smoothed = dist_smoother.smooth(filtered)

    coords = mds.fit_transform(smoothed)
    coords = align_with_x_axis(coords)
    
    if last_coords is None:
        if coords[2, 1] < 0:
            coords[:, 1] = -coords[:, 1]
            
        last_coords = coords.copy()
        initial_orientation = coords.copy()
        rospy.loginfo("初始化坐标系完成: ID1在ID0右侧")
    else:
        coords = detect_and_fix_flips(coords, last_coords)
        coords = enforce_consistent_orientation(coords)
        coords = correct_single_point_anomalies(coords, last_coords)
    
    coords = coord_smoother.smooth(coords)
    last_coords = coords.copy()
    
    if reference_frame is None and initial_orientation is not None:
        frame_stabilizer_count = getattr(callback, 'frame_count', 0) + 1
        setattr(callback, 'frame_count', frame_stabilizer_count)
        
        if frame_stabilizer_count >= 10:
            reference_frame = initial_orientation.copy()
            rospy.loginfo("参考框架已稳定确立")

    rospy.loginfo_throttle(0.5, "\n" + "\n".join(
        [f"ID {i}: ({coords[i][0]:.2f}, {coords[i][1]:.2f})" for i in range(NUM_NODES)]
    ))

    publish_poses(coords)
    publish_ids(coords)
    publish_axes(coords)

# ---------- ROS Publishers ----------
def publish_poses(coords):
    pose_array = PoseArray()
    pose_array.header.frame_id = "map"
    pose_array.header.stamp = rospy.Time.now()
    for i in range(NUM_NODES):  # 改为4个
        pose = Pose()
        pose.position.x = coords[i][0]
        pose.position.y = coords[i][1]
        pose.position.z = 0.0
        pose.orientation.w = 1.0
        pose_array.poses.append(pose)
    pose_array_pub.publish(pose_array)

def publish_ids(coords):
    marker_array = MarkerArray()
    for i in range(NUM_NODES):  # 改为4个
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = rospy.Time.now()
        marker.ns = "uwb_ids"
        marker.id = i
        marker.type = Marker.TEXT_VIEW_FACING
        marker.action = Marker.ADD
        marker.pose.position.x = coords[i][0]
        marker.pose.position.y = coords[i][1]
        marker.pose.position.z = 0.3
        marker.scale.z = 0.2
        marker.color.a = 1.0
        marker.color.r = marker.color.g = marker.color.b = 1.0
        marker.text = f"{i}"
        marker_array.markers.append(marker)
    id_pub.publish(marker_array)

def publish_axes(coords):
    marker_array = MarkerArray()
    colors = [
        (1.0, 0.0, 0.0), (0.7, 0.0, 0.0),
        (0.0, 1.0, 0.0), (0.0, 0.7, 0.0),
        (1.0, 0.5, 0.0)
    ]
    
    directions = [
        (coords[0,0], coords[0,1], coords[0,0]+0.5, coords[0,1]),
        (coords[0,0], coords[0,1], coords[0,0]-0.5, coords[0,1]),
        (coords[0,0], coords[0,1], coords[0,0], coords[0,1]+0.5),
        (coords[0,0], coords[0,1], coords[0,0], coords[0,1]-0.5),
        (coords[0,0], coords[0,1], coords[1,0], coords[1,1])
    ]
    
    for i, ((r, g, b), (x1, y1, x2, y2)) in enumerate(zip(colors, directions)):
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = rospy.Time.now()
        marker.ns = "axes"
        marker.id = i + 1
        marker.type = Marker.ARROW
        marker.action = Marker.ADD
        marker.scale.x = 0.5 if i != 4 else 0.2
        marker.scale.y = marker.scale.z = 0.05 if i != 4 else 0.03
        marker.color.a = 1.0
        marker.color.r, marker.color.g, marker.color.b = r, g, b
        marker.points = [Point(x=x1, y=y1, z=0.0), Point(x=x2, y=y2, z=0.0)]
        marker_array.markers.append(marker)
    
    axes_pub.publish(marker_array)

# ---------- Main ----------
if __name__ == "__main__":
    rospy.init_node('mds_pose_solver')

    pose_array_pub = rospy.Publisher('/pose_array', PoseArray, queue_size=10)
    id_pub = rospy.Publisher('/pose_id_marker', MarkerArray, queue_size=10)
    axes_pub = rospy.Publisher('/coordinate_axes', MarkerArray, queue_size=10)

    rospy.Subscriber('/uwb1/distance_matrix', Float32MultiArray, callback)

    # 初始化4x4卡尔曼滤波器矩阵
    kalman_filters = [[KalmanFilter1D() for _ in range(NUM_NODES)] for _ in range(NUM_NODES)]
    dist_smoother = AverageSmoother(window_size=5)
    coord_smoother = AverageSmoother(window_size=5)
    mds = FixedMDS(n_components=2, dissimilarity='precomputed', random_state=42)

    rospy.spin()
