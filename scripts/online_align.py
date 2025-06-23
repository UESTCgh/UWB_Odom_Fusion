#!/usr/bin/env python3
import rospy
import os
import numpy as np
from scipy.optimize import least_squares
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from visualization_msgs.msg import MarkerArray
from sensor_msgs import point_cloud2
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header
from collections import defaultdict
from functools import partial

# 轨迹对应关系配置（源ID -> 目标话题）
TRAJECTORY_PAIRS = {
    'id_0': '/vrpn_client_node/UAV_exploration450/pose',
    'id_1': '/vrpn_client_node/uav206/pose',
    'id_2': '/vrpn_client_node/MASK2/pose',
    'id_3': '/vrpn_client_node/MARK1/pose'
}

class OnlineICPNode:
    def __init__(self):
        rospy.init_node('online_icp_visualization')
        
        # 初始化参数
        self.log_dir = 'robot_logs'
        os.makedirs(self.log_dir, exist_ok=True)  # 创建日志目录
        self.max_points = 1000        # 每轨迹最大缓存点数
        self.min_align_points = 5     # 最小对齐点数
        self.initial_guess = np.zeros(3)  # 初始变换参数（tx, ty, theta）
        
        # 轨迹缓存（key: 轨迹ID_类型，value: {'timestamps': [], 'points': np.ndarray}）
        self.trajectory_cache = defaultdict(lambda: {'timestamps': [], 'points': np.empty((0, 2))})
        
        # 对齐轨迹发布者字典
        self.aligned_pubs = {}
        
        # 发布者初始化
        self.marker_pub = rospy.Publisher('icp_visualization', MarkerArray, queue_size=10)
        self.pointcloud_pub = rospy.Publisher('icp_pointcloud', PointCloud2, queue_size=10)
        
        # 订阅话题
        self.subscribe_topics()
        rospy.loginfo("在线ICP节点启动，等待轨迹数据...")

    def subscribe_topics(self):
        """订阅所有轨迹相关话题"""
        # 订阅目标轨迹（VRPN等）
        for src_id, topic in TRAJECTORY_PAIRS.items():
            rospy.Subscriber(topic, PoseStamped, self.trajectory_callback, callback_args=(src_id, 'target'))
            rospy.loginfo(f"已订阅目标轨迹: {topic} (ID:{src_id})")
        
        # 订阅源轨迹（Marker/里程计）
        rospy.Subscriber('/pose_id_marker', MarkerArray, self.marker_callback)
        rospy.Subscriber('/odom', Odometry, self.odometry_callback)

    def trajectory_callback(self, msg: PoseStamped, args):
        """处理目标轨迹数据（VRPN等设备）"""
        src_id, traj_type = args
        self._update_cache(src_id, traj_type, msg.header.stamp, (msg.pose.position.x, msg.pose.position.y))

    def marker_callback(self, msg: MarkerArray):
        """处理源轨迹数据（MarkerArray）"""
        for marker in msg.markers:
            src_id = f"id_{marker.id}"
            self._update_cache(src_id, 'source', marker.header.stamp, 
                            (marker.pose.position.x, marker.pose.position.y))
            self.run_icp(src_id)  # 新数据到达时触发对齐

    def odometry_callback(self, msg: Odometry):
        """处理里程计数据"""
        self._update_cache('odom', 'source', msg.header.stamp, 
                        (msg.pose.pose.position.x, msg.pose.pose.position.y))

    def _update_cache(self, src_id: str, traj_type: str, stamp, point: tuple):
        """更新轨迹缓存（带长度限制）"""
        key = f"{src_id}_{traj_type}"
        cache = self.trajectory_cache[key]
        
        # 添加新点（时间戳和点数同步）
        cache['timestamps'].append(rospy.Time.to_sec(stamp))
        new_point = np.array([[point[0], point[1]]])
        cache['points'] = np.concatenate([cache['points'], new_point])
        
        # 保持最大点数限制（FIFO机制）
        if self.max_points is not None and len(cache['points']) > self.max_points:
            cache['timestamps'].pop(0)
            cache['points'] = cache['points'][1:]
        
        # 写入原始数据日志（保留历史数据）
        with open(f"{self.log_dir}/{key}_raw.log", 'a') as f:
            f.write(f"{rospy.Time.to_sec(stamp):.6f} {point[0]:.6f} {point[1]:.6f}\n")

    def run_icp(self, src_id: str):
        """执行ICP对齐并更新可视化"""
        src_key = f"{src_id}_source"
        dst_key = f"{src_id}_target"
        
        src_data = self.trajectory_cache[src_key]
        dst_data = self.trajectory_cache[dst_key]
        
        src_ts = np.array(src_data['timestamps'])
        src_points = src_data['points']
        dst_ts = np.array(dst_data['timestamps'])
        dst_points = dst_data['points']
        
        # 检查基础点数
        if len(src_points) < self.min_align_points:
            rospy.logdebug(f"源轨迹点数不足（当前{len(src_points)}，需要{self.min_align_points}）")
            return
        if len(dst_points) < self.min_align_points:
            rospy.logwarn(f"目标轨迹点数不足（当前{len(dst_points)}，需要{self.min_align_points}），请检查{TRAJECTORY_PAIRS[src_id]}话题是否正常发布")
            return
        
        # 时间匹配点对（获取完整匹配点集）
        point_pairs = self._match_points_by_timestamp(src_ts, src_points, dst_ts, dst_points)
        if len(point_pairs) < self.min_align_points:
            rospy.logwarn(f"有效点对不足（{len(point_pairs)} < {self.min_align_points}）")
            return
        
        src_pairs, dst_pairs = zip(*point_pairs)
        src_pairs = np.array(src_pairs)
        dst_pairs = np.array(dst_pairs)
        
        try:
            # 执行ICP对齐（获取整条轨迹的对齐结果）
            aligned, tx, ty, theta, cost = self.icp_align(src_pairs, dst_pairs)
            self.initial_guess = [tx, ty, theta]  # 更新初始猜测
            
            # 计算整条轨迹的RPE RMSE
            if aligned.size == 0 or dst_pairs.size == 0:
                rospy.logwarn("对齐后轨迹为空，无法计算RMSE")
                rmse = np.nan
            else:
                # 计算x和y分量的误差平方和
                errors = aligned - dst_pairs
                squared_errors = np.sum(errors**2, axis=1)  # 每个点的(x² + y²)
                rmse = np.sqrt(np.mean(squared_errors))  # 均方根误差
            
            # 发布可视化（仅最新点）
            self.publish_visualization(src_id, src_pairs, dst_pairs, aligned)
            
            # 保存最新对齐点
            if len(aligned) > 0:
                latest_ts = src_ts[-1]
                latest_point = aligned[-1]
                self.save_aligned_trajectory(src_id, latest_ts, latest_point)
            
            # 发布对齐轨迹话题
            self.publish_aligned_trajectory(src_id, src_ts, aligned)
            
            # 日志输出（包含轨迹级RMSE）
            rospy.loginfo(
                f"对齐完成 | ID:{src_id[-1]} | "
                f"平移: ({tx:.2f}, {ty:.2f}) | 旋转: {theta:.2f}rad | "
                f"优化误差: {cost:.4f} | RPE RMSE: {rmse:.4f}m"
            )
            
        except Exception as e:
            rospy.logerr(f"ID:{src_id[-1]} ICP失败: {str(e)}")

    def save_aligned_trajectory(self, src_id: str, timestamp: float, point: np.ndarray):
        """保存最新对齐点到txt文件"""
        file_path = os.path.join(self.log_dir, f"{src_id}_latest_aligned.txt")
        try:
            with open(file_path, 'w') as f:  # 覆盖写入最新点
                f.write(f"{timestamp:.6f} {point[0]:.6f} {point[1]:.6f}\n")
            rospy.loginfo(f"已保存最新对齐点到{file_path}")
        except Exception as e:
            rospy.logerr(f"保存失败: {str(e)}")

    def publish_aligned_trajectory(self, src_id: str, timestamps: np.ndarray, points: np.ndarray):
        """发布对齐轨迹到ROS话题"""
        if src_id not in self.aligned_pubs:
            topic_name = f"/icp_aligned/{src_id}/pose"
            self.aligned_pubs[src_id] = rospy.Publisher(topic_name, PoseStamped, queue_size=10)
            rospy.loginfo(f"创建对齐轨迹发布者: {topic_name}")
        
        # 仅发布最新点
        if len(timestamps) > 0 and len(points) > 0:
            ts = timestamps[-1]
            x, y = points[-1]
            pose_msg = PoseStamped()
            pose_msg.header.stamp = rospy.Time.from_sec(ts)
            pose_msg.header.frame_id = "map"
            pose_msg.pose.position.x = x
            pose_msg.pose.position.y = y
            pose_msg.pose.position.z = 0.0
            pose_msg.pose.orientation.w = 1.0
            self.aligned_pubs[src_id].publish(pose_msg)

    def _match_points_by_timestamp(self, src_ts, src_points, dst_ts, dst_points):
        """根据时间戳匹配源和目标点"""
        point_pairs = []
        for ts, src_p in zip(src_ts, src_points):
            if not dst_ts.size:
                continue
            idx = np.argmin(np.abs(dst_ts - ts))
            dst_p = dst_points[idx]
            point_pairs.append((src_p, dst_p))
        return point_pairs

    def publish_visualization(self, src_id: str, src, dst, aligned):
        """发布最新点的可视化（仅显示最新一个点）"""
        # 获取最新点（若为空则使用原点）
        src_latest = src[-1:] if len(src) > 0 else np.array([[0, 0]])
        dst_latest = dst[-1:] if len(dst) > 0 else np.array([[0, 0]])
        aligned_latest = aligned[-1:] if len(aligned) > 0 else np.array([[0, 0]])
        
        # 源轨迹最新点：蓝色，大点
        src_cloud = self._create_pointcloud(
            src_latest, 
            "map", 
            color=(0, 0, 255),       # 纯蓝色
            alpha=1.0, 
            point_size=0.3
        )
        
        # 目标轨迹最新点：红色，大点
        dst_cloud = self._create_pointcloud(
            dst_latest, 
            "map", 
            color=(1, 0, 0),       # 纯红色
            alpha=1.0, 
            point_size=0.3
        )
        
        # 对齐轨迹最新点：绿色，更大点
        aligned_cloud = self._create_pointcloud(
            aligned_latest, 
            "map", 
            color=(0, 255, 0),       # 纯绿色
            alpha=1.0, 
            point_size=0.4
        )
        
        # self.pointcloud_pub.publish(src_cloud)
        self.pointcloud_pub.publish(dst_cloud)
        # self.pointcloud_pub.publish(aligned_cloud)

    def _create_pointcloud(self, points: np.ndarray, frame_id: str, color: tuple, 
                          alpha: float, point_size: float) -> PointCloud2:
        """生成单点点云（固定颜色，无渐变）"""
        header = Header(stamp=rospy.Time.now(), frame_id=frame_id)
        fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name="rgba", offset=12, datatype=PointField.UINT32, count=1),
        ]
        
        # 固定颜色处理
        rgba = np.zeros((len(points), 4), dtype=np.uint8)
        rgba[:, 0] = int(color[0] * 255)  # 红色分量
        rgba[:, 1] = int(color[1] * 255)  # 绿色分量
        rgba[:, 2] = int(color[2] * 255)  # 蓝色分量
        rgba[:, 3] = int(alpha * 255)     # 透明度（完全不透明）
        
        # 构建点云数据
        points_list = []
        for (x, y) in points:
            rgba_val = (rgba[0, 0] << 24) | (rgba[0, 1] << 16) | (rgba[0, 2] << 8) | rgba[0, 3]
            points_list.append((x, y, 0.0, rgba_val))
        
        return point_cloud2.create_cloud(header, fields, points_list)

    def icp_align(self, src: np.ndarray, dst: np.ndarray, method: str = 'point_to_point', max_iter: int = 100):
        """二维ICP对齐函数"""
        if len(src) < 1 or len(dst) < 1:
            raise ValueError("轨迹至少需要1个点才能对齐")
        
        if src.shape[0] != dst.shape[0]:
            raise ValueError("源和目标点对数量必须一致")
        
        if method == 'point_to_point':
            residuals = partial(self.p2p_residuals, src=src, dst=dst)
        elif method == 'point_to_line':
            residuals = partial(self.p2l_residuals, src=src, dst=dst)
        else:
            raise ValueError("method必须为'point_to_point'或'point_to_line'")
        
        result = least_squares(residuals, self.initial_guess, max_nfev=max_iter, 
                               ftol=1e-6, xtol=1e-6)
        
        if not result.success:
            rospy.logwarn(f"ICP优化未完全收敛（状态：{result.status}）")
        
        tx, ty, theta = result.x
        R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        aligned = src @ R.T + np.array([tx, ty])
        return aligned, tx, ty, theta, result.cost

    def p2p_residuals(self, params: np.ndarray, src: np.ndarray, dst: np.ndarray) -> np.ndarray:
        """点到点残差函数"""
        tx, ty, theta = params
        R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        transformed = src @ R.T + np.array([tx, ty])
        return (transformed - dst).ravel()

    def p2l_residuals(self, params: np.ndarray, src: np.ndarray, dst: np.ndarray) -> np.ndarray:
        """点到线残差函数"""
        tx, ty, theta = params
        R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        transformed = src @ R.T + np.array([tx, ty])
        
        residuals = []
        for p in transformed:
            distances = np.linalg.norm(dst - p, axis=1)
            idx = np.argsort(distances)
            if len(idx) < 2:
                closest_point = dst[idx[0]] if len(idx) > 0 else p
                residuals.append(p - closest_point)
                continue
            
            a, b = dst[idx[0]], dst[idx[1]]
            ab = b - a
            ab_norm_sq = np.dot(ab, ab)
            if ab_norm_sq < 1e-8:
                projection = a
            else:
                t = np.dot(p - a, ab) / ab_norm_sq
                t = np.clip(t, 0, 1)
                projection = a + t * ab
            residuals.append(p - projection)
        
        return np.array(residuals).ravel()

if __name__ == '__main__':
    try:
        node = OnlineICPNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        rospy.loginfo("节点关闭")
