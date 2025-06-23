#########################################################
# Organization:     UESTC (Shenzhen)                    #
# Author:           Hao,Guo                            #
# Email:            1417150646@qq.com                   #
# Github:           https://github.com/UESTCgh          #
#########################################################
 
#!/usr/bin/env python3
import rospy
import numpy as np
import argparse
from std_msgs.msg import Float32MultiArray
from geometry_msgs.msg import PoseArray, Pose, PoseStamped
from nav_msgs.msg import Path
from math import cos, sin
from scipy.optimize import least_squares
import threading
import os

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--id", type=int, default=3, help="要对齐的 odom/uwb 索引行")
    parser.add_argument("--move_thresh", type=float, default=0.02, help="移动判定阈值")
    return parser.parse_args()

class ID3ICPAligner:
    def __init__(self, args):
        self.id = args.id
        self.move_thresh = args.move_thresh
        rospy.init_node(f"id{self.id}_odom_to_uwb_icp")

        # 轨迹数组（不限制长度）
        self.odom_traj = []
        self.uwb_traj = []
        self.odom_raw_traj = []
        self.odom_icp_traj = []
        self.uwb_full_traj = []
        self.lock = threading.Lock()

        self.last_odom_point = None
        self.icp_enabled = False
        self.icp_tx, self.icp_ty, self.icp_theta = 0.0, 0.0, 0.0
        self.has_started = False

        # 订阅
        rospy.Subscriber("/uwb1/pose_matrix", Float32MultiArray, self.odom_callback)
        rospy.Subscriber("/uwb1/uwb_locator", PoseArray, self.uwb_callback)

        # 发布
        prefix = f"/id{self.id}"
        self.pub_pose = rospy.Publisher(f"{prefix}/odom_in_uwb", PoseArray, queue_size=1)
        self.pub_path_raw = rospy.Publisher(f"{prefix}/odom_raw_path", Path, queue_size=1)
        self.pub_path_icp = rospy.Publisher(f"{prefix}/odom_icp_path", Path, queue_size=1)
        self.pub_path_uwb = rospy.Publisher(f"{prefix}/uwb_path", Path, queue_size=1)
        self.pub_icp_T = rospy.Publisher(f"{prefix}/T_odom2uwb", PoseStamped, queue_size=1)

        rospy.Timer(rospy.Duration(0.1), self.align_and_publish)
        rospy.loginfo(f"ICP轨迹对齐节点启动，监听ID={self.id}")

    def odom_callback(self, msg):
        try:
            data = np.array(msg.data).reshape(-1, 4)
            if data.shape[0] > self.id:
                pt = data[self.id, :2]
                with self.lock:
                    self.odom_traj.append(pt)
                    if self.last_odom_point is not None:
                        dist = np.linalg.norm(pt - self.last_odom_point)
                        if dist > self.move_thresh:
                            if not self.has_started:
                                rospy.loginfo("Odom 初次移动，开始记录轨迹")
                                self.odom_raw_traj.clear()
                                self.uwb_full_traj.clear()
                                self.odom_icp_traj.clear()
                                self.has_started = True
                            self.icp_enabled = True
                    self.last_odom_point = pt
                    if self.has_started:
                        self.odom_raw_traj.append(pt.copy())
        except Exception as e:
            rospy.logerr(f"[odom_callback] 错误: {e}")

    def uwb_callback(self, msg):
        try:
            if len(msg.poses) > self.id:
                pt = msg.poses[self.id].position
                pt_arr = np.array([pt.x, pt.y])
                with self.lock:
                    self.uwb_traj.append(pt_arr)
                    if self.has_started and self.icp_enabled:
                        self.uwb_full_traj.append(pt_arr.copy())
        except Exception as e:
            rospy.logerr(f"[uwb_callback] 错误: {e}")

    def align_and_publish(self, _):
        with self.lock:
            if not self.has_started or len(self.odom_traj) < 5 or len(self.uwb_traj) < 5:
                return
            min_len = min(len(self.odom_traj), len(self.uwb_traj))
            src = np.array(self.odom_traj[-min_len:])
            dst = np.array(self.uwb_traj[-min_len:])

            if self.icp_enabled:
                def residual(params):
                    tx, ty, theta = params
                    R = np.array([[cos(theta), -sin(theta)], [sin(theta), cos(theta)]])
                    transformed = src @ R.T + np.array([tx, ty])
                    return (transformed - dst).ravel()

                result = least_squares(residual, x0=np.zeros(3), max_nfev=100)
                self.icp_tx, self.icp_ty, self.icp_theta = result.x

                # 计算每个点的变换后误差
                R = np.array([[cos(self.icp_theta), -sin(self.icp_theta)],
                            [sin(self.icp_theta), cos(self.icp_theta)]])
                transformed = src @ R.T + np.array([self.icp_tx, self.icp_ty])
                errors = np.linalg.norm(transformed - dst, axis=1)
                mean_error = np.mean(errors)
             
                # 发布变换后的位姿
                pose = PoseStamped()
                pose.header.stamp = rospy.Time.now()
                pose.header.frame_id = "uwb"
                pose.pose.position.x = self.icp_tx
                pose.pose.position.y = self.icp_ty
                pose.pose.orientation.z = np.sin(self.icp_theta / 2.0)
                pose.pose.orientation.w = np.cos(self.icp_theta / 2.0)
                self.pub_icp_T.publish(pose)

                rospy.loginfo_throttle(1.0,
                    f"[ICP] T{self.id} = [tx={self.icp_tx:.2f}, ty={self.icp_ty:.2f}, θ={self.icp_theta:.2f} rad], "
                    f"cost={result.cost:.4f}, mean_error={mean_error:.4f} m")
                self.icp_enabled = False

            R = np.array([[cos(self.icp_theta), -sin(self.icp_theta)], [sin(self.icp_theta), cos(self.icp_theta)]])
            aligned = np.array(self.odom_raw_traj) @ R.T + np.array([self.icp_tx, self.icp_ty])
            self.odom_icp_traj = aligned.tolist()

        self.publish_pose_array(self.odom_icp_traj)
        self.publish_paths()

    def publish_pose_array(self, points):
        pa = PoseArray()
        pa.header.stamp = rospy.Time.now()
        pa.header.frame_id = "uwb"
        for pt in points:
            pose = Pose()
            pose.position.x = pt[0]
            pose.position.y = pt[1]
            pose.orientation.w = 1.0
            pa.poses.append(pose)
        self.pub_pose.publish(pa)

    def publish_paths(self):
        now = rospy.Time.now()
        frame_id = "uwb"
        with self.lock:
            raw = list(self.odom_raw_traj)
            aligned = list(self.odom_icp_traj)
            uwb = list(self.uwb_full_traj)

        def create_path(traj_pts):
            path = Path()
            path.header.stamp = now
            path.header.frame_id = frame_id
            for pt in traj_pts:
                pose = PoseStamped()
                pose.header.stamp = now
                pose.header.frame_id = frame_id
                pose.pose.position.x = pt[0]
                pose.pose.position.y = pt[1]
                pose.pose.orientation.w = 1.0
                path.poses.append(pose)
            return path

        self.pub_path_raw.publish(create_path(raw))
        self.pub_path_icp.publish(create_path(aligned))
        self.pub_path_uwb.publish(create_path(uwb))

    def save_traj_as_tum(self, traj, filename, folder="trajectories"):
        # 创建保存文件夹（如果不存在）
        os.makedirs(folder, exist_ok=True)

        filepath = os.path.join(folder, filename)
        with open(filepath, 'w') as f:
            t = 0.0
            for pt in traj:
                tx, ty = pt
                tz = 0.0
                qx, qy, qz, qw = 0.0, 0.0, 0.0, 1.0  # 无旋转
                f.write(f"{t:.6f} {tx:.6f} {ty:.6f} {tz:.6f} {qx} {qy} {qz} {qw}\n")
                t += 0.1  # 模拟10Hz，可按实际调整
        rospy.loginfo(f"[TUM Export] 保存轨迹至: {filepath}")

if __name__ == "__main__":
    args = get_args()
    node = None
    try:
        node = ID3ICPAligner(args)
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
    finally:
        if node is not None:
            if len(node.odom_traj) > 0:
                node.save_traj_as_tum(node.odom_traj, f"id{node.id}_odom.tum")
            if len(node.uwb_traj) > 0:
                node.save_traj_as_tum(node.uwb_traj, f"id{node.id}_uwb.tum")

