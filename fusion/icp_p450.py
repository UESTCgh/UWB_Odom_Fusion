#########################################################
# Organization:     UESTC (Shenzhen)                    #
# Author:           Hao,Guo                            #
# Email:            1417150646@qq.com                   #
# Github:           https://github.com/UESTCgh          #
#########################################################

#!/usr/bin/env python3
import rospy
import numpy as np
from std_msgs.msg import Float32MultiArray
from geometry_msgs.msg import PoseArray, Pose, PoseStamped
from nav_msgs.msg import Path
from math import cos, sin
from scipy.optimize import least_squares
import threading

class ID3ICPAligner:
    def __init__(self):
        rospy.init_node("id3_odom_to_uwb_icp")

        # 轨迹数组（不限制长度）
        self.odom_traj = []
        self.uwb_traj = []
        self.odom_raw_traj = []    # 原始 odom（未对齐）
        self.odom_icp_traj = []    # 对齐后的 odom
        self.uwb_full_traj = []    # 完整 UWB 轨迹
        self.lock = threading.Lock()

        self.last_odom_point = None
        self.icp_enabled = False
        self.icp_tx, self.icp_ty, self.icp_theta = 0.0, 0.0, 0.0
        self.move_thresh = 0.02  # 2cm 为“移动”判定
        self.has_started = False  # 是否已触发首次移动


        # 订阅话题
        rospy.Subscriber("/uwb1/pose_matrix", Float32MultiArray, self.odom_callback)
        rospy.Subscriber("/uwb1/uwb_locator", PoseArray, self.uwb_callback)

        # 发布路径
        self.pub_pose = rospy.Publisher("/id3/odom_in_uwb", PoseArray, queue_size=1)
        self.pub_path_raw = rospy.Publisher("/id3/odom_raw_path", Path, queue_size=1)
        self.pub_path_icp = rospy.Publisher("/id3/odom_icp_path", Path, queue_size=1)
        self.pub_path_uwb = rospy.Publisher("/id3/uwb_path", Path, queue_size=1)
        self.pub_icp_T = rospy.Publisher("/T_odom2uwb", PoseStamped, queue_size=1)

        # 启动定时器进行对齐
        rospy.Timer(rospy.Duration(0.1), self.align_and_publish)

        rospy.loginfo("ICP轨迹对齐节点已启动")

    def odom_callback(self, msg):
        try:
            data = np.array(msg.data).reshape(-1, 4)
            if data.shape[0] > 3:
                pt = data[3, :2]
                with self.lock:
                    self.odom_traj.append(pt)

                    if self.last_odom_point is not None:
                        dist = np.linalg.norm(pt - self.last_odom_point)

                        if dist > self.move_thresh:
                            if not self.has_started:
                                rospy.loginfo("🚶 Odom 初次移动，开始记录轨迹")
                                self.odom_raw_traj.clear()
                                self.uwb_full_traj.clear()
                                self.odom_icp_traj.clear()
                                self.has_started = True

                            self.icp_enabled = True  # 每次动都触发一次 ICP

                    self.last_odom_point = pt

                    if self.has_started:
                        self.odom_raw_traj.append(pt.copy())
        except Exception as e:
            rospy.logerr(f"[odom_callback] 错误: {e}")



    def uwb_callback(self, msg):
        try:
            if len(msg.poses) > 3:
                pt = msg.poses[3].position
                pt_arr = np.array([pt.x, pt.y])
                with self.lock:
                    self.uwb_traj.append(pt_arr)

                    # 仅当 odom 在移动时记录用于 ICP 的 UWB 轨迹
                    if self.has_started and self.icp_enabled:
                        self.uwb_full_traj.append(pt_arr.copy())
        except Exception as e:
            rospy.logerr(f"[uwb_callback] 错误: {e}")


    def align_and_publish(self, _):
        with self.lock:
            if not self.has_started:
                return
            if len(self.odom_traj) < 5 or len(self.uwb_traj) < 5:
                return

            min_len = min(len(self.odom_traj), len(self.uwb_traj))
            src = np.array(self.odom_traj[-min_len:])
            dst = np.array(self.uwb_traj[-min_len:])

            # 仅在移动时重新执行 ICP
            if self.icp_enabled:
                def residual(params):
                    tx, ty, theta = params
                    R = np.array([[cos(theta), -sin(theta)], [sin(theta), cos(theta)]])
                    transformed = src @ R.T + np.array([tx, ty])
                    return (transformed - dst).ravel()

                result = least_squares(residual, x0=np.zeros(3), max_nfev=100)
                self.icp_tx, self.icp_ty, self.icp_theta = result.x

                # 发布当前对齐变换 T（odom 到 uwb）
                pose = PoseStamped()
                pose.header.stamp = rospy.Time.now()
                pose.header.frame_id = "uwb"

                pose.pose.position.x = self.icp_tx
                pose.pose.position.y = self.icp_ty
                pose.pose.position.z = 0.0

                # 将 theta 转为四元数
                theta = self.icp_theta
                pose.pose.orientation.z = np.sin(theta / 2.0)
                pose.pose.orientation.w = np.cos(theta / 2.0)

                self.pub_icp_T.publish(pose)

                self.icp_enabled = False  # 已计算，锁定当前变换

                rospy.loginfo_throttle(2.0,
                    f"[ICP] T = [tx={self.icp_tx:.2f}, ty={self.icp_ty:.2f}, θ={self.icp_theta:.2f} rad], cost={result.cost:.4f}")

            # 不管是否更新，只要有 T，就应用到全轨迹
            R = np.array([[cos(self.icp_theta), -sin(self.icp_theta)],
                        [sin(self.icp_theta),  cos(self.icp_theta)]])
            full_raw = np.array(self.odom_raw_traj)
            aligned = full_raw @ R.T + np.array([self.icp_tx, self.icp_ty])
            self.odom_icp_traj = aligned.tolist()

        self.publish_pose_array(aligned)
        self.publish_paths()


    def publish_pose_array(self, points):
        pa = PoseArray()
        pa.header.stamp = rospy.Time.now()
        pa.header.frame_id = "uwb"
        for pt in points:
            pose = Pose()
            pose.position.x = pt[0]
            pose.position.y = pt[1]
            pose.position.z = 0.0
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
                pose.pose.position.z = 0.0
                pose.pose.orientation.w = 1.0
                path.poses.append(pose)
            return path

        self.pub_path_raw.publish(create_path(raw))
        self.pub_path_icp.publish(create_path(aligned))
        self.pub_path_uwb.publish(create_path(uwb))

if __name__ == "__main__":
    try:
        ID3ICPAligner()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
