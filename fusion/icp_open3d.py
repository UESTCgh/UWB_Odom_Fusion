#########################################################
# Organization:     UESTC (Shenzhen)
# Author:           Hao,Guo
# Email:            1417150646@qq.com
# Github:           https://github.com/UESTCgh
#########################################################

#!/usr/bin/env python3
import rospy
import numpy as np
import argparse
from std_msgs.msg import Float32MultiArray
from geometry_msgs.msg import PoseArray, Pose, PoseStamped
from nav_msgs.msg import Path
from math import atan2
import threading
import open3d as o3d

# 使用Open3D ICP，并在运动不足时跳过计算
class ID3ICPAlignerOpen3D:
    def __init__(self, args):
        self.id = args.id
        self.move_thresh = args.move_thresh
        rospy.init_node(f"id{self.id}_odom_to_uwb_icp")

        self.odom_raw = []
        self.uwb_pts = []
        self.lock = threading.Lock()

        # 上次ICP触发时的odom点
        self.last_icp_point = None
        # 当前全局变换
        self.transform = np.eye(4)

        rospy.Subscriber("/uwb1/pose_matrix", Float32MultiArray, self.odom_cb)
        rospy.Subscriber("/uwb1/custom_matrix", PoseArray, self.uwb_cb)

        prefix = f"/id{self.id}"
        self.pub_icp_T = rospy.Publisher(f"{prefix}/T_odom2uwb", PoseStamped, queue_size=1)
        self.pub_pose = rospy.Publisher(f"{prefix}/odom_in_uwb", PoseArray, queue_size=1)
        self.pub_path_raw = rospy.Publisher(f"{prefix}/odom_raw_path", Path, queue_size=1)
        self.pub_path_icp = rospy.Publisher(f"{prefix}/odom_icp_path", Path, queue_size=1)
        self.pub_path_uwb = rospy.Publisher(f"{prefix}/uwb_path", Path, queue_size=1)

        rospy.Timer(rospy.Duration(0.1), self.do_icp)
        rospy.loginfo(f"Open3D ICP节点启动，监听ID={self.id}")

    def odom_cb(self, msg):
        data = np.array(msg.data).reshape(-1, 4)
        if data.shape[0] > self.id:
            x, y = data[self.id, :2]
            with self.lock:
                self.odom_raw.append([x, y, 0.0])

    def uwb_cb(self, msg):
        if len(msg.poses) > self.id:
            p = msg.poses[self.id].position
            with self.lock:
                self.uwb_pts.append([p.x, p.y, 0.0])

    def publish_paths(self, raw_pts, aligned_pts, uwb_pts):
        now = rospy.Time.now()
        frame_id = "uwb"

        def make_path(pts):
            path = Path(); path.header.stamp=now; path.header.frame_id=frame_id
            for v in pts:
                ps = PoseStamped(); ps.header.stamp = now; ps.header.frame_id = frame_id
                ps.pose.position.x=v[0]; ps.pose.position.y=v[1]; ps.pose.orientation.w=1.0
                path.poses.append(ps)
            return path

        # PoseArray for aligned odom
        pa = PoseArray(); pa.header.stamp=now; pa.header.frame_id=frame_id
        for v in aligned_pts:
            p = Pose(); p.position.x=v[0]; p.position.y=v[1]; p.orientation.w=1.0
            pa.poses.append(p)

        self.pub_pose.publish(pa)
        self.pub_path_raw.publish(make_path(raw_pts))
        self.pub_path_icp.publish(make_path(aligned_pts))
        self.pub_path_uwb.publish(make_path(uwb_pts))

    def do_icp(self, event):
        with self.lock:
            if len(self.odom_raw)<20 or len(self.uwb_pts)<20:
                return
            n = min(len(self.odom_raw), len(self.uwb_pts))
            src = np.array(self.odom_raw[-n:])

        # 检查是否移动足够
        cur_point = src[-1,:2]
        if self.last_icp_point is not None:
            if np.linalg.norm(cur_point - self.last_icp_point) < self.move_thresh:
                return  # 未移动，跳过ICP
        self.last_icp_point = cur_point

        with self.lock:
            dst = np.array(self.uwb_pts[-n:])

        # 构建点云并执行ICP
        pcd_src = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(src))
        pcd_dst = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(dst))
        reg = o3d.pipelines.registration.registration_icp(
            pcd_src, pcd_dst, self.move_thresh*5,
            self.transform,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=50)
        )
        self.transform = reg.transformation

        # 提取变换
        tx, ty = self.transform[0,3], self.transform[1,3]
        R2 = self.transform[:2,:2]
        theta = atan2(R2[1,0], R2[0,0])

        # 发布变换
        ps = PoseStamped(); ps.header.stamp=rospy.Time.now(); ps.header.frame_id="uwb"
        ps.pose.position.x=tx; ps.pose.position.y=ty
        ps.pose.orientation.z=np.sin(theta/2); ps.pose.orientation.w=np.cos(theta/2)
        self.pub_icp_T.publish(ps)

        rospy.loginfo_throttle(1.0, f"[ICP] tx={tx:.3f}, ty={ty:.3f}, θ={theta:.3f}")

        # 对raw进行对齐并发布轨迹
        aligned = ((self.transform[:3,:3]@src.T).T + self.transform[:3,3]).tolist()
        self.publish_paths(src.tolist(), aligned, dst.tolist())

if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--id',type=int,default=3); p.add_argument('--move_thresh',type=float,default=0.1)
    args = p.parse_args()
    node = ID3ICPAlignerOpen3D(args)
    rospy.spin()
