#!/usr/bin/env python
import rospy
from std_msgs.msg import Float32MultiArray
import numpy as np
import matplotlib.pyplot as plt
import signal
import sys

class DistanceMatrixTracker:
    def __init__(self):
        self.time_series = {}
        self.timestamps = []
        self.index_map = self.build_index_map()
        rospy.Subscriber("/uwb1/distance_matrix", Float32MultiArray, self.callback)
        rospy.loginfo("Tracking upper triangle of /uwb1/distance_matrix...")
        signal.signal(signal.SIGINT, self.signal_handler)

    def build_index_map(self):
        """Map (i,j) to flat index for 4x4 row-major matrix"""
        mapping = {}
        for i in range(4):
            for j in range(4):
                if i < j:
                    idx = i * 4 + j
                    key = f"Node{i}-Node{j}"
                    mapping[key] = idx
                    self.time_series[key] = []
        return mapping

    def callback(self, msg):
        data = msg.data
        if len(data) != 16:
            rospy.logwarn("Invalid data length: expected 16, got %d", len(data))
            return

        now = rospy.get_time()
        self.timestamps.append(now)
        for key, idx in self.index_map.items():
            self.time_series[key].append(data[idx])

    def signal_handler(self, sig, frame):
        rospy.loginfo("Shutting down and plotting data...")
        self.plot_data()
        sys.exit(0)

    def plot_data(self):
        if not self.timestamps:
            print("No data collected.")
            return

        time_offsets = np.array(self.timestamps) - self.timestamps[0]

        plt.figure(figsize=(10, 6))
        for key in sorted(self.time_series.keys()):
            plt.plot(time_offsets, self.time_series[key], label=key)

        plt.xlabel("Time (s)")
        plt.ylabel("Distance")
        plt.title("UWB Distance Changes (Upper Triangle Only)")
        plt.legend(loc='upper right', fontsize='small', ncol=2)
        plt.grid(True)
        plt.tight_layout()
        plt.show()

if __name__ == '__main__':
    rospy.init_node('distance_matrix_tracker_upper', anonymous=True)
    tracker = DistanceMatrixTracker()
    rospy.spin()
