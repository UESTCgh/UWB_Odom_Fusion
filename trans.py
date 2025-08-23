#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rosbag
import matplotlib.pyplot as plt

# 1. 打开 bag 文件
bag_path = "2025-07-22-14-00-46.bag"
bag = rosbag.Bag(bag_path)

# 2. 准备存储
timestamps = []
node_data = {}  # { (role, id): [dis_list] }

# 3. 读取消息并解析
for topic, msg, t in bag.read_messages(topics=['/uwb0/nodeframe2']):
    # msg 类型：nlink_parser/LinktrackNodeframe2
    # 基本字段（示例，不一定全部用到）
    role        = msg.role          # uint8
    node_id     = msg.id            # uint8
    local_time  = msg.local_time    # uint32
    system_time = msg.system_time   # uint32
    voltage     = msg.voltage       # float32
    # ... 其余字段略 ...

    # 解析 nodes 数组
    for node in msg.nodes:
        key = (node.role, node.id)
        if key not in node_data:
            node_data[key] = []
        node_data[key].append(node.dis)

    # 用于横轴或时间索引
    timestamps.append(t.to_sec())

bag.close()

# 4. 绘图：每个 (role,id) 一条曲线，展示距离随消息序号的变化
plt.figure(figsize=(10, 6))
for (role, nid), dis_list in node_data.items():
    plt.plot(dis_list, label=f"role={role}, id={nid}", marker='o')

plt.xlabel("消息序号")
plt.ylabel("距离 dis (m)")
plt.title("各节点距离变化曲线")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

