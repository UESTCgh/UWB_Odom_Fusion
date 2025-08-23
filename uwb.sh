#!/bin/bash
#启动UWB+通信驱动
sudo chmod 777 /dev/ttyCH343USB*
roslaunch nlink_parser linktrack_muti4.launch & sleep 2;
roslaunch distance uwb_distance_muti_data.launch;
# python3 ./fusion/run.py;
wait;

#  launch-prefix="gdb --args"