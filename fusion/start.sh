python3 uwb_odom_P4501.py --fixed_indices 0,1 &
# python3 uwb_odom_P4501.py &
# python3 uwb_odom_sliding.py &
python3 icp_change.py --id=0 &
python3 icp_change.py --id=1 &
python3 icp_change.py --id=2 &
python3 icp_change.py --id=3 &
roslaunch distance rviz.launch &
wait;
