# ESKF
ROS Error-State Kalman Filter based on PX4/ecl. Performs GPS/Vision Pose/Optical Flow/RangeFinder fusion with IMU

# Description
Multisensor fusion ROS node with delayed time horizon based on EKF2.

# Building ESKF

Prerequisites:
* Eigen - https://bitbucket.org/eigen/eigen
* Mavros - https://github.com/mavlink/mavros

Steps:
1. Clone repository in your `catkin` workspace -`git clone https://github.com/EliaTarasov/ESKF.git`
2. Run `catkin_make`
3. Edit launch file `launch/eskf.launch` according to topics you want to subscribe.
4. Run ROS node which provides subscribed topics.
5. Run `roslaunch eskf eskf.launch` to start eskf node.
6. Run `echo /eskf/pose` to display results.
