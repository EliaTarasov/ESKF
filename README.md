# ESKF
ROS Error-State Kalman Filter based on PX4/ecl. Performs GPS/Magnetometer/Vision Pose/Optical Flow/RangeFinder fusion with IMU

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
5. In case you want to use magnetometer for attitude correction make sure IMU has at least 3-4 times higher rate than magnetometer. 
   If not run `rosrun mavros mavcmd long 511 31 4000 0 0 0 0 0`.
6. Run `roslaunch eskf eskf.launch` to start eskf node.
7. Run `echo /eskf/pose` to display results.
