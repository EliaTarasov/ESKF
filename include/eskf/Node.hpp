#ifndef ESKF_NODE_HPP_
#define ESKF_NODE_HPP_

#include <ros/ros.h>
#include <sensor_msgs/Imu.h>
#include <nav_msgs/Odometry.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/PoseWithCovarianceStamped.h>
#include <mavros_msgs/OpticalFlowRad.h>
#include <mavros_msgs/ExtendedState.h>
#include <message_filters/subscriber.h>
#include <eskf/ESKF.hpp>

namespace eskf {

  class Node {
  public:
    static constexpr int default_publish_rate_ = 100;
    static constexpr int default_fusion_mask_ = MASK_EV;

    Node(const ros::NodeHandle& nh, const ros::NodeHandle& pnh);

  private:
    ros::NodeHandle nh_;

    // publishers
    ros::Publisher pubPose_;

    //  subsribers
    ros::Subscriber subImu_;
    ros::Subscriber subVisionPose_;
    ros::Subscriber subGpsPose_;
    ros::Subscriber subOpticalFlowPose_;
    ros::Subscriber subExtendedState_;

    // implementation
    eskf::ESKF eskf_;
    ros::Time prevStampImu_;
    ros::Time prevStampVisionPose_;
    ros::Time prevStampGpsPose_;
    ros::Time prevStampOpticalFlowPose_;
    ros::Timer pubTimer_;
    bool init_;

    //  callbacks
    void inputCallback(const sensor_msgs::ImuConstPtr&);
    void visionCallback(const geometry_msgs::PoseWithCovarianceStampedConstPtr&);
    void gpsCallback(const nav_msgs::OdometryConstPtr&);
    void opticalFlowCallback(const mavros_msgs::OpticalFlowRadConstPtr&);
    void extendedStateCallback(const mavros_msgs::ExtendedStateConstPtr&);
    void publishState(const ros::TimerEvent&);
  };
} //  namespace eskf

#endif // ESKF_NODE_HPP_
