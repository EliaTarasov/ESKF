#include <eskf/Node.hpp>

namespace eskf
{

Node::Node(const ros::NodeHandle &nh, const ros::NodeHandle &pnh) : nh_(pnh), init_(false) {
  ROS_INFO("Subscribing to imu.");
  subImu_ = nh_.subscribe<sensor_msgs::Imu>("imu", 1000, &Node::inputCallback, this, ros::TransportHints().tcpNoDelay(true));
  
  ROS_INFO("Subscribing to extended state");
  subExtendedState_ = nh_.subscribe("extended_state", 1, &Node::extendedStateCallback, this);

  int fusion_mask = default_fusion_mask_;
  ros::param::get("~fusion_mask", fusion_mask);

  if((fusion_mask & MASK_EV_POS) || (fusion_mask & MASK_EV_YAW) || (fusion_mask & MASK_EV_HGT)) {
    ROS_INFO("Subscribing to vision");
    subVisionPose_ = nh_.subscribe("vision", 1, &Node::visionCallback, this);
  } 
  if((fusion_mask & MASK_GPS_POS) || (fusion_mask & MASK_GPS_VEL) || (fusion_mask & MASK_GPS_HGT)) {
    ROS_INFO("Subscribing to gps");
    subGpsPose_ = nh_.subscribe("gps", 1, &Node::gpsCallback, this);
  } 
  if(fusion_mask & MASK_OPTICAL_FLOW) {
    ROS_INFO("Subscribing to optical_flow");
    subOpticalFlowPose_ = nh_.subscribe("optical_flow", 1, &Node::opticalFlowCallback, this);
  }
  if(fusion_mask & MASK_MAG_HEADING) {
    ROS_INFO("Subscribing to mag");
    subMagPose_ = nh_.subscribe("mag", 1, &Node::magCallback, this);
  }
  if(fusion_mask & MASK_RANGEFINDER) {
    ROS_INFO("Subscribing to rangefinder");
    subRangeFinderPose_ = nh_.subscribe("rangefinder", 1, &Node::rangeFinderCallback, this);
  }

  eskf_.setFusionMask(fusion_mask);

  pubPose_ = nh_.advertise<nav_msgs::Odometry>("pose", 1);

  int publish_rate = default_publish_rate_;
  ros::param::get("~publish_rate", publish_rate);
  pubTimer_ = nh_.createTimer(ros::Duration(1.0f/publish_rate), &Node::publishState, this);
}

void Node::inputCallback(const sensor_msgs::ImuConstPtr& imuMsg) {

  vec3 wm = vec3(imuMsg->angular_velocity.x, imuMsg->angular_velocity.y, imuMsg->angular_velocity.z); //  measured angular rate
  vec3 am = vec3(imuMsg->linear_acceleration.x, imuMsg->linear_acceleration.y, imuMsg->linear_acceleration.z); //  measured linear acceleration

  if (prevStampImu_.sec != 0) {
    const double delta = (imuMsg->header.stamp - prevStampImu_).toSec();

    if (!init_) {
      init_ = true;
      ROS_INFO("Initialized ESKF");
    }

    //  run kalman filter
    eskf_.run(wm, am, static_cast<uint64_t>(imuMsg->header.stamp.toSec()*1e6f), delta);
  }
  prevStampImu_ = imuMsg->header.stamp;
}
  
void Node::visionCallback(const geometry_msgs::PoseWithCovarianceStampedConstPtr& poseMsg) {
  if(prevStampVisionPose_.sec != 0) {
    const double delta = (poseMsg->header.stamp - prevStampVisionPose_).toSec();
    // get measurements
    quat z_q = quat(poseMsg->pose.pose.orientation.w, poseMsg->pose.pose.orientation.x, poseMsg->pose.pose.orientation.y, poseMsg->pose.pose.orientation.z);
    vec3 z_p = vec3(poseMsg->pose.pose.position.x, poseMsg->pose.pose.position.y, poseMsg->pose.pose.position.z);
    // update vision
    eskf_.updateVision(z_q, z_p, static_cast<uint64_t>(poseMsg->header.stamp.toSec()*1e6f), delta);
  }
  prevStampVisionPose_ = poseMsg->header.stamp;
}

void Node::gpsCallback(const nav_msgs::OdometryConstPtr& odomMsg) {
  if (prevStampGpsPose_.sec != 0) {
    const double delta = (odomMsg->header.stamp - prevStampGpsPose_).toSec();
    // get gps measurements
    vec3 z_v = vec3(odomMsg->twist.twist.linear.x, odomMsg->twist.twist.linear.y, odomMsg->twist.twist.linear.z);
    vec3 z_p = vec3(odomMsg->pose.pose.position.x, odomMsg->pose.pose.position.y, odomMsg->pose.pose.position.z);
    // update gps
    eskf_.updateGps(z_v, z_p, static_cast<uint64_t>(odomMsg->header.stamp.toSec() * 1e6f), delta);
  }
  prevStampGpsPose_ = odomMsg->header.stamp;
}

void Node::opticalFlowCallback(const mavros_msgs::OpticalFlowRadConstPtr& opticalFlowMsg) {
  if (prevStampOpticalFlowPose_.sec != 0) {
    const double delta = (opticalFlowMsg->header.stamp - prevStampOpticalFlowPose_).toSec();
    // get optical flow measurements
    vec2 int_xy = vec2(opticalFlowMsg->integrated_x, opticalFlowMsg->integrated_y);
    vec2 int_xy_gyro = vec2(opticalFlowMsg->integrated_xgyro, opticalFlowMsg->integrated_ygyro);
    uint32_t integration_time = opticalFlowMsg->integration_time_us;
    scalar_t distance = opticalFlowMsg->distance;
    uint8_t quality = opticalFlowMsg->quality;
    // update optical flow
    eskf_.updateOpticalFlow(int_xy, int_xy_gyro, integration_time, distance, quality, static_cast<uint64_t>(opticalFlowMsg->header.stamp.toSec() * 1e6f), delta);
  }
  prevStampOpticalFlowPose_ = opticalFlowMsg->header.stamp;
}

void Node::magCallback(const sensor_msgs::MagneticFieldConstPtr& magMsg) {
  if (prevStampMagPose_.sec != 0) {
    const double delta = (magMsg->header.stamp - prevStampMagPose_).toSec();
    // get mag measurements
    vec3 m = vec3(magMsg->magnetic_field.x * 1e4f, magMsg->magnetic_field.y * 1e4f, magMsg->magnetic_field.z * 1e4f);
    eskf_.updateMagnetometer(m, static_cast<uint64_t>(magMsg->header.stamp.toSec() * 1e6f), delta);
  }
  prevStampMagPose_ = magMsg->header.stamp;
}

void Node::rangeFinderCallback(const sensor_msgs::RangeConstPtr& rangeMsg) {
  if (prevStampRangeFinderPose_.sec != 0) {
    const double delta = (rangeMsg->header.stamp - prevStampRangeFinderPose_).toSec();
    // get rangefinder measurements
    eskf_.updateRangeFinder(rangeMsg->range, static_cast<uint64_t>(rangeMsg->header.stamp.toSec() * 1e6f), delta);
  }
  prevStampRangeFinderPose_ = rangeMsg->header.stamp;
}

void Node::extendedStateCallback(const mavros_msgs::ExtendedStateConstPtr& extendedStateMsg) {
  eskf_.updateLandedState(extendedStateMsg->landed_state & mavros_msgs::ExtendedState::LANDED_STATE_IN_AIR);
}

void Node::publishState(const ros::TimerEvent&) {

  // get kalman filter result
  const quat e2g = eskf_.getQuat();
  const vec3 position = eskf_.getPosition();
  const vec3 velocity = eskf_.getVelocity();

  static size_t trace_id_ = 0;
  std_msgs::Header header;
  header.frame_id = "/pose";
  header.seq = trace_id_++;
  header.stamp = ros::Time::now();

  nav_msgs::Odometry odom;
  odom.header = header;
  odom.pose.pose.position.x = position[0];
  odom.pose.pose.position.y = position[1];
  odom.pose.pose.position.z = position[2];
  odom.twist.twist.linear.x = velocity[0];
  odom.twist.twist.linear.y = velocity[1];
  odom.twist.twist.linear.z = velocity[2];
  odom.pose.pose.orientation.w = e2g.w();
  odom.pose.pose.orientation.x = e2g.x();
  odom.pose.pose.orientation.y = e2g.y();
  odom.pose.pose.orientation.z = e2g.z();

  pubPose_.publish(odom);
}

}