#ifndef NDEBUG
#define NDEBUG
#endif

#include <ESKF.hpp>
#include <cmath>

using namespace Eigen;

namespace eskf {

  ESKF::ESKF() {
    // zeros state_
    state_.quat_nominal = quat(1, 0, 0, 0);
    state_.vel = vec3(0, 0, 0);
    state_.pos = vec3(0, 0, 0);
    state_.gyro_bias = vec3(0, 0, 0);
    state_.accel_bias = vec3(0, 0, 0);
    state_.mag_I.setZero();
    state_.mag_B.setZero();

    //  zeros P_
    for (unsigned i = 0; i < k_num_states_; i++) {
      for (unsigned j = 0; j < k_num_states_; j++) {
        P_[i][j] = 0.0f;
      }
    }

    imu_down_sampled_.delta_ang.setZero();
    imu_down_sampled_.delta_vel.setZero();
    imu_down_sampled_.delta_ang_dt = 0.0f;
    imu_down_sampled_.delta_vel_dt = 0.0f;

    q_down_sampled_.w() = 1.0f;
    q_down_sampled_.x() = 0.0f;
    q_down_sampled_.y() = 0.0f;
    q_down_sampled_.z() = 0.0f;

    imu_buffer_.allocate(imu_buffer_length_);
    for (int index = 0; index < imu_buffer_length_; index++) {
      imuSample imu_sample_init = {};
      imu_buffer_.push(imu_sample_init);
    }

    ext_vision_buffer_.allocate(obs_buffer_length_);
    for (int index = 0; index < obs_buffer_length_; index++) {
      extVisionSample ext_vision_sample_init = {};
      ext_vision_buffer_.push(ext_vision_sample_init);
    }

    gps_buffer_.allocate(obs_buffer_length_);
    for (int index = 0; index < obs_buffer_length_; index++) {
      gpsSample gps_sample_init = {};
      gps_buffer_.push(gps_sample_init);
    }

    opt_flow_buffer_.allocate(obs_buffer_length_);
    for (int index = 0; index < obs_buffer_length_; index++) {
      optFlowSample opt_flow_sample_init = {};
      opt_flow_buffer_.push(opt_flow_sample_init);
    }

    range_buffer_.allocate(obs_buffer_length_);
    for (int index = 0; index < obs_buffer_length_; index++) {
      rangeSample range_sample_init = {};
      range_buffer_.push(range_sample_init);
    }

    mag_buffer_.allocate(obs_buffer_length_);
    for (int index = 0; index < obs_buffer_length_; index++) {
      magSample mag_sample_init = {};
      mag_buffer_.push(mag_sample_init);
    }

    dt_ekf_avg_ = 0.001f * (scalar_t)(FILTER_UPDATE_PERIOD_MS);

    ///< filter initialisation
    NED_origin_initialised_ = false;
    filter_initialised_ = false;
    terrain_initialised_ = false;

    imu_updated_ = false;
    memset(vel_pos_innov_, 0, 6*sizeof(scalar_t));
    last_known_posNED_ = vec3(0, 0, 0);
  }

  void ESKF::initialiseCovariance() {
    // define the initial angle uncertainty as variances for a rotation vector

    for (unsigned i = 0; i < k_num_states_; i++) {
      for (unsigned j = 0; j < k_num_states_; j++) {
	P_[i][j] = 0.0f;
      }
    }

    // calculate average prediction time step in sec
    float dt = 0.001f * (float)FILTER_UPDATE_PERIOD_MS;

    vec3 rot_vec_var;
    rot_vec_var(2) = rot_vec_var(1) = rot_vec_var(0) = sq(initial_tilt_err_);

    // update the quaternion state covariances
    initialiseQuatCovariances(rot_vec_var);

    // velocity
    P_[4][4] = sq(fmaxf(vel_noise_, 0.01f));
    P_[5][5] = P_[4][4];
    P_[6][6] = sq(1.5f) * P_[4][4];

    // position
    P_[7][7] = sq(fmaxf(pos_noise_, 0.01f));
    P_[8][8] = P_[7][7];
    P_[9][9] = sq(fmaxf(range_noise_, 0.01f));

    // gyro bias
    P_[10][10] = sq(switch_on_gyro_bias_ * dt);
    P_[11][11] = P_[10][10];
    P_[12][12] = P_[10][10];

    P_[13][13] = sq(switch_on_accel_bias_ * dt);
    P_[14][14] = P_[13][13];
    P_[15][15] = P_[13][13];
    // variances for optional states

    // earth frame and body frame magnetic field
    // set to observation variance
    for (uint8_t index = 16; index <= 21; index ++) {
      P_[index][index] = sq(mag_noise_);
    }
  }
  
  bool ESKF::initializeFilter() {
    scalar_t pitch = 0.0;
    scalar_t roll = 0.0;
    scalar_t yaw = 0.0;
    imuSample imu_init = imu_buffer_.get_newest();
    static vec3 delVel_sum(0, 0, 0); ///< summed delta velocity (m/sec)
    delVel_sum += imu_init.delta_vel;
    if (delVel_sum.norm() > 0.001) {
      delVel_sum.normalize();
      pitch = asin(delVel_sum(0));
      roll = atan2(-delVel_sum(1), -delVel_sum(2));
    } else {
      return false;
    }
    // calculate initial tilt alignment
    state_.quat_nominal = AngleAxis<scalar_t>(yaw, vec3::UnitZ()) * AngleAxis<scalar_t>(pitch, vec3::UnitY()) * AngleAxis<scalar_t>(roll, vec3::UnitX());
    // update transformation matrix from body to world frame
    R_to_earth_ = quat_to_invrotmat(state_.quat_nominal);
    initialiseCovariance();
    return true;    
  }
  
  bool ESKF::collect_imu(imuSample &imu) {
    // accumulate and downsample IMU data across a period FILTER_UPDATE_PERIOD_MS long

    // copy imu data to local variables
    imu_sample_new_.delta_ang	= imu.delta_ang;
    imu_sample_new_.delta_vel	= imu.delta_vel;
    imu_sample_new_.delta_ang_dt = imu.delta_ang_dt;
    imu_sample_new_.delta_vel_dt = imu.delta_vel_dt;
    imu_sample_new_.time_us	= imu.time_us;

    // accumulate the time deltas
    imu_down_sampled_.delta_ang_dt += imu.delta_ang_dt;
    imu_down_sampled_.delta_vel_dt += imu.delta_vel_dt;

    // use a quaternion to accumulate delta angle data
    // this quaternion represents the rotation from the start to end of the accumulation period
    quat delta_q(1, 0, 0, 0);
    quat res = from_axis_angle(imu.delta_ang);
    delta_q = delta_q * res;
    q_down_sampled_ = q_down_sampled_ * delta_q;
    q_down_sampled_.normalize();

    // rotate the accumulated delta velocity data forward each time so it is always in the updated rotation frame
    mat3 delta_R = quat2dcm(delta_q.inverse());
    imu_down_sampled_.delta_vel = delta_R * imu_down_sampled_.delta_vel;

    // accumulate the most recent delta velocity data at the updated rotation frame
    // assume effective sample time is halfway between the previous and current rotation frame
    imu_down_sampled_.delta_vel += (imu_sample_new_.delta_vel + delta_R * imu_sample_new_.delta_vel) * 0.5f;

    // if the target time delta between filter prediction steps has been exceeded
    // write the accumulated IMU data to the ring buffer
    scalar_t target_dt = (scalar_t)(FILTER_UPDATE_PERIOD_MS) / 1000;

    if (imu_down_sampled_.delta_ang_dt >= target_dt - imu_collection_time_adj_) {

      // accumulate the amount of time to advance the IMU collection time so that we meet the
      // average EKF update rate requirement
      imu_collection_time_adj_ += 0.01f * (imu_down_sampled_.delta_ang_dt - target_dt);
      imu_collection_time_adj_ = constrain(imu_collection_time_adj_, -0.5f * target_dt, 0.5f * target_dt);

      imu.delta_ang     = to_axis_angle(q_down_sampled_);
      imu.delta_vel     = imu_down_sampled_.delta_vel;
      imu.delta_ang_dt  = imu_down_sampled_.delta_ang_dt;
      imu.delta_vel_dt  = imu_down_sampled_.delta_vel_dt;

      imu_down_sampled_.delta_ang.setZero();
      imu_down_sampled_.delta_vel.setZero();
      imu_down_sampled_.delta_ang_dt = 0.0f;
      imu_down_sampled_.delta_vel_dt = 0.0f;
      q_down_sampled_.w() = 1.0f;
      q_down_sampled_.x() = q_down_sampled_.y() = q_down_sampled_.z() = 0.0f;

      return true;
    }

    min_obs_interval_us_ = (imu_sample_new_.time_us - imu_sample_delayed_.time_us) / (obs_buffer_length_ - 1);

    return false;
  }
  
  void ESKF::run(const vec3 &w, const vec3 &a, uint64_t time_us, scalar_t dt) {
    // convert FLU to FRD body frame IMU data
    vec3 gyro_b = q_FLU2FRD.toRotationMatrix() * w;
    vec3 accel_b = q_FLU2FRD.toRotationMatrix() * a;

    vec3 delta_ang = vec3(gyro_b.x(), gyro_b.y(), gyro_b.z()) * dt; // current delta angle  (rad)
    vec3 delta_vel = vec3(accel_b.x(), accel_b.y(), accel_b.z()) * dt; //current delta velocity (m/s)

    // copy data
    imuSample imu_sample_new = {};
    imu_sample_new.delta_ang = delta_ang;
    imu_sample_new.delta_vel = delta_vel;
    imu_sample_new.delta_ang_dt = dt;
    imu_sample_new.delta_vel_dt = dt;
    imu_sample_new.time_us = time_us;
    
    time_last_imu_ = time_us;
        
    if(collect_imu(imu_sample_new)) {
      imu_buffer_.push(imu_sample_new);
      imu_updated_ = true;
      // get the oldest data from the buffer
      imu_sample_delayed_ = imu_buffer_.get_oldest();
    } else {
      imu_updated_ = false;
      return;
    }
    
    if (!filter_initialised_) {
      filter_initialised_ = initializeFilter();

      if (!filter_initialised_) {
        return;
      }
    }
    
    if(!imu_updated_) return;
    
    // apply imu bias corrections
    vec3 corrected_delta_ang = imu_sample_delayed_.delta_ang - state_.gyro_bias;
    vec3 corrected_delta_vel = imu_sample_delayed_.delta_vel - state_.accel_bias; 
    
    // convert the delta angle to a delta quaternion
    quat dq;
    dq = from_axis_angle(corrected_delta_ang);
    // rotate the previous quaternion by the delta quaternion using a quaternion multiplication
    state_.quat_nominal = state_.quat_nominal * dq;
    // quaternions must be normalised whenever they are modified
    state_.quat_nominal.normalize();
    
    // save the previous value of velocity so we can use trapezoidal integration
    vec3 vel_last = state_.vel;
    
    // update transformation matrix from body to world frame
    R_to_earth_ = quat_to_invrotmat(state_.quat_nominal);
    
    // Calculate an earth frame delta velocity
    vec3 corrected_delta_vel_ef = R_to_earth_ * corrected_delta_vel;
        
    // calculate the increment in velocity using the current orientation
    state_.vel += corrected_delta_vel_ef;

    // compensate for acceleration due to gravity
    state_.vel(2) += kOneG * imu_sample_delayed_.delta_vel_dt;
        
    // predict position states via trapezoidal integration of velocity
    state_.pos += (vel_last + state_.vel) * imu_sample_delayed_.delta_vel_dt * 0.5f;
        
    constrainStates();
        
    // calculate an average filter update time
    scalar_t input = 0.5f * (imu_sample_delayed_.delta_vel_dt + imu_sample_delayed_.delta_ang_dt);

    // filter and limit input between -50% and +100% of nominal value
    input = constrain(input, 0.0005f * (scalar_t)(FILTER_UPDATE_PERIOD_MS), 0.002f * (scalar_t)(FILTER_UPDATE_PERIOD_MS));
    dt_ekf_avg_ = 0.99f * dt_ekf_avg_ + 0.01f * input;
    
    predictCovariance();
    controlFusionModes();
  }
  
  void ESKF::predictCovariance() {
    // error-state jacobian
    // assign intermediate state variables
    scalar_t q0 = state_.quat_nominal.w();
    scalar_t q1 = state_.quat_nominal.x();
    scalar_t q2 = state_.quat_nominal.y();
    scalar_t q3 = state_.quat_nominal.z();

    scalar_t dax = imu_sample_delayed_.delta_ang(0);
    scalar_t day = imu_sample_delayed_.delta_ang(1);
    scalar_t daz = imu_sample_delayed_.delta_ang(2);

    scalar_t dvx = imu_sample_delayed_.delta_vel(0);
    scalar_t dvy = imu_sample_delayed_.delta_vel(1);
    scalar_t dvz = imu_sample_delayed_.delta_vel(2);

    scalar_t dax_b = state_.gyro_bias(0);
    scalar_t day_b = state_.gyro_bias(1);
    scalar_t daz_b = state_.gyro_bias(2);

    scalar_t dvx_b = state_.accel_bias(0);
    scalar_t dvy_b = state_.accel_bias(1);
    scalar_t dvz_b = state_.accel_bias(2);
	  
    // compute noise variance for stationary processes
    scalar_t process_noise[k_num_states_] = {};
    
    scalar_t dt = constrain(imu_sample_delayed_.delta_ang_dt, 0.0005f * (scalar_t)(FILTER_UPDATE_PERIOD_MS), 0.002f * (scalar_t)(FILTER_UPDATE_PERIOD_MS));
    
    // convert rate of change of rate gyro bias (rad/s**2) as specified by the parameter to an expected change in delta angle (rad) since the last update
    scalar_t d_ang_bias_sig = dt * dt * constrain(gyro_bias_p_noise_, 0.0f, 1.0f);

    // convert rate of change of accelerometer bias (m/s**3) as specified by the parameter to an expected change in delta velocity (m/s) since the last update
    scalar_t d_vel_bias_sig = dt * dt * constrain(accel_bias_p_noise_, 0.0f, 1.0f);

    // Don't continue to grow the earth field variances if they are becoming too large or we are not doing 3-axis fusion as this can make the covariance matrix badly conditioned
    scalar_t mag_I_sig;

    if (mag_3D_ && (P_[16][16] + P_[17][17] + P_[18][18]) < 0.1f) {
      mag_I_sig = dt * constrain(mage_p_noise_, 0.0f, 1.0f);
    } else {
      mag_I_sig = 0.0f;
    }

    // Don't continue to grow the body field variances if they is becoming too large or we are not doing 3-axis fusion as this can make the covariance matrix badly conditioned
    scalar_t mag_B_sig;

    if (mag_3D_ && (P_[19][19] + P_[20][20] + P_[21][21]) < 0.1f) {
      mag_B_sig = dt * constrain(magb_p_noise_, 0.0f, 1.0f);
    } else {
      mag_B_sig = 0.0f;
    }

    // Construct the process noise variance diagonal for those states with a stationary process model
    // These are kinematic states and their error growth is controlled separately by the IMU noise variances
    for (unsigned i = 0; i <= 9; i++) {
      process_noise[i] = 0.0;
    }

    // delta angle bias states
    process_noise[12] = process_noise[11] = process_noise[10] = sq(d_ang_bias_sig);
    // delta_velocity bias states
    process_noise[15] = process_noise[14] = process_noise[13] = sq(d_vel_bias_sig);
    // earth frame magnetic field states
    process_noise[18] = process_noise[17] = process_noise[16] = sq(mag_I_sig);
    // body frame magnetic field states
    process_noise[21] = process_noise[20] = process_noise[19] = sq(mag_B_sig);

    // assign IMU noise variances
    // inputs to the system are 3 delta angles and 3 delta velocities
    scalar_t daxVar, dayVar, dazVar;
    scalar_t dvxVar, dvyVar, dvzVar;
    daxVar = dayVar = dazVar = sq(dt * gyro_noise_); // gyro prediction variance TODO get variance from sensor
    dvxVar = dvyVar = dvzVar = sq(dt * accel_noise_); //accel prediction variance TODO get variance from sensor

    // intermediate calculations
    scalar_t SF[21];
    SF[0] = dvz - dvz_b;
    SF[1] = dvy - dvy_b;
    SF[2] = dvx - dvx_b;
    SF[3] = 2*q1*SF[2] + 2*q2*SF[1] + 2*q3*SF[0];
    SF[4] = 2*q0*SF[1] - 2*q1*SF[0] + 2*q3*SF[2];
    SF[5] = 2*q0*SF[2] + 2*q2*SF[0] - 2*q3*SF[1];
    SF[6] = day/2 - day_b/2;
    SF[7] = daz/2 - daz_b/2;
    SF[8] = dax/2 - dax_b/2;
    SF[9] = dax_b/2 - dax/2;
    SF[10] = daz_b/2 - daz/2;
    SF[11] = day_b/2 - day/2;
    SF[12] = 2*q1*SF[1];
    SF[13] = 2*q0*SF[0];
    SF[14] = q1/2;
    SF[15] = q2/2;
    SF[16] = q3/2;
    SF[17] = sq(q3);
    SF[18] = sq(q2);
    SF[19] = sq(q1);
    SF[20] = sq(q0);
    
    scalar_t SG[8];
    SG[0] = q0/2;
    SG[1] = sq(q3);
    SG[2] = sq(q2);
    SG[3] = sq(q1);
    SG[4] = sq(q0);
    SG[5] = 2*q2*q3;
    SG[6] = 2*q1*q3;
    SG[7] = 2*q1*q2;
    
    scalar_t SQ[11];
    SQ[0] = dvzVar*(SG[5] - 2*q0*q1)*(SG[1] - SG[2] - SG[3] + SG[4]) - dvyVar*(SG[5] + 2*q0*q1)*(SG[1] - SG[2] + SG[3] - SG[4]) + dvxVar*(SG[6] - 2*q0*q2)*(SG[7] + 2*q0*q3);
    SQ[1] = dvzVar*(SG[6] + 2*q0*q2)*(SG[1] - SG[2] - SG[3] + SG[4]) - dvxVar*(SG[6] - 2*q0*q2)*(SG[1] + SG[2] - SG[3] - SG[4]) + dvyVar*(SG[5] + 2*q0*q1)*(SG[7] - 2*q0*q3);
    SQ[2] = dvzVar*(SG[5] - 2*q0*q1)*(SG[6] + 2*q0*q2) - dvyVar*(SG[7] - 2*q0*q3)*(SG[1] - SG[2] + SG[3] - SG[4]) - dvxVar*(SG[7] + 2*q0*q3)*(SG[1] + SG[2] - SG[3] - SG[4]);
    SQ[3] = (dayVar*q1*SG[0])/2 - (dazVar*q1*SG[0])/2 - (daxVar*q2*q3)/4;
    SQ[4] = (dazVar*q2*SG[0])/2 - (daxVar*q2*SG[0])/2 - (dayVar*q1*q3)/4;
    SQ[5] = (daxVar*q3*SG[0])/2 - (dayVar*q3*SG[0])/2 - (dazVar*q1*q2)/4;
    SQ[6] = (daxVar*q1*q2)/4 - (dazVar*q3*SG[0])/2 - (dayVar*q1*q2)/4;
    SQ[7] = (dazVar*q1*q3)/4 - (daxVar*q1*q3)/4 - (dayVar*q2*SG[0])/2;
    SQ[8] = (dayVar*q2*q3)/4 - (daxVar*q1*SG[0])/2 - (dazVar*q2*q3)/4;
    SQ[9] = sq(SG[0]);
    SQ[10] = sq(q1);
    
    scalar_t SPP[11];
    SPP[0] = SF[12] + SF[13] - 2*q2*SF[2];
    SPP[1] = SF[17] - SF[18] - SF[19] + SF[20];
    SPP[2] = SF[17] - SF[18] + SF[19] - SF[20];
    SPP[3] = SF[17] + SF[18] - SF[19] - SF[20];
    SPP[4] = 2*q0*q2 - 2*q1*q3;
    SPP[5] = 2*q0*q1 - 2*q2*q3;
    SPP[6] = 2*q0*q3 - 2*q1*q2;
    SPP[7] = 2*q0*q1 + 2*q2*q3;
    SPP[8] = 2*q0*q3 + 2*q1*q2;
    SPP[9] = 2*q0*q2 + 2*q1*q3;
    SPP[10] = SF[16];
    
    // covariance update
    // calculate variances and upper diagonal covariances for quaternion, velocity, position and gyro bias states
    scalar_t nextP[k_num_states_][k_num_states_];
    nextP[0][0] = P_[0][0] + P_[1][0]*SF[9] + P_[2][0]*SF[11] + P_[3][0]*SF[10] + P_[10][0]*SF[14] + P_[11][0]*SF[15] + P_[12][0]*SPP[10] + (daxVar*SQ[10])/4 + SF[9]*(P_[0][1] + P_[1][1]*SF[9] + P_[2][1]*SF[11] + P_[3][1]*SF[10] + P_[10][1]*SF[14] + P_[11][1]*SF[15] + P_[12][1]*SPP[10]) + SF[11]*(P_[0][2] + P_[1][2]*SF[9] + P_[2][2]*SF[11] + P_[3][2]*SF[10] + P_[10][2]*SF[14] + P_[11][2]*SF[15] + P_[12][2]*SPP[10]) + SF[10]*(P_[0][3] + P_[1][3]*SF[9] + P_[2][3]*SF[11] + P_[3][3]*SF[10] + P_[10][3]*SF[14] + P_[11][3]*SF[15] + P_[12][3]*SPP[10]) + SF[14]*(P_[0][10] + P_[1][10]*SF[9] + P_[2][10]*SF[11] + P_[3][10]*SF[10] + P_[10][10]*SF[14] + P_[11][10]*SF[15] + P_[12][10]*SPP[10]) + SF[15]*(P_[0][11] + P_[1][11]*SF[9] + P_[2][11]*SF[11] + P_[3][11]*SF[10] + P_[10][11]*SF[14] + P_[11][11]*SF[15] + P_[12][11]*SPP[10]) + SPP[10]*(P_[0][12] + P_[1][12]*SF[9] + P_[2][12]*SF[11] + P_[3][12]*SF[10] + P_[10][12]*SF[14] + P_[11][12]*SF[15] + P_[12][12]*SPP[10]) + (dayVar*sq(q2))/4 + (dazVar*sq(q3))/4;
    nextP[0][1] = P_[0][1] + SQ[8] + P_[1][1]*SF[9] + P_[2][1]*SF[11] + P_[3][1]*SF[10] + P_[10][1]*SF[14] + P_[11][1]*SF[15] + P_[12][1]*SPP[10] + SF[8]*(P_[0][0] + P_[1][0]*SF[9] + P_[2][0]*SF[11] + P_[3][0]*SF[10] + P_[10][0]*SF[14] + P_[11][0]*SF[15] + P_[12][0]*SPP[10]) + SF[7]*(P_[0][2] + P_[1][2]*SF[9] + P_[2][2]*SF[11] + P_[3][2]*SF[10] + P_[10][2]*SF[14] + P_[11][2]*SF[15] + P_[12][2]*SPP[10]) + SF[11]*(P_[0][3] + P_[1][3]*SF[9] + P_[2][3]*SF[11] + P_[3][3]*SF[10] + P_[10][3]*SF[14] + P_[11][3]*SF[15] + P_[12][3]*SPP[10]) - SF[15]*(P_[0][12] + P_[1][12]*SF[9] + P_[2][12]*SF[11] + P_[3][12]*SF[10] + P_[10][12]*SF[14] + P_[11][12]*SF[15] + P_[12][12]*SPP[10]) + SPP[10]*(P_[0][11] + P_[1][11]*SF[9] + P_[2][11]*SF[11] + P_[3][11]*SF[10] + P_[10][11]*SF[14] + P_[11][11]*SF[15] + P_[12][11]*SPP[10]) - (q0*(P_[0][10] + P_[1][10]*SF[9] + P_[2][10]*SF[11] + P_[3][10]*SF[10] + P_[10][10]*SF[14] + P_[11][10]*SF[15] + P_[12][10]*SPP[10]))/2;
    nextP[1][1] = P_[1][1] + P_[0][1]*SF[8] + P_[2][1]*SF[7] + P_[3][1]*SF[11] - P_[12][1]*SF[15] + P_[11][1]*SPP[10] + daxVar*SQ[9] - (P_[10][1]*q0)/2 + SF[8]*(P_[1][0] + P_[0][0]*SF[8] + P_[2][0]*SF[7] + P_[3][0]*SF[11] - P_[12][0]*SF[15] + P_[11][0]*SPP[10] - (P_[10][0]*q0)/2) + SF[7]*(P_[1][2] + P_[0][2]*SF[8] + P_[2][2]*SF[7] + P_[3][2]*SF[11] - P_[12][2]*SF[15] + P_[11][2]*SPP[10] - (P_[10][2]*q0)/2) + SF[11]*(P_[1][3] + P_[0][3]*SF[8] + P_[2][3]*SF[7] + P_[3][3]*SF[11] - P_[12][3]*SF[15] + P_[11][3]*SPP[10] - (P_[10][3]*q0)/2) - SF[15]*(P_[1][12] + P_[0][12]*SF[8] + P_[2][12]*SF[7] + P_[3][12]*SF[11] - P_[12][12]*SF[15] + P_[11][12]*SPP[10] - (P_[10][12]*q0)/2) + SPP[10]*(P_[1][11] + P_[0][11]*SF[8] + P_[2][11]*SF[7] + P_[3][11]*SF[11] - P_[12][11]*SF[15] + P_[11][11]*SPP[10] - (P_[10][11]*q0)/2) + (dayVar*sq(q3))/4 + (dazVar*sq(q2))/4 - (q0*(P_[1][10] + P_[0][10]*SF[8] + P_[2][10]*SF[7] + P_[3][10]*SF[11] - P_[12][10]*SF[15] + P_[11][10]*SPP[10] - (P_[10][10]*q0)/2))/2;
    nextP[0][2] = P_[0][2] + SQ[7] + P_[1][2]*SF[9] + P_[2][2]*SF[11] + P_[3][2]*SF[10] + P_[10][2]*SF[14] + P_[11][2]*SF[15] + P_[12][2]*SPP[10] + SF[6]*(P_[0][0] + P_[1][0]*SF[9] + P_[2][0]*SF[11] + P_[3][0]*SF[10] + P_[10][0]*SF[14] + P_[11][0]*SF[15] + P_[12][0]*SPP[10]) + SF[10]*(P_[0][1] + P_[1][1]*SF[9] + P_[2][1]*SF[11] + P_[3][1]*SF[10] + P_[10][1]*SF[14] + P_[11][1]*SF[15] + P_[12][1]*SPP[10]) + SF[8]*(P_[0][3] + P_[1][3]*SF[9] + P_[2][3]*SF[11] + P_[3][3]*SF[10] + P_[10][3]*SF[14] + P_[11][3]*SF[15] + P_[12][3]*SPP[10]) + SF[14]*(P_[0][12] + P_[1][12]*SF[9] + P_[2][12]*SF[11] + P_[3][12]*SF[10] + P_[10][12]*SF[14] + P_[11][12]*SF[15] + P_[12][12]*SPP[10]) - SPP[10]*(P_[0][10] + P_[1][10]*SF[9] + P_[2][10]*SF[11] + P_[3][10]*SF[10] + P_[10][10]*SF[14] + P_[11][10]*SF[15] + P_[12][10]*SPP[10]) - (q0*(P_[0][11] + P_[1][11]*SF[9] + P_[2][11]*SF[11] + P_[3][11]*SF[10] + P_[10][11]*SF[14] + P_[11][11]*SF[15] + P_[12][11]*SPP[10]))/2;
    nextP[1][2] = P_[1][2] + SQ[5] + P_[0][2]*SF[8] + P_[2][2]*SF[7] + P_[3][2]*SF[11] - P_[12][2]*SF[15] + P_[11][2]*SPP[10] - (P_[10][2]*q0)/2 + SF[6]*(P_[1][0] + P_[0][0]*SF[8] + P_[2][0]*SF[7] + P_[3][0]*SF[11] - P_[12][0]*SF[15] + P_[11][0]*SPP[10] - (P_[10][0]*q0)/2) + SF[10]*(P_[1][1] + P_[0][1]*SF[8] + P_[2][1]*SF[7] + P_[3][1]*SF[11] - P_[12][1]*SF[15] + P_[11][1]*SPP[10] - (P_[10][1]*q0)/2) + SF[8]*(P_[1][3] + P_[0][3]*SF[8] + P_[2][3]*SF[7] + P_[3][3]*SF[11] - P_[12][3]*SF[15] + P_[11][3]*SPP[10] - (P_[10][3]*q0)/2) + SF[14]*(P_[1][12] + P_[0][12]*SF[8] + P_[2][12]*SF[7] + P_[3][12]*SF[11] - P_[12][12]*SF[15] + P_[11][12]*SPP[10] - (P_[10][12]*q0)/2) - SPP[10]*(P_[1][10] + P_[0][10]*SF[8] + P_[2][10]*SF[7] + P_[3][10]*SF[11] - P_[12][10]*SF[15] + P_[11][10]*SPP[10] - (P_[10][10]*q0)/2) - (q0*(P_[1][11] + P_[0][11]*SF[8] + P_[2][11]*SF[7] + P_[3][11]*SF[11] - P_[12][11]*SF[15] + P_[11][11]*SPP[10] - (P_[10][11]*q0)/2))/2;
    nextP[2][2] = P_[2][2] + P_[0][2]*SF[6] + P_[1][2]*SF[10] + P_[3][2]*SF[8] + P_[12][2]*SF[14] - P_[10][2]*SPP[10] + dayVar*SQ[9] + (dazVar*SQ[10])/4 - (P_[11][2]*q0)/2 + SF[6]*(P_[2][0] + P_[0][0]*SF[6] + P_[1][0]*SF[10] + P_[3][0]*SF[8] + P_[12][0]*SF[14] - P_[10][0]*SPP[10] - (P_[11][0]*q0)/2) + SF[10]*(P_[2][1] + P_[0][1]*SF[6] + P_[1][1]*SF[10] + P_[3][1]*SF[8] + P_[12][1]*SF[14] - P_[10][1]*SPP[10] - (P_[11][1]*q0)/2) + SF[8]*(P_[2][3] + P_[0][3]*SF[6] + P_[1][3]*SF[10] + P_[3][3]*SF[8] + P_[12][3]*SF[14] - P_[10][3]*SPP[10] - (P_[11][3]*q0)/2) + SF[14]*(P_[2][12] + P_[0][12]*SF[6] + P_[1][12]*SF[10] + P_[3][12]*SF[8] + P_[12][12]*SF[14] - P_[10][12]*SPP[10] - (P_[11][12]*q0)/2) - SPP[10]*(P_[2][10] + P_[0][10]*SF[6] + P_[1][10]*SF[10] + P_[3][10]*SF[8] + P_[12][10]*SF[14] - P_[10][10]*SPP[10] - (P_[11][10]*q0)/2) + (daxVar*sq(q3))/4 - (q0*(P_[2][11] + P_[0][11]*SF[6] + P_[1][11]*SF[10] + P_[3][11]*SF[8] + P_[12][11]*SF[14] - P_[10][11]*SPP[10] - (P_[11][11]*q0)/2))/2;
    nextP[0][3] = P_[0][3] + SQ[6] + P_[1][3]*SF[9] + P_[2][3]*SF[11] + P_[3][3]*SF[10] + P_[10][3]*SF[14] + P_[11][3]*SF[15] + P_[12][3]*SPP[10] + SF[7]*(P_[0][0] + P_[1][0]*SF[9] + P_[2][0]*SF[11] + P_[3][0]*SF[10] + P_[10][0]*SF[14] + P_[11][0]*SF[15] + P_[12][0]*SPP[10]) + SF[6]*(P_[0][1] + P_[1][1]*SF[9] + P_[2][1]*SF[11] + P_[3][1]*SF[10] + P_[10][1]*SF[14] + P_[11][1]*SF[15] + P_[12][1]*SPP[10]) + SF[9]*(P_[0][2] + P_[1][2]*SF[9] + P_[2][2]*SF[11] + P_[3][2]*SF[10] + P_[10][2]*SF[14] + P_[11][2]*SF[15] + P_[12][2]*SPP[10]) + SF[15]*(P_[0][10] + P_[1][10]*SF[9] + P_[2][10]*SF[11] + P_[3][10]*SF[10] + P_[10][10]*SF[14] + P_[11][10]*SF[15] + P_[12][10]*SPP[10]) - SF[14]*(P_[0][11] + P_[1][11]*SF[9] + P_[2][11]*SF[11] + P_[3][11]*SF[10] + P_[10][11]*SF[14] + P_[11][11]*SF[15] + P_[12][11]*SPP[10]) - (q0*(P_[0][12] + P_[1][12]*SF[9] + P_[2][12]*SF[11] + P_[3][12]*SF[10] + P_[10][12]*SF[14] + P_[11][12]*SF[15] + P_[12][12]*SPP[10]))/2;
    nextP[1][3] = P_[1][3] + SQ[4] + P_[0][3]*SF[8] + P_[2][3]*SF[7] + P_[3][3]*SF[11] - P_[12][3]*SF[15] + P_[11][3]*SPP[10] - (P_[10][3]*q0)/2 + SF[7]*(P_[1][0] + P_[0][0]*SF[8] + P_[2][0]*SF[7] + P_[3][0]*SF[11] - P_[12][0]*SF[15] + P_[11][0]*SPP[10] - (P_[10][0]*q0)/2) + SF[6]*(P_[1][1] + P_[0][1]*SF[8] + P_[2][1]*SF[7] + P_[3][1]*SF[11] - P_[12][1]*SF[15] + P_[11][1]*SPP[10] - (P_[10][1]*q0)/2) + SF[9]*(P_[1][2] + P_[0][2]*SF[8] + P_[2][2]*SF[7] + P_[3][2]*SF[11] - P_[12][2]*SF[15] + P_[11][2]*SPP[10] - (P_[10][2]*q0)/2) + SF[15]*(P_[1][10] + P_[0][10]*SF[8] + P_[2][10]*SF[7] + P_[3][10]*SF[11] - P_[12][10]*SF[15] + P_[11][10]*SPP[10] - (P_[10][10]*q0)/2) - SF[14]*(P_[1][11] + P_[0][11]*SF[8] + P_[2][11]*SF[7] + P_[3][11]*SF[11] - P_[12][11]*SF[15] + P_[11][11]*SPP[10] - (P_[10][11]*q0)/2) - (q0*(P_[1][12] + P_[0][12]*SF[8] + P_[2][12]*SF[7] + P_[3][12]*SF[11] - P_[12][12]*SF[15] + P_[11][12]*SPP[10] - (P_[10][12]*q0)/2))/2;
    nextP[2][3] = P_[2][3] + SQ[3] + P_[0][3]*SF[6] + P_[1][3]*SF[10] + P_[3][3]*SF[8] + P_[12][3]*SF[14] - P_[10][3]*SPP[10] - (P_[11][3]*q0)/2 + SF[7]*(P_[2][0] + P_[0][0]*SF[6] + P_[1][0]*SF[10] + P_[3][0]*SF[8] + P_[12][0]*SF[14] - P_[10][0]*SPP[10] - (P_[11][0]*q0)/2) + SF[6]*(P_[2][1] + P_[0][1]*SF[6] + P_[1][1]*SF[10] + P_[3][1]*SF[8] + P_[12][1]*SF[14] - P_[10][1]*SPP[10] - (P_[11][1]*q0)/2) + SF[9]*(P_[2][2] + P_[0][2]*SF[6] + P_[1][2]*SF[10] + P_[3][2]*SF[8] + P_[12][2]*SF[14] - P_[10][2]*SPP[10] - (P_[11][2]*q0)/2) + SF[15]*(P_[2][10] + P_[0][10]*SF[6] + P_[1][10]*SF[10] + P_[3][10]*SF[8] + P_[12][10]*SF[14] - P_[10][10]*SPP[10] - (P_[11][10]*q0)/2) - SF[14]*(P_[2][11] + P_[0][11]*SF[6] + P_[1][11]*SF[10] + P_[3][11]*SF[8] + P_[12][11]*SF[14] - P_[10][11]*SPP[10] - (P_[11][11]*q0)/2) - (q0*(P_[2][12] + P_[0][12]*SF[6] + P_[1][12]*SF[10] + P_[3][12]*SF[8] + P_[12][12]*SF[14] - P_[10][12]*SPP[10] - (P_[11][12]*q0)/2))/2;
    nextP[3][3] = P_[3][3] + P_[0][3]*SF[7] + P_[1][3]*SF[6] + P_[2][3]*SF[9] + P_[10][3]*SF[15] - P_[11][3]*SF[14] + (dayVar*SQ[10])/4 + dazVar*SQ[9] - (P_[12][3]*q0)/2 + SF[7]*(P_[3][0] + P_[0][0]*SF[7] + P_[1][0]*SF[6] + P_[2][0]*SF[9] + P_[10][0]*SF[15] - P_[11][0]*SF[14] - (P_[12][0]*q0)/2) + SF[6]*(P_[3][1] + P_[0][1]*SF[7] + P_[1][1]*SF[6] + P_[2][1]*SF[9] + P_[10][1]*SF[15] - P_[11][1]*SF[14] - (P_[12][1]*q0)/2) + SF[9]*(P_[3][2] + P_[0][2]*SF[7] + P_[1][2]*SF[6] + P_[2][2]*SF[9] + P_[10][2]*SF[15] - P_[11][2]*SF[14] - (P_[12][2]*q0)/2) + SF[15]*(P_[3][10] + P_[0][10]*SF[7] + P_[1][10]*SF[6] + P_[2][10]*SF[9] + P_[10][10]*SF[15] - P_[11][10]*SF[14] - (P_[12][10]*q0)/2) - SF[14]*(P_[3][11] + P_[0][11]*SF[7] + P_[1][11]*SF[6] + P_[2][11]*SF[9] + P_[10][11]*SF[15] - P_[11][11]*SF[14] - (P_[12][11]*q0)/2) + (daxVar*sq(q2))/4 - (q0*(P_[3][12] + P_[0][12]*SF[7] + P_[1][12]*SF[6] + P_[2][12]*SF[9] + P_[10][12]*SF[15] - P_[11][12]*SF[14] - (P_[12][12]*q0)/2))/2;
    nextP[0][4] = P_[0][4] + P_[1][4]*SF[9] + P_[2][4]*SF[11] + P_[3][4]*SF[10] + P_[10][4]*SF[14] + P_[11][4]*SF[15] + P_[12][4]*SPP[10] + SF[5]*(P_[0][0] + P_[1][0]*SF[9] + P_[2][0]*SF[11] + P_[3][0]*SF[10] + P_[10][0]*SF[14] + P_[11][0]*SF[15] + P_[12][0]*SPP[10]) + SF[3]*(P_[0][1] + P_[1][1]*SF[9] + P_[2][1]*SF[11] + P_[3][1]*SF[10] + P_[10][1]*SF[14] + P_[11][1]*SF[15] + P_[12][1]*SPP[10]) - SF[4]*(P_[0][3] + P_[1][3]*SF[9] + P_[2][3]*SF[11] + P_[3][3]*SF[10] + P_[10][3]*SF[14] + P_[11][3]*SF[15] + P_[12][3]*SPP[10]) + SPP[0]*(P_[0][2] + P_[1][2]*SF[9] + P_[2][2]*SF[11] + P_[3][2]*SF[10] + P_[10][2]*SF[14] + P_[11][2]*SF[15] + P_[12][2]*SPP[10]) + SPP[3]*(P_[0][13] + P_[1][13]*SF[9] + P_[2][13]*SF[11] + P_[3][13]*SF[10] + P_[10][13]*SF[14] + P_[11][13]*SF[15] + P_[12][13]*SPP[10]) + SPP[6]*(P_[0][14] + P_[1][14]*SF[9] + P_[2][14]*SF[11] + P_[3][14]*SF[10] + P_[10][14]*SF[14] + P_[11][14]*SF[15] + P_[12][14]*SPP[10]) - SPP[9]*(P_[0][15] + P_[1][15]*SF[9] + P_[2][15]*SF[11] + P_[3][15]*SF[10] + P_[10][15]*SF[14] + P_[11][15]*SF[15] + P_[12][15]*SPP[10]);
    nextP[1][4] = P_[1][4] + P_[0][4]*SF[8] + P_[2][4]*SF[7] + P_[3][4]*SF[11] - P_[12][4]*SF[15] + P_[11][4]*SPP[10] - (P_[10][4]*q0)/2 + SF[5]*(P_[1][0] + P_[0][0]*SF[8] + P_[2][0]*SF[7] + P_[3][0]*SF[11] - P_[12][0]*SF[15] + P_[11][0]*SPP[10] - (P_[10][0]*q0)/2) + SF[3]*(P_[1][1] + P_[0][1]*SF[8] + P_[2][1]*SF[7] + P_[3][1]*SF[11] - P_[12][1]*SF[15] + P_[11][1]*SPP[10] - (P_[10][1]*q0)/2) - SF[4]*(P_[1][3] + P_[0][3]*SF[8] + P_[2][3]*SF[7] + P_[3][3]*SF[11] - P_[12][3]*SF[15] + P_[11][3]*SPP[10] - (P_[10][3]*q0)/2) + SPP[0]*(P_[1][2] + P_[0][2]*SF[8] + P_[2][2]*SF[7] + P_[3][2]*SF[11] - P_[12][2]*SF[15] + P_[11][2]*SPP[10] - (P_[10][2]*q0)/2) + SPP[3]*(P_[1][13] + P_[0][13]*SF[8] + P_[2][13]*SF[7] + P_[3][13]*SF[11] - P_[12][13]*SF[15] + P_[11][13]*SPP[10] - (P_[10][13]*q0)/2) + SPP[6]*(P_[1][14] + P_[0][14]*SF[8] + P_[2][14]*SF[7] + P_[3][14]*SF[11] - P_[12][14]*SF[15] + P_[11][14]*SPP[10] - (P_[10][14]*q0)/2) - SPP[9]*(P_[1][15] + P_[0][15]*SF[8] + P_[2][15]*SF[7] + P_[3][15]*SF[11] - P_[12][15]*SF[15] + P_[11][15]*SPP[10] - (P_[10][15]*q0)/2);
    nextP[2][4] = P_[2][4] + P_[0][4]*SF[6] + P_[1][4]*SF[10] + P_[3][4]*SF[8] + P_[12][4]*SF[14] - P_[10][4]*SPP[10] - (P_[11][4]*q0)/2 + SF[5]*(P_[2][0] + P_[0][0]*SF[6] + P_[1][0]*SF[10] + P_[3][0]*SF[8] + P_[12][0]*SF[14] - P_[10][0]*SPP[10] - (P_[11][0]*q0)/2) + SF[3]*(P_[2][1] + P_[0][1]*SF[6] + P_[1][1]*SF[10] + P_[3][1]*SF[8] + P_[12][1]*SF[14] - P_[10][1]*SPP[10] - (P_[11][1]*q0)/2) - SF[4]*(P_[2][3] + P_[0][3]*SF[6] + P_[1][3]*SF[10] + P_[3][3]*SF[8] + P_[12][3]*SF[14] - P_[10][3]*SPP[10] - (P_[11][3]*q0)/2) + SPP[0]*(P_[2][2] + P_[0][2]*SF[6] + P_[1][2]*SF[10] + P_[3][2]*SF[8] + P_[12][2]*SF[14] - P_[10][2]*SPP[10] - (P_[11][2]*q0)/2) + SPP[3]*(P_[2][13] + P_[0][13]*SF[6] + P_[1][13]*SF[10] + P_[3][13]*SF[8] + P_[12][13]*SF[14] - P_[10][13]*SPP[10] - (P_[11][13]*q0)/2) + SPP[6]*(P_[2][14] + P_[0][14]*SF[6] + P_[1][14]*SF[10] + P_[3][14]*SF[8] + P_[12][14]*SF[14] - P_[10][14]*SPP[10] - (P_[11][14]*q0)/2) - SPP[9]*(P_[2][15] + P_[0][15]*SF[6] + P_[1][15]*SF[10] + P_[3][15]*SF[8] + P_[12][15]*SF[14] - P_[10][15]*SPP[10] - (P_[11][15]*q0)/2);
    nextP[3][4] = P_[3][4] + P_[0][4]*SF[7] + P_[1][4]*SF[6] + P_[2][4]*SF[9] + P_[10][4]*SF[15] - P_[11][4]*SF[14] - (P_[12][4]*q0)/2 + SF[5]*(P_[3][0] + P_[0][0]*SF[7] + P_[1][0]*SF[6] + P_[2][0]*SF[9] + P_[10][0]*SF[15] - P_[11][0]*SF[14] - (P_[12][0]*q0)/2) + SF[3]*(P_[3][1] + P_[0][1]*SF[7] + P_[1][1]*SF[6] + P_[2][1]*SF[9] + P_[10][1]*SF[15] - P_[11][1]*SF[14] - (P_[12][1]*q0)/2) - SF[4]*(P_[3][3] + P_[0][3]*SF[7] + P_[1][3]*SF[6] + P_[2][3]*SF[9] + P_[10][3]*SF[15] - P_[11][3]*SF[14] - (P_[12][3]*q0)/2) + SPP[0]*(P_[3][2] + P_[0][2]*SF[7] + P_[1][2]*SF[6] + P_[2][2]*SF[9] + P_[10][2]*SF[15] - P_[11][2]*SF[14] - (P_[12][2]*q0)/2) + SPP[3]*(P_[3][13] + P_[0][13]*SF[7] + P_[1][13]*SF[6] + P_[2][13]*SF[9] + P_[10][13]*SF[15] - P_[11][13]*SF[14] - (P_[12][13]*q0)/2) + SPP[6]*(P_[3][14] + P_[0][14]*SF[7] + P_[1][14]*SF[6] + P_[2][14]*SF[9] + P_[10][14]*SF[15] - P_[11][14]*SF[14] - (P_[12][14]*q0)/2) - SPP[9]*(P_[3][15] + P_[0][15]*SF[7] + P_[1][15]*SF[6] + P_[2][15]*SF[9] + P_[10][15]*SF[15] - P_[11][15]*SF[14] - (P_[12][15]*q0)/2);
    nextP[4][4] = P_[4][4] + P_[0][4]*SF[5] + P_[1][4]*SF[3] - P_[3][4]*SF[4] + P_[2][4]*SPP[0] + P_[13][4]*SPP[3] + P_[14][4]*SPP[6] - P_[15][4]*SPP[9] + dvyVar*sq(SG[7] - 2*q0*q3) + dvzVar*sq(SG[6] + 2*q0*q2) + SF[5]*(P_[4][0] + P_[0][0]*SF[5] + P_[1][0]*SF[3] - P_[3][0]*SF[4] + P_[2][0]*SPP[0] + P_[13][0]*SPP[3] + P_[14][0]*SPP[6] - P_[15][0]*SPP[9]) + SF[3]*(P_[4][1] + P_[0][1]*SF[5] + P_[1][1]*SF[3] - P_[3][1]*SF[4] + P_[2][1]*SPP[0] + P_[13][1]*SPP[3] + P_[14][1]*SPP[6] - P_[15][1]*SPP[9]) - SF[4]*(P_[4][3] + P_[0][3]*SF[5] + P_[1][3]*SF[3] - P_[3][3]*SF[4] + P_[2][3]*SPP[0] + P_[13][3]*SPP[3] + P_[14][3]*SPP[6] - P_[15][3]*SPP[9]) + SPP[0]*(P_[4][2] + P_[0][2]*SF[5] + P_[1][2]*SF[3] - P_[3][2]*SF[4] + P_[2][2]*SPP[0] + P_[13][2]*SPP[3] + P_[14][2]*SPP[6] - P_[15][2]*SPP[9]) + SPP[3]*(P_[4][13] + P_[0][13]*SF[5] + P_[1][13]*SF[3] - P_[3][13]*SF[4] + P_[2][13]*SPP[0] + P_[13][13]*SPP[3] + P_[14][13]*SPP[6] - P_[15][13]*SPP[9]) + SPP[6]*(P_[4][14] + P_[0][14]*SF[5] + P_[1][14]*SF[3] - P_[3][14]*SF[4] + P_[2][14]*SPP[0] + P_[13][14]*SPP[3] + P_[14][14]*SPP[6] - P_[15][14]*SPP[9]) - SPP[9]*(P_[4][15] + P_[0][15]*SF[5] + P_[1][15]*SF[3] - P_[3][15]*SF[4] + P_[2][15]*SPP[0] + P_[13][15]*SPP[3] + P_[14][15]*SPP[6] - P_[15][15]*SPP[9]) + dvxVar*sq(SG[1] + SG[2] - SG[3] - SG[4]);
    nextP[0][5] = P_[0][5] + P_[1][5]*SF[9] + P_[2][5]*SF[11] + P_[3][5]*SF[10] + P_[10][5]*SF[14] + P_[11][5]*SF[15] + P_[12][5]*SPP[10] + SF[4]*(P_[0][0] + P_[1][0]*SF[9] + P_[2][0]*SF[11] + P_[3][0]*SF[10] + P_[10][0]*SF[14] + P_[11][0]*SF[15] + P_[12][0]*SPP[10]) + SF[3]*(P_[0][2] + P_[1][2]*SF[9] + P_[2][2]*SF[11] + P_[3][2]*SF[10] + P_[10][2]*SF[14] + P_[11][2]*SF[15] + P_[12][2]*SPP[10]) + SF[5]*(P_[0][3] + P_[1][3]*SF[9] + P_[2][3]*SF[11] + P_[3][3]*SF[10] + P_[10][3]*SF[14] + P_[11][3]*SF[15] + P_[12][3]*SPP[10]) - SPP[0]*(P_[0][1] + P_[1][1]*SF[9] + P_[2][1]*SF[11] + P_[3][1]*SF[10] + P_[10][1]*SF[14] + P_[11][1]*SF[15] + P_[12][1]*SPP[10]) - SPP[8]*(P_[0][13] + P_[1][13]*SF[9] + P_[2][13]*SF[11] + P_[3][13]*SF[10] + P_[10][13]*SF[14] + P_[11][13]*SF[15] + P_[12][13]*SPP[10]) + SPP[2]*(P_[0][14] + P_[1][14]*SF[9] + P_[2][14]*SF[11] + P_[3][14]*SF[10] + P_[10][14]*SF[14] + P_[11][14]*SF[15] + P_[12][14]*SPP[10]) + SPP[5]*(P_[0][15] + P_[1][15]*SF[9] + P_[2][15]*SF[11] + P_[3][15]*SF[10] + P_[10][15]*SF[14] + P_[11][15]*SF[15] + P_[12][15]*SPP[10]);
    nextP[1][5] = P_[1][5] + P_[0][5]*SF[8] + P_[2][5]*SF[7] + P_[3][5]*SF[11] - P_[12][5]*SF[15] + P_[11][5]*SPP[10] - (P_[10][5]*q0)/2 + SF[4]*(P_[1][0] + P_[0][0]*SF[8] + P_[2][0]*SF[7] + P_[3][0]*SF[11] - P_[12][0]*SF[15] + P_[11][0]*SPP[10] - (P_[10][0]*q0)/2) + SF[3]*(P_[1][2] + P_[0][2]*SF[8] + P_[2][2]*SF[7] + P_[3][2]*SF[11] - P_[12][2]*SF[15] + P_[11][2]*SPP[10] - (P_[10][2]*q0)/2) + SF[5]*(P_[1][3] + P_[0][3]*SF[8] + P_[2][3]*SF[7] + P_[3][3]*SF[11] - P_[12][3]*SF[15] + P_[11][3]*SPP[10] - (P_[10][3]*q0)/2) - SPP[0]*(P_[1][1] + P_[0][1]*SF[8] + P_[2][1]*SF[7] + P_[3][1]*SF[11] - P_[12][1]*SF[15] + P_[11][1]*SPP[10] - (P_[10][1]*q0)/2) - SPP[8]*(P_[1][13] + P_[0][13]*SF[8] + P_[2][13]*SF[7] + P_[3][13]*SF[11] - P_[12][13]*SF[15] + P_[11][13]*SPP[10] - (P_[10][13]*q0)/2) + SPP[2]*(P_[1][14] + P_[0][14]*SF[8] + P_[2][14]*SF[7] + P_[3][14]*SF[11] - P_[12][14]*SF[15] + P_[11][14]*SPP[10] - (P_[10][14]*q0)/2) + SPP[5]*(P_[1][15] + P_[0][15]*SF[8] + P_[2][15]*SF[7] + P_[3][15]*SF[11] - P_[12][15]*SF[15] + P_[11][15]*SPP[10] - (P_[10][15]*q0)/2);
    nextP[2][5] = P_[2][5] + P_[0][5]*SF[6] + P_[1][5]*SF[10] + P_[3][5]*SF[8] + P_[12][5]*SF[14] - P_[10][5]*SPP[10] - (P_[11][5]*q0)/2 + SF[4]*(P_[2][0] + P_[0][0]*SF[6] + P_[1][0]*SF[10] + P_[3][0]*SF[8] + P_[12][0]*SF[14] - P_[10][0]*SPP[10] - (P_[11][0]*q0)/2) + SF[3]*(P_[2][2] + P_[0][2]*SF[6] + P_[1][2]*SF[10] + P_[3][2]*SF[8] + P_[12][2]*SF[14] - P_[10][2]*SPP[10] - (P_[11][2]*q0)/2) + SF[5]*(P_[2][3] + P_[0][3]*SF[6] + P_[1][3]*SF[10] + P_[3][3]*SF[8] + P_[12][3]*SF[14] - P_[10][3]*SPP[10] - (P_[11][3]*q0)/2) - SPP[0]*(P_[2][1] + P_[0][1]*SF[6] + P_[1][1]*SF[10] + P_[3][1]*SF[8] + P_[12][1]*SF[14] - P_[10][1]*SPP[10] - (P_[11][1]*q0)/2) - SPP[8]*(P_[2][13] + P_[0][13]*SF[6] + P_[1][13]*SF[10] + P_[3][13]*SF[8] + P_[12][13]*SF[14] - P_[10][13]*SPP[10] - (P_[11][13]*q0)/2) + SPP[2]*(P_[2][14] + P_[0][14]*SF[6] + P_[1][14]*SF[10] + P_[3][14]*SF[8] + P_[12][14]*SF[14] - P_[10][14]*SPP[10] - (P_[11][14]*q0)/2) + SPP[5]*(P_[2][15] + P_[0][15]*SF[6] + P_[1][15]*SF[10] + P_[3][15]*SF[8] + P_[12][15]*SF[14] - P_[10][15]*SPP[10] - (P_[11][15]*q0)/2);
    nextP[3][5] = P_[3][5] + P_[0][5]*SF[7] + P_[1][5]*SF[6] + P_[2][5]*SF[9] + P_[10][5]*SF[15] - P_[11][5]*SF[14] - (P_[12][5]*q0)/2 + SF[4]*(P_[3][0] + P_[0][0]*SF[7] + P_[1][0]*SF[6] + P_[2][0]*SF[9] + P_[10][0]*SF[15] - P_[11][0]*SF[14] - (P_[12][0]*q0)/2) + SF[3]*(P_[3][2] + P_[0][2]*SF[7] + P_[1][2]*SF[6] + P_[2][2]*SF[9] + P_[10][2]*SF[15] - P_[11][2]*SF[14] - (P_[12][2]*q0)/2) + SF[5]*(P_[3][3] + P_[0][3]*SF[7] + P_[1][3]*SF[6] + P_[2][3]*SF[9] + P_[10][3]*SF[15] - P_[11][3]*SF[14] - (P_[12][3]*q0)/2) - SPP[0]*(P_[3][1] + P_[0][1]*SF[7] + P_[1][1]*SF[6] + P_[2][1]*SF[9] + P_[10][1]*SF[15] - P_[11][1]*SF[14] - (P_[12][1]*q0)/2) - SPP[8]*(P_[3][13] + P_[0][13]*SF[7] + P_[1][13]*SF[6] + P_[2][13]*SF[9] + P_[10][13]*SF[15] - P_[11][13]*SF[14] - (P_[12][13]*q0)/2) + SPP[2]*(P_[3][14] + P_[0][14]*SF[7] + P_[1][14]*SF[6] + P_[2][14]*SF[9] + P_[10][14]*SF[15] - P_[11][14]*SF[14] - (P_[12][14]*q0)/2) + SPP[5]*(P_[3][15] + P_[0][15]*SF[7] + P_[1][15]*SF[6] + P_[2][15]*SF[9] + P_[10][15]*SF[15] - P_[11][15]*SF[14] - (P_[12][15]*q0)/2);
    nextP[4][5] = P_[4][5] + SQ[2] + P_[0][5]*SF[5] + P_[1][5]*SF[3] - P_[3][5]*SF[4] + P_[2][5]*SPP[0] + P_[13][5]*SPP[3] + P_[14][5]*SPP[6] - P_[15][5]*SPP[9] + SF[4]*(P_[4][0] + P_[0][0]*SF[5] + P_[1][0]*SF[3] - P_[3][0]*SF[4] + P_[2][0]*SPP[0] + P_[13][0]*SPP[3] + P_[14][0]*SPP[6] - P_[15][0]*SPP[9]) + SF[3]*(P_[4][2] + P_[0][2]*SF[5] + P_[1][2]*SF[3] - P_[3][2]*SF[4] + P_[2][2]*SPP[0] + P_[13][2]*SPP[3] + P_[14][2]*SPP[6] - P_[15][2]*SPP[9]) + SF[5]*(P_[4][3] + P_[0][3]*SF[5] + P_[1][3]*SF[3] - P_[3][3]*SF[4] + P_[2][3]*SPP[0] + P_[13][3]*SPP[3] + P_[14][3]*SPP[6] - P_[15][3]*SPP[9]) - SPP[0]*(P_[4][1] + P_[0][1]*SF[5] + P_[1][1]*SF[3] - P_[3][1]*SF[4] + P_[2][1]*SPP[0] + P_[13][1]*SPP[3] + P_[14][1]*SPP[6] - P_[15][1]*SPP[9]) - SPP[8]*(P_[4][13] + P_[0][13]*SF[5] + P_[1][13]*SF[3] - P_[3][13]*SF[4] + P_[2][13]*SPP[0] + P_[13][13]*SPP[3] + P_[14][13]*SPP[6] - P_[15][13]*SPP[9]) + SPP[2]*(P_[4][14] + P_[0][14]*SF[5] + P_[1][14]*SF[3] - P_[3][14]*SF[4] + P_[2][14]*SPP[0] + P_[13][14]*SPP[3] + P_[14][14]*SPP[6] - P_[15][14]*SPP[9]) + SPP[5]*(P_[4][15] + P_[0][15]*SF[5] + P_[1][15]*SF[3] - P_[3][15]*SF[4] + P_[2][15]*SPP[0] + P_[13][15]*SPP[3] + P_[14][15]*SPP[6] - P_[15][15]*SPP[9]);
    nextP[5][5] = P_[5][5] + P_[0][5]*SF[4] + P_[2][5]*SF[3] + P_[3][5]*SF[5] - P_[1][5]*SPP[0] - P_[13][5]*SPP[8] + P_[14][5]*SPP[2] + P_[15][5]*SPP[5] + dvxVar*sq(SG[7] + 2*q0*q3) + dvzVar*sq(SG[5] - 2*q0*q1) + SF[4]*(P_[5][0] + P_[0][0]*SF[4] + P_[2][0]*SF[3] + P_[3][0]*SF[5] - P_[1][0]*SPP[0] - P_[13][0]*SPP[8] + P_[14][0]*SPP[2] + P_[15][0]*SPP[5]) + SF[3]*(P_[5][2] + P_[0][2]*SF[4] + P_[2][2]*SF[3] + P_[3][2]*SF[5] - P_[1][2]*SPP[0] - P_[13][2]*SPP[8] + P_[14][2]*SPP[2] + P_[15][2]*SPP[5]) + SF[5]*(P_[5][3] + P_[0][3]*SF[4] + P_[2][3]*SF[3] + P_[3][3]*SF[5] - P_[1][3]*SPP[0] - P_[13][3]*SPP[8] + P_[14][3]*SPP[2] + P_[15][3]*SPP[5]) - SPP[0]*(P_[5][1] + P_[0][1]*SF[4] + P_[2][1]*SF[3] + P_[3][1]*SF[5] - P_[1][1]*SPP[0] - P_[13][1]*SPP[8] + P_[14][1]*SPP[2] + P_[15][1]*SPP[5]) - SPP[8]*(P_[5][13] + P_[0][13]*SF[4] + P_[2][13]*SF[3] + P_[3][13]*SF[5] - P_[1][13]*SPP[0] - P_[13][13]*SPP[8] + P_[14][13]*SPP[2] + P_[15][13]*SPP[5]) + SPP[2]*(P_[5][14] + P_[0][14]*SF[4] + P_[2][14]*SF[3] + P_[3][14]*SF[5] - P_[1][14]*SPP[0] - P_[13][14]*SPP[8] + P_[14][14]*SPP[2] + P_[15][14]*SPP[5]) + SPP[5]*(P_[5][15] + P_[0][15]*SF[4] + P_[2][15]*SF[3] + P_[3][15]*SF[5] - P_[1][15]*SPP[0] - P_[13][15]*SPP[8] + P_[14][15]*SPP[2] + P_[15][15]*SPP[5]) + dvyVar*sq(SG[1] - SG[2] + SG[3] - SG[4]);
    nextP[0][6] = P_[0][6] + P_[1][6]*SF[9] + P_[2][6]*SF[11] + P_[3][6]*SF[10] + P_[10][6]*SF[14] + P_[11][6]*SF[15] + P_[12][6]*SPP[10] + SF[4]*(P_[0][1] + P_[1][1]*SF[9] + P_[2][1]*SF[11] + P_[3][1]*SF[10] + P_[10][1]*SF[14] + P_[11][1]*SF[15] + P_[12][1]*SPP[10]) - SF[5]*(P_[0][2] + P_[1][2]*SF[9] + P_[2][2]*SF[11] + P_[3][2]*SF[10] + P_[10][2]*SF[14] + P_[11][2]*SF[15] + P_[12][2]*SPP[10]) + SF[3]*(P_[0][3] + P_[1][3]*SF[9] + P_[2][3]*SF[11] + P_[3][3]*SF[10] + P_[10][3]*SF[14] + P_[11][3]*SF[15] + P_[12][3]*SPP[10]) + SPP[0]*(P_[0][0] + P_[1][0]*SF[9] + P_[2][0]*SF[11] + P_[3][0]*SF[10] + P_[10][0]*SF[14] + P_[11][0]*SF[15] + P_[12][0]*SPP[10]) + SPP[4]*(P_[0][13] + P_[1][13]*SF[9] + P_[2][13]*SF[11] + P_[3][13]*SF[10] + P_[10][13]*SF[14] + P_[11][13]*SF[15] + P_[12][13]*SPP[10]) - SPP[7]*(P_[0][14] + P_[1][14]*SF[9] + P_[2][14]*SF[11] + P_[3][14]*SF[10] + P_[10][14]*SF[14] + P_[11][14]*SF[15] + P_[12][14]*SPP[10]) - SPP[1]*(P_[0][15] + P_[1][15]*SF[9] + P_[2][15]*SF[11] + P_[3][15]*SF[10] + P_[10][15]*SF[14] + P_[11][15]*SF[15] + P_[12][15]*SPP[10]);
    nextP[1][6] = P_[1][6] + P_[0][6]*SF[8] + P_[2][6]*SF[7] + P_[3][6]*SF[11] - P_[12][6]*SF[15] + P_[11][6]*SPP[10] - (P_[10][6]*q0)/2 + SF[4]*(P_[1][1] + P_[0][1]*SF[8] + P_[2][1]*SF[7] + P_[3][1]*SF[11] - P_[12][1]*SF[15] + P_[11][1]*SPP[10] - (P_[10][1]*q0)/2) - SF[5]*(P_[1][2] + P_[0][2]*SF[8] + P_[2][2]*SF[7] + P_[3][2]*SF[11] - P_[12][2]*SF[15] + P_[11][2]*SPP[10] - (P_[10][2]*q0)/2) + SF[3]*(P_[1][3] + P_[0][3]*SF[8] + P_[2][3]*SF[7] + P_[3][3]*SF[11] - P_[12][3]*SF[15] + P_[11][3]*SPP[10] - (P_[10][3]*q0)/2) + SPP[0]*(P_[1][0] + P_[0][0]*SF[8] + P_[2][0]*SF[7] + P_[3][0]*SF[11] - P_[12][0]*SF[15] + P_[11][0]*SPP[10] - (P_[10][0]*q0)/2) + SPP[4]*(P_[1][13] + P_[0][13]*SF[8] + P_[2][13]*SF[7] + P_[3][13]*SF[11] - P_[12][13]*SF[15] + P_[11][13]*SPP[10] - (P_[10][13]*q0)/2) - SPP[7]*(P_[1][14] + P_[0][14]*SF[8] + P_[2][14]*SF[7] + P_[3][14]*SF[11] - P_[12][14]*SF[15] + P_[11][14]*SPP[10] - (P_[10][14]*q0)/2) - SPP[1]*(P_[1][15] + P_[0][15]*SF[8] + P_[2][15]*SF[7] + P_[3][15]*SF[11] - P_[12][15]*SF[15] + P_[11][15]*SPP[10] - (P_[10][15]*q0)/2);
    nextP[2][6] = P_[2][6] + P_[0][6]*SF[6] + P_[1][6]*SF[10] + P_[3][6]*SF[8] + P_[12][6]*SF[14] - P_[10][6]*SPP[10] - (P_[11][6]*q0)/2 + SF[4]*(P_[2][1] + P_[0][1]*SF[6] + P_[1][1]*SF[10] + P_[3][1]*SF[8] + P_[12][1]*SF[14] - P_[10][1]*SPP[10] - (P_[11][1]*q0)/2) - SF[5]*(P_[2][2] + P_[0][2]*SF[6] + P_[1][2]*SF[10] + P_[3][2]*SF[8] + P_[12][2]*SF[14] - P_[10][2]*SPP[10] - (P_[11][2]*q0)/2) + SF[3]*(P_[2][3] + P_[0][3]*SF[6] + P_[1][3]*SF[10] + P_[3][3]*SF[8] + P_[12][3]*SF[14] - P_[10][3]*SPP[10] - (P_[11][3]*q0)/2) + SPP[0]*(P_[2][0] + P_[0][0]*SF[6] + P_[1][0]*SF[10] + P_[3][0]*SF[8] + P_[12][0]*SF[14] - P_[10][0]*SPP[10] - (P_[11][0]*q0)/2) + SPP[4]*(P_[2][13] + P_[0][13]*SF[6] + P_[1][13]*SF[10] + P_[3][13]*SF[8] + P_[12][13]*SF[14] - P_[10][13]*SPP[10] - (P_[11][13]*q0)/2) - SPP[7]*(P_[2][14] + P_[0][14]*SF[6] + P_[1][14]*SF[10] + P_[3][14]*SF[8] + P_[12][14]*SF[14] - P_[10][14]*SPP[10] - (P_[11][14]*q0)/2) - SPP[1]*(P_[2][15] + P_[0][15]*SF[6] + P_[1][15]*SF[10] + P_[3][15]*SF[8] + P_[12][15]*SF[14] - P_[10][15]*SPP[10] - (P_[11][15]*q0)/2);
    nextP[3][6] = P_[3][6] + P_[0][6]*SF[7] + P_[1][6]*SF[6] + P_[2][6]*SF[9] + P_[10][6]*SF[15] - P_[11][6]*SF[14] - (P_[12][6]*q0)/2 + SF[4]*(P_[3][1] + P_[0][1]*SF[7] + P_[1][1]*SF[6] + P_[2][1]*SF[9] + P_[10][1]*SF[15] - P_[11][1]*SF[14] - (P_[12][1]*q0)/2) - SF[5]*(P_[3][2] + P_[0][2]*SF[7] + P_[1][2]*SF[6] + P_[2][2]*SF[9] + P_[10][2]*SF[15] - P_[11][2]*SF[14] - (P_[12][2]*q0)/2) + SF[3]*(P_[3][3] + P_[0][3]*SF[7] + P_[1][3]*SF[6] + P_[2][3]*SF[9] + P_[10][3]*SF[15] - P_[11][3]*SF[14] - (P_[12][3]*q0)/2) + SPP[0]*(P_[3][0] + P_[0][0]*SF[7] + P_[1][0]*SF[6] + P_[2][0]*SF[9] + P_[10][0]*SF[15] - P_[11][0]*SF[14] - (P_[12][0]*q0)/2) + SPP[4]*(P_[3][13] + P_[0][13]*SF[7] + P_[1][13]*SF[6] + P_[2][13]*SF[9] + P_[10][13]*SF[15] - P_[11][13]*SF[14] - (P_[12][13]*q0)/2) - SPP[7]*(P_[3][14] + P_[0][14]*SF[7] + P_[1][14]*SF[6] + P_[2][14]*SF[9] + P_[10][14]*SF[15] - P_[11][14]*SF[14] - (P_[12][14]*q0)/2) - SPP[1]*(P_[3][15] + P_[0][15]*SF[7] + P_[1][15]*SF[6] + P_[2][15]*SF[9] + P_[10][15]*SF[15] - P_[11][15]*SF[14] - (P_[12][15]*q0)/2);
    nextP[4][6] = P_[4][6] + SQ[1] + P_[0][6]*SF[5] + P_[1][6]*SF[3] - P_[3][6]*SF[4] + P_[2][6]*SPP[0] + P_[13][6]*SPP[3] + P_[14][6]*SPP[6] - P_[15][6]*SPP[9] + SF[4]*(P_[4][1] + P_[0][1]*SF[5] + P_[1][1]*SF[3] - P_[3][1]*SF[4] + P_[2][1]*SPP[0] + P_[13][1]*SPP[3] + P_[14][1]*SPP[6] - P_[15][1]*SPP[9]) - SF[5]*(P_[4][2] + P_[0][2]*SF[5] + P_[1][2]*SF[3] - P_[3][2]*SF[4] + P_[2][2]*SPP[0] + P_[13][2]*SPP[3] + P_[14][2]*SPP[6] - P_[15][2]*SPP[9]) + SF[3]*(P_[4][3] + P_[0][3]*SF[5] + P_[1][3]*SF[3] - P_[3][3]*SF[4] + P_[2][3]*SPP[0] + P_[13][3]*SPP[3] + P_[14][3]*SPP[6] - P_[15][3]*SPP[9]) + SPP[0]*(P_[4][0] + P_[0][0]*SF[5] + P_[1][0]*SF[3] - P_[3][0]*SF[4] + P_[2][0]*SPP[0] + P_[13][0]*SPP[3] + P_[14][0]*SPP[6] - P_[15][0]*SPP[9]) + SPP[4]*(P_[4][13] + P_[0][13]*SF[5] + P_[1][13]*SF[3] - P_[3][13]*SF[4] + P_[2][13]*SPP[0] + P_[13][13]*SPP[3] + P_[14][13]*SPP[6] - P_[15][13]*SPP[9]) - SPP[7]*(P_[4][14] + P_[0][14]*SF[5] + P_[1][14]*SF[3] - P_[3][14]*SF[4] + P_[2][14]*SPP[0] + P_[13][14]*SPP[3] + P_[14][14]*SPP[6] - P_[15][14]*SPP[9]) - SPP[1]*(P_[4][15] + P_[0][15]*SF[5] + P_[1][15]*SF[3] - P_[3][15]*SF[4] + P_[2][15]*SPP[0] + P_[13][15]*SPP[3] + P_[14][15]*SPP[6] - P_[15][15]*SPP[9]);
    nextP[5][6] = P_[5][6] + SQ[0] + P_[0][6]*SF[4] + P_[2][6]*SF[3] + P_[3][6]*SF[5] - P_[1][6]*SPP[0] - P_[13][6]*SPP[8] + P_[14][6]*SPP[2] + P_[15][6]*SPP[5] + SF[4]*(P_[5][1] + P_[0][1]*SF[4] + P_[2][1]*SF[3] + P_[3][1]*SF[5] - P_[1][1]*SPP[0] - P_[13][1]*SPP[8] + P_[14][1]*SPP[2] + P_[15][1]*SPP[5]) - SF[5]*(P_[5][2] + P_[0][2]*SF[4] + P_[2][2]*SF[3] + P_[3][2]*SF[5] - P_[1][2]*SPP[0] - P_[13][2]*SPP[8] + P_[14][2]*SPP[2] + P_[15][2]*SPP[5]) + SF[3]*(P_[5][3] + P_[0][3]*SF[4] + P_[2][3]*SF[3] + P_[3][3]*SF[5] - P_[1][3]*SPP[0] - P_[13][3]*SPP[8] + P_[14][3]*SPP[2] + P_[15][3]*SPP[5]) + SPP[0]*(P_[5][0] + P_[0][0]*SF[4] + P_[2][0]*SF[3] + P_[3][0]*SF[5] - P_[1][0]*SPP[0] - P_[13][0]*SPP[8] + P_[14][0]*SPP[2] + P_[15][0]*SPP[5]) + SPP[4]*(P_[5][13] + P_[0][13]*SF[4] + P_[2][13]*SF[3] + P_[3][13]*SF[5] - P_[1][13]*SPP[0] - P_[13][13]*SPP[8] + P_[14][13]*SPP[2] + P_[15][13]*SPP[5]) - SPP[7]*(P_[5][14] + P_[0][14]*SF[4] + P_[2][14]*SF[3] + P_[3][14]*SF[5] - P_[1][14]*SPP[0] - P_[13][14]*SPP[8] + P_[14][14]*SPP[2] + P_[15][14]*SPP[5]) - SPP[1]*(P_[5][15] + P_[0][15]*SF[4] + P_[2][15]*SF[3] + P_[3][15]*SF[5] - P_[1][15]*SPP[0] - P_[13][15]*SPP[8] + P_[14][15]*SPP[2] + P_[15][15]*SPP[5]);
    nextP[6][6] = P_[6][6] + P_[1][6]*SF[4] - P_[2][6]*SF[5] + P_[3][6]*SF[3] + P_[0][6]*SPP[0] + P_[13][6]*SPP[4] - P_[14][6]*SPP[7] - P_[15][6]*SPP[1] + dvxVar*sq(SG[6] - 2*q0*q2) + dvyVar*sq(SG[5] + 2*q0*q1) + SF[4]*(P_[6][1] + P_[1][1]*SF[4] - P_[2][1]*SF[5] + P_[3][1]*SF[3] + P_[0][1]*SPP[0] + P_[13][1]*SPP[4] - P_[14][1]*SPP[7] - P_[15][1]*SPP[1]) - SF[5]*(P_[6][2] + P_[1][2]*SF[4] - P_[2][2]*SF[5] + P_[3][2]*SF[3] + P_[0][2]*SPP[0] + P_[13][2]*SPP[4] - P_[14][2]*SPP[7] - P_[15][2]*SPP[1]) + SF[3]*(P_[6][3] + P_[1][3]*SF[4] - P_[2][3]*SF[5] + P_[3][3]*SF[3] + P_[0][3]*SPP[0] + P_[13][3]*SPP[4] - P_[14][3]*SPP[7] - P_[15][3]*SPP[1]) + SPP[0]*(P_[6][0] + P_[1][0]*SF[4] - P_[2][0]*SF[5] + P_[3][0]*SF[3] + P_[0][0]*SPP[0] + P_[13][0]*SPP[4] - P_[14][0]*SPP[7] - P_[15][0]*SPP[1]) + SPP[4]*(P_[6][13] + P_[1][13]*SF[4] - P_[2][13]*SF[5] + P_[3][13]*SF[3] + P_[0][13]*SPP[0] + P_[13][13]*SPP[4] - P_[14][13]*SPP[7] - P_[15][13]*SPP[1]) - SPP[7]*(P_[6][14] + P_[1][14]*SF[4] - P_[2][14]*SF[5] + P_[3][14]*SF[3] + P_[0][14]*SPP[0] + P_[13][14]*SPP[4] - P_[14][14]*SPP[7] - P_[15][14]*SPP[1]) - SPP[1]*(P_[6][15] + P_[1][15]*SF[4] - P_[2][15]*SF[5] + P_[3][15]*SF[3] + P_[0][15]*SPP[0] + P_[13][15]*SPP[4] - P_[14][15]*SPP[7] - P_[15][15]*SPP[1]) + dvzVar*sq(SG[1] - SG[2] - SG[3] + SG[4]);
    nextP[0][7] = P_[0][7] + P_[1][7]*SF[9] + P_[2][7]*SF[11] + P_[3][7]*SF[10] + P_[10][7]*SF[14] + P_[11][7]*SF[15] + P_[12][7]*SPP[10] + dt*(P_[0][4] + P_[1][4]*SF[9] + P_[2][4]*SF[11] + P_[3][4]*SF[10] + P_[10][4]*SF[14] + P_[11][4]*SF[15] + P_[12][4]*SPP[10]);
    nextP[1][7] = P_[1][7] + P_[0][7]*SF[8] + P_[2][7]*SF[7] + P_[3][7]*SF[11] - P_[12][7]*SF[15] + P_[11][7]*SPP[10] - (P_[10][7]*q0)/2 + dt*(P_[1][4] + P_[0][4]*SF[8] + P_[2][4]*SF[7] + P_[3][4]*SF[11] - P_[12][4]*SF[15] + P_[11][4]*SPP[10] - (P_[10][4]*q0)/2);
    nextP[2][7] = P_[2][7] + P_[0][7]*SF[6] + P_[1][7]*SF[10] + P_[3][7]*SF[8] + P_[12][7]*SF[14] - P_[10][7]*SPP[10] - (P_[11][7]*q0)/2 + dt*(P_[2][4] + P_[0][4]*SF[6] + P_[1][4]*SF[10] + P_[3][4]*SF[8] + P_[12][4]*SF[14] - P_[10][4]*SPP[10] - (P_[11][4]*q0)/2);
    nextP[3][7] = P_[3][7] + P_[0][7]*SF[7] + P_[1][7]*SF[6] + P_[2][7]*SF[9] + P_[10][7]*SF[15] - P_[11][7]*SF[14] - (P_[12][7]*q0)/2 + dt*(P_[3][4] + P_[0][4]*SF[7] + P_[1][4]*SF[6] + P_[2][4]*SF[9] + P_[10][4]*SF[15] - P_[11][4]*SF[14] - (P_[12][4]*q0)/2);
    nextP[4][7] = P_[4][7] + P_[0][7]*SF[5] + P_[1][7]*SF[3] - P_[3][7]*SF[4] + P_[2][7]*SPP[0] + P_[13][7]*SPP[3] + P_[14][7]*SPP[6] - P_[15][7]*SPP[9] + dt*(P_[4][4] + P_[0][4]*SF[5] + P_[1][4]*SF[3] - P_[3][4]*SF[4] + P_[2][4]*SPP[0] + P_[13][4]*SPP[3] + P_[14][4]*SPP[6] - P_[15][4]*SPP[9]);
    nextP[5][7] = P_[5][7] + P_[0][7]*SF[4] + P_[2][7]*SF[3] + P_[3][7]*SF[5] - P_[1][7]*SPP[0] - P_[13][7]*SPP[8] + P_[14][7]*SPP[2] + P_[15][7]*SPP[5] + dt*(P_[5][4] + P_[0][4]*SF[4] + P_[2][4]*SF[3] + P_[3][4]*SF[5] - P_[1][4]*SPP[0] - P_[13][4]*SPP[8] + P_[14][4]*SPP[2] + P_[15][4]*SPP[5]);
    nextP[6][7] = P_[6][7] + P_[1][7]*SF[4] - P_[2][7]*SF[5] + P_[3][7]*SF[3] + P_[0][7]*SPP[0] + P_[13][7]*SPP[4] - P_[14][7]*SPP[7] - P_[15][7]*SPP[1] + dt*(P_[6][4] + P_[1][4]*SF[4] - P_[2][4]*SF[5] + P_[3][4]*SF[3] + P_[0][4]*SPP[0] + P_[13][4]*SPP[4] - P_[14][4]*SPP[7] - P_[15][4]*SPP[1]);
    nextP[7][7] = P_[7][7] + P_[4][7]*dt + dt*(P_[7][4] + P_[4][4]*dt);
    nextP[0][8] = P_[0][8] + P_[1][8]*SF[9] + P_[2][8]*SF[11] + P_[3][8]*SF[10] + P_[10][8]*SF[14] + P_[11][8]*SF[15] + P_[12][8]*SPP[10] + dt*(P_[0][5] + P_[1][5]*SF[9] + P_[2][5]*SF[11] + P_[3][5]*SF[10] + P_[10][5]*SF[14] + P_[11][5]*SF[15] + P_[12][5]*SPP[10]);
    nextP[1][8] = P_[1][8] + P_[0][8]*SF[8] + P_[2][8]*SF[7] + P_[3][8]*SF[11] - P_[12][8]*SF[15] + P_[11][8]*SPP[10] - (P_[10][8]*q0)/2 + dt*(P_[1][5] + P_[0][5]*SF[8] + P_[2][5]*SF[7] + P_[3][5]*SF[11] - P_[12][5]*SF[15] + P_[11][5]*SPP[10] - (P_[10][5]*q0)/2);
    nextP[2][8] = P_[2][8] + P_[0][8]*SF[6] + P_[1][8]*SF[10] + P_[3][8]*SF[8] + P_[12][8]*SF[14] - P_[10][8]*SPP[10] - (P_[11][8]*q0)/2 + dt*(P_[2][5] + P_[0][5]*SF[6] + P_[1][5]*SF[10] + P_[3][5]*SF[8] + P_[12][5]*SF[14] - P_[10][5]*SPP[10] - (P_[11][5]*q0)/2);
    nextP[3][8] = P_[3][8] + P_[0][8]*SF[7] + P_[1][8]*SF[6] + P_[2][8]*SF[9] + P_[10][8]*SF[15] - P_[11][8]*SF[14] - (P_[12][8]*q0)/2 + dt*(P_[3][5] + P_[0][5]*SF[7] + P_[1][5]*SF[6] + P_[2][5]*SF[9] + P_[10][5]*SF[15] - P_[11][5]*SF[14] - (P_[12][5]*q0)/2);
    nextP[4][8] = P_[4][8] + P_[0][8]*SF[5] + P_[1][8]*SF[3] - P_[3][8]*SF[4] + P_[2][8]*SPP[0] + P_[13][8]*SPP[3] + P_[14][8]*SPP[6] - P_[15][8]*SPP[9] + dt*(P_[4][5] + P_[0][5]*SF[5] + P_[1][5]*SF[3] - P_[3][5]*SF[4] + P_[2][5]*SPP[0] + P_[13][5]*SPP[3] + P_[14][5]*SPP[6] - P_[15][5]*SPP[9]);
    nextP[5][8] = P_[5][8] + P_[0][8]*SF[4] + P_[2][8]*SF[3] + P_[3][8]*SF[5] - P_[1][8]*SPP[0] - P_[13][8]*SPP[8] + P_[14][8]*SPP[2] + P_[15][8]*SPP[5] + dt*(P_[5][5] + P_[0][5]*SF[4] + P_[2][5]*SF[3] + P_[3][5]*SF[5] - P_[1][5]*SPP[0] - P_[13][5]*SPP[8] + P_[14][5]*SPP[2] + P_[15][5]*SPP[5]);
    nextP[6][8] = P_[6][8] + P_[1][8]*SF[4] - P_[2][8]*SF[5] + P_[3][8]*SF[3] + P_[0][8]*SPP[0] + P_[13][8]*SPP[4] - P_[14][8]*SPP[7] - P_[15][8]*SPP[1] + dt*(P_[6][5] + P_[1][5]*SF[4] - P_[2][5]*SF[5] + P_[3][5]*SF[3] + P_[0][5]*SPP[0] + P_[13][5]*SPP[4] - P_[14][5]*SPP[7] - P_[15][5]*SPP[1]);
    nextP[7][8] = P_[7][8] + P_[4][8]*dt + dt*(P_[7][5] + P_[4][5]*dt);
    nextP[8][8] = P_[8][8] + P_[5][8]*dt + dt*(P_[8][5] + P_[5][5]*dt);
    nextP[0][9] = P_[0][9] + P_[1][9]*SF[9] + P_[2][9]*SF[11] + P_[3][9]*SF[10] + P_[10][9]*SF[14] + P_[11][9]*SF[15] + P_[12][9]*SPP[10] + dt*(P_[0][6] + P_[1][6]*SF[9] + P_[2][6]*SF[11] + P_[3][6]*SF[10] + P_[10][6]*SF[14] + P_[11][6]*SF[15] + P_[12][6]*SPP[10]);
    nextP[1][9] = P_[1][9] + P_[0][9]*SF[8] + P_[2][9]*SF[7] + P_[3][9]*SF[11] - P_[12][9]*SF[15] + P_[11][9]*SPP[10] - (P_[10][9]*q0)/2 + dt*(P_[1][6] + P_[0][6]*SF[8] + P_[2][6]*SF[7] + P_[3][6]*SF[11] - P_[12][6]*SF[15] + P_[11][6]*SPP[10] - (P_[10][6]*q0)/2);
    nextP[2][9] = P_[2][9] + P_[0][9]*SF[6] + P_[1][9]*SF[10] + P_[3][9]*SF[8] + P_[12][9]*SF[14] - P_[10][9]*SPP[10] - (P_[11][9]*q0)/2 + dt*(P_[2][6] + P_[0][6]*SF[6] + P_[1][6]*SF[10] + P_[3][6]*SF[8] + P_[12][6]*SF[14] - P_[10][6]*SPP[10] - (P_[11][6]*q0)/2);
    nextP[3][9] = P_[3][9] + P_[0][9]*SF[7] + P_[1][9]*SF[6] + P_[2][9]*SF[9] + P_[10][9]*SF[15] - P_[11][9]*SF[14] - (P_[12][9]*q0)/2 + dt*(P_[3][6] + P_[0][6]*SF[7] + P_[1][6]*SF[6] + P_[2][6]*SF[9] + P_[10][6]*SF[15] - P_[11][6]*SF[14] - (P_[12][6]*q0)/2);
    nextP[4][9] = P_[4][9] + P_[0][9]*SF[5] + P_[1][9]*SF[3] - P_[3][9]*SF[4] + P_[2][9]*SPP[0] + P_[13][9]*SPP[3] + P_[14][9]*SPP[6] - P_[15][9]*SPP[9] + dt*(P_[4][6] + P_[0][6]*SF[5] + P_[1][6]*SF[3] - P_[3][6]*SF[4] + P_[2][6]*SPP[0] + P_[13][6]*SPP[3] + P_[14][6]*SPP[6] - P_[15][6]*SPP[9]);
    nextP[5][9] = P_[5][9] + P_[0][9]*SF[4] + P_[2][9]*SF[3] + P_[3][9]*SF[5] - P_[1][9]*SPP[0] - P_[13][9]*SPP[8] + P_[14][9]*SPP[2] + P_[15][9]*SPP[5] + dt*(P_[5][6] + P_[0][6]*SF[4] + P_[2][6]*SF[3] + P_[3][6]*SF[5] - P_[1][6]*SPP[0] - P_[13][6]*SPP[8] + P_[14][6]*SPP[2] + P_[15][6]*SPP[5]);
    nextP[6][9] = P_[6][9] + P_[1][9]*SF[4] - P_[2][9]*SF[5] + P_[3][9]*SF[3] + P_[0][9]*SPP[0] + P_[13][9]*SPP[4] - P_[14][9]*SPP[7] - P_[15][9]*SPP[1] + dt*(P_[6][6] + P_[1][6]*SF[4] - P_[2][6]*SF[5] + P_[3][6]*SF[3] + P_[0][6]*SPP[0] + P_[13][6]*SPP[4] - P_[14][6]*SPP[7] - P_[15][6]*SPP[1]);
    nextP[7][9] = P_[7][9] + P_[4][9]*dt + dt*(P_[7][6] + P_[4][6]*dt);
    nextP[8][9] = P_[8][9] + P_[5][9]*dt + dt*(P_[8][6] + P_[5][6]*dt);
    nextP[9][9] = P_[9][9] + P_[6][9]*dt + dt*(P_[9][6] + P_[6][6]*dt);
    nextP[0][10] = P_[0][10] + P_[1][10]*SF[9] + P_[2][10]*SF[11] + P_[3][10]*SF[10] + P_[10][10]*SF[14] + P_[11][10]*SF[15] + P_[12][10]*SPP[10];
    nextP[1][10] = P_[1][10] + P_[0][10]*SF[8] + P_[2][10]*SF[7] + P_[3][10]*SF[11] - P_[12][10]*SF[15] + P_[11][10]*SPP[10] - (P_[10][10]*q0)/2;
    nextP[2][10] = P_[2][10] + P_[0][10]*SF[6] + P_[1][10]*SF[10] + P_[3][10]*SF[8] + P_[12][10]*SF[14] - P_[10][10]*SPP[10] - (P_[11][10]*q0)/2;
    nextP[3][10] = P_[3][10] + P_[0][10]*SF[7] + P_[1][10]*SF[6] + P_[2][10]*SF[9] + P_[10][10]*SF[15] - P_[11][10]*SF[14] - (P_[12][10]*q0)/2;
    nextP[4][10] = P_[4][10] + P_[0][10]*SF[5] + P_[1][10]*SF[3] - P_[3][10]*SF[4] + P_[2][10]*SPP[0] + P_[13][10]*SPP[3] + P_[14][10]*SPP[6] - P_[15][10]*SPP[9];
    nextP[5][10] = P_[5][10] + P_[0][10]*SF[4] + P_[2][10]*SF[3] + P_[3][10]*SF[5] - P_[1][10]*SPP[0] - P_[13][10]*SPP[8] + P_[14][10]*SPP[2] + P_[15][10]*SPP[5];
    nextP[6][10] = P_[6][10] + P_[1][10]*SF[4] - P_[2][10]*SF[5] + P_[3][10]*SF[3] + P_[0][10]*SPP[0] + P_[13][10]*SPP[4] - P_[14][10]*SPP[7] - P_[15][10]*SPP[1];
    nextP[7][10] = P_[7][10] + P_[4][10]*dt;
    nextP[8][10] = P_[8][10] + P_[5][10]*dt;
    nextP[9][10] = P_[9][10] + P_[6][10]*dt;
    nextP[10][10] = P_[10][10];
    nextP[0][11] = P_[0][11] + P_[1][11]*SF[9] + P_[2][11]*SF[11] + P_[3][11]*SF[10] + P_[10][11]*SF[14] + P_[11][11]*SF[15] + P_[12][11]*SPP[10];
    nextP[1][11] = P_[1][11] + P_[0][11]*SF[8] + P_[2][11]*SF[7] + P_[3][11]*SF[11] - P_[12][11]*SF[15] + P_[11][11]*SPP[10] - (P_[10][11]*q0)/2;
    nextP[2][11] = P_[2][11] + P_[0][11]*SF[6] + P_[1][11]*SF[10] + P_[3][11]*SF[8] + P_[12][11]*SF[14] - P_[10][11]*SPP[10] - (P_[11][11]*q0)/2;
    nextP[3][11] = P_[3][11] + P_[0][11]*SF[7] + P_[1][11]*SF[6] + P_[2][11]*SF[9] + P_[10][11]*SF[15] - P_[11][11]*SF[14] - (P_[12][11]*q0)/2;
    nextP[4][11] = P_[4][11] + P_[0][11]*SF[5] + P_[1][11]*SF[3] - P_[3][11]*SF[4] + P_[2][11]*SPP[0] + P_[13][11]*SPP[3] + P_[14][11]*SPP[6] - P_[15][11]*SPP[9];
    nextP[5][11] = P_[5][11] + P_[0][11]*SF[4] + P_[2][11]*SF[3] + P_[3][11]*SF[5] - P_[1][11]*SPP[0] - P_[13][11]*SPP[8] + P_[14][11]*SPP[2] + P_[15][11]*SPP[5];
    nextP[6][11] = P_[6][11] + P_[1][11]*SF[4] - P_[2][11]*SF[5] + P_[3][11]*SF[3] + P_[0][11]*SPP[0] + P_[13][11]*SPP[4] - P_[14][11]*SPP[7] - P_[15][11]*SPP[1];
    nextP[7][11] = P_[7][11] + P_[4][11]*dt;
    nextP[8][11] = P_[8][11] + P_[5][11]*dt;
    nextP[9][11] = P_[9][11] + P_[6][11]*dt;
    nextP[10][11] = P_[10][11];
    nextP[11][11] = P_[11][11];
    nextP[0][12] = P_[0][12] + P_[1][12]*SF[9] + P_[2][12]*SF[11] + P_[3][12]*SF[10] + P_[10][12]*SF[14] + P_[11][12]*SF[15] + P_[12][12]*SPP[10];
    nextP[1][12] = P_[1][12] + P_[0][12]*SF[8] + P_[2][12]*SF[7] + P_[3][12]*SF[11] - P_[12][12]*SF[15] + P_[11][12]*SPP[10] - (P_[10][12]*q0)/2;
    nextP[2][12] = P_[2][12] + P_[0][12]*SF[6] + P_[1][12]*SF[10] + P_[3][12]*SF[8] + P_[12][12]*SF[14] - P_[10][12]*SPP[10] - (P_[11][12]*q0)/2;
    nextP[3][12] = P_[3][12] + P_[0][12]*SF[7] + P_[1][12]*SF[6] + P_[2][12]*SF[9] + P_[10][12]*SF[15] - P_[11][12]*SF[14] - (P_[12][12]*q0)/2;
    nextP[4][12] = P_[4][12] + P_[0][12]*SF[5] + P_[1][12]*SF[3] - P_[3][12]*SF[4] + P_[2][12]*SPP[0] + P_[13][12]*SPP[3] + P_[14][12]*SPP[6] - P_[15][12]*SPP[9];
    nextP[5][12] = P_[5][12] + P_[0][12]*SF[4] + P_[2][12]*SF[3] + P_[3][12]*SF[5] - P_[1][12]*SPP[0] - P_[13][12]*SPP[8] + P_[14][12]*SPP[2] + P_[15][12]*SPP[5];
    nextP[6][12] = P_[6][12] + P_[1][12]*SF[4] - P_[2][12]*SF[5] + P_[3][12]*SF[3] + P_[0][12]*SPP[0] + P_[13][12]*SPP[4] - P_[14][12]*SPP[7] - P_[15][12]*SPP[1];
    nextP[7][12] = P_[7][12] + P_[4][12]*dt;
    nextP[8][12] = P_[8][12] + P_[5][12]*dt;
    nextP[9][12] = P_[9][12] + P_[6][12]*dt;
    nextP[10][12] = P_[10][12];
    nextP[11][12] = P_[11][12];
    nextP[12][12] = P_[12][12];
    
    // add process noise that is not from the IMU
    for (unsigned i = 0; i <= 12; i++) {
      nextP[i][i] += process_noise[i];
    }

    // calculate variances and upper diagonal covariances for IMU delta velocity bias states
    nextP[0][13] = P_[0][13] + P_[1][13]*SF[9] + P_[2][13]*SF[11] + P_[3][13]*SF[10] + P_[10][13]*SF[14] + P_[11][13]*SF[15] + P_[12][13]*SPP[10];
    nextP[1][13] = P_[1][13] + P_[0][13]*SF[8] + P_[2][13]*SF[7] + P_[3][13]*SF[11] - P_[12][13]*SF[15] + P_[11][13]*SPP[10] - (P_[10][13]*q0)/2;
    nextP[2][13] = P_[2][13] + P_[0][13]*SF[6] + P_[1][13]*SF[10] + P_[3][13]*SF[8] + P_[12][13]*SF[14] - P_[10][13]*SPP[10] - (P_[11][13]*q0)/2;
    nextP[3][13] = P_[3][13] + P_[0][13]*SF[7] + P_[1][13]*SF[6] + P_[2][13]*SF[9] + P_[10][13]*SF[15] - P_[11][13]*SF[14] - (P_[12][13]*q0)/2;
    nextP[4][13] = P_[4][13] + P_[0][13]*SF[5] + P_[1][13]*SF[3] - P_[3][13]*SF[4] + P_[2][13]*SPP[0] + P_[13][13]*SPP[3] + P_[14][13]*SPP[6] - P_[15][13]*SPP[9];
    nextP[5][13] = P_[5][13] + P_[0][13]*SF[4] + P_[2][13]*SF[3] + P_[3][13]*SF[5] - P_[1][13]*SPP[0] - P_[13][13]*SPP[8] + P_[14][13]*SPP[2] + P_[15][13]*SPP[5];
    nextP[6][13] = P_[6][13] + P_[1][13]*SF[4] - P_[2][13]*SF[5] + P_[3][13]*SF[3] + P_[0][13]*SPP[0] + P_[13][13]*SPP[4] - P_[14][13]*SPP[7] - P_[15][13]*SPP[1];
    nextP[7][13] = P_[7][13] + P_[4][13]*dt;
    nextP[8][13] = P_[8][13] + P_[5][13]*dt;
    nextP[9][13] = P_[9][13] + P_[6][13]*dt;
    nextP[10][13] = P_[10][13];
    nextP[11][13] = P_[11][13];
    nextP[12][13] = P_[12][13];
    nextP[13][13] = P_[13][13];
    nextP[0][14] = P_[0][14] + P_[1][14]*SF[9] + P_[2][14]*SF[11] + P_[3][14]*SF[10] + P_[10][14]*SF[14] + P_[11][14]*SF[15] + P_[12][14]*SPP[10];
    nextP[1][14] = P_[1][14] + P_[0][14]*SF[8] + P_[2][14]*SF[7] + P_[3][14]*SF[11] - P_[12][14]*SF[15] + P_[11][14]*SPP[10] - (P_[10][14]*q0)/2;
    nextP[2][14] = P_[2][14] + P_[0][14]*SF[6] + P_[1][14]*SF[10] + P_[3][14]*SF[8] + P_[12][14]*SF[14] - P_[10][14]*SPP[10] - (P_[11][14]*q0)/2;
    nextP[3][14] = P_[3][14] + P_[0][14]*SF[7] + P_[1][14]*SF[6] + P_[2][14]*SF[9] + P_[10][14]*SF[15] - P_[11][14]*SF[14] - (P_[12][14]*q0)/2;
    nextP[4][14] = P_[4][14] + P_[0][14]*SF[5] + P_[1][14]*SF[3] - P_[3][14]*SF[4] + P_[2][14]*SPP[0] + P_[13][14]*SPP[3] + P_[14][14]*SPP[6] - P_[15][14]*SPP[9];
    nextP[5][14] = P_[5][14] + P_[0][14]*SF[4] + P_[2][14]*SF[3] + P_[3][14]*SF[5] - P_[1][14]*SPP[0] - P_[13][14]*SPP[8] + P_[14][14]*SPP[2] + P_[15][14]*SPP[5];
    nextP[6][14] = P_[6][14] + P_[1][14]*SF[4] - P_[2][14]*SF[5] + P_[3][14]*SF[3] + P_[0][14]*SPP[0] + P_[13][14]*SPP[4] - P_[14][14]*SPP[7] - P_[15][14]*SPP[1];
    nextP[7][14] = P_[7][14] + P_[4][14]*dt;
    nextP[8][14] = P_[8][14] + P_[5][14]*dt;
    nextP[9][14] = P_[9][14] + P_[6][14]*dt;
    nextP[10][14] = P_[10][14];
    nextP[11][14] = P_[11][14];
    nextP[12][14] = P_[12][14];
    nextP[13][14] = P_[13][14];
    nextP[14][14] = P_[14][14];
    nextP[0][15] = P_[0][15] + P_[1][15]*SF[9] + P_[2][15]*SF[11] + P_[3][15]*SF[10] + P_[10][15]*SF[14] + P_[11][15]*SF[15] + P_[12][15]*SPP[10];
    nextP[1][15] = P_[1][15] + P_[0][15]*SF[8] + P_[2][15]*SF[7] + P_[3][15]*SF[11] - P_[12][15]*SF[15] + P_[11][15]*SPP[10] - (P_[10][15]*q0)/2;
    nextP[2][15] = P_[2][15] + P_[0][15]*SF[6] + P_[1][15]*SF[10] + P_[3][15]*SF[8] + P_[12][15]*SF[14] - P_[10][15]*SPP[10] - (P_[11][15]*q0)/2;
    nextP[3][15] = P_[3][15] + P_[0][15]*SF[7] + P_[1][15]*SF[6] + P_[2][15]*SF[9] + P_[10][15]*SF[15] - P_[11][15]*SF[14] - (P_[12][15]*q0)/2;
    nextP[4][15] = P_[4][15] + P_[0][15]*SF[5] + P_[1][15]*SF[3] - P_[3][15]*SF[4] + P_[2][15]*SPP[0] + P_[13][15]*SPP[3] + P_[14][15]*SPP[6] - P_[15][15]*SPP[9];
    nextP[5][15] = P_[5][15] + P_[0][15]*SF[4] + P_[2][15]*SF[3] + P_[3][15]*SF[5] - P_[1][15]*SPP[0] - P_[13][15]*SPP[8] + P_[14][15]*SPP[2] + P_[15][15]*SPP[5];
    nextP[6][15] = P_[6][15] + P_[1][15]*SF[4] - P_[2][15]*SF[5] + P_[3][15]*SF[3] + P_[0][15]*SPP[0] + P_[13][15]*SPP[4] - P_[14][15]*SPP[7] - P_[15][15]*SPP[1];
    nextP[7][15] = P_[7][15] + P_[4][15]*dt;
    nextP[8][15] = P_[8][15] + P_[5][15]*dt;
    nextP[9][15] = P_[9][15] + P_[6][15]*dt;
    nextP[10][15] = P_[10][15];
    nextP[11][15] = P_[11][15];
    nextP[12][15] = P_[12][15];
    nextP[13][15] = P_[13][15];
    nextP[14][15] = P_[14][15];
    nextP[15][15] = P_[15][15];

    // add process noise that is not from the IMU
    for (unsigned i = 13; i <= 15; i++) {
      nextP[i][i] += process_noise[i];
    }

    // Don't do covariance prediction on magnetic field states unless we are using 3-axis fusion
    if (mag_3D_) {
      // calculate variances and upper diagonal covariances for earth and body magnetic field states
      nextP[0][16] = P_[0][16] + P_[1][16]*SF[9] + P_[2][16]*SF[11] + P_[3][16]*SF[10] + P_[10][16]*SF[14] + P_[11][16]*SF[15] + P_[12][16]*SPP[10];
      nextP[1][16] = P_[1][16] + P_[0][16]*SF[8] + P_[2][16]*SF[7] + P_[3][16]*SF[11] - P_[12][16]*SF[15] + P_[11][16]*SPP[10] - (P_[10][16]*q0)/2;
      nextP[2][16] = P_[2][16] + P_[0][16]*SF[6] + P_[1][16]*SF[10] + P_[3][16]*SF[8] + P_[12][16]*SF[14] - P_[10][16]*SPP[10] - (P_[11][16]*q0)/2;
      nextP[3][16] = P_[3][16] + P_[0][16]*SF[7] + P_[1][16]*SF[6] + P_[2][16]*SF[9] + P_[10][16]*SF[15] - P_[11][16]*SF[14] - (P_[12][16]*q0)/2;
      nextP[4][16] = P_[4][16] + P_[0][16]*SF[5] + P_[1][16]*SF[3] - P_[3][16]*SF[4] + P_[2][16]*SPP[0] + P_[13][16]*SPP[3] + P_[14][16]*SPP[6] - P_[15][16]*SPP[9];
      nextP[5][16] = P_[5][16] + P_[0][16]*SF[4] + P_[2][16]*SF[3] + P_[3][16]*SF[5] - P_[1][16]*SPP[0] - P_[13][16]*SPP[8] + P_[14][16]*SPP[2] + P_[15][16]*SPP[5];
      nextP[6][16] = P_[6][16] + P_[1][16]*SF[4] - P_[2][16]*SF[5] + P_[3][16]*SF[3] + P_[0][16]*SPP[0] + P_[13][16]*SPP[4] - P_[14][16]*SPP[7] - P_[15][16]*SPP[1];
      nextP[7][16] = P_[7][16] + P_[4][16]*dt;
      nextP[8][16] = P_[8][16] + P_[5][16]*dt;
      nextP[9][16] = P_[9][16] + P_[6][16]*dt;
      nextP[10][16] = P_[10][16];
      nextP[11][16] = P_[11][16];
      nextP[12][16] = P_[12][16];
      nextP[13][16] = P_[13][16];
      nextP[14][16] = P_[14][16];
      nextP[15][16] = P_[15][16];
      nextP[16][16] = P_[16][16];
      nextP[0][17] = P_[0][17] + P_[1][17]*SF[9] + P_[2][17]*SF[11] + P_[3][17]*SF[10] + P_[10][17]*SF[14] + P_[11][17]*SF[15] + P_[12][17]*SPP[10];
      nextP[1][17] = P_[1][17] + P_[0][17]*SF[8] + P_[2][17]*SF[7] + P_[3][17]*SF[11] - P_[12][17]*SF[15] + P_[11][17]*SPP[10] - (P_[10][17]*q0)/2;
      nextP[2][17] = P_[2][17] + P_[0][17]*SF[6] + P_[1][17]*SF[10] + P_[3][17]*SF[8] + P_[12][17]*SF[14] - P_[10][17]*SPP[10] - (P_[11][17]*q0)/2;
      nextP[3][17] = P_[3][17] + P_[0][17]*SF[7] + P_[1][17]*SF[6] + P_[2][17]*SF[9] + P_[10][17]*SF[15] - P_[11][17]*SF[14] - (P_[12][17]*q0)/2;
      nextP[4][17] = P_[4][17] + P_[0][17]*SF[5] + P_[1][17]*SF[3] - P_[3][17]*SF[4] + P_[2][17]*SPP[0] + P_[13][17]*SPP[3] + P_[14][17]*SPP[6] - P_[15][17]*SPP[9];
      nextP[5][17] = P_[5][17] + P_[0][17]*SF[4] + P_[2][17]*SF[3] + P_[3][17]*SF[5] - P_[1][17]*SPP[0] - P_[13][17]*SPP[8] + P_[14][17]*SPP[2] + P_[15][17]*SPP[5];
      nextP[6][17] = P_[6][17] + P_[1][17]*SF[4] - P_[2][17]*SF[5] + P_[3][17]*SF[3] + P_[0][17]*SPP[0] + P_[13][17]*SPP[4] - P_[14][17]*SPP[7] - P_[15][17]*SPP[1];
      nextP[7][17] = P_[7][17] + P_[4][17]*dt;
      nextP[8][17] = P_[8][17] + P_[5][17]*dt;
      nextP[9][17] = P_[9][17] + P_[6][17]*dt;
      nextP[10][17] = P_[10][17];
      nextP[11][17] = P_[11][17];
      nextP[12][17] = P_[12][17];
      nextP[13][17] = P_[13][17];
      nextP[14][17] = P_[14][17];
      nextP[15][17] = P_[15][17];
      nextP[16][17] = P_[16][17];
      nextP[17][17] = P_[17][17];
      nextP[0][18] = P_[0][18] + P_[1][18]*SF[9] + P_[2][18]*SF[11] + P_[3][18]*SF[10] + P_[10][18]*SF[14] + P_[11][18]*SF[15] + P_[12][18]*SPP[10];
      nextP[1][18] = P_[1][18] + P_[0][18]*SF[8] + P_[2][18]*SF[7] + P_[3][18]*SF[11] - P_[12][18]*SF[15] + P_[11][18]*SPP[10] - (P_[10][18]*q0)/2;
      nextP[2][18] = P_[2][18] + P_[0][18]*SF[6] + P_[1][18]*SF[10] + P_[3][18]*SF[8] + P_[12][18]*SF[14] - P_[10][18]*SPP[10] - (P_[11][18]*q0)/2;
      nextP[3][18] = P_[3][18] + P_[0][18]*SF[7] + P_[1][18]*SF[6] + P_[2][18]*SF[9] + P_[10][18]*SF[15] - P_[11][18]*SF[14] - (P_[12][18]*q0)/2;
      nextP[4][18] = P_[4][18] + P_[0][18]*SF[5] + P_[1][18]*SF[3] - P_[3][18]*SF[4] + P_[2][18]*SPP[0] + P_[13][18]*SPP[3] + P_[14][18]*SPP[6] - P_[15][18]*SPP[9];
      nextP[5][18] = P_[5][18] + P_[0][18]*SF[4] + P_[2][18]*SF[3] + P_[3][18]*SF[5] - P_[1][18]*SPP[0] - P_[13][18]*SPP[8] + P_[14][18]*SPP[2] + P_[15][18]*SPP[5];
      nextP[6][18] = P_[6][18] + P_[1][18]*SF[4] - P_[2][18]*SF[5] + P_[3][18]*SF[3] + P_[0][18]*SPP[0] + P_[13][18]*SPP[4] - P_[14][18]*SPP[7] - P_[15][18]*SPP[1];
      nextP[7][18] = P_[7][18] + P_[4][18]*dt;
      nextP[8][18] = P_[8][18] + P_[5][18]*dt;
      nextP[9][18] = P_[9][18] + P_[6][18]*dt;
      nextP[10][18] = P_[10][18];
      nextP[11][18] = P_[11][18];
      nextP[12][18] = P_[12][18];
      nextP[13][18] = P_[13][18];
      nextP[14][18] = P_[14][18];
      nextP[15][18] = P_[15][18];
      nextP[16][18] = P_[16][18];
      nextP[17][18] = P_[17][18];
      nextP[18][18] = P_[18][18];
      nextP[0][19] = P_[0][19] + P_[1][19]*SF[9] + P_[2][19]*SF[11] + P_[3][19]*SF[10] + P_[10][19]*SF[14] + P_[11][19]*SF[15] + P_[12][19]*SPP[10];
      nextP[1][19] = P_[1][19] + P_[0][19]*SF[8] + P_[2][19]*SF[7] + P_[3][19]*SF[11] - P_[12][19]*SF[15] + P_[11][19]*SPP[10] - (P_[10][19]*q0)/2;
      nextP[2][19] = P_[2][19] + P_[0][19]*SF[6] + P_[1][19]*SF[10] + P_[3][19]*SF[8] + P_[12][19]*SF[14] - P_[10][19]*SPP[10] - (P_[11][19]*q0)/2;
      nextP[3][19] = P_[3][19] + P_[0][19]*SF[7] + P_[1][19]*SF[6] + P_[2][19]*SF[9] + P_[10][19]*SF[15] - P_[11][19]*SF[14] - (P_[12][19]*q0)/2;
      nextP[4][19] = P_[4][19] + P_[0][19]*SF[5] + P_[1][19]*SF[3] - P_[3][19]*SF[4] + P_[2][19]*SPP[0] + P_[13][19]*SPP[3] + P_[14][19]*SPP[6] - P_[15][19]*SPP[9];
      nextP[5][19] = P_[5][19] + P_[0][19]*SF[4] + P_[2][19]*SF[3] + P_[3][19]*SF[5] - P_[1][19]*SPP[0] - P_[13][19]*SPP[8] + P_[14][19]*SPP[2] + P_[15][19]*SPP[5];
      nextP[6][19] = P_[6][19] + P_[1][19]*SF[4] - P_[2][19]*SF[5] + P_[3][19]*SF[3] + P_[0][19]*SPP[0] + P_[13][19]*SPP[4] - P_[14][19]*SPP[7] - P_[15][19]*SPP[1];
      nextP[7][19] = P_[7][19] + P_[4][19]*dt;
      nextP[8][19] = P_[8][19] + P_[5][19]*dt;
      nextP[9][19] = P_[9][19] + P_[6][19]*dt;
      nextP[10][19] = P_[10][19];
      nextP[11][19] = P_[11][19];
      nextP[12][19] = P_[12][19];
      nextP[13][19] = P_[13][19];
      nextP[14][19] = P_[14][19];
      nextP[15][19] = P_[15][19];
      nextP[16][19] = P_[16][19];
      nextP[17][19] = P_[17][19];
      nextP[18][19] = P_[18][19];
      nextP[19][19] = P_[19][19];
      nextP[0][20] = P_[0][20] + P_[1][20]*SF[9] + P_[2][20]*SF[11] + P_[3][20]*SF[10] + P_[10][20]*SF[14] + P_[11][20]*SF[15] + P_[12][20]*SPP[10];
      nextP[1][20] = P_[1][20] + P_[0][20]*SF[8] + P_[2][20]*SF[7] + P_[3][20]*SF[11] - P_[12][20]*SF[15] + P_[11][20]*SPP[10] - (P_[10][20]*q0)/2;
      nextP[2][20] = P_[2][20] + P_[0][20]*SF[6] + P_[1][20]*SF[10] + P_[3][20]*SF[8] + P_[12][20]*SF[14] - P_[10][20]*SPP[10] - (P_[11][20]*q0)/2;
      nextP[3][20] = P_[3][20] + P_[0][20]*SF[7] + P_[1][20]*SF[6] + P_[2][20]*SF[9] + P_[10][20]*SF[15] - P_[11][20]*SF[14] - (P_[12][20]*q0)/2;
      nextP[4][20] = P_[4][20] + P_[0][20]*SF[5] + P_[1][20]*SF[3] - P_[3][20]*SF[4] + P_[2][20]*SPP[0] + P_[13][20]*SPP[3] + P_[14][20]*SPP[6] - P_[15][20]*SPP[9];
      nextP[5][20] = P_[5][20] + P_[0][20]*SF[4] + P_[2][20]*SF[3] + P_[3][20]*SF[5] - P_[1][20]*SPP[0] - P_[13][20]*SPP[8] + P_[14][20]*SPP[2] + P_[15][20]*SPP[5];
      nextP[6][20] = P_[6][20] + P_[1][20]*SF[4] - P_[2][20]*SF[5] + P_[3][20]*SF[3] + P_[0][20]*SPP[0] + P_[13][20]*SPP[4] - P_[14][20]*SPP[7] - P_[15][20]*SPP[1];
      nextP[7][20] = P_[7][20] + P_[4][20]*dt;
      nextP[8][20] = P_[8][20] + P_[5][20]*dt;
      nextP[9][20] = P_[9][20] + P_[6][20]*dt;
      nextP[10][20] = P_[10][20];
      nextP[11][20] = P_[11][20];
      nextP[12][20] = P_[12][20];
      nextP[13][20] = P_[13][20];
      nextP[14][20] = P_[14][20];
      nextP[15][20] = P_[15][20];
      nextP[16][20] = P_[16][20];
      nextP[17][20] = P_[17][20];
      nextP[18][20] = P_[18][20];
      nextP[19][20] = P_[19][20];
      nextP[20][20] = P_[20][20];
      nextP[0][21] = P_[0][21] + P_[1][21]*SF[9] + P_[2][21]*SF[11] + P_[3][21]*SF[10] + P_[10][21]*SF[14] + P_[11][21]*SF[15] + P_[12][21]*SPP[10];
      nextP[1][21] = P_[1][21] + P_[0][21]*SF[8] + P_[2][21]*SF[7] + P_[3][21]*SF[11] - P_[12][21]*SF[15] + P_[11][21]*SPP[10] - (P_[10][21]*q0)/2;
      nextP[2][21] = P_[2][21] + P_[0][21]*SF[6] + P_[1][21]*SF[10] + P_[3][21]*SF[8] + P_[12][21]*SF[14] - P_[10][21]*SPP[10] - (P_[11][21]*q0)/2;
      nextP[3][21] = P_[3][21] + P_[0][21]*SF[7] + P_[1][21]*SF[6] + P_[2][21]*SF[9] + P_[10][21]*SF[15] - P_[11][21]*SF[14] - (P_[12][21]*q0)/2;
      nextP[4][21] = P_[4][21] + P_[0][21]*SF[5] + P_[1][21]*SF[3] - P_[3][21]*SF[4] + P_[2][21]*SPP[0] + P_[13][21]*SPP[3] + P_[14][21]*SPP[6] - P_[15][21]*SPP[9];
      nextP[5][21] = P_[5][21] + P_[0][21]*SF[4] + P_[2][21]*SF[3] + P_[3][21]*SF[5] - P_[1][21]*SPP[0] - P_[13][21]*SPP[8] + P_[14][21]*SPP[2] + P_[15][21]*SPP[5];
      nextP[6][21] = P_[6][21] + P_[1][21]*SF[4] - P_[2][21]*SF[5] + P_[3][21]*SF[3] + P_[0][21]*SPP[0] + P_[13][21]*SPP[4] - P_[14][21]*SPP[7] - P_[15][21]*SPP[1];
      nextP[7][21] = P_[7][21] + P_[4][21]*dt;
      nextP[8][21] = P_[8][21] + P_[5][21]*dt;
      nextP[9][21] = P_[9][21] + P_[6][21]*dt;
      nextP[10][21] = P_[10][21];
      nextP[11][21] = P_[11][21];
      nextP[12][21] = P_[12][21];
      nextP[13][21] = P_[13][21];
      nextP[14][21] = P_[14][21];
      nextP[15][21] = P_[15][21];
      nextP[16][21] = P_[16][21];
      nextP[17][21] = P_[17][21];
      nextP[18][21] = P_[18][21];
      nextP[19][21] = P_[19][21];
      nextP[20][21] = P_[20][21];
      nextP[21][21] = P_[21][21];

      // add process noise that is not from the IMU
      for (unsigned i = 16; i <= 21; i++) {
        nextP[i][i] += process_noise[i];
      }
    }

    // stop position covariance growth if our total position variance reaches 100m
    if ((P_[7][7] + P_[8][8]) > 1.0e4) {
      for (uint8_t i = 7; i <= 8; i++) {
        for (uint8_t j = 0; j < k_num_states_; j++) {
          nextP[i][j] = P_[i][j];
          nextP[j][i] = P_[j][i];
        }
      }
    }

    // covariance matrix is symmetrical, so copy upper half to lower half
    for (unsigned row = 1; row < k_num_states_; row++) {
      for (unsigned column = 0 ; column < row; column++) {
        P_[row][column] = P_[column][row] = nextP[column][row];
      }
    }

    // copy variances (diagonals)
    for (unsigned i = 0; i < k_num_states_; i++) {
      P_[i][i] = nextP[i][i];
    }
    
    fixCovarianceErrors();
  }
  
  void ESKF::controlFusionModes() {
    
    gps_data_ready_ = gps_buffer_.pop_first_older_than(imu_sample_delayed_.time_us, &gps_sample_delayed_);
    vision_data_ready_ = ext_vision_buffer_.pop_first_older_than(imu_sample_delayed_.time_us, &ev_sample_delayed_);
    flow_data_ready_ = opt_flow_buffer_.pop_first_older_than(imu_sample_delayed_.time_us, &opt_flow_sample_delayed_);
    mag_data_ready_ = mag_buffer_.pop_first_older_than(imu_sample_delayed_.time_us, &mag_sample_delayed_);

    R_rng_to_earth_2_2_ = R_to_earth_(2, 0) * sin_tilt_rng_ + R_to_earth_(2, 2) * cos_tilt_rng_;
    range_data_ready_ = range_buffer_.pop_first_older_than(imu_sample_delayed_.time_us, &range_sample_delayed_) && (R_rng_to_earth_2_2_ > range_cos_max_tilt_);
    
    controlHeightSensorTimeouts();

    // For efficiency, fusion of direct state observations for position and velocity is performed sequentially
    // in a single function using sensor data from multiple sources
    controlVelPosFusion();

    controlExternalVisionFusion();
    controlGpsFusion();
    controlOpticalFlowFusion();
    controlMagFusion();
    
    runTerrainEstimator();
  }

  void ESKF::controlOpticalFlowFusion() {
    // Accumulate autopilot gyro data across the same time interval as the flow sensor
    imu_del_ang_of_ += imu_sample_delayed_.delta_ang - state_.gyro_bias;
    delta_time_of_ += imu_sample_delayed_.delta_ang_dt;
    
    if(flow_data_ready_) {
       // we are not yet using flow data
      if ((fusion_mask_ & MASK_OPTICAL_FLOW) && (!opt_flow_)) {
        opt_flow_ = true;
        printf("ESKF commencing optical flow fusion\n");
      }
      
      // Only fuse optical flow if valid body rate compensation data is available
      if (calcOptFlowBodyRateComp()) {
        bool flow_quality_good = (opt_flow_sample_delayed_.quality >= flow_quality_min_);

        if (!flow_quality_good && !in_air_) {
          // when on the ground with poor flow quality, assume zero ground relative velocity and LOS rate
          flow_rad_xy_comp_ = vec2(0,0);
        } else {
          // compensate for body motion to give a LOS rate
          flow_rad_xy_comp_(0) = opt_flow_sample_delayed_.flowRadXY(0) - opt_flow_sample_delayed_.gyroXY(0);
          flow_rad_xy_comp_(1) = opt_flow_sample_delayed_.flowRadXY(1) - opt_flow_sample_delayed_.gyroXY(1);
        }
      } else {
        // don't use this flow data and wait for the next data to arrive
        flow_data_ready_ = false;
      }
    }
    
    // Wait until the midpoint of the flow sample has fallen behind the fusion time horizon
    if (flow_data_ready_ && (imu_sample_delayed_.time_us > opt_flow_sample_delayed_.time_us - uint32_t(1e6f * opt_flow_sample_delayed_.dt) / 2)) {
      // Fuse optical flow LOS rate observations into the main filter only if height above ground has been updated recently but use a relaxed time criteria to enable it to coast through bad range finder data
      if (opt_flow_ && (time_last_imu_ - time_last_hagl_fuse_ < (uint64_t)10e6)) {
        fuseOptFlow();
      }
      flow_data_ready_ = false;
    }
  }
  
  // calculate optical flow body angular rate compensation
  // returns false if bias corrected body rate data is unavailable
  bool ESKF::calcOptFlowBodyRateComp() {
    // reset the accumulators if the time interval is too large
    if (delta_time_of_ > 1.0f) {
      imu_del_ang_of_.setZero();
      delta_time_of_ = 0.0f;
      return false;
    }

    // Use the EKF gyro data if optical flow sensor gyro data is not available for clarification of the sign see definition of flowSample and imuSample in common.h
    opt_flow_sample_delayed_.gyroXY(0) = -imu_del_ang_of_(0);
    opt_flow_sample_delayed_.gyroXY(1) = -imu_del_ang_of_(1);

    // reset the accumulators
    imu_del_ang_of_.setZero();
    delta_time_of_ = 0.0f;
    return true;
}
  
  void ESKF::runTerrainEstimator() {
    // Perform initialisation check
    if (!terrain_initialised_) {
      terrain_initialised_ = initHagl();
    } else {
      // predict the state variance growth where the state is the vertical position of the terrain underneath the vehicle

      // process noise due to errors in vehicle height estimate
      terrain_var_ += sq(imu_sample_delayed_.delta_vel_dt * terrain_p_noise_);

      // process noise due to terrain gradient
      terrain_var_ += sq(imu_sample_delayed_.delta_vel_dt * terrain_gradient_) * (sq(state_.vel(0)) + sq(state_.vel(1)));

      // limit the variance to prevent it becoming badly conditioned
      terrain_var_ = constrain(terrain_var_, 0.0f, 1e4f);

      // Fuse range finder data if available
      if (range_data_ready_) {
        // determine if we should use the hgt observation
        if ((fusion_mask_ & MASK_RANGEFINDER) && (!rng_hgt_)) {
          if (time_last_imu_ - time_last_range_ < 2 * RANGE_MAX_INTERVAL) {
            rng_hgt_ = true;
            printf("ESKF commencing rng fusion\n");
          }
        }

        if(rng_hgt_) {
          fuseHagl();
        }

        // update range sensor angle parameters in case they have changed
        // we do this here to avoid doing those calculations at a high rate
        sin_tilt_rng_ = sinf(rng_sens_pitch_);
        cos_tilt_rng_ = cosf(rng_sens_pitch_);
        fuse_height_ = true;
      }

      //constrain _terrain_vpos to be a minimum of _params.rng_gnd_clearance larger than _state.pos(2)
      if (terrain_vpos_ - state_.pos(2) < rng_gnd_clearance_) {
        terrain_vpos_ = rng_gnd_clearance_ + state_.pos(2);
      }
    }
  }

  void ESKF::fuseHagl() {
    // If the vehicle is excessively tilted, do not try to fuse range finder observations
    if (R_rng_to_earth_2_2_ > range_cos_max_tilt_) {
      // get a height above ground measurement from the range finder assuming a flat earth
      scalar_t meas_hagl = range_sample_delayed_.rng * R_rng_to_earth_2_2_;

      // predict the hagl from the vehicle position and terrain height
      scalar_t pred_hagl = terrain_vpos_ - state_.pos(2);

      // calculate the innovation
      hagl_innov_ = pred_hagl - meas_hagl;

      // calculate the observation variance adding the variance of the vehicles own height uncertainty
      scalar_t obs_variance = fmaxf(P_[9][9], 0.0f) + sq(range_noise_) + sq(range_noise_scaler_ * range_sample_delayed_.rng);

      // calculate the innovation variance - limiting it to prevent a badly conditioned fusion
      hagl_innov_var_ = fmaxf(terrain_var_ + obs_variance, obs_variance);

      // perform an innovation consistency check and only fuse data if it passes
      scalar_t gate_size = fmaxf(range_innov_gate_, 1.0f);
      terr_test_ratio_ = sq(hagl_innov_) / (sq(gate_size) * hagl_innov_var_);

      if (terr_test_ratio_ <= 1.0f) {
        // calculate the Kalman gain
        scalar_t gain = terrain_var_ / hagl_innov_var_;
        // correct the state
        terrain_vpos_ -= gain * hagl_innov_;
        // correct the variance
        terrain_var_ = fmaxf(terrain_var_ * (1.0f - gain), 0.0f);
        // record last successful fusion event
        time_last_hagl_fuse_ = time_last_imu_;
      } else {
        // If we have been rejecting range data for too long, reset to measurement
        if (time_last_imu_ - time_last_hagl_fuse_ > (uint64_t)10E6) {
          terrain_vpos_ = state_.pos(2) + meas_hagl;
          terrain_var_ = obs_variance;
        }
      } 
    }
  }

  void ESKF::fuseOptFlow() {
    scalar_t optflow_test_ratio[2] = {0};

    // get latest estimated orientation
    scalar_t q0 = state_.quat_nominal.w();
    scalar_t q1 = state_.quat_nominal.x();
    scalar_t q2 = state_.quat_nominal.y();
    scalar_t q3 = state_.quat_nominal.z();

    // get latest velocity in earth frame
    scalar_t vn = state_.vel(0);
    scalar_t ve = state_.vel(1);
    scalar_t vd = state_.vel(2);

    // calculate the optical flow observation variance
    // calculate the observation noise variance - scaling noise linearly across flow quality range
    scalar_t R_LOS_best = fmaxf(flow_noise_, 0.05f);
    scalar_t R_LOS_worst = fmaxf(flow_noise_qual_min_, 0.05f);

    // calculate a weighting that varies between 1 when flow quality is best and 0 when flow quality is worst
    scalar_t weighting = (255.0f - (scalar_t)flow_quality_min_);

    if (weighting >= 1.0f) {
      weighting = constrain(((scalar_t)opt_flow_sample_delayed_.quality - (scalar_t)flow_quality_min_) / weighting, 0.0f, 1.0f);
    } else {
      weighting = 0.0f;
    }

    // take the weighted average of the observation noie for the best and wort flow quality
    scalar_t R_LOS = sq(R_LOS_best * weighting + R_LOS_worst * (1.0f - weighting));

    scalar_t H_LOS[2][k_num_states_] = {}; // Optical flow observation Jacobians
    scalar_t Kfusion[k_num_states_][2] = {}; // Optical flow Kalman gains

    // constrain height above ground to be above minimum height when sitting on ground
    scalar_t heightAboveGndEst = max((terrain_vpos_ - state_.pos(2)), rng_gnd_clearance_);
    
    mat3 earth_to_body = R_to_earth_.transpose();
    
    // rotate into body frame
    vec3 vel_body = earth_to_body * state_.vel;

    // calculate range from focal point to centre of image
    scalar_t range = heightAboveGndEst / earth_to_body(2, 2); // absolute distance to the frame region in view

    // calculate optical LOS rates using optical flow rates that have had the body angular rate contribution removed
    // correct for gyro bias errors in the data used to do the motion compensation
    // Note the sign convention used: A positive LOS rate is a RH rotaton of the scene about that axis.
    vec2 opt_flow_rate;
    opt_flow_rate(0) = flow_rad_xy_comp_(0) / opt_flow_sample_delayed_.dt;// + _flow_gyro_bias(0);
    opt_flow_rate(1) = flow_rad_xy_comp_(1) / opt_flow_sample_delayed_.dt;// + _flow_gyro_bias(1);

    if (opt_flow_rate.norm() < flow_max_rate_) {
      flow_innov_[0] =  vel_body(1) / range - opt_flow_rate(0); // flow around the X axis
      flow_innov_[1] = -vel_body(0) / range - opt_flow_rate(1); // flow around the Y axis
    } else {
      return;
    }

    // Fuse X and Y axis measurements sequentially assuming observation errors are uncorrelated
    // Calculate Obser ation Jacobians and Kalman gans for each measurement axis
    for (uint8_t obs_index = 0; obs_index <= 1; obs_index++) {
      if (obs_index == 0) {
	// calculate X axis observation Jacobian
	float t2 = 1.0f / range;
	H_LOS[0][0] = t2*(q1*vd*2.0f+q0*ve*2.0f-q3*vn*2.0f);
	H_LOS[0][1] = t2*(q0*vd*2.0f-q1*ve*2.0f+q2*vn*2.0f);
	H_LOS[0][2] = t2*(q3*vd*2.0f+q2*ve*2.0f+q1*vn*2.0f);
	H_LOS[0][3] = -t2*(q2*vd*-2.0f+q3*ve*2.0f+q0*vn*2.0f);
	H_LOS[0][4] = -t2*(q0*q3*2.0f-q1*q2*2.0f);
	H_LOS[0][5] = t2*(q0*q0-q1*q1+q2*q2-q3*q3);
	H_LOS[0][6] = t2*(q0*q1*2.0f+q2*q3*2.0f);

	// calculate intermediate variables for the X observaton innovatoin variance and Kalman gains
	float t3 = q1*vd*2.0f;
	float t4 = q0*ve*2.0f;
	float t11 = q3*vn*2.0f;
	float t5 = t3+t4-t11;
	float t6 = q0*q3*2.0f;
	float t29 = q1*q2*2.0f;
	float t7 = t6-t29;
	float t8 = q0*q1*2.0f;
	float t9 = q2*q3*2.0f;
	float t10 = t8+t9;
	float t12 = P_[0][0]*t2*t5;
	float t13 = q0*vd*2.0f;
	float t14 = q2*vn*2.0f;
	float t28 = q1*ve*2.0f;
	float t15 = t13+t14-t28;
	float t16 = q3*vd*2.0f;
	float t17 = q2*ve*2.0f;
	float t18 = q1*vn*2.0f;
	float t19 = t16+t17+t18;
	float t20 = q3*ve*2.0f;
	float t21 = q0*vn*2.0f;
	float t30 = q2*vd*2.0f;
	float t22 = t20+t21-t30;
	float t23 = q0*q0;
	float t24 = q1*q1;
	float t25 = q2*q2;
	float t26 = q3*q3;
	float t27 = t23-t24+t25-t26;
	float t31 = P_[1][1]*t2*t15;
	float t32 = P_[6][0]*t2*t10;
	float t33 = P_[1][0]*t2*t15;
	float t34 = P_[2][0]*t2*t19;
	float t35 = P_[5][0]*t2*t27;
	float t79 = P_[4][0]*t2*t7;
	float t80 = P_[3][0]*t2*t22;
	float t36 = t12+t32+t33+t34+t35-t79-t80;
	float t37 = t2*t5*t36;
	float t38 = P_[6][1]*t2*t10;
	float t39 = P_[0][1]*t2*t5;
	float t40 = P_[2][1]*t2*t19;
	float t41 = P_[5][1]*t2*t27;
	float t81 = P_[4][1]*t2*t7;
	float t82 = P_[3][1]*t2*t22;
	float t42 = t31+t38+t39+t40+t41-t81-t82;
	float t43 = t2*t15*t42;
	float t44 = P_[6][2]*t2*t10;
	float t45 = P_[0][2]*t2*t5;
	float t46 = P_[1][2]*t2*t15;
	float t47 = P_[2][2]*t2*t19;
	float t48 = P_[5][2]*t2*t27;
	float t83 = P_[4][2]*t2*t7;
	float t84 = P_[3][2]*t2*t22;
	float t49 = t44+t45+t46+t47+t48-t83-t84;
	float t50 = t2*t19*t49;
	float t51 = P_[6][3]*t2*t10;
	float t52 = P_[0][3]*t2*t5;
	float t53 = P_[1][3]*t2*t15;
	float t54 = P_[2][3]*t2*t19;
	float t55 = P_[5][3]*t2*t27;
	float t85 = P_[4][3]*t2*t7;
	float t86 = P_[3][3]*t2*t22;
	float t56 = t51+t52+t53+t54+t55-t85-t86;
	float t57 = P_[6][5]*t2*t10;
	float t58 = P_[0][5]*t2*t5;
	float t59 = P_[1][5]*t2*t15;
	float t60 = P_[2][5]*t2*t19;
	float t61 = P_[5][5]*t2*t27;
	float t88 = P_[4][5]*t2*t7;
	float t89 = P_[3][5]*t2*t22;
	float t62 = t57+t58+t59+t60+t61-t88-t89;
	float t63 = t2*t27*t62;
	float t64 = P_[6][4]*t2*t10;
	float t65 = P_[0][4]*t2*t5;
	float t66 = P_[1][4]*t2*t15;
	float t67 = P_[2][4]*t2*t19;
	float t68 = P_[5][4]*t2*t27;
	float t90 = P_[4][4]*t2*t7;
	float t91 = P_[3][4]*t2*t22;
	float t69 = t64+t65+t66+t67+t68-t90-t91;
	float t70 = P_[6][6]*t2*t10;
	float t71 = P_[0][6]*t2*t5;
	float t72 = P_[1][6]*t2*t15;
	float t73 = P_[2][6]*t2*t19;
	float t74 = P_[5][6]*t2*t27;
	float t93 = P_[4][6]*t2*t7;
	float t94 = P_[3][6]*t2*t22;
	float t75 = t70+t71+t72+t73+t74-t93-t94;
	float t76 = t2*t10*t75;
	float t87 = t2*t22*t56;
	float t92 = t2*t7*t69;
	float t77 = R_LOS+t37+t43+t50+t63+t76-t87-t92;
	float t78;

	// calculate innovation variance for X axis observation and protect against a badly conditioned calculation
	if (t77 >= R_LOS) {
	  t78 = 1.0f / t77;
	  flow_innov_var_[0] = t77;
	} else {
	  // we need to reinitialise the covariance matrix and abort this fusion step
	  initialiseCovariance();
	  return;
	}

	// calculate Kalman gains for X-axis observation
	Kfusion[0][0] = t78*(t12-P_[0][4]*t2*t7+P_[0][1]*t2*t15+P_[0][6]*t2*t10+P_[0][2]*t2*t19-P_[0][3]*t2*t22+P_[0][5]*t2*t27);
	Kfusion[1][0] = t78*(t31+P_[1][0]*t2*t5-P_[1][4]*t2*t7+P_[1][6]*t2*t10+P_[1][2]*t2*t19-P_[1][3]*t2*t22+P_[1][5]*t2*t27);
	Kfusion[2][0] = t78*(t47+P_[2][0]*t2*t5-P_[2][4]*t2*t7+P_[2][1]*t2*t15+P_[2][6]*t2*t10-P_[2][3]*t2*t22+P_[2][5]*t2*t27);
	Kfusion[3][0] = t78*(-t86+P_[3][0]*t2*t5-P_[3][4]*t2*t7+P_[3][1]*t2*t15+P_[3][6]*t2*t10+P_[3][2]*t2*t19+P_[3][5]*t2*t27);
	Kfusion[4][0] = t78*(-t90+P_[4][0]*t2*t5+P_[4][1]*t2*t15+P_[4][6]*t2*t10+P_[4][2]*t2*t19-P_[4][3]*t2*t22+P_[4][5]*t2*t27);
	Kfusion[5][0] = t78*(t61+P_[5][0]*t2*t5-P_[5][4]*t2*t7+P_[5][1]*t2*t15+P_[5][6]*t2*t10+P_[5][2]*t2*t19-P_[5][3]*t2*t22);
	Kfusion[6][0] = t78*(t70+P_[6][0]*t2*t5-P_[6][4]*t2*t7+P_[6][1]*t2*t15+P_[6][2]*t2*t19-P_[6][3]*t2*t22+P_[6][5]*t2*t27);
	Kfusion[7][0] = t78*(P_[7][0]*t2*t5-P_[7][4]*t2*t7+P_[7][1]*t2*t15+P_[7][6]*t2*t10+P_[7][2]*t2*t19-P_[7][3]*t2*t22+P_[7][5]*t2*t27);
	Kfusion[8][0] = t78*(P_[8][0]*t2*t5-P_[8][4]*t2*t7+P_[8][1]*t2*t15+P_[8][6]*t2*t10+P_[8][2]*t2*t19-P_[8][3]*t2*t22+P_[8][5]*t2*t27);
	Kfusion[9][0] = t78*(P_[9][0]*t2*t5-P_[9][4]*t2*t7+P_[9][1]*t2*t15+P_[9][6]*t2*t10+P_[9][2]*t2*t19-P_[9][3]*t2*t22+P_[9][5]*t2*t27);
	Kfusion[10][0] = t78*(P_[10][0]*t2*t5-P_[10][4]*t2*t7+P_[10][1]*t2*t15+P_[10][6]*t2*t10+P_[10][2]*t2*t19-P_[10][3]*t2*t22+P_[10][5]*t2*t27);
	Kfusion[11][0] = t78*(P_[11][0]*t2*t5-P_[11][4]*t2*t7+P_[11][1]*t2*t15+P_[11][6]*t2*t10+P_[11][2]*t2*t19-P_[11][3]*t2*t22+P_[11][5]*t2*t27);
	Kfusion[12][0] = t78*(P_[12][0]*t2*t5-P_[12][4]*t2*t7+P_[12][1]*t2*t15+P_[12][6]*t2*t10+P_[12][2]*t2*t19-P_[12][3]*t2*t22+P_[12][5]*t2*t27);
	Kfusion[13][0] = t78*(P_[13][0]*t2*t5-P_[13][4]*t2*t7+P_[13][1]*t2*t15+P_[13][6]*t2*t10+P_[13][2]*t2*t19-P_[13][3]*t2*t22+P_[13][5]*t2*t27);
	Kfusion[14][0] = t78*(P_[14][0]*t2*t5-P_[14][4]*t2*t7+P_[14][1]*t2*t15+P_[14][6]*t2*t10+P_[14][2]*t2*t19-P_[14][3]*t2*t22+P_[14][5]*t2*t27);
	Kfusion[15][0] = t78*(P_[15][0]*t2*t5-P_[15][4]*t2*t7+P_[15][1]*t2*t15+P_[15][6]*t2*t10+P_[15][2]*t2*t19-P_[15][3]*t2*t22+P_[15][5]*t2*t27);

	// run innovation consistency checks
	optflow_test_ratio[0] = sq(flow_innov_[0]) / (sq(max(flow_innov_gate_, 1.0f)) * flow_innov_var_[0]);

      } else if (obs_index == 1) {

	// calculate Y axis observation Jacobian
	float t2 = 1.0f / range;
	H_LOS[1][0] = -t2*(q2*vd*-2.0f+q3*ve*2.0f+q0*vn*2.0f);
	H_LOS[1][1] = -t2*(q3*vd*2.0f+q2*ve*2.0f+q1*vn*2.0f);
	H_LOS[1][2] = t2*(q0*vd*2.0f-q1*ve*2.0f+q2*vn*2.0f);
	H_LOS[1][3] = -t2*(q1*vd*2.0f+q0*ve*2.0f-q3*vn*2.0f);
	H_LOS[1][4] = -t2*(q0*q0+q1*q1-q2*q2-q3*q3);
	H_LOS[1][5] = -t2*(q0*q3*2.0f+q1*q2*2.0f);
	H_LOS[1][6] = t2*(q0*q2*2.0f-q1*q3*2.0f);

	// calculate intermediate variables for the Y observaton innovatoin variance and Kalman gains
	float t3 = q3*ve*2.0f;
	float t4 = q0*vn*2.0f;
	float t11 = q2*vd*2.0f;
	float t5 = t3+t4-t11;
	float t6 = q0*q3*2.0f;
	float t7 = q1*q2*2.0f;
	float t8 = t6+t7;
	float t9 = q0*q2*2.0f;
	float t28 = q1*q3*2.0f;
	float t10 = t9-t28;
	float t12 = P_[0][0]*t2*t5;
	float t13 = q3*vd*2.0f;
	float t14 = q2*ve*2.0f;
	float t15 = q1*vn*2.0f;
	float t16 = t13+t14+t15;
	float t17 = q0*vd*2.0f;
	float t18 = q2*vn*2.0f;
	float t29 = q1*ve*2.0f;
	float t19 = t17+t18-t29;
	float t20 = q1*vd*2.0f;
	float t21 = q0*ve*2.0f;
	float t30 = q3*vn*2.0f;
	float t22 = t20+t21-t30;
	float t23 = q0*q0;
	float t24 = q1*q1;
	float t25 = q2*q2;
	float t26 = q3*q3;
	float t27 = t23+t24-t25-t26;
	float t31 = P_[1][1]*t2*t16;
	float t32 = P_[5][0]*t2*t8;
	float t33 = P_[1][0]*t2*t16;
	float t34 = P_[3][0]*t2*t22;
	float t35 = P_[4][0]*t2*t27;
	float t80 = P_[6][0]*t2*t10;
	float t81 = P_[2][0]*t2*t19;
	float t36 = t12+t32+t33+t34+t35-t80-t81;
	float t37 = t2*t5*t36;
	float t38 = P_[5][1]*t2*t8;
	float t39 = P_[0][1]*t2*t5;
	float t40 = P_[3][1]*t2*t22;
	float t41 = P_[4][1]*t2*t27;
	float t82 = P_[6][1]*t2*t10;
	float t83 = P_[2][1]*t2*t19;
	float t42 = t31+t38+t39+t40+t41-t82-t83;
	float t43 = t2*t16*t42;
	float t44 = P_[5][2]*t2*t8;
	float t45 = P_[0][2]*t2*t5;
	float t46 = P_[1][2]*t2*t16;
	float t47 = P_[3][2]*t2*t22;
	float t48 = P_[4][2]*t2*t27;
	float t79 = P_[2][2]*t2*t19;
	float t84 = P_[6][2]*t2*t10;
	float t49 = t44+t45+t46+t47+t48-t79-t84;
	float t50 = P_[5][3]*t2*t8;
	float t51 = P_[0][3]*t2*t5;
	float t52 = P_[1][3]*t2*t16;
	float t53 = P_[3][3]*t2*t22;
	float t54 = P_[4][3]*t2*t27;
	float t86 = P_[6][3]*t2*t10;
	float t87 = P_[2][3]*t2*t19;
	float t55 = t50+t51+t52+t53+t54-t86-t87;
	float t56 = t2*t22*t55;
	float t57 = P_[5][4]*t2*t8;
	float t58 = P_[0][4]*t2*t5;
	float t59 = P_[1][4]*t2*t16;
	float t60 = P_[3][4]*t2*t22;
	float t61 = P_[4][4]*t2*t27;
	float t88 = P_[6][4]*t2*t10;
	float t89 = P_[2][4]*t2*t19;
	float t62 = t57+t58+t59+t60+t61-t88-t89;
	float t63 = t2*t27*t62;
	float t64 = P_[5][5]*t2*t8;
	float t65 = P_[0][5]*t2*t5;
	float t66 = P_[1][5]*t2*t16;
	float t67 = P_[3][5]*t2*t22;
	float t68 = P_[4][5]*t2*t27;
	float t90 = P_[6][5]*t2*t10;
	float t91 = P_[2][5]*t2*t19;
	float t69 = t64+t65+t66+t67+t68-t90-t91;
	float t70 = t2*t8*t69;
	float t71 = P_[5][6]*t2*t8;
	float t72 = P_[0][6]*t2*t5;
	float t73 = P_[1][6]*t2*t16;
	float t74 = P_[3][6]*t2*t22;
	float t75 = P_[4][6]*t2*t27;
	float t92 = P_[6][6]*t2*t10;
	float t93 = P_[2][6]*t2*t19;
	float t76 = t71+t72+t73+t74+t75-t92-t93;
	float t85 = t2*t19*t49;
	float t94 = t2*t10*t76;
	float t77 = R_LOS+t37+t43+t56+t63+t70-t85-t94;
	float t78;
	
	// calculate innovation variance for Y axis observation and protect against a badly conditioned calculation
	if (t77 >= R_LOS) {
	  t78 = 1.0f / t77;
	  flow_innov_var_[1] = t77;
	} else {
	  // we need to reinitialise the covariance matrix and abort this fusion step
	  initialiseCovariance();
	  return;
	}

	// calculate Kalman gains for Y-axis observation
	Kfusion[0][1] = -t78*(t12+P_[0][5]*t2*t8-P_[0][6]*t2*t10+P_[0][1]*t2*t16-P_[0][2]*t2*t19+P_[0][3]*t2*t22+P_[0][4]*t2*t27);
	Kfusion[1][1] = -t78*(t31+P_[1][0]*t2*t5+P_[1][5]*t2*t8-P_[1][6]*t2*t10-P_[1][2]*t2*t19+P_[1][3]*t2*t22+P_[1][4]*t2*t27);
	Kfusion[2][1] = -t78*(-t79+P_[2][0]*t2*t5+P_[2][5]*t2*t8-P_[2][6]*t2*t10+P_[2][1]*t2*t16+P_[2][3]*t2*t22+P_[2][4]*t2*t27);
	Kfusion[3][1] = -t78*(t53+P_[3][0]*t2*t5+P_[3][5]*t2*t8-P_[3][6]*t2*t10+P_[3][1]*t2*t16-P_[3][2]*t2*t19+P_[3][4]*t2*t27);
	Kfusion[4][1] = -t78*(t61+P_[4][0]*t2*t5+P_[4][5]*t2*t8-P_[4][6]*t2*t10+P_[4][1]*t2*t16-P_[4][2]*t2*t19+P_[4][3]*t2*t22);
	Kfusion[5][1] = -t78*(t64+P_[5][0]*t2*t5-P_[5][6]*t2*t10+P_[5][1]*t2*t16-P_[5][2]*t2*t19+P_[5][3]*t2*t22+P_[5][4]*t2*t27);
	Kfusion[6][1] = -t78*(-t92+P_[6][0]*t2*t5+P_[6][5]*t2*t8+P_[6][1]*t2*t16-P_[6][2]*t2*t19+P_[6][3]*t2*t22+P_[6][4]*t2*t27);
	Kfusion[7][1] = -t78*(P_[7][0]*t2*t5+P_[7][5]*t2*t8-P_[7][6]*t2*t10+P_[7][1]*t2*t16-P_[7][2]*t2*t19+P_[7][3]*t2*t22+P_[7][4]*t2*t27);
	Kfusion[8][1] = -t78*(P_[8][0]*t2*t5+P_[8][5]*t2*t8-P_[8][6]*t2*t10+P_[8][1]*t2*t16-P_[8][2]*t2*t19+P_[8][3]*t2*t22+P_[8][4]*t2*t27);
	Kfusion[9][1] = -t78*(P_[9][0]*t2*t5+P_[9][5]*t2*t8-P_[9][6]*t2*t10+P_[9][1]*t2*t16-P_[9][2]*t2*t19+P_[9][3]*t2*t22+P_[9][4]*t2*t27);
	Kfusion[10][1] = -t78*(P_[10][0]*t2*t5+P_[10][5]*t2*t8-P_[10][6]*t2*t10+P_[10][1]*t2*t16-P_[10][2]*t2*t19+P_[10][3]*t2*t22+P_[10][4]*t2*t27);
	Kfusion[11][1] = -t78*(P_[11][0]*t2*t5+P_[11][5]*t2*t8-P_[11][6]*t2*t10+P_[11][1]*t2*t16-P_[11][2]*t2*t19+P_[11][3]*t2*t22+P_[11][4]*t2*t27);
	Kfusion[12][1] = -t78*(P_[12][0]*t2*t5+P_[12][5]*t2*t8-P_[12][6]*t2*t10+P_[12][1]*t2*t16-P_[12][2]*t2*t19+P_[12][3]*t2*t22+P_[12][4]*t2*t27);
	Kfusion[13][1] = -t78*(P_[13][0]*t2*t5+P_[13][5]*t2*t8-P_[13][6]*t2*t10+P_[13][1]*t2*t16-P_[13][2]*t2*t19+P_[13][3]*t2*t22+P_[13][4]*t2*t27);
	Kfusion[14][1] = -t78*(P_[14][0]*t2*t5+P_[14][5]*t2*t8-P_[14][6]*t2*t10+P_[14][1]*t2*t16-P_[14][2]*t2*t19+P_[14][3]*t2*t22+P_[14][4]*t2*t27);
	Kfusion[15][1] = -t78*(P_[15][0]*t2*t5+P_[15][5]*t2*t8-P_[15][6]*t2*t10+P_[15][1]*t2*t16-P_[15][2]*t2*t19+P_[15][3]*t2*t22+P_[15][4]*t2*t27);

	// run innovation consistency check
	optflow_test_ratio[1] = sq(flow_innov_[1]) / (sq(max(flow_innov_gate_, 1.0f)) * flow_innov_var_[1]);
      }
    }

    // record the innovation test pass/fail
    bool flow_fail = false;

    for (uint8_t obs_index = 0; obs_index <= 1; obs_index++) {
      if (optflow_test_ratio[obs_index] > 1.0f) {
       flow_fail = true;
      }
    }

    // if either axis fails we abort the fusion
    if (flow_fail) {
      return;
    }

    for (uint8_t obs_index = 0; obs_index <= 1; obs_index++) {

      // copy the Kalman gain vector for the axis we are fusing
      float gain[k_num_states_];

      for (unsigned row = 0; row <= k_num_states_ - 1; row++) {
	gain[row] = Kfusion[row][obs_index];
      }

      // apply covariance correction via P_new = (I -K*H)*P_
      // first calculate expression for KHP
      // then calculate P_ - KHP
      float KHP[k_num_states_][k_num_states_];
      float KH[7];

      for (unsigned row = 0; row < k_num_states_; row++) {
	KH[0] = gain[row] * H_LOS[obs_index][0];
	KH[1] = gain[row] * H_LOS[obs_index][1];
	KH[2] = gain[row] * H_LOS[obs_index][2];
	KH[3] = gain[row] * H_LOS[obs_index][3];
	KH[4] = gain[row] * H_LOS[obs_index][4];
	KH[5] = gain[row] * H_LOS[obs_index][5];
	KH[6] = gain[row] * H_LOS[obs_index][6];

	for (unsigned column = 0; column < k_num_states_; column++) {
	  float tmp = KH[0] * P_[0][column];
	  tmp += KH[1] * P_[1][column];
	  tmp += KH[2] * P_[2][column];
	  tmp += KH[3] * P_[3][column];
	  tmp += KH[4] * P_[4][column];
	  tmp += KH[5] * P_[5][column];
	  tmp += KH[6] * P_[6][column];
	  KHP[row][column] = tmp;
	}
      }

      // if the covariance correction will result in a negative variance, then
      // the covariance marix is unhealthy and must be corrected
      bool healthy = true;

      for (int i = 0; i < k_num_states_; i++) {
	if (P_[i][i] < KHP[i][i]) {
	  // zero rows and columns
	  zeroRows(P_, i, i);
	  zeroCols(P_, i, i);

	  //flag as unhealthy
	  healthy = false;
	}
      }

      // only apply covariance and state corrrections if healthy
      if (healthy) {
	// apply the covariance corrections
	for (unsigned row = 0; row < k_num_states_; row++) {
	  for (unsigned column = 0; column < k_num_states_; column++) {
	    P_[row][column] = P_[row][column] - KHP[row][column];
	  }
	}

	// correct the covariance marix for gross errors
	fixCovarianceErrors();

	// apply the state corrections
	fuse(gain, flow_innov_[obs_index]);
      }
    }
  }

  void ESKF::controlMagFusion() {
    if(fusion_mask_ & MASK_MAG_INHIBIT) {
      mag_use_inhibit_ = true;
      fuseHeading();
    } else if(mag_data_ready_) {
      // determine if we should use the yaw observation
      if ((fusion_mask_ & MASK_MAG_HEADING) && (!mag_hdg_)) {
        if (time_last_imu_ - time_last_mag_ < 2 * MAG_INTERVAL) {
          mag_hdg_ = true;
          printf("ESKF commencing mag yaw fusion\n");
        }
      }

      if(mag_hdg_) {
        fuseHeading();
      }
    }
  }

  void ESKF::controlExternalVisionFusion() {
    if(vision_data_ready_) {
      // Fuse available NED position data into the main filter
      if ((fusion_mask_ & MASK_EV_POS) && (!ev_pos_)) {
        // check for an external vision measurement that has fallen behind the fusion time horizon
        if (time_last_imu_ - time_last_ext_vision_ < 2 * EV_MAX_INTERVAL) {
          ev_pos_ = true;
          printf("ESKF commencing external vision position fusion\n");
        }
        // reset the position if we are not already aiding using GPS, else use a relative position method for fusing the position data
        if (gps_pos_) {
	  //
        } else {
          resetPosition();
          resetVelocity();
        }
      }

      // determine if we should use the yaw observation
      if ((fusion_mask_ & MASK_EV_YAW) && (!ev_yaw_)) {
        if (time_last_imu_ - time_last_ext_vision_ < 2 * EV_MAX_INTERVAL) {
          // reset the yaw angle to the value from the observaton quaternion
          vec3 euler_init = dcm2vec(quat2dcm(state_.quat_nominal));

          // get initial yaw from the observation quaternion
          const extVisionSample &ev_newest = ext_vision_buffer_.get_newest();
          vec3 euler_obs = dcm2vec(quat2dcm(ev_newest.quatNED));
          euler_init(2) = euler_obs(2);

          // calculate initial quaternion states for the ekf
          state_.quat_nominal = from_axis_angle(euler_init);

          ev_yaw_ = true;
          printf("ESKF commencing external vision yaw fusion\n");
        }
      }
      
      // determine if we should use the hgt observation
      if ((fusion_mask_ & MASK_EV_HGT) && (!ev_hgt_)) {
        // don't start using EV data unless data is arriving frequently
        if (time_last_imu_ - time_last_ext_vision_ < 2 * EV_MAX_INTERVAL) {
          ev_hgt_ = true;
          printf("ESKF commencing external vision hgt fusion\n");
          if(rng_hgt_) {
            //
          } else {
            resetHeight();
          }
        }
      }
      
      if (ev_hgt_) {
        fuse_height_ = true;
      }
      
      if (ev_pos_) {
        fuse_pos_ = true;
      }
      
      if(fuse_height_ || fuse_pos_) {
        fuseVelPosHeight();
        fuse_pos_ = fuse_height_ = false;
      }

      if (ev_yaw_) {
        fuseHeading();
      }
    }
  }

  void ESKF::controlGpsFusion() {
    if (gps_data_ready_) {
      if ((fusion_mask_ & MASK_GPS_POS) && (!gps_pos_)) {
        gps_pos_ = true;
        printf("ESKF commencing GPS pos fusion\n");
      }
      if(gps_pos_) {
        fuse_pos_ = true;
        fuse_vert_vel_ = true;
        fuse_hor_vel_ = true;
        time_last_gps_ = time_last_imu_;
      }
      if ((fusion_mask_ & MASK_GPS_HGT) && (!gps_hgt_)) {
        gps_hgt_ = true;
        printf("ESKF commencing GPS hgt fusion\n");
      }
      if(gps_pos_) {
        fuse_height_ = true;
        time_last_gps_ = time_last_imu_;
      }
    }
  }

  void ESKF::controlVelPosFusion() {
    if (!gps_pos_ && !opt_flow_ && !ev_pos_) {
      // Fuse synthetic position observations every 200msec
      if ((time_last_imu_ - time_last_fake_gps_ > (uint64_t)2e5) || fuse_height_) {
        // Reset position and velocity states if we re-commence this aiding method
        if ((time_last_imu_ - time_last_fake_gps_) > (uint64_t)4e5) {
          resetPosition();
          resetVelocity();

          if (time_last_fake_gps_ != 0) {
            printf("ESKF stopping navigation\n");
          }
        }

        fuse_pos_ = true;
        fuse_hor_vel_ = false;
        fuse_vert_vel_ = false;
        time_last_fake_gps_ = time_last_imu_;

        vel_pos_innov_[0] = 0.0f;
        vel_pos_innov_[1] = 0.0f;
        vel_pos_innov_[2] = 0.0f;
        vel_pos_innov_[3] = state_.pos(0) - last_known_posNED_(0);
        vel_pos_innov_[4] = state_.pos(1) - last_known_posNED_(1);

        // glitch protection is not required so set gate to a large value
        posInnovGateNE_ = 100.0f;
      }
    }
    
    // Fuse available NED velocity and position data into the main filter
    if (fuse_height_ || fuse_pos_ || fuse_hor_vel_ || fuse_vert_vel_) {
      fuseVelPosHeight();
    }
  }

  void ESKF::controlHeightSensorTimeouts() {
    /*
     * Handle the case where we have not fused height measurements recently and
     * uncertainty exceeds the max allowable. Reset using the best available height
     * measurement source, continue using it after the reset and declare the current
     * source failed if we have switched.
    */

    // check if height has been inertial deadreckoning for too long
    bool hgt_fusion_timeout = ((time_last_imu_ - time_last_hgt_fuse_) > 5e6);

    // reset the vertical position and velocity states
    if ((P_[9][9] > sq(hgt_reset_lim)) && (hgt_fusion_timeout)) {
      // boolean that indicates we will do a height reset
      bool reset_height = false;

      // handle the case where we are using external vision data for height
      if (ev_hgt_) {
        // check if vision data is available
        extVisionSample ev_init = ext_vision_buffer_.get_newest();
        bool ev_data_available = ((time_last_imu_ - ev_init.time_us) < 2 * EV_MAX_INTERVAL);

        // reset to ev data if it is available
        bool reset_to_ev = ev_data_available;

        if (reset_to_ev) {
          // request a reset
          reset_height = true;
        } else {
          // we have nothing to reset to
          reset_height = false;
        }
      } else if (gps_hgt_) {
        // check if gps data is available
        gpsSample gps_init = gps_buffer_.get_newest();
        bool gps_data_available = ((time_last_imu_ - gps_init.time_us) < 2 * GPS_MAX_INTERVAL);

        // reset to ev data if it is available
        bool reset_to_gps = gps_data_available;

        if (reset_to_gps) {
          // request a reset
          reset_height = true;
        } else {
          // we have nothing to reset to
          reset_height = false;
        }
      }
      
      // Reset vertical position and velocity states to the last measurement
      if (reset_height) {
        resetHeight();
        // Reset the timout timer
        time_last_hgt_fuse_ = time_last_imu_;
      }
    }
  }

  void ESKF::resetPosition() {
    if (gps_pos_) {
      // this reset is only called if we have new gps data at the fusion time horizon
      state_.pos(0) = gps_sample_delayed_.pos(0);
      state_.pos(1) = gps_sample_delayed_.pos(1);

      // use GPS accuracy to reset variances
      setDiag(P_, 7, 8, sq(gps_sample_delayed_.hacc));

    } else if (ev_pos_) {
      // this reset is only called if we have new ev data at the fusion time horizon
      state_.pos(0) = ev_sample_delayed_.posNED(0);
      state_.pos(1) = ev_sample_delayed_.posNED(1);

      // use EV accuracy to reset variances
      setDiag(P_, 7, 8, sq(ev_sample_delayed_.posErr));

    } else if (opt_flow_) {
      if (!in_air_) {
        // we are likely starting OF for the first time so reset the horizontal position
        state_.pos(0) = 0.0f;
        state_.pos(1) = 0.0f;
      } else {
        // set to the last known position
        state_.pos(0) = last_known_posNED_(0);
        state_.pos(1) = last_known_posNED_(1);
      }
      // estimate is relative to initial positon in this mode, so we start with zero error.
      zeroCols(P_, 7, 8);
      zeroRows(P_, 7, 8);
    } else {
      // Used when falling back to non-aiding mode of operation
      state_.pos(0) = last_known_posNED_(0);
      state_.pos(1) = last_known_posNED_(1);
      setDiag(P_, 7, 8, sq(pos_noaid_noise_));
    }
  }

  void ESKF::resetVelocity() {
    
  }

  void ESKF::resetHeight() {
    // reset the vertical position
    if (ev_hgt_) {
      // initialize vertical position with newest measurement
      extVisionSample ev_newest = ext_vision_buffer_.get_newest();

      // use the most recent data if it's time offset from the fusion time horizon is smaller
      int32_t dt_newest = ev_newest.time_us - imu_sample_delayed_.time_us;
      int32_t dt_delayed = ev_sample_delayed_.time_us - imu_sample_delayed_.time_us;

      if (std::abs(dt_newest) < std::abs(dt_delayed)) {
        state_.pos(2) = ev_newest.posNED(2);
      } else {
        state_.pos(2) = ev_sample_delayed_.posNED(2);
      }
    } else if(gps_hgt_) {
      // Get the most recent GPS data
      const gpsSample &gps_newest = gps_buffer_.get_newest();
      if (time_last_imu_ - gps_newest.time_us < 2 * GPS_MAX_INTERVAL) {
        state_.pos(2) = gps_newest.hgt;

        // reset the associated covarince values
        zeroRows(P_, 9, 9);
        zeroCols(P_, 9, 9);

        // the state variance is the same as the observation
        P_[9][9] = sq(gps_newest.hacc);
      }
    }

    // reset the vertical velocity covariance values
    zeroRows(P_, 6, 6);
    zeroCols(P_, 6, 6);

    // we don't know what the vertical velocity is, so set it to zero
    state_.vel(2) = 0.0f;

    // Set the variance to a value large enough to allow the state to converge quickly
    // that does not destabilise the filter
    P_[6][6] = 10.0f;
  }
    
  void ESKF::fuseVelPosHeight() {
    bool fuse_map[6] = {}; // map of booleans true when [VN,VE,VD,PN,PE,PD] observations are available
    bool innov_check_pass_map[6] = {}; // true when innovations consistency checks pass for [PN,PE,PD] observations
    scalar_t R[6] = {}; // observation variances for [VN,VE,VD,PN,PE,PD]
    scalar_t gate_size[6] = {}; // innovation consistency check gate sizes for [VN,VE,VD,PN,PE,PD] observations
    scalar_t Kfusion[k_num_states_] = {}; // Kalman gain vector for any single observation - sequential fusion is used
    
    // calculate innovations, innovations gate sizes and observation variances
    if (fuse_hor_vel_) {
      // enable fusion for NE velocity axes
      fuse_map[0] = fuse_map[1] = true;
      velObsVarNE_(1) = velObsVarNE_(0) = sq(fmaxf(gps_sample_delayed_.sacc, gps_vel_noise_));
      
      // Set observation noise variance and innovation consistency check gate size for the NE position observations
      R[0] = velObsVarNE_(0);
      R[1] = velObsVarNE_(1);
      
      hvelInnovGate_ = fmaxf(vel_innov_gate_, 1.0f);

      gate_size[1] = gate_size[0] = hvelInnovGate_;
    }

    if (fuse_vert_vel_) {
      fuse_map[2] = true;
      // observation variance - use receiver reported accuracy with parameter setting the minimum value
      R[2] = fmaxf(gps_vel_noise_, 0.01f);
      // use scaled horizontal speed accuracy assuming typical ratio of VDOP/HDOP
      R[2] = 1.5f * fmaxf(R[2], gps_sample_delayed_.sacc);
      R[2] = R[2] * R[2];
      // innovation gate size
      gate_size[2] = fmaxf(vel_innov_gate_, 1.0f);
    }
    
    if (fuse_pos_) {
      fuse_map[3] = fuse_map[4] = true;
      
      if(gps_pos_) {
        // calculate observation process noise
        scalar_t lower_limit = fmaxf(gps_pos_noise_, 0.01f);
        scalar_t upper_limit = fmaxf(pos_noaid_noise_, lower_limit);
        posObsNoiseNE_ = constrain(gps_sample_delayed_.hacc, lower_limit, upper_limit);
        velObsVarNE_(1) = velObsVarNE_(0) = sq(fmaxf(gps_sample_delayed_.sacc, gps_vel_noise_));

        // calculate innovations
        vel_pos_innov_[0] = state_.vel(0) - gps_sample_delayed_.vel(0);
        vel_pos_innov_[1] = state_.vel(1) - gps_sample_delayed_.vel(1);
        vel_pos_innov_[2] = state_.vel(2) - gps_sample_delayed_.vel(2);
        vel_pos_innov_[3] = state_.pos(0) - gps_sample_delayed_.pos(0);
        vel_pos_innov_[4] = state_.pos(1) - gps_sample_delayed_.pos(1);

        // observation 1-STD error
        R[3] = sq(posObsNoiseNE_);

        // set innovation gate size
        gate_size[3] = fmaxf(5.0, 1.0f);

      } else if (ev_pos_) {
        // calculate innovations
        // use the absolute position
        vel_pos_innov_[3] = state_.pos(0) - ev_sample_delayed_.posNED(0);
        vel_pos_innov_[4] = state_.pos(1) - ev_sample_delayed_.posNED(1);

        // observation 1-STD error
        R[3] = fmaxf(0.05f, 0.01f);

        // innovation gate size
        gate_size[3] = fmaxf(5.0f, 1.0f);

      } else {
        // No observations - use a static position to constrain drift
        if (in_air_) {
          R[3] = fmaxf(10.0f, 0.5f);
        } else {
          R[3] = 0.5f;
        }
        vel_pos_innov_[3] = state_.pos(0) - last_known_posNED_(0);
        vel_pos_innov_[4] = state_.pos(1) - last_known_posNED_(1);

        // glitch protection is not required so set gate to a large value
        gate_size[3] = 100.0f;

        vel_pos_innov_[5] = state_.pos(2) - last_known_posNED_(2);
        fuse_map[5] = true;
        R[5] = 0.5f;
        R[5] = R[5] * R[5];
        // innovation gate size
        gate_size[5] = 100.0f;
      }

      // convert North position noise to variance
      R[3] = R[3] * R[3];

      // copy North axis values to East axis
      R[4] = R[3];
      gate_size[4] = gate_size[3];
    }

    if (fuse_height_) {
      if(ev_hgt_) {
        fuse_map[5] = true;
        // calculate the innovation assuming the external vision observaton is in local NED frame
        vel_pos_innov_[5] = state_.pos(2) - ev_sample_delayed_.posNED(2);
        // observation variance - defined externally
        R[5] = fmaxf(0.05f, 0.01f);
        R[5] = R[5] * R[5];
        // innovation gate size
        gate_size[5] = fmaxf(5.0f, 1.0f);
      } else if(gps_hgt_) {
        // vertical position innovation - gps measurement has opposite sign to earth z axis
        vel_pos_innov_[5] = state_.pos(2) - gps_sample_delayed_.hgt;
        // observation variance - receiver defined and parameter limited
        // use scaled horizontal position accuracy assuming typical ratio of VDOP/HDOP
        scalar_t lower_limit = fmaxf(gps_pos_noise_, 0.01f);
        scalar_t upper_limit = fmaxf(pos_noaid_noise_, lower_limit);
        R[5] = 1.5f * constrain(gps_sample_delayed_.vacc, lower_limit, upper_limit);
        R[5] = R[5] * R[5];
        // innovation gate size
        gate_size[5] = fmaxf(5.0, 1.0f);
      } else if ((rng_hgt_) && (R_rng_to_earth_2_2_ > range_cos_max_tilt_)) {
        fuse_map[5] = true;
        // use range finder with tilt correction
        vel_pos_innov_[5] = state_.pos(2) - (-max(range_sample_delayed_.rng * R_rng_to_earth_2_2_, rng_gnd_clearance_)) - 0.1f;
        // observation variance - user parameter defined
        R[5] = fmaxf((sq(range_noise_) + sq(range_noise_scaler_ * range_sample_delayed_.rng)) * sq(R_rng_to_earth_2_2_), 0.01f);
        // innovation gate size
        gate_size[5] = fmaxf(range_innov_gate_, 1.0f);
      }
    }

    // calculate innovation test ratios
    for (unsigned obs_index = 0; obs_index < 6; obs_index++) {
      if (fuse_map[obs_index]) {
        // compute the innovation variance SK = HPH + R
        unsigned state_index = obs_index + 4;	// we start with vx and this is the 4. state
        vel_pos_innov_var_[obs_index] = P_[state_index][state_index] + R[obs_index];
        // Compute the ratio of innovation to gate size
        vel_pos_test_ratio_[obs_index] = sq(vel_pos_innov_[obs_index]) / (sq(gate_size[obs_index]) * vel_pos_innov_var_[obs_index]);
      }
    }

    // check position, velocity and height innovations
    // treat 2D position and height as separate sensors
    bool pos_check_pass = ((vel_pos_test_ratio_[3] <= 1.0f) && (vel_pos_test_ratio_[4] <= 1.0f));
    innov_check_pass_map[3] = innov_check_pass_map[4] = pos_check_pass;
    innov_check_pass_map[5] = (vel_pos_test_ratio_[5] <= 1.0f);

    for (unsigned obs_index = 0; obs_index < 6; obs_index++) {
      // skip fusion if not requested or checks have failed
      if (!fuse_map[obs_index] || !innov_check_pass_map[obs_index]) {
        continue;
      }

      unsigned state_index = obs_index + 4;	// we start with vx and this is the 4. state

      // calculate kalman gain K = PHS, where S = 1/innovation variance
      for (int row = 0; row < k_num_states_; row++) {
        Kfusion[row] = P_[row][state_index] / vel_pos_innov_var_[obs_index];
      }

      // update covarinace matrix via Pnew = (I - KH)P
      float KHP[k_num_states_][k_num_states_];
      for (unsigned row = 0; row < k_num_states_; row++) {
        for (unsigned column = 0; column < k_num_states_; column++) {
          KHP[row][column] = Kfusion[row] * P_[state_index][column];
        }
      }

      // if the covariance correction will result in a negative variance, then
      // the covariance marix is unhealthy and must be corrected
      bool healthy = true;
      for (int i = 0; i < k_num_states_; i++) {
        if (P_[i][i] < KHP[i][i]) {
          // zero rows and columns
          zeroRows(P_,i,i);
          zeroCols(P_,i,i);

          //flag as unhealthy
          healthy = false;
        } 
      }

      // only apply covariance and state corrrections if healthy
      if (healthy) {
        // apply the covariance corrections
        for (unsigned row = 0; row < k_num_states_; row++) {
          for (unsigned column = 0; column < k_num_states_; column++) {
            P_[row][column] = P_[row][column] - KHP[row][column];
          }
        }

        // correct the covariance marix for gross errors
        fixCovarianceErrors();

        // apply the state corrections
        fuse(Kfusion, vel_pos_innov_[obs_index]);
      }
    }
  }

  void ESKF::updateVision(const quat& q, const vec3& p, uint64_t time_usec, scalar_t dt) {
    // transform orientation from (ENU2FLU) to (NED2FRD):
    quat q_nb = q_NED2ENU * q * q_FLU2FRD;

    // transform position from local ENU to local NED frame
    vec3 pos_nb = q_NED2ENU.inverse().toRotationMatrix() * p;

    // limit data rate to prevent data being lost
    if (time_usec - time_last_ext_vision_ > min_obs_interval_us_) {
      extVisionSample ev_sample_new;
      // calculate the system time-stamp for the mid point of the integration period
      // copy required data
      ev_sample_new.angErr = 0.05f;
      ev_sample_new.posErr = 0.05f;
      ev_sample_new.quatNED = q_nb;
      ev_sample_new.posNED = pos_nb;
      ev_sample_new.time_us = time_usec - ev_delay_ms_ * 1000;
      time_last_ext_vision_ = time_usec;
      // push to buffer
      ext_vision_buffer_.push(ev_sample_new);
    }
  }

  void ESKF::updateGps(const vec3& v, const vec3& p, uint64_t time_us, scalar_t dt) {
    // transform linear velocity from local ENU to body FRD frame
    vec3 vel_nb = q_NED2ENU.inverse().toRotationMatrix() * v;

    // transform position from local ENU to local NED frame
    vec3 pos_nb = q_NED2ENU.inverse().toRotationMatrix() * p;

    // check for arrival of new sensor data at the fusion time horizon
    if (time_us - time_last_gps_ > min_obs_interval_us_) {
      gpsSample gps_sample_new;
      gps_sample_new.time_us = time_us - gps_delay_ms_ * 1000;

      gps_sample_new.time_us -= FILTER_UPDATE_PERIOD_MS * 1000 / 2;
      time_last_gps_ = time_us;

      gps_sample_new.time_us = max(gps_sample_new.time_us, imu_sample_delayed_.time_us);
      gps_sample_new.vel = vel_nb;
      gps_sample_new.hacc = 1.0;
      gps_sample_new.vacc = 1.0;
      gps_sample_new.sacc = 0.0;
      
      gps_sample_new.pos(0) = pos_nb(0);
      gps_sample_new.pos(1) = pos_nb(1);
      gps_sample_new.hgt = pos_nb(2);
      gps_buffer_.push(gps_sample_new);
    }
  }

  void ESKF::updateOpticalFlow(const vec2& int_xy, const vec2& int_xy_gyro, uint32_t integration_time_us, scalar_t distance, uint8_t quality, uint64_t time_us, scalar_t dt) {
    // convert integrated flow and integrated gyro from flu to frd
    ///< integrated flow in PX4 Body (FRD) coordinates
    ///< integrated gyro in PX4 Body (FRD) coordinates
    vec2 int_xy_b;
    vec2 int_xy_gyro_b;

    int_xy_b(0) = int_xy(0);
    int_xy_b(1) = -int_xy(1);

    int_xy_gyro_b = vec2(0,0); //replace embedded camera gyro to imu gyro

    // check for arrival of new sensor data at the fusion time horizon
    if (time_us - time_last_opt_flow_ > min_obs_interval_us_) {
      // check if enough integration time and fail if integration time is less than 50% of min arrival interval because too much data is being lost
      scalar_t delta_time = 1e-6f * (scalar_t)integration_time_us;
      scalar_t delta_time_min = 5e-7f * (scalar_t)min_obs_interval_us_;
      bool delta_time_good = delta_time >= delta_time_min;

      if (!delta_time_good) {
        // protect against overflow casued by division with very small delta_time
        delta_time = delta_time_min;
      }

      // check magnitude is within sensor limits use this to prevent use of a saturated flow sensor when there are other aiding sources available
      scalar_t flow_rate_magnitude;
      bool flow_magnitude_good = true;
      if (delta_time_good) {
        flow_rate_magnitude = int_xy_b.norm() / delta_time;
        flow_magnitude_good = (flow_rate_magnitude <= flow_max_rate_);
      }

      bool relying_on_flow = opt_flow_ && !gps_pos_ && !ev_pos_;

      // check quality metric
      bool flow_quality_good = (quality >= flow_quality_min_);
      // Always use data when on ground to allow for bad quality due to unfocussed sensors and operator handling
      // If flow quality fails checks on ground, assume zero flow rate after body rate compensation
      if ((delta_time_good && flow_quality_good && (flow_magnitude_good || relying_on_flow)) || !in_air_) {
        optFlowSample opt_flow_sample_new;
        opt_flow_sample_new.time_us = time_us - flow_delay_ms_ * 1000;

        // copy the quality metric returned by the PX4Flow sensor
        opt_flow_sample_new.quality = quality;

        // NOTE: the EKF uses the reverse sign convention to the flow sensor. EKF assumes positive LOS rate is produced by a RH rotation of the image about the sensor axis.
        // copy the optical and gyro measured delta angles
        opt_flow_sample_new.gyroXY = - vec2(int_xy_gyro_b.x(), int_xy_gyro_b.y());
        opt_flow_sample_new.flowRadXY = - vec2(int_xy_b.x(), int_xy_b.y());

        // convert integration interval to seconds
        opt_flow_sample_new.dt = delta_time;

        time_last_opt_flow_ = time_us;

        opt_flow_buffer_.push(opt_flow_sample_new);

        rangeSample range_sample_new;
        range_sample_new.rng = distance;
        range_sample_new.time_us = time_us - range_delay_ms_ * 1000;
        time_last_range_ = time_us;

        range_buffer_.push(range_sample_new);
      }
    }
  }

  void ESKF::updateRangeFinder(scalar_t range, uint64_t time_us, scalar_t dt) {
    // check for arrival of new range data at the fusion time horizon
    if (time_us - time_last_range_ > min_obs_interval_us_) {
      rangeSample range_sample_new;
      range_sample_new.rng = range;
      range_sample_new.time_us = time_us - range_delay_ms_ * 1000;
      time_last_range_ = time_us;

      range_buffer_.push(range_sample_new);
    }
  }

  void ESKF::updateMagnetometer(const vec3& m, uint64_t time_usec, scalar_t dt) {
    // convert FLU to FRD body frame IMU data
    vec3 m_nb = q_FLU2FRD.toRotationMatrix() * m;

    // limit data rate to prevent data being lost
    if ((time_usec - time_last_mag_) > min_obs_interval_us_) {
      magSample mag_sample_new;
      mag_sample_new.time_us = time_usec - mag_delay_ms_ * 1000;

      mag_sample_new.time_us -= FILTER_UPDATE_PERIOD_MS * 1000 / 2;
      time_last_mag_ = time_usec;
      mag_sample_new.mag = m_nb;
      mag_buffer_.push(mag_sample_new);
    }
  }

  void ESKF::updateLandedState(uint8_t landed_state) {
    in_air_ = landed_state;
  }

  bool ESKF::initHagl() {
    // get most recent range measurement from buffer
    const rangeSample &latest_measurement = range_buffer_.get_newest();

    if ((time_last_imu_ - latest_measurement.time_us) < (uint64_t)2e5 && R_rng_to_earth_2_2_ > range_cos_max_tilt_) {
      // if we have a fresh measurement, use it to initialise the terrain estimator
      terrain_vpos_ = state_.pos(2) + latest_measurement.rng * R_rng_to_earth_2_2_;
      // initialise state variance to variance of measurement
      terrain_var_ = sq(range_noise_);
      // success
      return true;
    } else if (!in_air_) {
      // if on ground we assume a ground clearance
      terrain_vpos_ = state_.pos(2) + rng_gnd_clearance_;
      // Use the ground clearance value as our uncertainty
      terrain_var_ = rng_gnd_clearance_;
      // ths is a guess
      return false;
    } else {
      // no information - cannot initialise
      return false;
    }
  }

  void ESKF::fuseHeading() {
    // assign intermediate state variables
    scalar_t q0 = state_.quat_nominal.w();
    scalar_t q1 = state_.quat_nominal.x();
    scalar_t q2 = state_.quat_nominal.y();
    scalar_t q3 = state_.quat_nominal.z();

    scalar_t predicted_hdg, measured_hdg;
    scalar_t H_YAW[4];
    vec3 mag_earth_pred;

    // determine if a 321 or 312 Euler sequence is best
    if (fabsf(R_to_earth_(2, 0)) < fabsf(R_to_earth_(2, 1))) {
      // calculate observation jacobian when we are observing the first rotation in a 321 sequence
      scalar_t t9 = q0*q3;
      scalar_t t10 = q1*q2;
      scalar_t t2 = t9+t10;
      scalar_t t3 = q0*q0;
      scalar_t t4 = q1*q1;
      scalar_t t5 = q2*q2;
      scalar_t t6 = q3*q3;
      scalar_t t7 = t3+t4-t5-t6;
      scalar_t t8 = t7*t7;
      if (t8 > 1e-6f) {
        t8 = 1.0f/t8;
      } else {
        return;
      }
      scalar_t t11 = t2*t2;
      scalar_t t12 = t8*t11*4.0f;
      scalar_t t13 = t12+1.0f;
      scalar_t t14;
      if (fabsf(t13) > 1e-6f) {
        t14 = 1.0f/t13;
      } else {
        return;
      }

      H_YAW[0] = t8*t14*(q3*t3-q3*t4+q3*t5+q3*t6+q0*q1*q2*2.0f)*-2.0f;
      H_YAW[1] = t8*t14*(-q2*t3+q2*t4+q2*t5+q2*t6+q0*q1*q3*2.0f)*-2.0f;
      H_YAW[2] = t8*t14*(q1*t3+q1*t4+q1*t5-q1*t6+q0*q2*q3*2.0f)*2.0f;
      H_YAW[3] = t8*t14*(q0*t3+q0*t4-q0*t5+q0*t6+q1*q2*q3*2.0f)*2.0f;

      // rotate the magnetometer measurement into earth frame
      vec3 euler321 = dcm2vec(quat2dcm(state_.quat_nominal));
      predicted_hdg = euler321(2); // we will need the predicted heading to calculate the innovation

      // calculate the observed yaw angle
      if (mag_hdg_) {
        // Set the yaw angle to zero and rotate the measurements into earth frame using the zero yaw angle
        euler321(2) = 0.0f;
        mat3 R_to_earth = quat2dcm(from_euler(euler321));

        // rotate the magnetometer measurements into earth frame using a zero yaw angle
        if (mag_3D_) {
          // don't apply bias corrections if we are learning them
          mag_earth_pred = R_to_earth * mag_sample_delayed_.mag;
        } else {
          mag_earth_pred = R_to_earth * (mag_sample_delayed_.mag - state_.mag_B);
        }

        // the angle of the projection onto the horizontal gives the yaw angle
        measured_hdg = -atan2f(mag_earth_pred(1), mag_earth_pred(0)) + mag_declination_;

      } else if (ev_yaw_) {

        // calculate the yaw angle for a 321 sequence
        // Expressions obtained from yaw_input_321.c produced by https://github.com/PX4/ecl/blob/master/matlab/scripts/Inertial%20Nav%20EKF/quat2yaw321.m
        scalar_t Tbn_1_0 = 2.0f*(ev_sample_delayed_.quatNED.w() * ev_sample_delayed_.quatNED.z() + ev_sample_delayed_.quatNED.x() * ev_sample_delayed_.quatNED.y());
        scalar_t Tbn_0_0 = sq(ev_sample_delayed_.quatNED.w()) + sq(ev_sample_delayed_.quatNED.x()) - sq(ev_sample_delayed_.quatNED.y()) - sq(ev_sample_delayed_.quatNED.z());
        measured_hdg = atan2f(Tbn_1_0,Tbn_0_0);

      } else return;
    }
    else {
      // calculate observaton jacobian when we are observing a rotation in a 312 sequence
      scalar_t t9 = q0*q3;
      scalar_t t10 = q1*q2;
      scalar_t t2 = t9-t10;
      scalar_t t3 = q0*q0;
      scalar_t t4 = q1*q1;
      scalar_t t5 = q2*q2;
      scalar_t t6 = q3*q3;
      scalar_t t7 = t3-t4+t5-t6;
      scalar_t t8 = t7*t7;
      if (t8 > 1e-6f) {
        t8 = 1.0f/t8;
      } else {
        return;
      }
      scalar_t t11 = t2*t2;
      scalar_t t12 = t8*t11*4.0f;
      scalar_t t13 = t12+1.0f;
      scalar_t t14;
      if (fabsf(t13) > 1e-6f) {
        t14 = 1.0f/t13;
      } else {
        return;
      }

      H_YAW[0] = t8*t14*(q3*t3+q3*t4-q3*t5+q3*t6-q0*q1*q2*2.0f)*-2.0f;
      H_YAW[1] = t8*t14*(q2*t3+q2*t4+q2*t5-q2*t6-q0*q1*q3*2.0f)*-2.0f;
      H_YAW[2] = t8*t14*(-q1*t3+q1*t4+q1*t5+q1*t6-q0*q2*q3*2.0f)*2.0f;
      H_YAW[3] = t8*t14*(q0*t3-q0*t4+q0*t5+q0*t6-q1*q2*q3*2.0f)*2.0f;

      scalar_t yaw = atan2f(-R_to_earth_(0, 1), R_to_earth_(1, 1)); // first rotation (yaw)
      scalar_t roll = asinf(R_to_earth_(2, 1)); // second rotation (roll)
      scalar_t pitch = atan2f(-R_to_earth_(2, 0), R_to_earth_(2, 2)); // third rotation (pitch)

      predicted_hdg = yaw; // we will need the predicted heading to calculate the innovation

      // calculate the observed yaw angle
      if (mag_hdg_) {
        // Set the first rotation (yaw) to zero and rotate the measurements into earth frame
        yaw = 0.0f;

        // Calculate the body to earth frame rotation matrix from the euler angles using a 312 rotation sequence
        // Equations from Tbn_312.c produced by https://github.com/PX4/ecl/blob/master/matlab/scripts/Inertial%20Nav%20EKF/quat2yaw312.m
        mat3 R_to_earth;
        scalar_t sy = sinf(yaw);
        scalar_t cy = cosf(yaw);
        scalar_t sp = sinf(pitch);
        scalar_t cp = cosf(pitch);
        scalar_t sr = sinf(roll);
        scalar_t cr = cosf(roll);
        R_to_earth(0,0) = cy*cp-sy*sp*sr;
        R_to_earth(0,1) = -sy*cr;
        R_to_earth(0,2) = cy*sp+sy*cp*sr;
        R_to_earth(1,0) = sy*cp+cy*sp*sr;
        R_to_earth(1,1) = cy*cr;
        R_to_earth(1,2) = sy*sp-cy*cp*sr;
        R_to_earth(2,0) = -sp*cr;
        R_to_earth(2,1) = sr;
        R_to_earth(2,2) = cp*cr;

        // rotate the magnetometer measurements into earth frame using a zero yaw angle
        if (mag_3D_) {
          // don't apply bias corrections if we are learning them
          mag_earth_pred = R_to_earth * mag_sample_delayed_.mag;
        } else {
          mag_earth_pred = R_to_earth * (mag_sample_delayed_.mag - state_.mag_B);
        }

        // the angle of the projection onto the horizontal gives the yaw angle
        measured_hdg = -atan2f(mag_earth_pred(1), mag_earth_pred(0)) + mag_declination_;

      } else if (ev_yaw_) {
        // calculate the yaw angle for a 312 sequence
        // Values from yaw_input_312.c file produced by https://github.com/PX4/ecl/blob/master/matlab/scripts/Inertial%20Nav%20EKF/quat2yaw312.m
        scalar_t Tbn_0_1_neg = 2.0f * (ev_sample_delayed_.quatNED.w() * ev_sample_delayed_.quatNED.z() - ev_sample_delayed_.quatNED.x() * ev_sample_delayed_.quatNED.y());
        scalar_t Tbn_1_1 = sq(ev_sample_delayed_.quatNED.w()) - sq(ev_sample_delayed_.quatNED.x()) + sq(ev_sample_delayed_.quatNED.y()) - sq(ev_sample_delayed_.quatNED.z());
        measured_hdg = atan2f(Tbn_0_1_neg, Tbn_1_1);

      } else return;
    }

    scalar_t R_YAW; // calculate observation variance
    if (mag_hdg_) {
      // using magnetic heading tuning parameter
      R_YAW = sq(fmaxf(mag_heading_noise_, 1.0e-2f));
    } else if (ev_yaw_)
      // using error estimate from external vision data
      R_YAW = sq(fmaxf(ev_sample_delayed_.angErr, 1.0e-2f));
    else return;

    // Calculate innovation variance and Kalman gains, taking advantage of the fact that only the first 3 elements in H are non zero
    // calculate the innovaton variance
    scalar_t PH[4];
    scalar_t heading_innov_var = R_YAW;
    for (unsigned row = 0; row <= 3; row++) {
      PH[row] = 0.0f;
      for (uint8_t col = 0; col <= 3; col++) {
	PH[row] += P_[row][col] * H_YAW[col];
      }
      heading_innov_var += H_YAW[row] * PH[row];
    }

    scalar_t heading_innov_var_inv;

    // check if the innovation variance calculation is badly conditioned
    if (heading_innov_var >= R_YAW) {
      // the innovation variance contribution from the state covariances is not negative, no fault
      heading_innov_var_inv = 1.0f / heading_innov_var;
    } else {
      // the innovation variance contribution from the state covariances is negative which means the covariance matrix is badly conditioned
      // we reinitialise the covariance matrix and abort this fusion step
      initialiseCovariance();
      return;
    }

    // calculate the Kalman gains
    // only calculate gains for states we are using
    scalar_t Kfusion[k_num_states_] = {};

    for (uint8_t row = 0; row < k_num_states_; row++) {
      Kfusion[row] = 0.0f;
      for (uint8_t col = 0; col <= 3; col++) {
        Kfusion[row] += P_[row][col] * H_YAW[col];
      }
      Kfusion[row] *= heading_innov_var_inv;
    }

    // wrap the heading to the interval between +-pi
    measured_hdg = wrap_pi(measured_hdg);

    if (mag_use_inhibit_) {
      // The magnetomer cannot be trusted but we need to fuse a heading to prevent a badly conditoned covariance matrix developing over time.
      if (!vehicle_at_rest_) {
        // Vehicle is not at rest so fuse a zero innovation and record the predicted heading to use as an observation when movement ceases.
        heading_innov_ = 0.0f;
        vehicle_at_rest_prev_ = false;
      } else {
        // Vehicle is at rest so use the last moving prediciton as an observation to prevent the heading from drifting and to enable yaw gyro bias learning before takeoff.
        if (!vehicle_at_rest_prev_ || !mag_use_inhibit_prev_) {
          last_static_yaw_ = predicted_hdg;
          vehicle_at_rest_prev_ = true;
        }
        // calculate the innovation
        heading_innov_ = predicted_hdg - last_static_yaw_;
        R_YAW = 0.01f;
        heading_innov_gate_ = 5.0f;
      }
    } else {
      // calculate the innovation
      heading_innov_ = predicted_hdg - measured_hdg;
      last_static_yaw_ = predicted_hdg;
    }
    mag_use_inhibit_prev_ = mag_use_inhibit_;

    // wrap the innovation to the interval between +-pi
    heading_innov_ = wrap_pi(heading_innov_);

    // innovation test ratio
    scalar_t yaw_test_ratio = sq(heading_innov_) / (sq(heading_innov_gate_) * heading_innov_var);

    // set the vision yaw unhealthy if the test fails
    if (yaw_test_ratio > 1.0f) {
      if(in_air_)
        return;
      else {
        // constrain the innovation to the maximum set by the gate
        scalar_t gate_limit = sqrtf(sq(heading_innov_gate_) * heading_innov_var);
        heading_innov_ = constrain(heading_innov_, -gate_limit, gate_limit);
      }
    }
    
    // apply covariance correction via P_new = (I -K*H)*P
    // first calculate expression for KHP
    // then calculate P - KHP
    float KHP[k_num_states_][k_num_states_];
    float KH[4];
    for (unsigned row = 0; row < k_num_states_; row++) {

      KH[0] = Kfusion[row] * H_YAW[0];
      KH[1] = Kfusion[row] * H_YAW[1];
      KH[2] = Kfusion[row] * H_YAW[2];
      KH[3] = Kfusion[row] * H_YAW[3];

      for (unsigned column = 0; column < k_num_states_; column++) {
        float tmp = KH[0] * P_[0][column];
        tmp += KH[1] * P_[1][column];
        tmp += KH[2] * P_[2][column];
        tmp += KH[3] * P_[3][column];
        KHP[row][column] = tmp;
      }
    }

    // if the covariance correction will result in a negative variance, then
    // the covariance marix is unhealthy and must be corrected
    bool healthy = true;
    for (int i = 0; i < k_num_states_; i++) {
      if (P_[i][i] < KHP[i][i]) {
        // zero rows and columns
        zeroRows(P_,i,i);
        zeroCols(P_,i,i);

        //flag as unhealthy
        healthy = false;
      }
    }

    // only apply covariance and state corrrections if healthy
    if (healthy) {
      // apply the covariance corrections
      for (unsigned row = 0; row < k_num_states_; row++) {
        for (unsigned column = 0; column < k_num_states_; column++) {
          P_[row][column] = P_[row][column] - KHP[row][column];
        }
      }

      // correct the covariance marix for gross errors
      fixCovarianceErrors();

      // apply the state corrections
      fuse(Kfusion, heading_innov_);
    }
  }

  void ESKF::fixCovarianceErrors() {
    scalar_t P_lim[7] = {};
    P_lim[0] = 1.0f;		// quaternion max var
    P_lim[1] = 1e6f;		// velocity max var
    P_lim[2] = 1e6f;		// positiion max var
    P_lim[3] = 1.0f;		// gyro bias max var
    P_lim[4] = 1.0f;		// delta velocity z bias max var
    P_lim[5] = 1.0f;		// earth mag field max var
    P_lim[6] = 1.0f;		// body mag field max var
    
    for (int i = 0; i <= 3; i++) {
      // quaternion states
      P_[i][i] = constrain(P_[i][i], 0.0f, P_lim[0]);
    }
    
    for (int i = 4; i <= 6; i++) {
      // NED velocity states
      P_[i][i] = constrain(P_[i][i], 0.0f, P_lim[1]);
    }

    for (int i = 7; i <= 9; i++) {
      // NED position states
      P_[i][i] = constrain(P_[i][i], 0.0f, P_lim[2]);
    }
    
    for (int i = 10; i <= 12; i++) {
      // gyro bias states
      P_[i][i] = constrain(P_[i][i], 0.0f, P_lim[3]);
    }
    
    // force symmetry on the quaternion, velocity, positon and gyro bias state covariances
    makeSymmetrical(P_,0,12);

    // Find the maximum delta velocity bias state variance and request a covariance reset if any variance is below the safe minimum
    const scalar_t minSafeStateVar = 1e-9f;
    scalar_t maxStateVar = minSafeStateVar;
    bool resetRequired = false;

    for (uint8_t stateIndex = 13; stateIndex <= 15; stateIndex++) {
      if (P_[stateIndex][stateIndex] > maxStateVar) {
        maxStateVar = P_[stateIndex][stateIndex];
      } else if (P_[stateIndex][stateIndex] < minSafeStateVar) {
        resetRequired = true;
      }
    }

    // To ensure stability of the covariance matrix operations, the ratio of a max and min variance must
    // not exceed 100 and the minimum variance must not fall below the target minimum
    // Also limit variance to a maximum equivalent to a 0.1g uncertainty
    const scalar_t minStateVarTarget = 5E-8f;
    scalar_t minAllowedStateVar = fmaxf(0.01f * maxStateVar, minStateVarTarget);

    for (uint8_t stateIndex = 13; stateIndex <= 15; stateIndex++) {
      P_[stateIndex][stateIndex] = constrain(P_[stateIndex][stateIndex], minAllowedStateVar, sq(0.1f * CONSTANTS_ONE_G * dt_ekf_avg_));
    }

    // If any one axis has fallen below the safe minimum, all delta velocity covariance terms must be reset to zero
    if (resetRequired) {
      scalar_t delVelBiasVar[3];

      // store all delta velocity bias variances
      for (uint8_t stateIndex = 13; stateIndex <= 15; stateIndex++) {
        delVelBiasVar[stateIndex - 13] = P_[stateIndex][stateIndex];
      }

      // reset all delta velocity bias covariances
      zeroCols(P_, 13, 15);

      // restore all delta velocity bias variances
      for (uint8_t stateIndex = 13; stateIndex <= 15; stateIndex++) {
        P_[stateIndex][stateIndex] = delVelBiasVar[stateIndex - 13];
      }
    }

    // Run additional checks to see if the delta velocity bias has hit limits in a direction that is clearly wrong
    // calculate accel bias term aligned with the gravity vector
    //scalar_t dVel_bias_lim = 0.9f * acc_bias_lim * dt_ekf_avg_;
    scalar_t down_dvel_bias = 0.0f;

    for (uint8_t axis_index = 0; axis_index < 3; axis_index++) {
      down_dvel_bias += state_.accel_bias(axis_index) * R_to_earth_(2, axis_index);
    }

    // check that the vertical componenent of accel bias is consistent with both the vertical position and velocity innovation
    //bool bad_acc_bias = (fabsf(down_dvel_bias) > dVel_bias_lim && down_dvel_bias * vel_pos_innov_[2] < 0.0f && down_dvel_bias * vel_pos_innov_[5] < 0.0f);

    // if we have failed for 7 seconds continuously, reset the accel bias covariances to fix bad conditioning of
    // the covariance matrix but preserve the variances (diagonals) to allow bias learning to continue
    if (time_last_imu_ - time_acc_bias_check_ > (uint64_t)7e6) {
      scalar_t varX = P_[13][13];
      scalar_t varY = P_[14][14];
      scalar_t varZ = P_[15][15];
      zeroRows(P_, 13, 15);
      zeroCols(P_, 13, 15);
      P_[13][13] = varX;
      P_[14][14] = varY;
      P_[15][15] = varZ;
      //ECL_WARN("EKF invalid accel bias - resetting covariance");
    } else {
      // ensure the covariance values are symmetrical
      makeSymmetrical(P_, 13, 15);
    }

    // magnetic field states
    if (!mag_3D_) {
      zeroRows(P_, 16, 21);
      zeroCols(P_, 16, 21);
    } else {
      // constrain variances
      for (int i = 16; i <= 18; i++) {
        P_[i][i] = constrain(P_[i][i], 0.0f, P_lim[5]);
      }

      for (int i = 19; i <= 21; i++) {
        P_[i][i] = constrain(P_[i][i], 0.0f, P_lim[6]);
      }

      // force symmetry
      makeSymmetrical(P_, 16, 21);
    }
  }
  
  // This function forces the covariance matrix to be symmetric
  void ESKF::makeSymmetrical(scalar_t (&cov_mat)[k_num_states_][k_num_states_], uint8_t first, uint8_t last) {
    for (unsigned row = first; row <= last; row++) {
      for (unsigned column = 0; column < row; column++) {
        float tmp = (cov_mat[row][column] + cov_mat[column][row]) / 2;
        cov_mat[row][column] = tmp;
        cov_mat[column][row] = tmp;
      }
    }
  }
  
  // fuse measurement
  void ESKF::fuse(scalar_t *K, scalar_t innovation) {
    state_.quat_nominal.w() = state_.quat_nominal.w() - K[0] * innovation;
    state_.quat_nominal.x() = state_.quat_nominal.x() - K[1] * innovation;
    state_.quat_nominal.y() = state_.quat_nominal.y() - K[2] * innovation;
    state_.quat_nominal.z() = state_.quat_nominal.z() - K[3] * innovation;
    
    state_.quat_nominal.normalize();

    for (unsigned i = 0; i < 3; i++) {
      state_.vel(i) = state_.vel(i) - K[i + 4] * innovation;
    }

    for (unsigned i = 0; i < 3; i++) {
      state_.pos(i) = state_.pos(i) - K[i + 7] * innovation;
    }

    for (unsigned i = 0; i < 3; i++) {
      state_.gyro_bias(i) = state_.gyro_bias(i) - K[i + 10] * innovation;
    }

    for (unsigned i = 0; i < 3; i++) {
      state_.accel_bias(i) = state_.accel_bias(i) - K[i + 13] * innovation;
    }
    
    for (unsigned i = 0; i < 3; i++) {
      state_.mag_I(i) = state_.mag_I(i) - K[i + 16] * innovation;
    }

    for (unsigned i = 0; i < 3; i++) {
      state_.mag_B(i) = state_.mag_B(i) - K[i + 19] * innovation;
    }
  }

  quat ESKF::getQuat() {
    // transform orientation from (NED2FRD) to (ENU2FLU)
    return q_NED2ENU.conjugate() * state_.quat_nominal * q_FLU2FRD.conjugate(); 
  }

  vec3 ESKF::getPosition() {
    // transform position from local NED to local ENU frame
    return q_NED2ENU.toRotationMatrix() * state_.pos;
  }

  vec3 ESKF::getVelocity() {
    // transform velocity from local NED to local ENU frame
    return q_NED2ENU.toRotationMatrix() * state_.vel;
  }

  // initialise the quaternion covariances using rotation vector variances
  void ESKF::initialiseQuatCovariances(const vec3& rot_vec_var) {
    // calculate an equivalent rotation vector from the quaternion
    scalar_t q0,q1,q2,q3;
    if (state_.quat_nominal.w() >= 0.0f) {
      q0 = state_.quat_nominal.w();
      q1 = state_.quat_nominal.x();
      q2 = state_.quat_nominal.y();
      q3 = state_.quat_nominal.z();
    } else {
      q0 = -state_.quat_nominal.w();
      q1 = -state_.quat_nominal.x();
      q2 = -state_.quat_nominal.y();
      q3 = -state_.quat_nominal.z();
    }
    scalar_t delta = 2.0f*acosf(q0);
    scalar_t scaler = (delta/sinf(delta*0.5f));
    scalar_t rotX = scaler*q1;
    scalar_t rotY = scaler*q2;
    scalar_t rotZ = scaler*q3;

    // autocode generated using matlab symbolic toolbox
    scalar_t t2 = rotX*rotX;
    scalar_t t4 = rotY*rotY;
    scalar_t t5 = rotZ*rotZ;
    scalar_t t6 = t2+t4+t5;
    if (t6 > 1e-9f) {
      scalar_t t7 = sqrtf(t6);
      scalar_t t8 = t7*0.5f;
      scalar_t t3 = sinf(t8);
      scalar_t t9 = t3*t3;
      scalar_t t10 = 1.0f/t6;
      scalar_t t11 = 1.0f/sqrtf(t6);
      scalar_t t12 = cosf(t8);
      scalar_t t13 = 1.0f/powf(t6,1.5f);
      scalar_t t14 = t3*t11;
      scalar_t t15 = rotX*rotY*t3*t13;
      scalar_t t16 = rotX*rotZ*t3*t13;
      scalar_t t17 = rotY*rotZ*t3*t13;
      scalar_t t18 = t2*t10*t12*0.5f;
      scalar_t t27 = t2*t3*t13;
      scalar_t t19 = t14+t18-t27;
      scalar_t t23 = rotX*rotY*t10*t12*0.5f;
      scalar_t t28 = t15-t23;
      scalar_t t20 = rotY*rot_vec_var(1)*t3*t11*t28*0.5f;
      scalar_t t25 = rotX*rotZ*t10*t12*0.5f;
      scalar_t t31 = t16-t25;
      scalar_t t21 = rotZ*rot_vec_var(2)*t3*t11*t31*0.5f;
      scalar_t t22 = t20+t21-rotX*rot_vec_var(0)*t3*t11*t19*0.5f;
      scalar_t t24 = t15-t23;
      scalar_t t26 = t16-t25;
      scalar_t t29 = t4*t10*t12*0.5f;
      scalar_t t34 = t3*t4*t13;
      scalar_t t30 = t14+t29-t34;
      scalar_t t32 = t5*t10*t12*0.5f;
      scalar_t t40 = t3*t5*t13;
      scalar_t t33 = t14+t32-t40;
      scalar_t t36 = rotY*rotZ*t10*t12*0.5f;
      scalar_t t39 = t17-t36;
      scalar_t t35 = rotZ*rot_vec_var(2)*t3*t11*t39*0.5f;
      scalar_t t37 = t15-t23;
      scalar_t t38 = t17-t36;
      scalar_t t41 = rot_vec_var(0)*(t15-t23)*(t16-t25);
      scalar_t t42 = t41-rot_vec_var(1)*t30*t39-rot_vec_var(2)*t33*t39;
      scalar_t t43 = t16-t25;
      scalar_t t44 = t17-t36;

      // zero all the quaternion covariances
      zeroRows(P_,0,3);
      zeroCols(P_,0,3);

      // Update the quaternion internal covariances using auto-code generated using matlab symbolic toolbox
      P_[0][0] = rot_vec_var(0)*t2*t9*t10*0.25f+rot_vec_var(1)*t4*t9*t10*0.25f+rot_vec_var(2)*t5*t9*t10*0.25f;
      P_[0][1] = t22;
      P_[0][2] = t35+rotX*rot_vec_var(0)*t3*t11*(t15-rotX*rotY*t10*t12*0.5f)*0.5f-rotY*rot_vec_var(1)*t3*t11*t30*0.5f;
      P_[0][3] = rotX*rot_vec_var(0)*t3*t11*(t16-rotX*rotZ*t10*t12*0.5f)*0.5f+rotY*rot_vec_var(1)*t3*t11*(t17-rotY*rotZ*t10*t12*0.5f)*0.5f-rotZ*rot_vec_var(2)*t3*t11*t33*0.5f;
      P_[1][0] = t22;
      P_[1][1] = rot_vec_var(0)*(t19*t19)+rot_vec_var(1)*(t24*t24)+rot_vec_var(2)*(t26*t26);
      P_[1][2] = rot_vec_var(2)*(t16-t25)*(t17-rotY*rotZ*t10*t12*0.5f)-rot_vec_var(0)*t19*t28-rot_vec_var(1)*t28*t30;
      P_[1][3] = rot_vec_var(1)*(t15-t23)*(t17-rotY*rotZ*t10*t12*0.5f)-rot_vec_var(0)*t19*t31-rot_vec_var(2)*t31*t33;
      P_[2][0] = t35-rotY*rot_vec_var(1)*t3*t11*t30*0.5f+rotX*rot_vec_var(0)*t3*t11*(t15-t23)*0.5f;
      P_[2][1] = rot_vec_var(2)*(t16-t25)*(t17-t36)-rot_vec_var(0)*t19*t28-rot_vec_var(1)*t28*t30;
      P_[2][2] = rot_vec_var(1)*(t30*t30)+rot_vec_var(0)*(t37*t37)+rot_vec_var(2)*(t38*t38);
      P_[2][3] = t42;
      P_[3][0] = rotZ*rot_vec_var(2)*t3*t11*t33*(-0.5f)+rotX*rot_vec_var(0)*t3*t11*(t16-t25)*0.5f+rotY*rot_vec_var(1)*t3*t11*(t17-t36)*0.5f;
      P_[3][1] = rot_vec_var(1)*(t15-t23)*(t17-t36)-rot_vec_var(0)*t19*t31-rot_vec_var(2)*t31*t33;
      P_[3][2] = t42;
      P_[3][3] = rot_vec_var(2)*(t33*t33)+rot_vec_var(0)*(t43*t43)+rot_vec_var(1)*(t44*t44);
    } else {
      // the equations are badly conditioned so use a small angle approximation
      P_[0][0] = 0.0f;
      P_[0][1] = 0.0f;
      P_[0][2] = 0.0f;
      P_[0][3] = 0.0f;
      P_[1][0] = 0.0f;
      P_[1][1] = 0.25f * rot_vec_var(0);
      P_[1][2] = 0.0f;
      P_[1][3] = 0.0f;
      P_[2][0] = 0.0f;
      P_[2][1] = 0.0f;
      P_[2][2] = 0.25f * rot_vec_var(1);
      P_[2][3] = 0.0f;
      P_[3][0] = 0.0f;
      P_[3][1] = 0.0f;
      P_[3][2] = 0.0f;
      P_[3][3] = 0.25f * rot_vec_var(2);
    }
  }
  
  // zero specified range of rows in the state covariance matrix
  void ESKF::zeroRows(scalar_t (&cov_mat)[k_num_states_][k_num_states_], uint8_t first, uint8_t last) {
    uint8_t row;
    for (row = first; row <= last; row++) {
      memset(&cov_mat[row][0], 0, sizeof(cov_mat[0][0]) * k_num_states_);
    }
  }

  // zero specified range of columns in the state covariance matrix
  void ESKF::zeroCols(scalar_t (&cov_mat)[k_num_states_][k_num_states_], uint8_t first, uint8_t last) {
    uint8_t row;
    for (row = 0; row <= k_num_states_-1; row++) {
      memset(&cov_mat[row][first], 0, sizeof(cov_mat[0][0]) * (1 + last - first));
    }
  }

  void ESKF::setDiag(scalar_t (&cov_mat)[k_num_states_][k_num_states_], uint8_t first, uint8_t last, scalar_t variance) {
    // zero rows and columns
    zeroRows(cov_mat, first, last);
    zeroCols(cov_mat, first, last);

    // set diagonals
    for (uint8_t row = first; row <= last; row++) {
      cov_mat[row][row] = variance;
    }
  }

  void ESKF::constrainStates() {
    state_.quat_nominal.w() = constrain(state_.quat_nominal.w(), -1.0f, 1.0f);
    state_.quat_nominal.x() = constrain(state_.quat_nominal.x(), -1.0f, 1.0f);
    state_.quat_nominal.y() = constrain(state_.quat_nominal.y(), -1.0f, 1.0f);
    state_.quat_nominal.z() = constrain(state_.quat_nominal.z(), -1.0f, 1.0f);
	  
    for (int i = 0; i < 3; i++) {
      state_.vel(i) = constrain(state_.vel(i), -1000.0f, 1000.0f);
    }

    for (int i = 0; i < 3; i++) {
      state_.pos(i) = constrain(state_.pos(i), -1.e6f, 1.e6f);
    }

    for (int i = 0; i < 3; i++) {
      state_.gyro_bias(i) = constrain(state_.gyro_bias(i), -0.349066f * dt_ekf_avg_, 0.349066f * dt_ekf_avg_);
    }

    for (int i = 0; i < 3; i++) {
      state_.accel_bias(i) = constrain(state_.accel_bias(i), -acc_bias_lim * dt_ekf_avg_, acc_bias_lim * dt_ekf_avg_);
    }
    
    for (int i = 0; i < 3; i++) {
      state_.mag_I(i) = constrain(state_.mag_I(i), -1.0f, 1.0f);
    }

    for (int i = 0; i < 3; i++) {
      state_.mag_B(i) = constrain(state_.mag_B(i), -0.5f, 0.5f);
    }
  }

  void ESKF::setFusionMask(int fusion_mask) {
    fusion_mask_ = fusion_mask;
  }
} //  namespace eskf
