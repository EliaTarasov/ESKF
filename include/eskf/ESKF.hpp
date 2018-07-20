#ifndef ESKF_H_
#define ESKF_H_

#include <Eigen/Core>
#include <Eigen/Dense>
#include <ros/time.h>
#include <ros/ros.h>

#include <RingBuffer.hpp>

#include <ignition/math.hh>

#define EV_MAX_INTERVAL		2e5	///< Maximum allowable time interval between external vision system measurements (uSec)
#define GPS_MAX_INTERVAL	1e6	///< Maximum allowable time interval between external gps system measurements (uSec)
#define OPTICAL_FLOW_INTERVAL	2e5	///< Maximum allowable time interval between optical flow system measurements (uSec)

#define MASK_EV_POS 1<<0
#define MASK_EV_YAW 1<<1
#define MASK_EV_HGT 1<<2
#define MASK_GPS_POS 1<<3
#define MASK_GPS_VEL 1<<4
#define MASK_GPS_HGT 1<<5
#define MASK_MAG_YAW 1<<6
#define MASK_OPTICAL_FLOW 1<<7
#define MASK_LIDAR 1<<8
#define MASK_EV (MASK_EV_POS | MASK_EV_YAW | MASK_EV_HGT)
#define MASK_GPS (MASK_GPS_POS | MASK_GPS_VEL | MASK_GPS_HGT)

namespace eskf {

  typedef float scalar_t;
  typedef Eigen::Matrix<scalar_t, 3, 1> vec3; /// Vector in R3
  typedef Eigen::Matrix<scalar_t, 2, 1> vec2; /// Vector in R2
  typedef Eigen::Matrix<scalar_t, 3, 3> mat3; /// Matrix in R3
  typedef Eigen::Quaternion<scalar_t> quat;   /// Member of S4

  class ESKF {
  public:

    static constexpr int k_num_states_ = 16;

    ESKF();

    void setFusionMask(int fusion_mask);

    void run(const vec3& w, const vec3& a, uint64_t time_us, scalar_t dt);
    void updateVision(const quat& q, const vec3& p, uint64_t time_us, scalar_t dt);
    void updateGps(const vec3& v, const vec3& p, uint64_t time_us, scalar_t dt);
    void updateOpticalFlow(const vec2& int_xy, const vec2& int_xy_gyro, uint32_t integration_time_us, scalar_t distance, uint8_t quality, uint64_t time_us, scalar_t dt);
    void updateLandedState(uint8_t landed_state);

    quat getQuat();
    vec3 getXYZ();

  private:

    void constrainStates();
    bool initializeFilter();
    void initialiseQuatCovariances(const vec3& rot_vec_var);
    void zeroRows(scalar_t (&cov_mat)[k_num_states_][k_num_states_], uint8_t first, uint8_t last);
    void zeroCols(scalar_t (&cov_mat)[k_num_states_][k_num_states_], uint8_t first, uint8_t last);
    void makeSymmetrical(scalar_t (&cov_mat)[k_num_states_][k_num_states_], uint8_t first, uint8_t last);
    void setDiag(scalar_t (&cov_mat)[k_num_states_][k_num_states_], uint8_t first, uint8_t last, scalar_t variance);
    void fuse(scalar_t *K, scalar_t innovation);
    void fixCovarianceErrors();
    void initialiseCovariance();
    void predictCovariance();
    void fuseVelPosHeight();
    void resetHeight();
    void resetPosition();
    void resetVelocity();
    void fuseHeading();
    void fuseOptFlow();
    void fuseHagl();
    bool initHagl();
    void runTerrainEstimator();
    void controlFusionModes();
    void controlMagFusion();
    void controlOpticalFlowFusion();
    void controlGpsFusion();
    void controlVelPosFusion();
    void controlExternalVisionFusion();
    void controlHeightSensorTimeouts();
    mat3 quat_to_invrotmat(const quat &q);
    quat from_axis_angle(vec3 vec);
    quat from_axis_angle(const vec3 &axis, scalar_t theta);
    vec3 to_axis_angle(const quat& q);
    mat3 quat2dcm(const quat& q);
    vec3 dcm2vec(const mat3& dcm);
    bool calcOptFlowBodyRateComp();

    /* State vector:
     * Attitude quaternion
     * NED velocity
     * NED position
     * Delta Angle bias - rad (X,Y,Z)
     * Delta Velocity bias
    */
    struct state {
      quat  quat_nominal; ///< quaternion defining the rotaton from NED to XYZ frame
      vec3  vel;          ///< NED velocity in earth frame in m/s
      vec3  pos;          ///< NED position in earth frame in m
      vec3  gyro_bias;    ///< delta angle bias estimate in rad
      vec3  accel_bias;   ///< delta velocity bias estimate in m/s
    } state_;

    struct imuSample {
      vec3    delta_ang;      ///< delta angle in body frame (integrated gyro measurements) (rad)
      vec3    delta_vel;      ///< delta velocity in body frame (integrated accelerometer measurements) (m/sec)
      scalar_t  delta_ang_dt; ///< delta angle integration period (sec)
      scalar_t  delta_vel_dt; ///< delta velocity integration period (sec)
      uint64_t  time_us;      ///< timestamp of the measurement (uSec)
    };

    struct extVisionSample {
      vec3 posNED;       ///< measured NED body position relative to the local origin (m)
      quat quatNED;      ///< measured quaternion orientation defining rotation from NED to body frame
      scalar_t posErr;   ///< 1-Sigma spherical position accuracy (m)
      scalar_t angErr;   ///< 1-Sigma angular error (rad)
      uint64_t time_us;  ///< timestamp of the measurement (uSec)
    };

    struct gpsSample {
      vec2    	pos;	///< NE earth frame gps horizontal position measurement (m)
      vec3 	vel;	///< NED earth frame gps velocity measurement (m/sec)
      scalar_t  hgt;	///< gps height measurement (m)
      scalar_t  hacc;	///< 1-std horizontal position error (m)
      scalar_t  vacc;	///< 1-std vertical position error (m)
      scalar_t  sacc;	///< 1-std speed error (m/sec)
      uint64_t  time_us;	///< timestamp of the measurement (uSec)
    };

    struct optFlowSample {
      uint8_t  quality;	///< quality indicator between 0 and 255
      vec2 flowRadXY;	///< measured delta angle of the image about the X and Y body axes (rad), RH rotaton is positive
      vec2 flowRadXYcomp;	///< measured delta angle of the image about the X and Y body axes after removal of body rotation (rad), RH rotation is positive
      vec2 gyroXY;	///< measured delta angle of the inertial frame about the body axes obtained from rate gyro measurements (rad), RH rotation is positive
      scalar_t    dt;		///< amount of integration time (sec)
      uint64_t time_us;	///< timestamp of the integration period mid-point (uSec)
    };

    struct rangeSample {
      scalar_t       rng;	///< range (distance to ground) measurement (m)
      uint64_t    time_us;	///< timestamp of the measurement (uSec)
    };

    int fusion_mask_ = MASK_EV_POS && MASK_EV_YAW && MASK_EV_HGT; /// < ekf fusion mask (see launch file), vision pose set by default 

    bool collect_imu(imuSample& imu);

    const unsigned FILTER_UPDATE_PERIOD_MS = 12;	///< ekf prediction period in milliseconds - this should ideally be an integer multiple of the IMU time delta

    imuSample imu_sample_new_ {};	///< imu sample capturing the newest imu data
    imuSample imu_down_sampled_ {};  	///< down sampled imu data (sensor rate -> filter update rate)
    vec3 delVel_sum_; 			///< summed delta velocity (m/sec)
    RingBuffer<imuSample> imu_buffer_;
    mat3 R_to_earth_;
     
    quat q_down_sampled_;
    scalar_t imu_collection_time_adj_ {0.0f};	///< the amount of time the IMU collection needs to be advanced to meet the target set by FILTER_UPDATE_PERIOD_MS (sec)
    imuSample imu_sample_delayed_ {};	// captures the imu sample on the delayed time horizon
    extVisionSample ev_sample_delayed_ {}; 
    RingBuffer<extVisionSample> ext_vision_buffer_;
    scalar_t ev_delay_ms_ {100.0f};		///< off-board vision measurement delay relative to the IMU (mSec)
    scalar_t flow_delay_ms_ {5.0f};		///< optical flow measurement delay relative to the IMU (mSec) - this is to the middle of the optical flow integration interval
    uint64_t time_last_opt_flow_ {0};
    uint64_t time_last_ext_vision_ {0};
    uint64_t time_last_gps_ {0};
    uint64_t time_last_imu_ {0};
    uint64_t time_last_hgt_fuse_ {0};
    uint64_t time_last_range_ {0};
    uint64_t time_last_fake_gps_ {0};
    unsigned min_obs_interval_us_ {0}; // minimum time interval between observations that will guarantee data is not lost (usec)
    scalar_t dt_ekf_avg_ {0.001f * FILTER_UPDATE_PERIOD_MS}; ///< average update rate of the ekf

    gpsSample gps_sample_delayed_ {};	// captures the gps sample on the delayed time horizon
    RingBuffer<gpsSample> gps_buffer_;

    RingBuffer<rangeSample> range_buffer_;
    scalar_t range_delay_ms_{5.0f};		///< range finder measurement delay relative to the IMU (mSec)
    scalar_t range_noise_{0.1f};		///< observation noise for range finder measurements (m)
    scalar_t range_innov_gate_{5.0f};		///< range finder fusion innovation consistency gate size (STD)
    scalar_t rng_gnd_clearance_{0.1f};		///< minimum valid value for range when on ground (m)
    scalar_t rng_sens_pitch_{0.0f};		///< Pitch offset of the range sensor (rad). Sensor points out along Z axis when offset is zero. Positive rotation is RH about Y axis.
    scalar_t range_noise_scaler_{0.0f};		///< scaling from range measurement to noise (m/m)
    scalar_t vehicle_variance_scaler_{0.0f};	///< gain applied to vehicle height variance used in calculation of height above ground observation variance
    scalar_t max_hagl_for_range_aid_{5.0f};	///< maximum height above ground for which we allow to use the range finder as height source (if range_aid == 1)
    scalar_t max_vel_for_range_aid_{1.0f};	///< maximum ground velocity for which we allow to use the range finder as height source (if range_aid == 1)
    int32_t range_aid_{0};			///< allow switching primary height source to range finder if certian conditions are met
    scalar_t range_aid_innov_gate_{1.0f}; 	///< gate size used for innovation consistency checks for range aid fusion
    scalar_t range_cos_max_tilt_{0.7071f};	///< cosine of the maximum tilt angle from the vertical that permits use of range finder data
    scalar_t terrain_p_noise_{5.0f};		///< process noise for terrain offset (m/sec)
    scalar_t terrain_gradient_{0.5f};		///< gradient of terrain used to estimate process noise due to changing position (m/m)
    rangeSample range_sample_delayed_{};
    scalar_t terr_test_ratio_{0.0f};		// height above terrain measurement innovation consistency check ratio

    // Terrain height state estimation
    scalar_t terrain_vpos_{0.0f};		///< estimated vertical position of the terrain underneath the vehicle in local NED frame (m)
    scalar_t terrain_var_{1e4f};		///< variance of terrain position estimate (m**2)
    scalar_t hagl_innov_{0.0f};		///< innovation of the last height above terrain measurement (m)
    scalar_t hagl_innov_var_{0.0f};		///< innovation variance for the last height above terrain measurement (m**2)
    uint64_t time_last_hagl_fuse_{0};		///< last system time that the hagl measurement failed it's checks (uSec)
    bool terrain_initialised_{false};	///< true when the terrain estimator has been intialised
    scalar_t sin_tilt_rng_{0.0f};		///< sine of the range finder tilt rotation about the Y body axis
    scalar_t cos_tilt_rng_{1.0f};		///< cosine of the range finder tilt rotation about the Y body axis
    scalar_t R_rng_to_earth_2_2_{0.0f};	///< 2,2 element of the rotation matrix from sensor frame to earth frame
    bool hagl_valid_{false};		///< true when the height above ground estimate is valid
    
    optFlowSample opt_flow_sample_delayed_ {};	// captures the optical flow sample on the delayed time horizon
    
    RingBuffer<optFlowSample> opt_flow_buffer_;
    scalar_t flow_max_rate_ {2.5}; ///< maximum angular flow rate that the optical flow sensor can measure (rad/s)
    int32_t flow_quality_min_ {1};
    scalar_t flow_noise_ {0.15f};		///< observation noise for optical flow LOS rate measurements (rad/sec)
    scalar_t flow_noise_qual_min_ {0.5f};	///< observation noise for optical flow LOS rate measurements when flow sensor quality is at the minimum useable (rad/sec)
    vec2 flow_rad_xy_comp_ {};
    scalar_t flow_innov_[2] {};		///< flow measurement innovation (rad/sec)
    scalar_t flow_innov_var_[2] {};	///< flow innovation variance ((rad/sec)**2)
    scalar_t flow_innov_gate_{3.0f};	///< optical flow fusion innovation consistency gate size (STD)
    vec3 flow_gyro_bias_;	///< bias errors in optical flow sensor rate gyro outputs (rad/sec)
    vec3 imu_del_ang_of_;	///< bias corrected delta angle measurements accumulated across the same time frame as the optical flow rates (rad)
    scalar_t delta_time_of_{0.0f};	///< time in sec that _imu_del_ang_of was accumulated over (sec)

    bool imu_updated_;
    bool filter_initialised_;
    const int obs_buffer_length_ = 9;
    const int imu_buffer_length_ = 15;

    scalar_t P_[k_num_states_][k_num_states_]; /// System covariance matrix

    static constexpr scalar_t kOneG = 9.80665;  /// Earth gravity (m/s^2)
    static constexpr scalar_t acc_bias_lim = 0.4; ///< maximum accel bias magnitude (m/sec**2)
    static constexpr scalar_t hgt_reset_lim = 0.0f; ///< 
    static constexpr scalar_t gndclearance = 0.1f;

    // process noise
    scalar_t gyro_bias_p_noise_ {1.0e-3};		///< process noise for IMU rate gyro bias prediction (rad/sec**2)
    scalar_t accel_bias_p_noise_ {3.0e-3};	///< process noise for IMU accelerometer bias prediction (m/sec**3)

    // input noise
    scalar_t gyro_noise_ {1.5e-2};		///< IMU angular rate noise used for covariance prediction (rad/sec)
    scalar_t accel_noise_ {3.5e-1};		///< IMU acceleration noise use for covariance prediction (m/sec**2)

    // initialization errors
    scalar_t switch_on_gyro_bias_ {0.01f};		///< 1-sigma gyro bias uncertainty at switch on (rad/sec)
    scalar_t switch_on_accel_bias_ {0.2f};	///< 1-sigma accelerometer bias uncertainty at switch on (m/sec**2)
    scalar_t initial_tilt_err_ {0.01f};		///< 1-sigma tilt error after initial alignment using gravity vector (rad)

    scalar_t vel_noise_ {0.5f};	///< minimum allowed observation noise for velocity fusion (m/sec)
    scalar_t pos_noise_ {0.5f};		///< minimum allowed observation noise for position fusion (m)
    scalar_t baro_noise_ {2.0f};			///< observation noise for barometric height fusion (m)
    scalar_t vel_pos_test_ratio_[6] {};  // velocity and position innovation consistency check ratios
    scalar_t vel_pos_innov_[6] {};	///< ROS velocity and position innovations: (m**2)
    scalar_t vel_pos_innov_var_[6] {};	///< ROS velocity and position innovation variances: (m**2)

    scalar_t heading_innov_gate_ {2.6f};		///< heading fusion innovation consistency gate size (STD)
    scalar_t gps_vel_noise_ {5.0e-1f};		///< minimum allowed observation noise for gps velocity fusion (m/sec)
    scalar_t gps_pos_noise_ {0.5f};		///< minimum allowed observation noise for gps position fusion (m)
    scalar_t pos_noaid_noise_ {10.0f};		///< observation noise for non-aiding position fusion (m)

    scalar_t posNE_innov_gate_ {5.0f};		///< GPS horizontal position innovation consistency gate size (STD)
    scalar_t vel_innov_gate_ {5.0f};		///< GPS velocity innovation consistency gate size (STD)

    scalar_t posObsNoiseNE_ {0.0f};		///< 1-STD observtion noise used for the fusion of NE position data (m)
    scalar_t posInnovGateNE_ {1.0f};		///< Number of standard deviations used for the NE position fusion innovation consistency check

    vec2 velObsVarNE_;		///< 1-STD observation noise variance used for the fusion of NE velocity data (m/sec)**2
    scalar_t hvelInnovGate_ {1.0f};		///< Number of standard deviations used for the horizontal velocity fusion innovation consistency check

    /**
      * @brief Quaternion for rotation between ENU and NED frames
      *
      * NED to ENU: +PI/2 rotation about Z (Down) followed by a +PI rotation around X (old North/new East)
      * ENU to NED: +PI/2 rotation about Z (Up) followed by a +PI rotation about X (old East/new North)
      */
    const ignition::math::Quaterniond q_ng = ignition::math::Quaterniond(0, 0.70711, 0.70711, 0);

    /**
      * @brief Quaternion for rotation between body FLU and body FRD frames
      *
      * +PI rotation around X (Forward) axis rotates from Forward, Right, Down (aircraft)
      * to Forward, Left, Up (base_link) frames and vice-versa.
      */
    const ignition::math::Quaterniond q_br = ignition::math::Quaterniond(0, 1, 0, 0);
    
    bool gps_data_ready_ = false;
    bool vision_data_ready_ = false;
    bool range_data_ready_ = false;
    bool flow_data_ready_ = false;
    
    bool fuse_pos_ = false;
    bool fuse_height_ = false;
    bool opt_flow_ = false;
    bool ev_pos_ = false;
    bool ev_yaw_ = false;
    bool ev_hgt_ = false;
    bool gps_pos_ = false;
    bool gps_vel_ = false;
    bool gps_hgt_ = false;
    bool fuse_hor_vel_ = false;
    bool fuse_vert_vel_ = false;
    bool rng_hgt_ = false;
    bool in_air_ = false;
    vec3 last_known_posNED_;
  };
}

#endif /* defined(ESKF_H_) */
