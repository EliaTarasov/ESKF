#ifndef ESKF_H_
#define ESKF_H_

#include <common.h>
#include <RingBuffer.hpp>

namespace eskf {

  class ESKF {
  public:

    static constexpr int k_num_states_ = 22;

    ESKF();

    void setFusionMask(int fusion_mask);

    void run(const vec3& w, const vec3& a, uint64_t time_us, scalar_t dt);
    void updateVision(const quat& q, const vec3& p, uint64_t time_us, scalar_t dt);
    void updateGps(const vec3& v, const vec3& p, uint64_t time_us, scalar_t dt);
    void updateOpticalFlow(const vec2& int_xy, const vec2& int_xy_gyro, uint32_t integration_time_us, scalar_t distance, uint8_t quality, uint64_t time_us, scalar_t dt);
    void updateRangeFinder(scalar_t range, uint64_t time_us, scalar_t dt);
    void updateMagnetometer(const vec3& m, uint64_t time_us, scalar_t dt);
    void updateLandedState(uint8_t landed_state);

    quat getQuat();
    vec3 getPosition();
    vec3 getVelocity();

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
    bool calcOptFlowBodyRateComp();
    bool collect_imu(imuSample& imu);

    ///< state vector
    state state_;

    ///< FIFO buffers
    RingBuffer<imuSample> imu_buffer_;
    RingBuffer<extVisionSample> ext_vision_buffer_;
    RingBuffer<gpsSample> gps_buffer_;
    RingBuffer<rangeSample> range_buffer_;
    RingBuffer<optFlowSample> opt_flow_buffer_;
    RingBuffer<magSample> mag_buffer_;

    ///< FIFO buffers lengths
    const int obs_buffer_length_ {9};
    const int imu_buffer_length_ {15};

    ///< delayed samples
    imuSample imu_sample_delayed_ {};	// captures the imu sample on the delayed time horizon
    extVisionSample ev_sample_delayed_ {}; // captures the external vision sample on the delayed time horizon
    gpsSample gps_sample_delayed_ {};	// captures the gps sample on the delayed time horizon
    rangeSample range_sample_delayed_{};  // captures the range sample on the delayed time horizon
    optFlowSample opt_flow_sample_delayed_ {};	// captures the optical flow sample on the delayed time horizon
    magSample mag_sample_delayed_ {}; //captures magnetometer sample on the delayed time horizon

    ///< new samples
    imuSample imu_sample_new_ {};  ///< imu sample capturing the newest imu data

    ///< downsampled
    imuSample imu_down_sampled_ {};  	///< down sampled imu data (sensor rate -> filter update rate)
    quat q_down_sampled_; ///< down sampled rotation data (sensor rate -> filter update rate)

    ///< masks
    int gps_check_mask = MASK_GPS_NSATS && MASK_GPS_GDOP && MASK_GPS_HACC && MASK_GPS_VACC && MASK_GPS_SACC && MASK_GPS_HDRIFT && MASK_GPS_VDRIFT && MASK_GPS_HSPD && MASK_GPS_VSPD; ///< GPS checks by default
    int fusion_mask_ = MASK_EV_POS && MASK_EV_YAW && MASK_EV_HGT; /// < ekf fusion mask (see launch file), vision pose set by default 

    ///< timestamps
    uint64_t time_last_opt_flow_ {0};
    uint64_t time_last_ext_vision_ {0};
    uint64_t time_last_gps_ {0};
    uint64_t time_last_imu_ {0};
    uint64_t time_last_mag_ {0};
    uint64_t time_last_hgt_fuse_ {0};
    uint64_t time_last_range_ {0};
    uint64_t time_last_fake_gps_ {0};
    uint64_t time_last_hagl_fuse_{0};		///< last system time that the hagl measurement failed it's checks (uSec)
    uint64_t time_acc_bias_check_{0};

    ///< sensors delays
    scalar_t gps_delay_ms_ {110.0f};		///< gps measurement delay relative to the IMU (mSec)
    scalar_t ev_delay_ms_ {100.0f};		///< off-board vision measurement delay relative to the IMU (mSec)
    scalar_t flow_delay_ms_ {5.0f};		///< optical flow measurement delay relative to the IMU (mSec) - this is to the middle of the optical flow integration interval
    scalar_t range_delay_ms_{5.0f};		///< range finder measurement delay relative to the IMU (mSec)
    scalar_t mag_delay_ms_{0.0f};

    ///< frames rotations
    mat3 R_to_earth_;  ///< Rotation (DCM) from FRD to NED

    /**
      * Quaternion for rotation between ENU and NED frames
      *
      * NED to ENU: +PI/2 rotation about Z (Down) followed by a +PI rotation around X (old North/new East)
      * ENU to NED: +PI/2 rotation about Z (Up) followed by a +PI rotation about X (old East/new North)
      */
    const quat q_NED2ENU = quat(0, 0.70711, 0.70711, 0);

    /**
      * Quaternion for rotation between body FLU and body FRD frames
      *
      * FLU to FRD: +PI rotation about X(forward)
      * FRD to FLU: -PI rotation about X(forward)
      */
    const quat q_FLU2FRD = quat(0, 1, 0, 0);

    ///< filter times
    const unsigned FILTER_UPDATE_PERIOD_MS = 12;	///< ekf prediction period in milliseconds - this should ideally be an integer multiple of the IMU time delta
    scalar_t dt_ekf_avg_ {0.001f * FILTER_UPDATE_PERIOD_MS}; ///< average update rate of the ekf
    scalar_t imu_collection_time_adj_ {0.0f};	///< the amount of time the IMU collection needs to be advanced to meet the target set by FILTER_UPDATE_PERIOD_MS (sec)
    unsigned min_obs_interval_us_ {0}; // minimum time interval between observations that will guarantee data is not lost (usec)

    ///< filter initialisation
    bool NED_origin_initialised_; ///< true when the NED origin has been initialised
    bool terrain_initialised_;	///< true when the terrain estimator has been intialised
    bool filter_initialised_;	///< true when the filter has been initialised

    ///< Covariance
    scalar_t P_[k_num_states_][k_num_states_];	///< System covariance matrix

    // process noise
    const scalar_t gyro_bias_p_noise_ {1.0e-3};		///< process noise for IMU rate gyro bias prediction (rad/sec**2)
    const scalar_t accel_bias_p_noise_ {3.0e-3};	///< process noise for IMU accelerometer bias prediction (m/sec**3)

    // input noise
    const scalar_t gyro_noise_ {1.5e-2};	///< IMU angular rate noise used for covariance prediction (rad/sec)
    const scalar_t accel_noise_ {3.5e-1};	///< IMU acceleration noise use for covariance prediction (m/sec**2)

    ///< Measurement (observation) noise
    const scalar_t range_noise_{0.1f};		///< observation noise for range finder measurements (m)
    const scalar_t flow_noise_ {0.15f};		///< observation noise for optical flow LOS rate measurements (rad/sec)
    const scalar_t gps_vel_noise_ {5.0e-1f};	///< minimum allowed observation noise for gps velocity fusion (m/sec)
    const scalar_t gps_pos_noise_ {0.5f};	///< minimum allowed observation noise for gps position fusion (m)
    const scalar_t pos_noaid_noise_ {10.0f};	///< observation noise for non-aiding position fusion (m)
    const scalar_t vel_noise_ {0.5f};		///< minimum allowed observation noise for velocity fusion (m/sec)
    const scalar_t pos_noise_ {0.5f};		///< minimum allowed observation noise for position fusion (m)
    const scalar_t terrain_p_noise_{5.0f};	///< process noise for terrain offset (m/sec)
    scalar_t posObsNoiseNE_ {0.0f};	///< 1-STD observtion noise used for the fusion of NE position data (m)
    
    ///< Measurement (observation) variance
    scalar_t terrain_var_{1e4f};		///< variance of terrain position estimate (m**2)
    vec2 velObsVarNE_;		///< 1-STD observation noise variance used for the fusion of NE velocity data (m/sec)**2

    ///< initialization errors
    const scalar_t switch_on_gyro_bias_ {0.01f};	///< 1-sigma gyro bias uncertainty at switch on (rad/sec)
    const scalar_t switch_on_accel_bias_ {0.2f};	///< 1-sigma accelerometer bias uncertainty at switch on (m/sec**2)
    const scalar_t initial_tilt_err_ {0.01f};		///< 1-sigma tilt error after initial alignment using gravity vector (rad)

    ///< innovation gates
    const scalar_t range_innov_gate_{5.0f};		///< range finder fusion innovation consistency gate size (STD)
    const scalar_t range_aid_innov_gate_{1.0f}; 	///< gate size used for innovation consistency checks for range aid fusion   
    const scalar_t flow_innov_gate_{3.0f};		///< optical flow fusion innovation consistency gate size (STD)
    scalar_t heading_innov_gate_ {2.6f};		///< heading fusion innovation consistency gate size (STD)
    const scalar_t posNE_innov_gate_ {5.0f};		///< GPS horizontal position innovation consistency gate size (STD)
    const scalar_t vel_innov_gate_ {5.0f};		///< GPS velocity innovation consistency gate size (STD)
    scalar_t hvelInnovGate_ {1.0f};		///< Number of standard deviations used for the horizontal velocity fusion innovation consistency check
    scalar_t posInnovGateNE_ {1.0f};		///< Number of standard deviations used for the NE position fusion innovation consistency check

    ///< innovations
    scalar_t heading_innov_ {0.0f};     ///< innovation of the last heading measurement (rad)
    scalar_t hagl_innov_{0.0f};		///< innovation of the last height above terrain measurement (m)
    scalar_t flow_innov_[2] {};		///< flow measurement innovation (rad/sec)
    scalar_t vel_pos_innov_[6] {};	///< velocity and position innovations: (m/s and m)

    ///< innovation variances
    scalar_t hagl_innov_var_{0.0f};	///< innovation variance for the last height above terrain measurement (m**2)
    scalar_t vel_pos_innov_var_[6] {};	///< velocity and position innovation variances: (m**2)
    scalar_t flow_innov_var_[2] {};	///< flow innovation variance ((rad/sec)**2)

    ///< test ratios
    scalar_t terr_test_ratio_{0.0f};		// height above terrain measurement innovation consistency check ratio
    scalar_t vel_pos_test_ratio_[6] {};  // velocity and position innovation consistency check ratios

    ///< range specific params
    const scalar_t rng_gnd_clearance_{0.1f};		///< minimum valid value for range when on ground (m)
    const scalar_t rng_sens_pitch_{0.0f};		///< Pitch offset of the range sensor (rad). Sensor points out along Z axis when offset is zero. Positive rotation is RH about Y axis.
    scalar_t range_noise_scaler_{0.0f};		///< scaling from range measurement to noise (m/m)
    scalar_t vehicle_variance_scaler_{0.0f};	///< gain applied to vehicle height variance used in calculation of height above ground observation variance
    const scalar_t max_hagl_for_range_aid_{5.0f};	///< maximum height above ground for which we allow to use the range finder as height source (if range_aid == 1)
    const scalar_t max_vel_for_range_aid_{1.0f};	///< maximum ground velocity for which we allow to use the range finder as height source (if range_aid == 1)
    int32_t range_aid_{0};			///< allow switching primary height source to range finder if certian conditions are met
    const scalar_t range_cos_max_tilt_{0.7071f};	///< cosine of the maximum tilt angle from the vertical that permits use of range finder data
    scalar_t terrain_gradient_{0.5f};		///< gradient of terrain used to estimate process noise due to changing position (m/m)
    scalar_t terrain_vpos_{0.0f};		///< estimated vertical position of the terrain underneath the vehicle in local NED frame (m)
    scalar_t sin_tilt_rng_{0.0f};		///< sine of the range finder tilt rotation about the Y body axis
    scalar_t cos_tilt_rng_{1.0f};		///< cosine of the range finder tilt rotation about the Y body axis
    scalar_t R_rng_to_earth_2_2_{0.0f};	///< 2,2 element of the rotation matrix from sensor frame to earth frame
    bool hagl_valid_{false};		///< true when the height above ground estimate is valid

    ///< optical flow specific params 
    const scalar_t flow_max_rate_ {2.5}; ///< maximum angular flow rate that the optical flow sensor can measure (rad/s)
    int32_t flow_quality_min_ {1};
    const scalar_t flow_noise_qual_min_ {0.5f};	///< observation noise for optical flow LOS rate measurements when flow sensor quality is at the minimum useable (rad/sec)
    vec2 flow_rad_xy_comp_ {};
    vec3 flow_gyro_bias_;	///< bias errors in optical flow sensor rate gyro outputs (rad/sec)
    vec3 imu_del_ang_of_;	///< bias corrected delta angle measurements accumulated across the same time frame as the optical flow rates (rad)
    scalar_t delta_time_of_{0.0f};	///< time in sec that _imu_del_ang_of was accumulated over (sec)

    ///< mag specific params
    scalar_t mag_declination_{0.0f};
    scalar_t mag_heading_noise_{3.0e-1f};
    scalar_t mage_p_noise_{1.0e-3f};		///< process noise for earth magnetic field prediction (Gauss/sec)
    scalar_t magb_p_noise_{1.0e-4f};		///< process noise for body magnetic field prediction (Gauss/sec)
    scalar_t mag_noise_{5.0e-2f};		///< measurement noise used for 3-axis magnetoemeter fusion (Gauss)

    bool imu_updated_;
    vec3 last_known_posNED_;

    static constexpr scalar_t kOneG = 9.80665;  /// Earth gravity (m/s^2)
    static constexpr scalar_t acc_bias_lim = 0.4; ///< maximum accel bias magnitude (m/sec**2)
    static constexpr scalar_t hgt_reset_lim = 0.0f; ///< 
    static constexpr scalar_t CONSTANTS_ONE_G = 9.80665f; // m/s^2

    bool mag_use_inhibit_{false};		///< true when magnetomer use is being inhibited
    bool mag_use_inhibit_prev_{false};		///< true when magnetomer use was being inhibited the previous frame
    scalar_t last_static_yaw_{0.0f};		///< last yaw angle recorded when on ground motion checks were passing (rad)

    ///< flags on receive updates
    bool gps_data_ready_ = false;
    bool vision_data_ready_ = false;
    bool range_data_ready_ = false;
    bool flow_data_ready_ = false;
    bool mag_data_ready_ = false;

    ///< flags on fusion modes
    bool fuse_pos_ = false;
    bool fuse_height_ = false;
    bool mag_hdg_ = false;
    bool mag_3D_ = false;
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

    ///< flags on vehicle's state
    bool in_air_ = false;
    bool vehicle_at_rest_ = !in_air_; // true when the vehicle is at rest
    bool vehicle_at_rest_prev_ {false}; ///< true when the vehicle was at rest the previous time the status was checked
  };
}

#endif /* defined(ESKF_H_) */
