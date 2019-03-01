#ifndef COMMON_H_
#define COMMON_H_

#include <Eigen/Core>
#include <Eigen/Dense>

#define EV_MAX_INTERVAL		2e5	///< Maximum allowable time interval between external vision system measurements (uSec)
#define GPS_MAX_INTERVAL	1e6	///< Maximum allowable time interval between external gps system measurements (uSec)
#define OPTICAL_FLOW_INTERVAL	2e5	///< Maximum allowable time interval between optical flow system measurements (uSec)
#define MAG_INTERVAL		2e5	///< Maximum allowable time interval between mag system measurements (uSec)
#define RANGE_MAX_INTERVAL	1e6	///< Maximum allowable time interval between rangefinder system measurements (uSec)

#define MASK_EV_POS 1<<0
#define MASK_EV_YAW 1<<1
#define MASK_EV_HGT 1<<2
#define MASK_GPS_POS 1<<3
#define MASK_GPS_VEL 1<<4
#define MASK_GPS_HGT 1<<5
#define MASK_MAG_INHIBIT 1<<6
#define MASK_OPTICAL_FLOW 1<<7
#define MASK_RANGEFINDER 1<<8
#define MASK_MAG_HEADING 1<<9
#define MASK_EV (MASK_EV_POS | MASK_EV_YAW | MASK_EV_HGT)
#define MASK_GPS (MASK_GPS_POS | MASK_GPS_VEL | MASK_GPS_HGT)

// GPS pre-flight check bit locations
#define MASK_GPS_NSATS  (1<<0)
#define MASK_GPS_GDOP   (1<<1)
#define MASK_GPS_HACC   (1<<2)
#define MASK_GPS_VACC   (1<<3)
#define MASK_GPS_SACC   (1<<4)
#define MASK_GPS_HDRIFT (1<<5)
#define MASK_GPS_VDRIFT (1<<6)
#define MASK_GPS_HSPD   (1<<7)
#define MASK_GPS_VSPD   (1<<8)

namespace eskf {

  typedef float scalar_t;
  typedef Eigen::Matrix<scalar_t, 3, 1> vec3; /// Vector in R3
  typedef Eigen::Matrix<scalar_t, 2, 1> vec2; /// Vector in R2
  typedef Eigen::Matrix<scalar_t, 3, 3> mat3; /// Matrix in R3
  typedef Eigen::Quaternion<scalar_t> quat;   /// Member of S4

  /* State vector:
   * Attitude quaternion
   * NED velocity
   * NED position
   * Delta Angle bias - rad
   * Delta Velocity bias - m/s
  */

  struct state {
    quat  quat_nominal; ///< quaternion defining the rotaton from NED to XYZ frame
    vec3  vel;          ///< NED velocity in earth frame in m/s
    vec3  pos;          ///< NED position in earth frame in m
    vec3  gyro_bias;    ///< delta angle bias estimate in rad
    vec3  accel_bias;   ///< delta velocity bias estimate in m/s
    vec3  mag_I;	///< NED earth magnetic field in gauss
    vec3  mag_B;	///< magnetometer bias estimate in body frame in gauss
  };

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

  struct magSample {
    vec3     mag;	///< NED magnetometer body frame measurements (Gauss)
    uint64_t   time_us;	///< timestamp of the measurement (uSec)
  };

  struct gps_message {
    uint64_t time_usec;
    int32_t lat;		///< Latitude in 1E-7 degrees
    int32_t lon;		///< Longitude in 1E-7 degrees
    int32_t alt;		///< Altitude in 1E-3 meters (millimeters) above MSL
    float yaw;		///< yaw angle. NaN if not set (used for dual antenna GPS), (rad, [-PI, PI])
    float yaw_offset;	///< Heading/Yaw offset for dual antenna GPS - refer to description for GPS_YAW_OFFSET
    uint8_t fix_type;	///< 0-1: no fix, 2: 2D fix, 3: 3D fix, 4: RTCM code differential, 5: Real-Time Kinematic
    float eph;		///< GPS horizontal position accuracy in m
    float epv;		///< GPS vertical position accuracy in m
    float sacc;		///< GPS speed accuracy in m/s
    float vel_m_s;		///< GPS ground speed (m/sec)
    float vel_ned[3];	///< GPS ground speed NED
    bool vel_ned_valid;	///< GPS ground speed is valid
    uint8_t nsats;		///< number of satellites used
    float gdop;		///< geometric dilution of precision
  };

  // publish the status of various GPS quality checks
  union gps_check_fail_status_u {
    struct {
      uint16_t fix    : 1; ///< 0 - true if the fix type is insufficient (no 3D solution)
      uint16_t nsats  : 1; ///< 1 - true if number of satellites used is insufficient
      uint16_t gdop   : 1; ///< 2 - true if geometric dilution of precision is insufficient
      uint16_t hacc   : 1; ///< 3 - true if reported horizontal accuracy is insufficient
      uint16_t vacc   : 1; ///< 4 - true if reported vertical accuracy is insufficient
      uint16_t sacc   : 1; ///< 5 - true if reported speed accuracy is insufficient
      uint16_t hdrift : 1; ///< 6 - true if horizontal drift is excessive (can only be used when stationary on ground)
      uint16_t vdrift : 1; ///< 7 - true if vertical drift is excessive (can only be used when stationary on ground)
      uint16_t hspeed : 1; ///< 8 - true if horizontal speed is excessive (can only be used when stationary on ground)
      uint16_t vspeed : 1; ///< 9 - true if vertical speed error is excessive
    } flags;
    uint16_t value;
  };

  ///< common math functions 
  template <typename T> inline T sq(T var) {
    return var * var;
  }

  template <typename T> inline T max(T val1, T val2) {
    return (val1 > val2) ? val1 : val2;
  }

  template<typename Scalar>
  static inline constexpr const Scalar &constrain(const Scalar &val, const Scalar &min_val, const Scalar &max_val) {
    return (val < min_val) ? min_val : ((val > max_val) ? max_val : val);
  }

  template<typename Type>
  Type wrap_pi(Type x) {
    while (x >= Type(M_PI)) {
      x -= Type(2.0 * M_PI);
    }

    while (x < Type(-M_PI)) {
      x += Type(2.0 * M_PI);
    }
    return x;
  }

  ///< tf functions
  inline quat from_axis_angle(const vec3 &axis, scalar_t theta) {
    quat q;

    if (theta < scalar_t(1e-10)) {
      q.w() = scalar_t(1.0);
      q.x() = q.y() = q.z() = 0;
    }

    scalar_t magnitude = sin(theta / 2.0f);

    q.w() = cos(theta / 2.0f);
    q.x() = axis(0) * magnitude;
    q.y() = axis(1) * magnitude;
    q.z() = axis(2) * magnitude;
    
    return q;
  }

  inline quat from_axis_angle(vec3 vec) {
    quat q;
    scalar_t theta = vec.norm();

    if (theta < scalar_t(1e-10)) {
      q.w() = scalar_t(1.0);
      q.x() = q.y() = q.z() = 0;
      return q;
    }

    vec3 tmp = vec / theta;
    return from_axis_angle(tmp, theta);
  }

  inline vec3 to_axis_angle(const quat& q) {
    scalar_t axis_magnitude = scalar_t(sqrt(q.x() * q.x() + q.y() * q.y() + q.z() * q.z()));
    vec3 vec;
    vec(0) = q.x();
    vec(1) = q.y();
    vec(2) = q.z();

    if (axis_magnitude >= scalar_t(1e-10)) {
      vec = vec / axis_magnitude;
      vec = vec * wrap_pi(scalar_t(2.0) * atan2(axis_magnitude, q.w()));
    }

    return vec;
  }

  inline mat3 quat2dcm(const quat& q) {
    mat3 dcm;
    scalar_t a = q.w();
    scalar_t b = q.x();
    scalar_t c = q.y();
    scalar_t d = q.z();
    scalar_t aSq = a * a;
    scalar_t bSq = b * b;
    scalar_t cSq = c * c;
    scalar_t dSq = d * d;
    dcm(0, 0) = aSq + bSq - cSq - dSq;
    dcm(0, 1) = 2 * (b * c - a * d);
    dcm(0, 2) = 2 * (a * c + b * d);
    dcm(1, 0) = 2 * (b * c + a * d);
    dcm(1, 1) = aSq - bSq + cSq - dSq;
    dcm(1, 2) = 2 * (c * d - a * b);
    dcm(2, 0) = 2 * (b * d - a * c);
    dcm(2, 1) = 2 * (a * b + c * d);
    dcm(2, 2) = aSq - bSq - cSq + dSq;
    return dcm;
  }

  inline vec3 dcm2vec(const mat3& dcm) {
    scalar_t phi_val = atan2(dcm(2, 1), dcm(2, 2));
    scalar_t theta_val = asin(-dcm(2, 0));
    scalar_t psi_val = atan2(dcm(1, 0), dcm(0, 0));
    scalar_t pi = M_PI;

    if (fabs(theta_val - pi / 2) < 1.0e-3) {
      phi_val = 0.0;
      psi_val = atan2(dcm(1, 2), dcm(0, 2));
    } else if (fabs(theta_val + pi / 2) < 1.0e-3) {
      phi_val = 0.0;
      psi_val = atan2(-dcm(1, 2), -dcm(0, 2));
    }
    return vec3(phi_val, theta_val, psi_val);
  }

  // calculate the inverse rotation matrix from a quaternion rotation
  inline mat3 quat_to_invrotmat(const quat &q) {
    scalar_t q00 = q.w() * q.w();
    scalar_t q11 = q.x() * q.x();
    scalar_t q22 = q.y() * q.y();
    scalar_t q33 = q.z() * q.z();
    scalar_t q01 = q.w() * q.x();
    scalar_t q02 = q.w() * q.y();
    scalar_t q03 = q.w() * q.z();
    scalar_t q12 = q.x() * q.y();
    scalar_t q13 = q.x() * q.z();
    scalar_t q23 = q.y() * q.z();

    mat3 dcm;
    dcm(0, 0) = q00 + q11 - q22 - q33;
    dcm(1, 1) = q00 - q11 + q22 - q33;
    dcm(2, 2) = q00 - q11 - q22 + q33;
    dcm(0, 1) = 2.0f * (q12 - q03);
    dcm(0, 2) = 2.0f * (q13 + q02);
    dcm(1, 0) = 2.0f * (q12 + q03);
    dcm(1, 2) = 2.0f * (q23 - q01);
    dcm(2, 0) = 2.0f * (q13 - q02);
    dcm(2, 1) = 2.0f * (q23 + q01);
    
    return dcm;
  }

  /**
   * Constructor from euler angles
   *
   * This sets the instance to a quaternion representing coordinate transformation from
   * frame 2 to frame 1 where the rotation from frame 1 to frame 2 is described
   * by a 3-2-1 intrinsic Tait-Bryan rotation sequence.
   *
   * @param euler euler angle instance
   */
  inline quat from_euler(const vec3& euler) {
    quat q;
    scalar_t cosPhi_2 = cos(euler(0) / 2.0);
    scalar_t cosTheta_2 = cos(euler(1) / 2.0);
    scalar_t cosPsi_2 = cos(euler(2) / 2.0);
    scalar_t sinPhi_2 = sin(euler(0) / 2.0);
    scalar_t sinTheta_2 = sin(euler(1) / 2.0);
    scalar_t sinPsi_2 = sin(euler(2) / 2.0);
    q.w() = cosPhi_2 * cosTheta_2 * cosPsi_2 +
           sinPhi_2 * sinTheta_2 * sinPsi_2;
    q.x() = sinPhi_2 * cosTheta_2 * cosPsi_2 -
           cosPhi_2 * sinTheta_2 * sinPsi_2;
    q.y() = cosPhi_2 * sinTheta_2 * cosPsi_2 +
           sinPhi_2 * cosTheta_2 * sinPsi_2;
    q.z() = cosPhi_2 * cosTheta_2 * sinPsi_2 -
           sinPhi_2 * sinTheta_2 * cosPsi_2;
    return q;
  }
}

#endif