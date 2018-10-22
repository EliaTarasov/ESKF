#ifndef GEO_H_
#define GEO_H_

#include <stdbool.h>
#include <stdint.h>

static constexpr double CONSTANTS_RADIUS_OF_EARTH = 6371000;					// meters (m)
static constexpr float  CONSTANTS_RADIUS_OF_EARTH_F = CONSTANTS_RADIUS_OF_EARTH;		// meters (m)

/* lat/lon are in radians */
struct map_projection_reference_s {
	uint64_t timestamp;
	double lat_rad;
	double lon_rad;
	double sin_lat;
	double cos_lat;
	bool init_done;
};

struct globallocal_converter_reference_s {
	float alt;
	bool init_done;
};

/**
 * Checks if global projection was initialized
 * @return true if map was initialized before, false else
 */
bool map_projection_global_initialized();

/**
 * Checks if projection given as argument was initialized
 * @return true if map was initialized before, false else
 */
bool map_projection_initialized(const struct map_projection_reference_s *ref);

/**
 * Get the timestamp of the global map projection
 * @return the timestamp of the map_projection
 */
uint64_t map_projection_global_timestamp(void);

/**
 * Get the timestamp of the map projection given by the argument
 * @return the timestamp of the map_projection
 */
uint64_t map_projection_timestamp(const struct map_projection_reference_s *ref);

/**
 * Writes the reference values of the global projection to ref_lat and ref_lon
 * @return 0 if map_projection_init was called before, -1 else
 */
int map_projection_global_reference(double *ref_lat_rad, double *ref_lon_rad);

/**
 * Writes the reference values of the projection given by the argument to ref_lat and ref_lon
 * @return 0 if map_projection_init was called before, -1 else
 */
int map_projection_reference(const struct map_projection_reference_s *ref, double *ref_lat_rad, double *ref_lon_rad);

/**
 * Initializes the global map transformation.
 *
 * Initializes the transformation between the geographic coordinate system and
 * the azimuthal equidistant plane
 * @param lat in degrees (47.1234567°, not 471234567°)
 * @param lon in degrees (8.1234567°, not 81234567°)
 */
int map_projection_global_init(double lat_0, double lon_0, uint64_t timestamp);

/**
 * Initializes the map transformation given by the argument.
 *
 * Initializes the transformation between the geographic coordinate system and
 * the azimuthal equidistant plane
 * @param lat in degrees (47.1234567°, not 471234567°)
 * @param lon in degrees (8.1234567°, not 81234567°)
 */
int map_projection_init_timestamped(struct map_projection_reference_s *ref, double lat_0, double lon_0, uint64_t timestamp);

/**
 * Initializes the map transformation given by the argument and sets the timestamp to now.
 *
 * Initializes the transformation between the geographic coordinate system and
 * the azimuthal equidistant plane
 * @param lat in degrees (47.1234567°, not 471234567°)
 * @param lon in degrees (8.1234567°, not 81234567°)
 */
int map_projection_init(struct map_projection_reference_s *ref, double lat_0, double lon_0);

/**
 * Transforms a point in the geographic coordinate system to the local
 * azimuthal equidistant plane using the global projection
 * @param x north
 * @param y east
 * @param lat in degrees (47.1234567°, not 471234567°)
 * @param lon in degrees (8.1234567°, not 81234567°)
 * @return 0 if map_projection_init was called before, -1 else
 */
int map_projection_global_project(double lat, double lon, float *x, float *y);

/* Transforms a point in the geographic coordinate system to the local
 * azimuthal equidistant plane using the projection given by the argument
* @param x north
* @param y east
* @param lat in degrees (47.1234567°, not 471234567°)
* @param lon in degrees (8.1234567°, not 81234567°)
* @return 0 if map_projection_init was called before, -1 else
*/
int map_projection_project(const struct map_projection_reference_s *ref, double lat, double lon, float *x, float *y);

/**
 * Transforms a point in the local azimuthal equidistant plane to the
 * geographic coordinate system using the global projection
 *
 * @param x north
 * @param y east
 * @param lat in degrees (47.1234567°, not 471234567°)
 * @param lon in degrees (8.1234567°, not 81234567°)
 * @return 0 if map_projection_init was called before, -1 else
 */
int map_projection_global_reproject(float x, float y, double *lat, double *lon);

/**
 * Transforms a point in the local azimuthal equidistant plane to the
 * geographic coordinate system using the projection given by the argument
 *
 * @param x north
 * @param y east
 * @param lat in degrees (47.1234567°, not 471234567°)
 * @param lon in degrees (8.1234567°, not 81234567°)
 * @return 0 if map_projection_init was called before, -1 else
 */
int map_projection_reproject(const struct map_projection_reference_s *ref, float x, float y, double *lat, double *lon);

/**
 * Get reference position of the global map projection
 */
int map_projection_global_getref(double *lat_0, double *lon_0);

/**
 * Initialize the global mapping between global position (spherical) and local position (NED).
 */
int globallocalconverter_init(double lat_0, double lon_0, float alt_0, uint64_t timestamp);

/**
 * Checks if globallocalconverter was initialized
 * @return true if map was initialized before, false else
 */
bool globallocalconverter_initialized(void);

/**
 * Convert from global position coordinates to local position coordinates using the global reference
 */
int globallocalconverter_tolocal(double lat, double lon, float alt, float *x, float *y, float *z);

/**
 * Convert from local position coordinates to global position coordinates using the global reference
 */
int globallocalconverter_toglobal(float x, float y, float z,  double *lat, double *lon, float *alt);

/**
 * Get reference position of the global to local converter
 */
int globallocalconverter_getref(double *lat_0, double *lon_0, float *alt_0);

#endif