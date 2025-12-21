/**
 * @file monotonicity_utils.h
 * @brief Monotonicity detection and enforcement utilities
 *
 * Features:
 * - Welford's online algorithm for numerically stable correlation
 * - Optimized slope and trend detection
 * - Efficient monotonicity violation detection
 *
 * References:
 * - Welford, B. P. (1962). "Note on a method for calculating corrected sums of
 *   squares and products". Technometrics, 4(3), 419-420.
 */

#ifndef OPTIMAL_BINNING_MONOTONICITY_UTILS_H
#define OPTIMAL_BINNING_MONOTONICITY_UTILS_H

#include "optimal_binning_common.h"
#include "safe_math.h"
#include <vector>

namespace OptimalBinning {

/**
 * @brief Detect monotonic trend direction from WoE values (uses Welford)
 *
 * Uses Welford's algorithm for numerically stable slope calculation.
 *
 * @param woe_values Vector of WoE values in bin order
 * @return Detected trend (ASCENDING or DESCENDING)
 */
inline MonotonicTrend detect_monotonic_direction(const std::vector<double>& woe_values) {
  return detect_trend_welford_woe(woe_values);
}

/**
 * @brief Detect monotonic trend from feature-target correlation (Welford's algorithm)
 *
 * Uses Welford's online algorithm for numerically stable computation of
 * correlation coefficient. Superior to naive summation for large datasets.
 *
 * Reference:
 * Welford, B. P. (1962). "Note on a method for calculating corrected sums
 * of squares and products". Technometrics, 4(3), 419-420.
 *
 * @param feature Feature values
 * @param target Binary target values
 * @return Detected trend based on correlation
 */
template<typename T>
inline MonotonicTrend detect_trend_from_correlation(
    const std::vector<double>& feature,
    const std::vector<T>& target
) {
  if (feature.size() != target.size() || feature.empty()) {
    return MonotonicTrend::ASCENDING;
  }

  size_t n = feature.size();

  // Welford's online algorithm for mean and variance
  double mean_f = 0.0, mean_t = 0.0;
  double M2_f = 0.0, M2_t = 0.0;
  double M_ft = 0.0;  // Covariance accumulator

  for (size_t i = 0; i < n; ++i) {
    double f = feature[i];
    double t = static_cast<double>(target[i]);

    // Update means
    double delta_f = f - mean_f;
    double delta_t = t - mean_t;

    mean_f += delta_f / (i + 1);
    mean_t += delta_t / (i + 1);

    // Update sum of squared deviations
    double delta_f_new = f - mean_f;
    double delta_t_new = t - mean_t;

    M2_f += delta_f * delta_f_new;
    M2_t += delta_t * delta_t_new;

    // Update covariance accumulator
    M_ft += delta_f * delta_t_new;
  }

  // Compute variances and covariance
  if (n < 2) {
    return MonotonicTrend::ASCENDING;
  }

  double var_f = M2_f / (n - 1);
  double var_t = M2_t / (n - 1);
  double cov_ft = M_ft / (n - 1);

  // Compute correlation
  double denom = std::sqrt(var_f * var_t);
  if (denom < EPSILON) {
    return MonotonicTrend::ASCENDING;
  }

  double correlation = cov_ft / denom;

  // Clamp to [-1, 1] to handle numerical errors
  correlation = clamp(correlation, -1.0, 1.0);

  return (correlation >= 0) ? MonotonicTrend::ASCENDING : MonotonicTrend::DESCENDING;
}

/**
 * @brief Welford's algorithm variant for WoE slope detection
 *
 * Specialized version that works with pre-computed WoE values.
 *
 * @param woe_values Vector of WoE values
 * @return Detected trend direction
 */
inline MonotonicTrend detect_trend_welford_woe(const std::vector<double>& woe_values) {
  if (woe_values.size() < 2) {
    return MonotonicTrend::ASCENDING;
  }

  size_t n = woe_values.size();

  // Create index vector [0, 1, 2, ..., n-1]
  std::vector<double> indices(n);
  for (size_t i = 0; i < n; ++i) {
    indices[i] = static_cast<double>(i);
  }

  // Use Welford's algorithm
  double mean_x = 0.0, mean_y = 0.0;
  double M2_x = 0.0, M2_y = 0.0;
  double M_xy = 0.0;

  for (size_t i = 0; i < n; ++i) {
    double x = indices[i];
    double y = woe_values[i];

    double delta_x = x - mean_x;
    double delta_y = y - mean_y;

    mean_x += delta_x / (i + 1);
    mean_y += delta_y / (i + 1);

    double delta_x_new = x - mean_x;
    double delta_y_new = y - mean_y;

    M2_x += delta_x * delta_x_new;
    M2_y += delta_y * delta_y_new;
    M_xy += delta_x * delta_y_new;
  }

  // Compute slope
  if (M2_x < EPSILON) {
    return MonotonicTrend::ASCENDING;
  }

  double slope = M_xy / M2_x;

  return (slope >= 0) ? MonotonicTrend::ASCENDING : MonotonicTrend::DESCENDING;
}

/**
 * @brief Check if WoE values are monotonic
 * 
 * @param woe_values Vector of WoE values
 * @param trend Required trend direction
 * @param tolerance Tolerance for violation detection
 * @return true if monotonic in specified direction
 */
inline bool is_monotonic(
    const std::vector<double>& woe_values,
    MonotonicTrend trend,
    double tolerance = EPSILON
) {
  if (woe_values.size() < 2) return true;
  if (trend == MonotonicTrend::NONE) return true;
  
  bool ascending = (trend == MonotonicTrend::ASCENDING);
  
  for (size_t i = 1; i < woe_values.size(); ++i) {
    if (ascending) {
      if (woe_values[i] < woe_values[i-1] - tolerance) {
        return false;
      }
    } else {
      if (woe_values[i] > woe_values[i-1] + tolerance) {
        return false;
      }
    }
  }
  
  return true;
}

/**
 * @brief Find index of first monotonicity violation
 * 
 * @param woe_values Vector of WoE values
 * @param trend Required trend direction
 * @return Index of first violation, or -1 if none
 */
inline int find_monotonicity_violation(
    const std::vector<double>& woe_values,
    MonotonicTrend trend
) {
  if (woe_values.size() < 2) return -1;
  if (trend == MonotonicTrend::NONE) return -1;
  
  bool ascending = (trend == MonotonicTrend::ASCENDING);
  
  for (size_t i = 1; i < woe_values.size(); ++i) {
    if (ascending && woe_values[i] < woe_values[i-1] - EPSILON) {
      return static_cast<int>(i);
    }
    if (!ascending && woe_values[i] > woe_values[i-1] + EPSILON) {
      return static_cast<int>(i);
    }
  }
  
  return -1;
}

/**
 * @brief Find the pair of adjacent bins with smallest WoE difference
 * 
 * Used for merging bins to reduce count while preserving IV.
 * 
 * @param woe_values Vector of WoE values
 * @return Index of first bin in the pair with smallest difference
 */
inline int find_smallest_woe_diff(const std::vector<double>& woe_values) {
  if (woe_values.size() < 2) return -1;
  
  double min_diff = std::numeric_limits<double>::max();
  int min_idx = 0;
  
  for (size_t i = 0; i < woe_values.size() - 1; ++i) {
    double diff = std::abs(woe_values[i+1] - woe_values[i]);
    if (diff < min_diff) {
      min_diff = diff;
      min_idx = static_cast<int>(i);
    }
  }
  
  return min_idx;
}

} // namespace OptimalBinning

#endif // OPTIMAL_BINNING_MONOTONICITY_UTILS_H
