/**
 * @file optimal_binning_common.h
 * @brief Common constants, types, and utilities for Optimal Binning algorithms
 * 
 * This header provides unified definitions used across all binning algorithms
 * to ensure consistency and eliminate code duplication.
 */

#ifndef OPTIMAL_BINNING_COMMON_H
#define OPTIMAL_BINNING_COMMON_H

#include <Rcpp.h>
#include <vector>
#include <string>
#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>

namespace OptimalBinning {

// =============================================================================
// NUMERICAL CONSTANTS
// =============================================================================

/// Default epsilon for numerical stability (most algorithms use 1e-10)
constexpr double EPSILON = 1e-10;

/// Stricter epsilon for correlation calculations
constexpr double EPSILON_STRICT = 1e-12;

/// Maximum Weight of Evidence value to prevent extreme values
constexpr double MAX_WOE = 20.0;

/// Minimum Weight of Evidence value to prevent extreme values
constexpr double MIN_WOE = -20.0;

/// Default Bayesian prior strength for smoothing
constexpr double BAYESIAN_PRIOR_STRENGTH = 0.5;

/// Default Laplace smoothing parameter
constexpr double DEFAULT_LAPLACE_ALPHA = 0.5;

// =============================================================================
// ENUMERATIONS
// =============================================================================

/**
 * @brief Smoothing methods for WoE/IV calculations
 * 
 * NONE: Only EPSILON protection, no explicit smoothing
 * LAPLACE: Add smoothing_param to counts
 * BAYESIAN: Use Bayesian prior based on overall proportions
 */
enum class SmoothingMethod {
  NONE,
  LAPLACE,
  BAYESIAN
};

/**
 * @brief Monotonicity trend options
 * 
 * AUTO: Automatically detect trend from data
 * ASCENDING: WoE must increase with bin order
 * DESCENDING: WoE must decrease with bin order
 * NONE: No monotonicity enforcement
 */
enum class MonotonicTrend {
  AUTO,
  ASCENDING,
  DESCENDING,
  NONE
};

// =============================================================================
// CONVERSION UTILITIES
// =============================================================================

/**
 * @brief Convert string to MonotonicTrend enum
 * @param trend_str String representation ("auto", "ascending", "descending", "none")
 * @return MonotonicTrend enum value
 */
inline MonotonicTrend string_to_monotonic_trend(const std::string& trend_str) {
  if (trend_str == "ascending") return MonotonicTrend::ASCENDING;
  if (trend_str == "descending") return MonotonicTrend::DESCENDING;
  if (trend_str == "none") return MonotonicTrend::NONE;
  return MonotonicTrend::AUTO; // Default
}

/**
 * @brief Convert MonotonicTrend enum to string
 * @param trend MonotonicTrend enum value
 * @return String representation
 */
inline std::string monotonic_trend_to_string(MonotonicTrend trend) {
  switch (trend) {
    case MonotonicTrend::ASCENDING: return "ascending";
    case MonotonicTrend::DESCENDING: return "descending";
    case MonotonicTrend::NONE: return "none";
    default: return "auto";
  }
}

} // namespace OptimalBinning

#endif // OPTIMAL_BINNING_COMMON_H
