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

/**
 * @brief Grid on which a Gaussian KDE was evaluated, and the density on it.
 *
 * Callers that only need the density *at the sample points* should use
 * gaussian_kde_sorted(). Callers that differentiate the density need the grid
 * itself: interpolating a smooth curve onto the sample points makes it
 * piecewise linear, so finite differences between adjacent observations are
 * constant inside a segment and the sign changes that mark extrema and
 * inflection points disappear.
 */
struct KdeGrid {
  std::vector<double> x;
  std::vector<double> density;
};

inline KdeGrid gaussian_kde_grid(const std::vector<double>& sorted_x,
                                 double bandwidth,
                                 std::size_t grid_size = 512);

inline KdeGrid gaussian_kde_grid(const std::vector<double>& sorted_x,
                                 double bandwidth,
                                 std::size_t grid_size) {
  KdeGrid out;
  const std::size_t n = sorted_x.size();
  if (n == 0 || !(bandwidth > 0.0)) {
    return out;
  }

  const double lo = sorted_x.front();
  const double hi = sorted_x.back();
  const double span = hi - lo;

  // A degenerate range collapses the grid to a single point; the kernel then
  // contributes its peak there, which is what the double loop also produced.
  if (!(span > EPSILON)) {
    out.x.assign(1, lo);
    out.density.assign(1, 1.0 / (bandwidth * std::sqrt(2.0 * M_PI)));
    return out;
  }

  std::size_t G = grid_size < 16 ? 16 : grid_size;
  const double dx = span / static_cast<double>(G - 1);

  // Linear binning: split each observation's weight between its two
  // neighbouring grid points, which preserves the first moment of the sample.
  std::vector<double> mass(G, 0.0);
  for (std::size_t i = 0; i < n; ++i) {
    double pos = (sorted_x[i] - lo) / dx;
    if (pos < 0.0) pos = 0.0;
    if (pos > static_cast<double>(G - 1)) pos = static_cast<double>(G - 1);
    const std::size_t g0 = static_cast<std::size_t>(pos);
    const double frac = pos - static_cast<double>(g0);
    if (g0 + 1 < G) {
      mass[g0] += 1.0 - frac;
      mass[g0 + 1] += frac;
    } else {
      mass[G - 1] += 1.0;
    }
  }

  // Kernel weights depend only on grid distance, so they are computed once.
  std::vector<double> kern(G, 0.0);
  for (std::size_t d = 0; d < G; ++d) {
    const double z = (static_cast<double>(d) * dx) / bandwidth;
    kern[d] = std::exp(-0.5 * z * z);
  }

  const double norm = static_cast<double>(n) * bandwidth * std::sqrt(2.0 * M_PI);
  out.x.resize(G);
  out.density.resize(G);
  for (std::size_t g = 0; g < G; ++g) {
    double acc = 0.0;
    for (std::size_t k = 0; k < G; ++k) {
      if (mass[k] == 0.0) continue;
      acc += mass[k] * kern[g > k ? g - k : k - g];
    }
    out.x[g] = lo + static_cast<double>(g) * dx;
    out.density[g] = acc / norm;
  }

  return out;
}

/**
 * @brief Gaussian kernel density estimate at the sample points, in linear time.
 *
 * Both OBN_LDB and OBN_LPDB estimated the density by evaluating the Gaussian
 * kernel of every observation against every other one -- an O(n^2) double loop
 * with an exp() call in its body, written twice. Measured scaling was n^2.00
 * for both, which put a single 10^6-row variable at roughly 72 minutes and made
 * the two algorithms unusable at any realistic size. It now lives once, here.
 *
 * The estimator is the standard linear-binning one that R's own density() uses:
 * distribute the sample onto a grid, convolve, interpolate back. Cost is
 * O(n + G^2) with G fixed, so the sort that precedes it dominates again.
 *
 * The result is a grid approximation rather than a bit-identical reproduction
 * of the double loop. Callers that *differentiate* the density must use
 * gaussian_kde_grid() instead: interpolating onto the sample points makes the
 * curve piecewise linear, and finite differences taken between adjacent
 * observations then vanish inside each segment.
 *
 * @param sorted_x   Sample values, ascending. Not re-sorted here.
 * @param bandwidth  Kernel bandwidth; must be positive.
 * @param grid_size  Grid points spanning the sample range.
 * @return Density at each element of sorted_x, same length and normalisation
 *   as the double loop it replaces.
 */
inline std::vector<double> gaussian_kde_sorted(const std::vector<double>& sorted_x,
                                               double bandwidth,
                                               std::size_t grid_size = 512) {
  const std::size_t n = sorted_x.size();
  std::vector<double> density(n, 0.0);
  if (n <= 1 || !(bandwidth > 0.0)) {
    return density;
  }

  const KdeGrid grid = gaussian_kde_grid(sorted_x, bandwidth, grid_size);
  if (grid.x.empty()) {
    return density;
  }
  if (grid.x.size() == 1) {
    std::fill(density.begin(), density.end(), grid.density[0]);
    return density;
  }

  const double lo = grid.x.front();
  const double dx = grid.x[1] - grid.x[0];
  const std::size_t G = grid.x.size();
  for (std::size_t i = 0; i < n; ++i) {
    double pos = (sorted_x[i] - lo) / dx;
    if (pos < 0.0) pos = 0.0;
    if (pos > static_cast<double>(G - 1)) pos = static_cast<double>(G - 1);
    const std::size_t g0 = static_cast<std::size_t>(pos);
    const double frac = pos - static_cast<double>(g0);
    density[i] = (g0 + 1 < G)
      ? grid.density[g0] * (1.0 - frac) + grid.density[g0 + 1] * frac
      : grid.density[G - 1];
  }

  return density;
}

} // namespace OptimalBinning

#endif // OPTIMAL_BINNING_COMMON_H
