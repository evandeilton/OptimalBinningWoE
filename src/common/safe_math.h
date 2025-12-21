/**
 * @file safe_math.h
 * @brief Numerically stable mathematical operations
 *
 * Provides high-precision mathematical operations with:
 * - Underflow/overflow protection
 * - Compensated summation algorithms (Kahan, Neumaier)
 * - Log-space arithmetic (log-sum-exp trick)
 * - Constexpr functions for compile-time optimization
 */

#ifndef OPTIMAL_BINNING_SAFE_MATH_H
#define OPTIMAL_BINNING_SAFE_MATH_H

#include "optimal_binning_common.h"
#include <vector>
#include <algorithm>
#include <numeric>

namespace OptimalBinning {

// =============================================================================
// BASIC SAFE OPERATIONS (constexpr for compile-time optimization)
// =============================================================================

/**
 * @brief Safe logarithm with underflow protection
 * @param x Input value
 * @param epsilon Minimum value before log (default: EPSILON)
 * @return log(max(x, epsilon))
 */
constexpr double safe_log(double x, double epsilon = EPSILON) {
  return std::log(x > epsilon ? x : epsilon);
}

/**
 * @brief Safe exponential with overflow protection
 * @param x Input value
 * @param max_exp Maximum exponent (default: 700.0 for double)
 * @return exp(min(x, max_exp))
 */
constexpr double safe_exp(double x, double max_exp = 700.0) {
  return std::exp(x < max_exp ? x : max_exp);
}

/**
 * @brief Safe division with zero denominator protection
 * @param num Numerator
 * @param denom Denominator
 * @param epsilon Minimum denominator magnitude
 * @return num/denom with protection against division by zero
 */
constexpr double safe_divide(double num, double denom, double epsilon = EPSILON) {
  return (std::abs(denom) > epsilon) ? (num / denom) : 0.0;
}

/**
 * @brief Clamp value to range [min_val, max_val]
 * @param value Value to clamp
 * @param min_val Minimum allowed value
 * @param max_val Maximum allowed value
 * @return Clamped value
 */
constexpr double clamp(double value, double min_val, double max_val) {
  return (value < min_val) ? min_val : ((value > max_val) ? max_val : value);
}

/**
 * @brief Check if a value is finite and not NaN
 * @param value Value to check
 * @return true if value is finite and not NaN
 */
constexpr bool is_valid_number(double value) {
  return std::isfinite(value) && !std::isnan(value);
}

/**
 * @brief Adaptive epsilon based on value scale
 * @param value Reference value for scale
 * @return Epsilon appropriate for the value's magnitude
 */
constexpr double adaptive_epsilon(double value) {
  double scale_eps = std::numeric_limits<double>::epsilon() * std::abs(value);
  return (scale_eps > EPSILON) ? scale_eps : EPSILON;
}

// =============================================================================
// LOG-SPACE ARITHMETIC (prevents overflow in exp operations)
// =============================================================================

/**
 * @brief Log-sum-exp trick for numerical stability
 *
 * Computes log(exp(x) + exp(y)) without overflow.
 *
 * Standard formula:
 *   log(exp(x) + exp(y)) can overflow if x or y is large
 *
 * Stable formula:
 *   max_val = max(x, y)
 *   log(exp(x) + exp(y)) = max_val + log(exp(x - max_val) + exp(y - max_val))
 *
 * @param x First log-value
 * @param y Second log-value
 * @return log(exp(x) + exp(y))
 */
inline double log_sum_exp(double x, double y) {
  double max_val = std::max(x, y);
  return max_val + std::log(std::exp(x - max_val) + std::exp(y - max_val));
}

/**
 * @brief Log-sum-exp for vector of log-values
 *
 * Computes log(Σ exp(log_values[i])) without overflow.
 *
 * @param log_values Vector of values in log-space
 * @return log(Σ exp(log_values[i]))
 */
inline double log_sum_exp(const std::vector<double>& log_values) {
  if (log_values.empty()) return -std::numeric_limits<double>::infinity();
  if (log_values.size() == 1) return log_values[0];

  double max_val = *std::max_element(log_values.begin(), log_values.end());

  // Handle -infinity case
  if (std::isinf(max_val) && max_val < 0) {
    return max_val;
  }

  double sum = 0.0;
  for (double lv : log_values) {
    sum += std::exp(lv - max_val);
  }

  return max_val + std::log(sum);
}

// =============================================================================
// COMPENSATED SUMMATION (Kahan & Neumaier algorithms)
// =============================================================================

/**
 * @brief Kahan summation algorithm for compensated sum
 *
 * Reduces numerical error in sum from O(nε) to O(ε).
 *
 * Reference:
 * Kahan, W. (1965). "Further remarks on reducing truncation errors".
 * Communications of the ACM, 8(1), 40.
 *
 * @param values Vector of values to sum
 * @return Compensated sum with reduced error
 */
inline double kahan_sum(const std::vector<double>& values) {
  if (values.empty()) return 0.0;

  double sum = 0.0;
  double c = 0.0;  // Running compensation for lost low-order bits

  for (double value : values) {
    double y = value - c;    // Compensated value
    double t = sum + y;      // New sum
    c = (t - sum) - y;       // Update compensation
    sum = t;
  }

  return sum;
}

/**
 * @brief Neumaier variant of Kahan summation (more robust)
 *
 * Handles cases where Kahan summation can still accumulate error.
 * Generally more accurate than Kahan at small additional cost.
 *
 * Reference:
 * Neumaier, A. (1974). "Rundungsfehleranalyse einiger Verfahren
 * zur Summation endlicher Summen". ZAMM, 54, 39-51.
 *
 * @param values Vector of values to sum
 * @return Compensated sum with minimal error
 */
inline double neumaier_sum(const std::vector<double>& values) {
  if (values.empty()) return 0.0;

  double sum = values[0];
  double c = 0.0;

  for (size_t i = 1; i < values.size(); ++i) {
    double t = sum + values[i];

    if (std::abs(sum) >= std::abs(values[i])) {
      c += (sum - t) + values[i];  // sum is larger
    } else {
      c += (values[i] - t) + sum;  // values[i] is larger
    }

    sum = t;
  }

  return sum + c;
}

/**
 * @brief Choose best summation algorithm based on context
 *
 * Uses Neumaier for high-precision needs, Kahan for speed,
 * standard sum for very small vectors.
 *
 * @param values Vector of values to sum
 * @param high_precision If true, use Neumaier; else use Kahan
 * @return Compensated sum
 */
inline double compensated_sum(const std::vector<double>& values,
                              bool high_precision = false) {
  if (values.size() < 10) {
    // For small vectors, compensation overhead not worth it
    return std::accumulate(values.begin(), values.end(), 0.0);
  }

  return high_precision ? neumaier_sum(values) : kahan_sum(values);
}

} // namespace OptimalBinning

#endif // OPTIMAL_BINNING_SAFE_MATH_H
