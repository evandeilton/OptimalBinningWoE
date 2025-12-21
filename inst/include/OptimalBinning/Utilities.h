#ifndef OPTIMAL_BINNING_UTILITIES_H
#define OPTIMAL_BINNING_UTILITIES_H

#include <cmath>
#include <algorithm>
#include <limits>

namespace OptimalBinning {

// Constants
constexpr double EPSILON = 1e-10;
constexpr double MAX_WOE = 20.0;   // Cap for numerical stability
constexpr double MIN_WOE = -20.0;  // Cap for numerical stability
constexpr double DEFAULT_LAPLACE_SMOOTHING = 0.5;

/**
 * @brief Safe logarithm with underflow protection
 * @param x Input value
 * @param epsilon Minimum value before log (default: 1e-12)
 * @return log(max(x, epsilon))
 */
inline double safe_log(double x, double epsilon = 1e-12) {
  return std::log(std::max(x, epsilon));
}

/**
 * @brief Safe division with zero denominator protection
 * @param num Numerator
 * @param denom Denominator
 * @param epsilon Minimum denominator magnitude (default: 1e-12)
 * @return num/denom with protection against division by zero
 */
inline double safe_divide(double num, double denom, double epsilon = 1e-12) {
  if (std::abs(denom) < epsilon) {
    return 0.0;
  }
  return num / denom;
}

/**
 * @brief Clamp value to range [min_val, max_val]
 * @param value Value to clamp
 * @param min_val Minimum allowed value
 * @param max_val Maximum allowed value
 * @return Clamped value
 */
inline double clamp(double value, double min_val, double max_val) {
  return std::max(min_val, std::min(value, max_val));
}

/**
 * @brief Calculate Weight of Evidence (WoE) with Laplace smoothing
 * 
 * WoE = ln((P(X|Y=1) / P(X|Y=0)))
 * 
 * @param count_pos Count of positives in bin
 * @param count_neg Count of negatives in bin
 * @param total_pos Total positives in dataset
 * @param total_neg Total negatives in dataset
 * @param laplace_smoothing Smoothing parameter (default: 0.5)
 * @return Weight of Evidence value (capped between MIN_WOE and MAX_WOE)
 */
inline double calculate_woe(int count_pos, int count_neg, 
                            int total_pos, int total_neg,
                            double laplace_smoothing = DEFAULT_LAPLACE_SMOOTHING) {
  // Apply Laplace smoothing
  double smoothed_pos = static_cast<double>(count_pos) + laplace_smoothing;
  double smoothed_neg = static_cast<double>(count_neg) + laplace_smoothing;
  
  double total_smoothed_pos = static_cast<double>(total_pos) + laplace_smoothing;
  double total_smoothed_neg = static_cast<double>(total_neg) + laplace_smoothing;
  
  double dist_pos = smoothed_pos / total_smoothed_pos;
  double dist_neg = smoothed_neg / total_smoothed_neg;
  
  // Calculate WoE with protection against edge cases
  if (dist_pos <= 0.0 && dist_neg <= 0.0) {
    return 0.0;
  } else if (dist_pos <= 0.0) {
    return MIN_WOE;  // Capped negative infinity
  } else if (dist_neg <= 0.0) {
    return MAX_WOE;  // Capped positive infinity
  } else {
    double woe = std::log(dist_pos / dist_neg);
    return clamp(woe, MIN_WOE, MAX_WOE);
  }
}

/**
 * @brief Calculate Information Value (IV) contribution for a bin
 * 
 * IV_i = (P(X_i|Y=1) - P(X_i|Y=0)) * WoE_i
 * 
 * @param woe Weight of Evidence for the bin
 * @param count_pos Count of positives in bin
 * @param count_neg Count of negatives in bin
 * @param total_pos Total positives in dataset
 * @param total_neg Total negatives in dataset
 * @param laplace_smoothing Smoothing parameter (default: 0.5)
 * @return Information Value contribution
 */
inline double calculate_iv(double woe, int count_pos, int count_neg,
                          int total_pos, int total_neg,
                          double laplace_smoothing = DEFAULT_LAPLACE_SMOOTHING) {
  if (!std::isfinite(woe)) {
    return 0.0;
  }
  
  // Apply Laplace smoothing
  double smoothed_pos = static_cast<double>(count_pos) + laplace_smoothing;
  double smoothed_neg = static_cast<double>(count_neg) + laplace_smoothing;
  
  double total_smoothed_pos = static_cast<double>(total_pos) + laplace_smoothing;
  double total_smoothed_neg = static_cast<double>(total_neg) + laplace_smoothing;
  
  double dist_pos = smoothed_pos / total_smoothed_pos;
  double dist_neg = smoothed_neg / total_smoothed_neg;
  
  return (dist_pos - dist_neg) * woe;
}

/**
 * @brief Check if WoE values are monotonic
 * 
 * @param woe_values Vector of WoE values
 * @return true if monotonically increasing or decreasing
 */
template<typename Container>
bool is_monotonic(const Container& woe_values) {
  if (woe_values.size() <= 1) {
    return true;
  }
  
  bool increasing = true;
  bool decreasing = true;
  
  for (size_t i = 1; i < woe_values.size(); ++i) {
    if (woe_values[i] < woe_values[i-1]) {
      increasing = false;
    }
    if (woe_values[i] > woe_values[i-1]) {
      decreasing = false;
    }
    if (!increasing && !decreasing) {
      return false;
    }
  }
  
  return true;
}

/**
 * @brief Calculate chi-square statistic between two bins
 * 
 * @param bin1_pos Positive count in bin 1
 * @param bin1_neg Negative count in bin 1
 * @param bin2_pos Positive count in bin 2
 * @param bin2_neg Negative count in bin 2
 * @param use_continuity_correction Whether to apply continuity correction
 * @return Chi-square statistic
 */
inline double calculate_chi_square(int bin1_pos, int bin1_neg,
                                   int bin2_pos, int bin2_neg,
                                   bool use_continuity_correction = true) {
  const int total_pos = bin1_pos + bin2_pos;
  const int total_neg = bin1_neg + bin2_neg;
  const int total = total_pos + total_neg;
  
  if (total == 0 || total_pos == 0 || total_neg == 0) {
    return 0.0;
  }
  
  const int bin1_total = bin1_pos + bin1_neg;
  const int bin2_total = bin2_pos + bin2_neg;
  
  const double expected_pos1 = static_cast<double>(bin1_total * total_pos) / total;
  const double expected_neg1 = static_cast<double>(bin1_total * total_neg) / total;
  const double expected_pos2 = static_cast<double>(bin2_total * total_pos) / total;
  const double expected_neg2 = static_cast<double>(bin2_total * total_neg) / total;
  
  double chi_square = 0.0;
  
  if (use_continuity_correction) {
    if (expected_pos1 > EPSILON) {
      chi_square += std::pow(std::abs(bin1_pos - expected_pos1) - 0.5, 2.0) / expected_pos1;
    }
    if (expected_neg1 > EPSILON) {
      chi_square += std::pow(std::abs(bin1_neg - expected_neg1) - 0.5, 2.0) / expected_neg1;
    }
    if (expected_pos2 > EPSILON) {
      chi_square += std::pow(std::abs(bin2_pos - expected_pos2) - 0.5, 2.0) / expected_pos2;
    }
    if (expected_neg2 > EPSILON) {
      chi_square += std::pow(std::abs(bin2_neg - expected_neg2) - 0.5, 2.0) / expected_neg2;
    }
  } else {
    if (expected_pos1 > EPSILON) {
      chi_square += std::pow(bin1_pos - expected_pos1, 2.0) / expected_pos1;
    }
    if (expected_neg1 > EPSILON) {
      chi_square += std::pow(bin1_neg - expected_neg1, 2.0) / expected_neg1;
    }
    if (expected_pos2 > EPSILON) {
      chi_square += std::pow(bin2_pos - expected_pos2, 2.0) / expected_pos2;
    }
    if (expected_neg2 > EPSILON) {
      chi_square += std::pow(bin2_neg - expected_neg2, 2.0) / expected_neg2;
    }
  }
  
  return chi_square;
}

} // namespace OptimalBinning

#endif // OPTIMAL_BINNING_UTILITIES_H

