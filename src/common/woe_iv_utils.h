/**
 * @file woe_iv_utils.h
 * @brief Weight of Evidence and Information Value calculations
 *
 * Provides unified WoE/IV calculations supporting all smoothing methods:
 * - NONE: EPSILON-only protection
 * - LAPLACE: Add smoothing parameter to counts (Laplace, 1814)
 * - BAYESIAN: Use Bayesian prior based on overall proportions (Good, 1950)
 *
 * Scientific References:
 * - Good, I. J. (1950). "Probability and the Weighing of Evidence".
 * - Kullback, S., & Leibler, R. A. (1951). "On information and sufficiency".
 * - Siddiqi, N. (2006). "Credit Risk Scorecards" (Chapter 3: WoE and IV).
 */

#ifndef OPTIMAL_BINNING_WOE_IV_UTILS_H
#define OPTIMAL_BINNING_WOE_IV_UTILS_H

#include "optimal_binning_common.h"
#include "safe_math.h"

namespace OptimalBinning {

/**
 * @brief Compute Weight of Evidence for a bin
 * 
 * @param pos Positive count in bin
 * @param neg Negative count in bin
 * @param total_pos Total positive count across all bins
 * @param total_neg Total negative count across all bins
 * @param method Smoothing method (NONE, LAPLACE, BAYESIAN)
 * @param smoothing_param Smoothing parameter (used by LAPLACE and BAYESIAN)
 * @return Weight of Evidence value
 */
inline double compute_woe(
    int pos, int neg,
    int total_pos, int total_neg,
    SmoothingMethod method = SmoothingMethod::NONE,
    double smoothing_param = DEFAULT_LAPLACE_ALPHA
) {
  double dist_pos, dist_neg;
  
  switch (method) {
    case SmoothingMethod::LAPLACE: {
      // Laplace smoothing: add smoothing_param to counts
      double smoothed_pos = pos + smoothing_param;
      double smoothed_neg = neg + smoothing_param;
      // Total bins contribution approximated (use 2 * smoothing_param for single bin)
      dist_pos = smoothed_pos / (total_pos + 2 * smoothing_param);
      dist_neg = smoothed_neg / (total_neg + 2 * smoothing_param);
      break;
    }
    
    case SmoothingMethod::BAYESIAN: {
      // Bayesian smoothing with prior based on overall proportions
      double prior_pos = smoothing_param * static_cast<double>(total_pos) / (total_pos + total_neg);
      double prior_neg = smoothing_param - prior_pos;
      dist_pos = (pos + prior_pos) / (total_pos + smoothing_param);
      dist_neg = (neg + prior_neg) / (total_neg + smoothing_param);
      break;
    }
    
    default: // SmoothingMethod::NONE
      // Just EPSILON protection
      dist_pos = std::max(static_cast<double>(pos) / std::max(total_pos, 1), EPSILON);
      dist_neg = std::max(static_cast<double>(neg) / std::max(total_neg, 1), EPSILON);
      break;
  }
  
  // Compute WoE with bounds checking
  if (dist_pos < EPSILON) dist_pos = EPSILON;
  if (dist_neg < EPSILON) dist_neg = EPSILON;
  
  double woe = std::log(dist_pos / dist_neg);
  
  // Clamp to prevent extreme values
  return clamp(woe, MIN_WOE, MAX_WOE);
}

/**
 * @brief Compute Weight of Evidence with Laplace smoothing (bins-aware version)
 * 
 * This version accounts for total number of bins in the smoothing calculation.
 * 
 * @param pos Positive count in bin
 * @param neg Negative count in bin
 * @param total_pos Total positive count
 * @param total_neg Total negative count
 * @param num_bins Number of bins
 * @param smoothing_param Laplace smoothing parameter
 */
inline double compute_woe_laplace(
    int pos, int neg,
    int total_pos, int total_neg,
    size_t num_bins,
    double smoothing_param = DEFAULT_LAPLACE_ALPHA
) {
  double smoothed_pos = pos + smoothing_param;
  double smoothed_neg = neg + smoothing_param;
  double total_smoothed_pos = total_pos + num_bins * smoothing_param;
  double total_smoothed_neg = total_neg + num_bins * smoothing_param;
  
  double dist_pos = smoothed_pos / total_smoothed_pos;
  double dist_neg = smoothed_neg / total_smoothed_neg;
  
  if (dist_pos < EPSILON) dist_pos = EPSILON;
  if (dist_neg < EPSILON) dist_neg = EPSILON;
  
  double woe = std::log(dist_pos / dist_neg);
  return clamp(woe, MIN_WOE, MAX_WOE);
}

/**
 * @brief Compute Information Value for a bin
 * 
 * @param pos Positive count in bin
 * @param neg Negative count in bin
 * @param total_pos Total positive count
 * @param total_neg Total negative count
 * @param woe Pre-computed WoE value (if available, otherwise computed)
 * @param method Smoothing method
 * @param smoothing_param Smoothing parameter
 * @return Information Value contribution for this bin
 */
inline double compute_iv(
    int pos, int neg,
    int total_pos, int total_neg,
    double woe = std::numeric_limits<double>::quiet_NaN(),
    SmoothingMethod method = SmoothingMethod::NONE,
    double smoothing_param = DEFAULT_LAPLACE_ALPHA
) {
  // Compute WoE if not provided
  if (std::isnan(woe)) {
    woe = compute_woe(pos, neg, total_pos, total_neg, method, smoothing_param);
  }
  
  double dist_pos, dist_neg;
  
  switch (method) {
    case SmoothingMethod::LAPLACE: {
      double smoothed_pos = pos + smoothing_param;
      double smoothed_neg = neg + smoothing_param;
      dist_pos = smoothed_pos / (total_pos + 2 * smoothing_param);
      dist_neg = smoothed_neg / (total_neg + 2 * smoothing_param);
      break;
    }
    
    case SmoothingMethod::BAYESIAN: {
      double prior_pos = smoothing_param * static_cast<double>(total_pos) / (total_pos + total_neg);
      double prior_neg = smoothing_param - prior_pos;
      dist_pos = (pos + prior_pos) / (total_pos + smoothing_param);
      dist_neg = (neg + prior_neg) / (total_neg + smoothing_param);
      break;
    }
    
    default:
      dist_pos = std::max(static_cast<double>(pos) / std::max(total_pos, 1), EPSILON);
      dist_neg = std::max(static_cast<double>(neg) / std::max(total_neg, 1), EPSILON);
      break;
  }
  
  double iv = (dist_pos - dist_neg) * woe;
  
  // Return 0 for invalid IV
  return is_valid_number(iv) ? iv : 0.0;
}

/**
 * @brief Compute WoE and IV together (output parameters)
 *
 * @param pos Positive count
 * @param neg Negative count
 * @param total_pos Total positive count
 * @param total_neg Total negative count
 * @param woe Output: Weight of Evidence
 * @param iv Output: Information Value
 * @param method Smoothing method
 * @param smoothing_param Smoothing parameter
 */
inline void compute_woe_iv(
    int pos, int neg,
    int total_pos, int total_neg,
    double& woe, double& iv,
    SmoothingMethod method = SmoothingMethod::NONE,
    double smoothing_param = DEFAULT_LAPLACE_ALPHA
) {
  woe = compute_woe(pos, neg, total_pos, total_neg, method, smoothing_param);
  iv = compute_iv(pos, neg, total_pos, total_neg, woe, method, smoothing_param);
}

// =============================================================================
// AGGREGATION FUNCTIONS (with compensated summation)
// =============================================================================

/**
 * @brief Compute total Information Value with compensated summation
 *
 * Uses Kahan summation for numerical stability when summing IV values.
 * Reduces error from O(nε) to O(ε).
 *
 * @param iv_values Vector of IV contributions from each bin
 * @param high_precision If true, use Neumaier summation; else use Kahan
 * @return Total IV with compensated sum
 */
inline double compute_total_iv(const std::vector<double>& iv_values,
                               bool high_precision = false) {
  return compensated_sum(iv_values, high_precision);
}

/**
 * @brief Compute total IV from bin containers
 *
 * Template function that works with any bin structure having .iv field.
 *
 * @tparam BinType Type of bin structure (NumericalBin, CategoricalBin, etc.)
 * @param bins Vector of bins
 * @param high_precision If true, use Neumaier summation
 * @return Total IV
 */
template<typename BinType>
inline double compute_total_iv_bins(const std::vector<BinType>& bins,
                                   bool high_precision = false) {
  if (bins.empty()) return 0.0;

  std::vector<double> iv_values;
  iv_values.reserve(bins.size());

  for (const auto& bin : bins) {
    iv_values.push_back(bin.iv);
  }

  return compensated_sum(iv_values, high_precision);
}

} // namespace OptimalBinning

#endif // OPTIMAL_BINNING_WOE_IV_UTILS_H
