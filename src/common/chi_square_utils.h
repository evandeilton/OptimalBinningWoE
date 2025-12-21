/**
 * @file chi_square_utils.h
 * @brief Chi-square calculations for ChiMerge algorithms
 * 
 * Used by: OptimalBinningNumericalCM, OptimalBinningCategoricalCM
 */

#ifndef OPTIMAL_BINNING_CHI_SQUARE_UTILS_H
#define OPTIMAL_BINNING_CHI_SQUARE_UTILS_H

#include "optimal_binning_common.h"
#include <unordered_map>

namespace OptimalBinning {

/// Chi-square critical values for DF=1 at various significance levels
const std::unordered_map<double, double> CHI_SQUARE_CRITICAL_VALUES = {
  {0.995, 7.879},
  {0.99, 6.635},
  {0.975, 5.024},
  {0.95, 3.841},
  {0.90, 2.706},
  {0.80, 1.642},
  {0.70, 1.074},
  {0.50, 0.455},
  {0.30, 0.148},
  {0.20, 0.064},
  {0.10, 0.016},
  {0.05, 0.004},
  {0.01, 0.0002},
  {0.001, 0.00001}
};

/**
 * @brief Get chi-square critical value for given significance level
 * 
 * @param significance Significance level (alpha)
 * @return Critical value for DF=1
 */
inline double get_chi_critical_value(double significance) {
  auto it = CHI_SQUARE_CRITICAL_VALUES.find(significance);
  if (it != CHI_SQUARE_CRITICAL_VALUES.end()) {
    return it->second;
  }
  
  // Find closest threshold
  double closest = 0.95;
  double min_diff = std::numeric_limits<double>::max();
  
  for (const auto& [thresh, val] : CHI_SQUARE_CRITICAL_VALUES) {
    double diff = std::abs(thresh - significance);
    if (diff < min_diff) {
      min_diff = diff;
      closest = thresh;
    }
  }
  
  return CHI_SQUARE_CRITICAL_VALUES.at(closest);
}

/**
 * @brief Calculate chi-square statistic between two bins
 * 
 * Uses Yates' continuity correction.
 * 
 * @param pos1 Positive count in bin 1
 * @param neg1 Negative count in bin 1
 * @param pos2 Positive count in bin 2
 * @param neg2 Negative count in bin 2
 * @param continuity_correction Whether to apply Yates' correction
 * @return Chi-square statistic
 */
inline double calculate_chi_square(
    int pos1, int neg1,
    int pos2, int neg2,
    bool continuity_correction = true
) {
  int total_pos = pos1 + pos2;
  int total_neg = neg1 + neg2;
  int total = total_pos + total_neg;
  int n1 = pos1 + neg1;
  int n2 = pos2 + neg2;
  
  if (total == 0 || total_pos == 0 || total_neg == 0 || n1 == 0 || n2 == 0) {
    return 0.0;
  }
  
  double expected_pos1 = static_cast<double>(n1 * total_pos) / total;
  double expected_neg1 = static_cast<double>(n1 * total_neg) / total;
  double expected_pos2 = static_cast<double>(n2 * total_pos) / total;
  double expected_neg2 = static_cast<double>(n2 * total_neg) / total;
  
  double chi_square = 0.0;
  auto add_term = [&chi_square, continuity_correction](double observed, double expected) {
    if (expected > EPSILON) {
      double diff = std::abs(observed - expected);
      if (continuity_correction) diff -= 0.5;
      diff = std::max(0.0, diff);
      chi_square += (diff * diff) / expected;
    }
  };
  
  add_term(pos1, expected_pos1);
  add_term(neg1, expected_neg1);
  add_term(pos2, expected_pos2);
  add_term(neg2, expected_neg2);
  
  return chi_square;
}

/**
 * @brief Linear cache for adjacent bin chi-square values
 */
class ChiSquareCache {
private:
  std::vector<double> cache_;
  size_t num_bins_;
  static constexpr double INVALID_VALUE = -1.0;
  
public:
  ChiSquareCache() : num_bins_(0) {}
  
  explicit ChiSquareCache(size_t n) : num_bins_(n) {
    resize(n);
  }
  
  void resize(size_t n) {
    num_bins_ = n;
    size_t cache_size = (n > 1) ? (n - 1) : 0;
    cache_.assign(cache_size, INVALID_VALUE);
  }
  
  double get(size_t i) const {
    if (i >= cache_.size()) return INVALID_VALUE;
    return cache_[i];
  }
  
  void set(size_t i, double value) {
    if (i < cache_.size() && value >= 0) {
      cache_[i] = value;
    }
  }
  
  void invalidate(size_t i) {
    if (i < cache_.size()) cache_[i] = INVALID_VALUE;
  }
  
  void invalidate_after_merge(size_t merge_idx) {
    if (merge_idx > 0 && merge_idx - 1 < cache_.size()) {
      cache_[merge_idx - 1] = INVALID_VALUE;
    }
    if (merge_idx < cache_.size()) {
      cache_[merge_idx] = INVALID_VALUE;
    }
  }
  
  void clear() {
    std::fill(cache_.begin(), cache_.end(), INVALID_VALUE);
  }
  
  static bool is_valid(double value) {
    return value >= 0;
  }
};

} // namespace OptimalBinning

#endif // OPTIMAL_BINNING_CHI_SQUARE_UTILS_H
