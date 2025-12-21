/**
 * @file validation_utils.h
 * @brief Input validation utilities for Optimal Binning algorithms
 */

#ifndef OPTIMAL_BINNING_VALIDATION_UTILS_H
#define OPTIMAL_BINNING_VALIDATION_UTILS_H

#include "optimal_binning_common.h"
#include <stdexcept>

namespace OptimalBinning {

/**
 * @brief Validate that target vector contains only binary values (0/1)
 * 
 * @param target Target vector
 * @throws std::invalid_argument if target contains values other than 0 or 1
 * @throws std::invalid_argument if target doesn't contain both 0 and 1
 */
template<typename T>
inline void validate_binary_target(const std::vector<T>& target) {
  if (target.empty()) {
    throw std::invalid_argument("Target vector cannot be empty");
  }
  
  bool has_zero = false;
  bool has_one = false;
  
  for (const auto& t : target) {
    if (t == 0) {
      has_zero = true;
    } else if (t == 1) {
      has_one = true;
    } else {
      throw std::invalid_argument("Target must contain only values 0 and 1");
    }
    
    if (has_zero && has_one) break; // Optimization
  }
  
  if (!has_zero) {
    throw std::invalid_argument("Target must contain at least one 0 value");
  }
  if (!has_one) {
    throw std::invalid_argument("Target must contain at least one 1 value");
  }
}

/**
 * @brief Validate Rcpp IntegerVector as binary target
 */
inline void validate_binary_target_rcpp(const Rcpp::IntegerVector& target) {
  if (target.size() == 0) {
    Rcpp::stop("Target vector cannot be empty");
  }
  
  bool has_zero = false, has_one = false;
  
  for (int i = 0; i < target.size(); ++i) {
    if (target[i] == 0) has_zero = true;
    else if (target[i] == 1) has_one = true;
    else Rcpp::stop("Target must contain only values 0 and 1");
    
    if (has_zero && has_one) break;
  }
  
  if (!has_zero || !has_one) {
    Rcpp::stop("Target must contain both 0 and 1 values");
  }
}

/**
 * @brief Validate that feature and target have the same size
 * 
 * @param feature_size Size of feature vector
 * @param target_size Size of target vector
 * @throws std::invalid_argument if sizes don't match
 */
inline void validate_feature_target_size(size_t feature_size, size_t target_size) {
  if (feature_size != target_size) {
    throw std::invalid_argument(
      "Feature and target must have the same size. Got feature size: " +
      std::to_string(feature_size) + ", target size: " + std::to_string(target_size)
    );
  }
}

/**
 * @brief Validate bin parameters
 * 
 * @param min_bins Minimum number of bins
 * @param max_bins Maximum number of bins
 * @param bin_cutoff Minimum bin frequency cutoff
 * @throws std::invalid_argument if parameters are invalid
 */
inline void validate_bin_parameters(int min_bins, int max_bins, double bin_cutoff) {
  if (min_bins < 2) {
    throw std::invalid_argument("min_bins must be at least 2, got: " + std::to_string(min_bins));
  }
  
  if (max_bins < min_bins) {
    throw std::invalid_argument(
      "max_bins (" + std::to_string(max_bins) + 
      ") must be >= min_bins (" + std::to_string(min_bins) + ")"
    );
  }
  
  if (bin_cutoff <= 0.0 || bin_cutoff >= 1.0) {
    throw std::invalid_argument(
      "bin_cutoff must be between 0 and 1 (exclusive), got: " + std::to_string(bin_cutoff)
    );
  }
}

/**
 * @brief Validate additional algorithm parameters
 * 
 * @param max_n_prebins Maximum number of pre-bins
 * @param convergence_threshold Convergence threshold
 * @param max_iterations Maximum iterations
 */
inline void validate_algorithm_parameters(
    int max_n_prebins,
    double convergence_threshold,
    int max_iterations
) {
  if (max_n_prebins < 2) {
    throw std::invalid_argument("max_n_prebins must be >= 2");
  }
  
  if (convergence_threshold <= 0) {
    throw std::invalid_argument("convergence_threshold must be positive");
  }
  
  if (max_iterations <= 0) {
    throw std::invalid_argument("max_iterations must be positive");
  }
}

/**
 * @brief Check numerical feature for NaN/Inf values
 * 
 * @param feature Numerical feature vector
 * @return Pair of (count_nan, count_inf)
 */
inline std::pair<int, int> check_numerical_feature(const std::vector<double>& feature) {
  int nan_count = 0;
  int inf_count = 0;
  
  for (double f : feature) {
    if (std::isnan(f)) nan_count++;
    else if (std::isinf(f)) inf_count++;
  }
  
  return {nan_count, inf_count};
}

} // namespace OptimalBinning

#endif // OPTIMAL_BINNING_VALIDATION_UTILS_H
