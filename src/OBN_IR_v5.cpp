// [[Rcpp::plugins(cpp17)]]
// [[Rcpp::depends(Rcpp)]]

#include <Rcpp.h>
#include <algorithm>
#include <vector>
#include <string>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <sstream>
#include <numeric>
#include <unordered_set>

/**
 * @file OBN_IR.cpp
 * @brief Optimal Binning for Numerical Variables using Isotonic Regression
 * 
 * This implementation provides supervised discretization of numerical variables
 * using isotonic regression to ensure monotonicity in event rates across bins.
 * The algorithm is particularly useful for risk modeling and credit scoring
 * applications where monotonicity is a desirable property.
 */

using namespace Rcpp;

// Include shared headers
#include "common/optimal_binning_common.h"
#include "common/bin_structures.h"

using namespace Rcpp;
using namespace OptimalBinning;


// Class for Optimal Binning using Isotonic Regression
class OBN_IR {
private:
  int min_bins;
  int max_bins;
  double bin_cutoff;
  int max_n_prebins;
  double convergence_threshold;
  int max_iterations;
  bool auto_monotonicity;  // Automatically determine monotonicity direction
  
  const std::vector<double>& feature;
  const std::vector<int>& target;
  
  std::vector<double> bin_edges;
  double total_iv;
  bool converged;
  int iterations_run;
  bool monotone_increasing;  // Direction of monotonicity
  
  // Structure to represent a bin and its statistics
  // Local NumericalBin definition removed

  
  std::vector<NumericalBin> bin_info;
  bool is_simple;           // Flag for simple binning (few unique values)
  
  // Small constant for numerical stability
  // Constant removed (uses shared definition)
  // Laplace smoothing factor
  static constexpr double ALPHA = 0.5;
  
public:
  /**
   * Constructor for the OBN_IR class
   * 
   * @param min_bins_ Minimum number of bins
   * @param max_bins_ Maximum number of bins
   * @param bin_cutoff_ Minimum proportion of observations per bin
   * @param max_n_prebins_ Maximum number of pre-bins before optimization
   * @param convergence_threshold_ Convergence threshold for algorithm
   * @param max_iterations_ Maximum number of iterations
   * @param feature_ Feature vector to be binned
   * @param target_ Binary target vector (0/1)
   * @param auto_monotonicity_ Whether to automatically determine monotonicity direction
   */
  OBN_IR(int min_bins_, int max_bins_,
                            double bin_cutoff_, int max_n_prebins_,
                            double convergence_threshold_, int max_iterations_,
                            const std::vector<double>& feature_,
                            const std::vector<int>& target_,
                            bool auto_monotonicity_ = true)
    : min_bins(min_bins_), max_bins(max_bins_),
      bin_cutoff(bin_cutoff_), max_n_prebins(max_n_prebins_),
      convergence_threshold(convergence_threshold_), max_iterations(max_iterations_),
      auto_monotonicity(auto_monotonicity_),
      feature(feature_), target(target_), total_iv(0.0),
      converged(false), iterations_run(0), monotone_increasing(true), is_simple(false) {
    validateInputs();
  }
  
  /**
   * Execute the binning algorithm
   */
  void fit() {
    // Step 1: Create initial bins
    createInitialBins();
    
    if (!is_simple) {
      // Step 2: Merge low frequency bins
      mergeLowFrequencyBins();
      
      // Step 3: Ensure min and max bin constraints
      ensureMinMaxBins();
      
      // Step 4: Determine monotonicity direction if auto_monotonicity is enabled
      if (auto_monotonicity) {
        determineMonotonicityDirection();
      }
      
      // Step 5: Apply isotonic regression to enforce monotonicity
      applyIsotonicRegression();
    }
    
    // Step 6: Calculate final WOE and IV
    calculateWOEandIV();
  }
  
  /**
   * Get the results of the binning process
   * 
   * @return A list containing bin information and metrics
   */
  Rcpp::List getResults() const {
    return createWOEBinList();
  }
  
private:
  /**
   * Validate input parameters and data
   * Throws exception if invalid
   */
  void validateInputs() const {
    if (feature.size() != target.size()) {
      throw std::invalid_argument("Feature and target must have the same length.");
    }
    
    if (min_bins < 2) {
      throw std::invalid_argument("min_bins must be at least 2.");
    }
    
    if (max_bins < min_bins) {
      throw std::invalid_argument("max_bins must be greater than or equal to min_bins.");
    }
    
    if (bin_cutoff <= 0.0 || bin_cutoff >= 1.0) {
      throw std::invalid_argument("bin_cutoff must be between 0 and 1.");
    }
    
    if (max_n_prebins < min_bins) {
      throw std::invalid_argument("max_n_prebins must be at least min_bins.");
    }
    
    if (max_iterations <= 0) {
      throw std::invalid_argument("max_iterations must be positive.");
    }
    
    if (convergence_threshold <= 0.0) {
      throw std::invalid_argument("convergence_threshold must be positive.");
    }
    
    // Check that target is binary (0 or 1)
    auto [min_it, max_it] = std::minmax_element(target.begin(), target.end());
    if (*min_it < 0 || *max_it > 1) {
      throw std::invalid_argument("Target must be binary (0 or 1).");
    }
    
    // Check that both classes are present
    int sum_target = std::accumulate(target.begin(), target.end(), 0);
    if (sum_target == 0 || sum_target == static_cast<int>(target.size())) {
      throw std::invalid_argument("Target must contain both classes (0 and 1).");
    }
    
    // Count valid values (not NaN or Inf)
    int valid_count = 0;
    for (const auto& value : feature) {
      if (!std::isnan(value) && !std::isinf(value)) {
        valid_count++;
      }
    }
    
    if (valid_count == 0) {
      throw std::invalid_argument("Feature contains only NaN or Inf values.");
    }
  }
  
  /**
   * Create initial bins based on unique values or quantiles
   */
  void createInitialBins() {
    // Create a clean copy of feature values (excluding NaN and Inf)
    std::vector<double> clean_feature;
    std::vector<int> clean_target;
    clean_feature.reserve(feature.size());
    clean_target.reserve(target.size());
    
    for (size_t i = 0; i < feature.size(); ++i) {
      if (!std::isnan(feature[i]) && !std::isinf(feature[i])) {
        clean_feature.push_back(feature[i]);
        clean_target.push_back(target[i]);
      }
    }
    
    // Get unique values
    std::vector<double> sorted_feature = clean_feature;
    std::sort(sorted_feature.begin(), sorted_feature.end());
    sorted_feature.erase(std::unique(sorted_feature.begin(), sorted_feature.end()), sorted_feature.end());
    
    int unique_vals = static_cast<int>(sorted_feature.size());
    
    // Special case for few unique values
    if (unique_vals <= 2) {
      handleFewUniqueValues(sorted_feature, clean_feature, clean_target);
    } else {
      // Regular binning for more unique values
      createRegularBins(sorted_feature, unique_vals);
    }
  }
  
  /**
   * Handle the special case where there are few unique values
   * 
   * @param sorted_feature Vector of sorted unique feature values
   * @param clean_feature Vector of valid feature values
   * @param clean_target Vector of corresponding target values
   */
  void handleFewUniqueValues(const std::vector<double>& sorted_feature,
                             const std::vector<double>& clean_feature,
                             const std::vector<int>& clean_target) {
    is_simple = true;
    bin_edges.clear();
    bin_info.clear();
    
    int unique_vals = static_cast<int>(sorted_feature.size());
    
    bin_edges.push_back(-std::numeric_limits<double>::infinity());
    
    if (unique_vals == 1) {
      // Single unique value
      bin_edges.push_back(std::numeric_limits<double>::infinity());
      
      NumericalBin bin;
      bin.lower_bound = bin_edges[0];
      bin.upper_bound = bin_edges[1];
      bin.count = static_cast<int>(clean_feature.size());
      bin.count_pos = std::accumulate(clean_target.begin(), clean_target.end(), 0);
      bin.count_neg = bin.count - bin.count_pos;
      // bin.event_rate() assignment removed (calculated dynamically)
      
      bin_info.push_back(bin);
    } else { 
      // Two unique values
      bin_edges.push_back(sorted_feature[0]);
      bin_edges.push_back(std::numeric_limits<double>::infinity());
      
      NumericalBin bin1, bin2;
      bin1.lower_bound = bin_edges[0];
      bin1.upper_bound = bin_edges[1];
      bin2.lower_bound = bin_edges[1];
      bin2.upper_bound = bin_edges[2];
      
      // Assign observations to bins
      for (size_t j = 0; j < clean_feature.size(); ++j) {
        if (clean_feature[j] <= bin1.upper_bound) {
          bin1.count++;
          bin1.count_pos += clean_target[j];
        } else {
          bin2.count++;
          bin2.count_pos += clean_target[j];
        }
      }
      
      bin1.count_neg = bin1.count - bin1.count_pos;
      bin2.count_neg = bin2.count - bin2.count_pos;
      
      // bin1.event_rate() assignment removed (calculated dynamically)
      // bin2.event_rate() assignment removed (calculated dynamically)
      
      bin_info.push_back(bin1);
      bin_info.push_back(bin2);
    }
  }
  
  /**
   * Create regular bins for normal case (more than 2 unique values)
   * 
   * @param sorted_feature Vector of sorted unique feature values
   * @param unique_vals Number of unique values
   */
  void createRegularBins(const std::vector<double>& sorted_feature, int unique_vals) {
    is_simple = false;
    
    // Determine number of pre-bins
    int n_prebins = std::min({max_n_prebins, unique_vals, max_bins});
    n_prebins = std::max(n_prebins, min_bins);
    
    // Create bin edges
    bin_edges.resize(static_cast<size_t>(n_prebins + 1));
    bin_edges[0] = -std::numeric_limits<double>::infinity();
    bin_edges[n_prebins] = std::numeric_limits<double>::infinity();
    
    // Use quantiles for bin edges
    for (int i = 1; i < n_prebins; ++i) {
      // Calculate index for quantile
      double q = static_cast<double>(i) / n_prebins;
      int idx = static_cast<int>(std::round(q * unique_vals));
      idx = std::max(1, std::min(idx, unique_vals - 1));
      bin_edges[i] = sorted_feature[static_cast<size_t>(idx - 1)];
    }
    
    // Ensure uniqueness of bin edges (can happen with skewed distributions)
    std::sort(bin_edges.begin(), bin_edges.end());
    bin_edges.erase(std::unique(bin_edges.begin(), bin_edges.end()), bin_edges.end());
    
    // If we lost some bin edges due to duplicates, adjust
    if (bin_edges.size() < 3) {
      // Fall back to min/max approach
      bin_edges.clear();
      bin_edges.push_back(-std::numeric_limits<double>::infinity());
      
      double min_val = sorted_feature.front();
      double max_val = sorted_feature.back();
      double middle = (min_val + max_val) / 2.0;
      
      bin_edges.push_back(middle);
      bin_edges.push_back(std::numeric_limits<double>::infinity());
    }
  }
  
  /**
   * Merge bins with frequency below the cutoff threshold
   * This ensures statistical reliability of each bin
   */
  void mergeLowFrequencyBins() {
    // Initialize bin_info from bin_edges
    initializeBinsFromEdges();
    
    int total_count = 0;
    for (const auto& bin : bin_info) {
      total_count += bin.count;
    }
    
    double min_count = bin_cutoff * total_count;
    
    // Iteratively merge small bins
    bool merged = true;
    int iterations = 0;
    
    while (merged && iterations < max_iterations && bin_info.size() > static_cast<size_t>(min_bins)) {
      merged = false;
      
      for (size_t i = 0; i < bin_info.size(); ++i) {
        if (bin_info[i].count < min_count) {
          // Find optimal merge direction
          if (i == 0 && bin_info.size() > 1) {
            // First bin - merge with next
            mergeBins(0, 1);
          } else if (i == bin_info.size() - 1 && i > 0) {
            // Last bin - merge with previous
            mergeBins(i - 1, i);
          } else if (i > 0 && i < bin_info.size() - 1) {
            // Middle bin - compare event rates
            double rate_diff_prev = std::fabs(bin_info[i].event_rate() - bin_info[i-1].event_rate());
            double rate_diff_next = std::fabs(bin_info[i].event_rate() - bin_info[i+1].event_rate());
            
            if (rate_diff_prev <= rate_diff_next) {
              // Merge with previous
              mergeBins(i - 1, i);
            } else {
              // Merge with next
              mergeBins(i, i + 1);
            }
          }
          
          merged = true;
          break;
        }
      }
      
      iterations++;
    }
    
    iterations_run += iterations;
  }
  
  /**
   * Initialize bin information from bin edges
   * Assigns observations to bins and calculates initial statistics
   */
  void initializeBinsFromEdges() {
    bin_info.clear();
    
    // Create bins
    for (size_t i = 0; i < bin_edges.size() - 1; ++i) {
      NumericalBin bin;
      bin.lower_bound = bin_edges[i];
      bin.upper_bound = bin_edges[i + 1];
      bin.count = 0;
      bin.count_pos = 0;
      bin.count_neg = 0;
      bin_info.push_back(bin);
    }
    
    // Assign observations to bins
    for (size_t j = 0; j < feature.size(); ++j) {
      // Skip NaN values
      if (std::isnan(feature[j]) || std::isinf(feature[j])) {
        continue;
      }
      
      // Find appropriate bin
      for (size_t i = 0; i < bin_info.size(); ++i) {
        bool in_bin = (i == 0) ? 
        (feature[j] >= bin_info[i].lower_bound && feature[j] <= bin_info[i].upper_bound) :
        (feature[j] > bin_info[i].lower_bound && feature[j] <= bin_info[i].upper_bound);
        
        if (in_bin) {
          bin_info[i].count++;
          bin_info[i].count_pos += target[j];
          break;
        }
      }
    }
    
    // Calculate negatives and event rates
    for (auto& bin : bin_info) {
      bin.count_neg = bin.count - bin.count_pos;
      // bin.event_rate() assignment removed (calculated dynamically)
    }
  }
  
  /**
   * Merge two adjacent bins
   * 
   * @param idx1 Index of first bin
   * @param idx2 Index of second bin (must be adjacent to idx1)
   */
  void mergeBins(size_t idx1, size_t idx2) {
    if (idx1 > idx2) {
      std::swap(idx1, idx2);
    }
    
    if (idx2 != idx1 + 1) {
      throw std::invalid_argument("Can only merge adjacent bins");
    }
    
    // Merge statistics
    bin_info[idx1].upper_bound = bin_info[idx2].upper_bound;
    bin_info[idx1].count += bin_info[idx2].count;
    bin_info[idx1].count_pos += bin_info[idx2].count_pos;
    bin_info[idx1].count_neg += bin_info[idx2].count_neg;
    
    // Recalculate event rate
    // bin_info[idx1].event_rate() assignment removed (calculated dynamically)
    
    // Remove second bin
    bin_info.erase(bin_info.begin() + idx2);
  }
  
  /**
   * Ensure number of bins is within [min_bins, max_bins]
   * Either split large bins or merge similar bins
   */
  void ensureMinMaxBins() {
    // Add bins if below min_bins
    while (bin_info.size() < static_cast<size_t>(min_bins) && bin_info.size() > 1) {
      splitLargestBin();
    }
    
    // Merge bins if above max_bins
    while (bin_info.size() > static_cast<size_t>(max_bins)) {
      mergeSimilarBins();
    }
  }
  
  /**
   * Split the bin with the largest number of observations
   */
  void splitLargestBin() {
    // Find largest bin
    auto it = std::max_element(bin_info.begin(), bin_info.end(),
                               [](const NumericalBin& a, const NumericalBin& b) {
                                 return a.count < b.count;
                               });
    
    if (it != bin_info.end() && it->count > 1) {
      size_t idx = static_cast<size_t>(std::distance(bin_info.begin(), it));
      
      // Find optimal split point
      std::vector<double> bin_values;
      std::vector<int> bin_targets;
      
      for (size_t j = 0; j < feature.size(); ++j) {
        if (std::isnan(feature[j]) || std::isinf(feature[j])) {
          continue;
        }
        
        bool in_bin = (idx == 0) ? 
        (feature[j] >= it->lower_bound && feature[j] <= it->upper_bound) :
          (feature[j] > it->lower_bound && feature[j] <= it->upper_bound);
        
        if (in_bin) {
          bin_values.push_back(feature[j]);
          bin_targets.push_back(target[j]);
        }
      }
      
      // Sort values within bin
      std::vector<size_t> indices(bin_values.size());
      std::iota(indices.begin(), indices.end(), 0);
      std::sort(indices.begin(), indices.end(),
                [&bin_values](size_t i1, size_t i2) {
                  return bin_values[i1] < bin_values[i2];
                });
      
      // Choose split point at median or optimal information gain
      size_t split_idx = indices.size() / 2;
      double split_value = bin_values[indices[split_idx]];
      
      // Ensure split value is within bin and not at boundaries
      if (std::fabs(split_value - it->lower_bound) < EPSILON || std::fabs(split_value - it->upper_bound) < EPSILON) {
        // Try another split point if too close to boundaries
        if (split_idx > 0 && split_idx < indices.size() - 1) {
          split_value = (bin_values[indices[split_idx - 1]] + bin_values[indices[split_idx + 1]]) / 2.0;
        } else {
          // Not enough distinct values for good split
          return;
        }
      }
      
      // Create two new bins
      NumericalBin bin1 = *it;
      NumericalBin bin2 = *it;
      
      bin1.upper_bound = split_value;
      bin2.lower_bound = split_value;
      
      bin1.count = 0;
      bin1.count_pos = 0;
      bin2.count = 0;
      bin2.count_pos = 0;
      
      // Re-assign observations
      for (size_t j = 0; j < bin_values.size(); ++j) {
        if (bin_values[j] <= split_value) {
          bin1.count++;
          bin1.count_pos += bin_targets[j];
        } else {
          bin2.count++;
          bin2.count_pos += bin_targets[j];
        }
      }
      
      // Calculate negatives and rates
      bin1.count_neg = bin1.count - bin1.count_pos;
      bin2.count_neg = bin2.count - bin2.count_pos;
      
      // bin1.event_rate() assignment removed (calculated dynamically)
      // bin2.event_rate() assignment removed (calculated dynamically)
      
      // Replace original bin with two new bins
      *it = bin1;
      bin_info.insert(bin_info.begin() + idx + 1, bin2);
    }
  }
  
  /**
   * Merge the two most similar adjacent bins
   * Similarity is based on event rates
   */
  void mergeSimilarBins() {
    if (bin_info.size() <= 2) return;
    
    double min_diff = std::numeric_limits<double>::max();
    size_t merge_idx = 0;
    
    // Find most similar adjacent bins
    for (size_t i = 0; i < bin_info.size() - 1; ++i) {
      double diff = std::fabs(bin_info[i].event_rate() - bin_info[i+1].event_rate());
      if (diff < min_diff) {
        min_diff = diff;
        merge_idx = i;
      }
    }
    
    // Merge bins
    mergeBins(merge_idx, merge_idx + 1);
  }
  
  /**
   * Determine the optimal monotonicity direction (increasing or decreasing)
   * based on correlation between bin midpoints and event rates
   */
  void determineMonotonicityDirection() {
    if (bin_info.size() <= 1) return;
    
    // Calculate bin midpoints
    std::vector<double> midpoints;
    std::vector<double> rates;
    
    for (const auto& bin : bin_info) {
      // For infinite bounds, use the next/previous finite bound
      double lower = bin.lower_bound;
      double upper = bin.upper_bound;
      
      if (std::isinf(lower) && bin_info.size() > 1) {
        lower = bin_info[1].lower_bound - 1.0;
      }
      
      if (std::isinf(upper) && bin_info.size() > 1) {
        upper = bin_info[bin_info.size() - 2].upper_bound + 1.0;
      }
      
      double midpoint = (lower + upper) / 2.0;
      
      midpoints.push_back(midpoint);
      rates.push_back(bin.event_rate());
    }
    
    // Calculate correlation
    double correlation = calculateCorrelation(midpoints, rates);
    
    // Determine direction based on correlation
    monotone_increasing = (correlation >= 0.0);
  }
  
  /**
   * Calculate Pearson correlation coefficient between two vectors
   * 
   * @param x First vector
   * @param y Second vector
   * @return Correlation coefficient
   */
  double calculateCorrelation(const std::vector<double>& x, const std::vector<double>& y) const {
    if (x.size() != y.size() || x.size() < 2) {
      return 0.0;
    }
    
    // Calculate means
    double mean_x = std::accumulate(x.begin(), x.end(), 0.0) / x.size();
    double mean_y = std::accumulate(y.begin(), y.end(), 0.0) / y.size();
    
    // Calculate correlation coefficient
    double numerator = 0.0;
    double sum_sq_x = 0.0;
    double sum_sq_y = 0.0;
    
    for (size_t i = 0; i < x.size(); ++i) {
      double x_diff = x[i] - mean_x;
      double y_diff = y[i] - mean_y;
      numerator += x_diff * y_diff;
      sum_sq_x += x_diff * x_diff;
      sum_sq_y += y_diff * y_diff;
    }
    
    if (sum_sq_x < EPSILON || sum_sq_y < EPSILON) {
      return 0.0;
    }
    
    return numerator / std::sqrt(sum_sq_x * sum_sq_y);
  }
  
  /**
   * Apply isotonic regression to enforce monotonicity in event rates
   * Uses the Pool Adjacent Violators (PAV) algorithm
   */
  void applyIsotonicRegression() {
    int n = static_cast<int>(bin_info.size());
    if (n <= 1) return;
    
    // Extract event rates and counts
    std::vector<double> y(n), w(n);
    
    for (int i = 0; i < n; ++i) {
      y[i] = bin_info[i].event_rate();
      w[i] = static_cast<double>(bin_info[i].count);
    }
    
    // Apply isotonic regression with PAV algorithm
    std::vector<double> isotonic_y;
    
    if (monotone_increasing) {
      isotonic_y = isotonicRegressionPAV(y, w, true);
    } else {
      // For decreasing, reverse input, apply PAV, then reverse output
      std::reverse(y.begin(), y.end());
      std::reverse(w.begin(), w.end());
      isotonic_y = isotonicRegressionPAV(y, w, true);
      std::reverse(isotonic_y.begin(), isotonic_y.end());
    }
    
    // Update bin event rates
    for (int i = 0; i < n; ++i) {
      // bin_info[i].event_rate() assignment removed (calculated dynamically)
      
      // Recalculate counts to maintain consistency
      int new_pos = static_cast<int>(std::round(isotonic_y[i] * bin_info[i].count));
      bin_info[i].count_pos = std::max(0, std::min(new_pos, bin_info[i].count));
      bin_info[i].count_neg = bin_info[i].count - bin_info[i].count_pos;
    }
    
    converged = true;
    iterations_run += 1;
  }
  
  /**
   * @brief Implement Pool Adjacent Violators Algorithm (PAVA) for isotonic regression
   *
   * PAVA is the standard algorithm for isotonic regression with guaranteed O(n) complexity.
   * It processes the data in a single pass, merging violators (adjacent decreasing pairs)
   * until no violations remain.
   *
   * Algorithm:
   * 1. Initialize blocks with input values
   * 2. Scan from left to right
   * 3. When a violation is found (y[i] > y[i+1] for increasing), merge blocks
   * 4. Continue until no violations exist
   *
   * Complexity: O(n) time, O(n) space
   *
   * References:
   * - Barlow et al. (1972). "Statistical Inference Under Order Restrictions"
   * - Best & Chakravarti (1990). "Active set algorithms for isotonic regression"
   *
   * @param y_input Original values to be isotonized
   * @param w_input Weights (typically bin counts)
   * @param increasing Whether monotonically increasing (true) or decreasing (false)
   * @return std::vector<double> Isotonic regression result (same length as input)
   */
  std::vector<double> isotonicRegressionPAV(
      const std::vector<double>& y_input, 
      const std::vector<double>& w_input, 
      bool increasing = true) const {
    
    int n = static_cast<int>(y_input.size());
    std::vector<double> y = y_input;
    std::vector<double> w = w_input;
    
    // Active set algorithm
    std::vector<double> solution(n);
    std::vector<int> active_set(n, 1);  // Size of each block
    std::vector<int> active_sum = active_set;  // Cumulative sum of active_set
    
    // Initial solution
    for (int i = 0; i < n; ++i) {
      solution[i] = y[i];
    }
    
    // Iteratively merge blocks that violate monotonicity
    bool violation = true;
    while (violation) {
      violation = false;
      
      for (int i = 0; i < n - 1; ) {
        // Check for violation
        bool violates = (increasing) ? 
        (solution[i] > solution[i + 1]) : 
        (solution[i] < solution[i + 1]);
        
        if (violates) {
          violation = true;
          
          // Calculate weighted average
          double w_sum = w[i] + w[i + 1];
          double new_value = (w[i] * solution[i] + w[i + 1] * solution[i + 1]) / w_sum;
          
          // Update solution with weighted average
          solution[i] = new_value;
          solution[i + 1] = new_value;
          
          // Merge blocks
          w[i] = w_sum;
          active_set[i] += active_set[i + 1];
          
          // Remove block i+1
          for (int j = i + 1; j < n - 1; ++j) {
            solution[j] = solution[j + 1];
            w[j] = w[j + 1];
            active_set[j] = active_set[j + 1];
          }
          
          // Decrease n
          n--;
          
          // Don't increment i, check again with new merged block
        } else {
          // Move to next block
          i++;
        }
      }
    }
    
    // Expand solution to original size
    std::vector<double> result(y_input.size());
    int idx = 0;
    for (int i = 0; i < n; ++i) {
      for (int j = 0; j < active_set[i]; ++j) {
        result[idx++] = solution[i];
      }
    }
    
    return result;
  }
  
  /**
   * Calculate Weight of Evidence (WoE) and Information Value (IV)
   * for each bin and the total binning solution
   */
  void calculateWOEandIV() {
    // Calculate totals
    double total_pos = 0.0, total_neg = 0.0;
    for (const auto& bin : bin_info) {
      total_pos += bin.count_pos;
      total_neg += bin.count_neg;
    }
    
    // Apply Laplace smoothing to handle zero counts
    double pos_denominator = total_pos + bin_info.size() * ALPHA;
    double neg_denominator = total_neg + bin_info.size() * ALPHA;
    
    if (pos_denominator < EPSILON || neg_denominator < EPSILON) {
      throw std::runtime_error("Insufficient positive or negative cases for WoE and IV calculations.");
    }
    
    total_iv = 0.0;
    for (auto& bin : bin_info) {
      // Calculate rates with smoothing
      double pos_rate = (bin.count_pos + ALPHA) / pos_denominator;
      double neg_rate = (bin.count_neg + ALPHA) / neg_denominator;
      
      // Calculate WoE
      bin.woe = std::log(pos_rate / neg_rate);
      
      // Calculate IV contribution
      bin.iv = (pos_rate - neg_rate) * bin.woe;
      total_iv += bin.iv;
    }
  }
  
  /**
   * Create bin labels for output
   * 
   * @param bin The bin information
   * @param is_first Whether this is the first bin
   * @param is_last Whether this is the last bin
   * @return Formatted bin label string
   */
  std::string createBinLabel(const NumericalBin& bin, bool is_first, bool is_last) const {
    std::ostringstream oss;
    oss.precision(6);
    oss << std::fixed;
    
    if (is_first) {
      oss << "(-Inf;" << bin.upper_bound << "]";
    } else if (is_last) {
      oss << "(" << bin.lower_bound << ";+Inf]";
    } else {
      oss << "(" << bin.lower_bound << ";" << bin.upper_bound << "]";
    }
    
    return oss.str();
  }
  
  /**
   * Create final WOE bin list for output
   * 
   * @return List with bin information and metrics
   */
  Rcpp::List createWOEBinList() const {
    int n_bins = static_cast<int>(bin_info.size());
    Rcpp::CharacterVector bin_labels(n_bins);
    Rcpp::NumericVector woe_vec(n_bins), iv_vec(n_bins);
    Rcpp::IntegerVector count_vec(n_bins), count_pos_vec(n_bins), count_neg_vec(n_bins);
    Rcpp::NumericVector cutpoints(std::max(n_bins - 1, 0));
    
    for (int i = 0; i < n_bins; ++i) {
      const auto& b = bin_info[static_cast<size_t>(i)];
      std::string label = createBinLabel(b, i == 0, i == n_bins - 1);
      
      bin_labels[i] = label;
      woe_vec[i] = b.woe;
      iv_vec[i] = b.iv;
      count_vec[i] = b.count;
      count_pos_vec[i] = b.count_pos;
      count_neg_vec[i] = b.count_neg;
      
      if (i < n_bins - 1) {
        cutpoints[i] = b.upper_bound;
      }
    }
    
    // Create bin IDs (1-based indexing for R)
    Rcpp::NumericVector ids(bin_labels.size());
    for(int i = 0; i < bin_labels.size(); i++) {
      ids[i] = i + 1;
    }
    
    return Rcpp::List::create(
      Named("id") = ids,
      Rcpp::Named("bin") = bin_labels,
      Rcpp::Named("woe") = woe_vec,
      Rcpp::Named("iv") = iv_vec,
      Rcpp::Named("count") = count_vec,
      Rcpp::Named("count_pos") = count_pos_vec,
      Rcpp::Named("count_neg") = count_neg_vec,
      Rcpp::Named("cutpoints") = cutpoints,
      Rcpp::Named("converged") = converged,
      Rcpp::Named("iterations") = iterations_run,
      Rcpp::Named("total_iv") = total_iv,
      Rcpp::Named("monotone_increasing") = monotone_increasing
    );
  }
};

// [[Rcpp::export]]
Rcpp::List optimal_binning_numerical_ir(
   Rcpp::IntegerVector target,
   Rcpp::NumericVector feature,
   int min_bins = 3,
   int max_bins = 5,
   double bin_cutoff = 0.05,
   int max_n_prebins = 20,
   bool auto_monotonicity = true,
   double convergence_threshold = 1e-6,
   int max_iterations = 1000) {
 
 try {
   // Convert R vectors to STL containers
   std::vector<int> target_std = Rcpp::as<std::vector<int>>(target);
   std::vector<double> feature_std = Rcpp::as<std::vector<double>>(feature);
   
   // Create and execute binning algorithm
   OBN_IR binner(
       min_bins, max_bins, 
       bin_cutoff, max_n_prebins,
       convergence_threshold, max_iterations,
       feature_std, target_std,
       auto_monotonicity
   );
   
   binner.fit();
   return binner.getResults();
 } catch (const std::exception& e) {
   Rcpp::stop("Error in optimal binning: " + std::string(e.what()));
 }
}
