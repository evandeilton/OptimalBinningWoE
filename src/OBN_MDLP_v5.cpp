// [[Rcpp::plugins(cpp11)]]
// [[Rcpp::depends(Rcpp)]]

#include <Rcpp.h>
#include <vector>
#include <algorithm>
#include <cmath>
#include <limits>
#include <string>
#include <sstream>
#include <iomanip>
#include <set>
#include <numeric>
#include <unordered_map>

using namespace Rcpp;

// Include shared headers
#include "common/optimal_binning_common.h"
#include "common/bin_structures.h"
#include "common/entropy_utils.h"

using namespace Rcpp;
using namespace OptimalBinning;


/**
 * @brief Optimal Binning for Numerical Features using MDLP (Minimum Description Length Principle)
 * 
 * This class implements the Minimum Description Length Principle for binning numerical features.
 * The algorithm seeks to find the optimal number of bins that minimizes information loss
 * while maintaining model interpretability. It includes:
 * 
 * 1. Pre-binning based on frequency or quantiles
 * 2. Iterative bin merging based on MDL cost optimization
 * 3. Handling of rare bins (those with frequency below a threshold)
 * 4. Monotonicity enforcement of Weight of Evidence (WoE)
 * 5. Efficient handling of edge cases and numeric stability
 * 6. Comprehensive validation of the binning results
 * 
 * The algorithm is particularly well-suited for credit scoring and risk modeling applications
 * where interpretability and robustness are critical.
 */
class OBN_MDLP {
private:
  // Input data and parameters
  std::vector<double> feature;
  std::vector<int> target;
  int min_bins;
  int max_bins;
  double bin_cutoff;
  int max_n_prebins;
  double convergence_threshold;
  int max_iterations;
  double laplace_smoothing;
  
  // Output and state tracking
  int iterations_run;
  bool converged;
  int n_unique; // Number of unique feature values
  
  /**
   * @brief NumericalBin structure to store bin information
   * 
   * Each bin contains:
   * - Lower and upper bounds of the numeric range
   * - Count statistics (total, positives, negatives)
   * - Weight of Evidence (WoE) and Information Value (IV) metrics
   */
  // Local NumericalBin definition removed

  
  // Vector to store the bins created by the algorithm
  std::vector<NumericalBin> bins;
  
  /**
   * @brief Calculate the entropy of a bin using optimized LUT
   *
   * Uses entropy_binary() from entropy_utils.h which employs a lookup table
   * for counts 0-100 (30-50% speedup). Falls back to runtime calculation for
   * larger counts.
   *
   * Entropy measures the impurity/uncertainty within a bin:
   * H(S) = -p log2(p) - q log2(q)
   * where p is the proportion of positive cases and q is the proportion of negative cases
   * 
   * @param pos Number of positive cases in the bin
   * @param neg Number of negative cases in the bin
   * @return double Entropy value (0 if pure bin)
   */
  double calculate_entropy(int pos, int neg) const {
    // Use optimized entropy_binary() from entropy_utils.h
    // This uses a pre-computed lookup table for counts 0-100 (30-50% speedup)
    return entropy_binary(pos, neg);
  }
  
  /**
   * @brief Calculate the MDL (Minimum Description Length) cost of a binning
   * 
   * The MDL principle combines:
   * 1. Model complexity (cost of encoding the model)
   * 2. Data fit quality (cost of encoding data given the model)
   * 
   * A lower MDL cost indicates a better binning solution.
   * 
   * @param current_bins Vector of bins to evaluate
   * @return double MDL cost value
   */
  double calculate_mdl_cost(const std::vector<NumericalBin>& current_bins) const {
    // Calculate total statistics for all bins
    int total_count = 0;
    int total_pos = 0;
    
    for (const auto& bin : current_bins) {
      total_count += bin.count;
      total_pos += bin.count_pos;
    }
    
    int total_neg = total_count - total_pos;
    
    // If all samples are positive or all are negative, return infinity (no meaningful split)
    if (total_pos == 0 || total_neg == 0) {
      return std::numeric_limits<double>::infinity();
    }
    
    // Model cost: Cost of encoding the model complexity (number of bins)
    // More bins = higher complexity = higher cost
    double model_cost = std::log2(static_cast<double>(current_bins.size()) - 1.0);
    
    // Data cost: Initial entropy of all data
    double data_cost = total_count * calculate_entropy(total_pos, total_neg);
    
    // Subtract the weighted entropy of each bin from the data cost
    // This represents the information gain from the binning
    for (const auto& bin : current_bins) {
      data_cost -= bin.count * calculate_entropy(bin.count_pos, bin.count_neg);
    }
    
    // Total MDL cost is the sum of model and data costs
    return model_cost + data_cost;
  }
  
  /**
   * @brief Merge two adjacent bins
   * 
   * Combines the statistics of two adjacent bins into the left bin and removes the right bin.
   * 
   * @param index Index of the left bin to merge with the bin at index+1
   */
  void merge_bins(size_t index) {
    if (index >= bins.size() - 1) {
      Rcpp::warning("Invalid bin index for merging: %d", index);
      return;
    }
    
    NumericalBin& left = bins[index];
    NumericalBin& right = bins[index + 1];
    
    // Update bin boundaries
    left.upper_bound = right.upper_bound;
    
    // Combine statistics
    left.count += right.count;
    left.count_pos += right.count_pos;
    left.count_neg += right.count_neg;
    
    // Update entropy of the merged bin
    left.entropy = calculate_entropy(left.count_pos, left.count_neg);
    
    // Remove the right bin from the vector
    bins.erase(bins.begin() + index + 1);
  }
  
  /**
   * @brief Calculate Weight of Evidence (WoE) and Information Value (IV) for all bins
   * 
   * WoE measures the predictive power of a bin:
   * WoE = ln(% of positives / % of negatives)
   * 
   * IV measures the overall predictive power of the variable:
   * IV = Σ (% of positives - % of negatives) * WoE
   * 
   * This implementation includes Laplace smoothing to handle rare events.
   */
  void calculate_woe_iv() {
    // Count totals across all bins
    int total_pos = 0, total_neg = 0;
    for (const auto& bin : bins) {
      total_pos += bin.count_pos;
      total_neg += bin.count_neg;
    }
    
    // Validate that we have both positive and negative samples
    if (total_pos == 0 || total_neg == 0) {
      Rcpp::stop("All target values are the same. Cannot compute WoE/IV.");
    }
    
    // Calculate WoE and IV for each bin
    double total_iv = 0.0;
    
    for (auto &bin : bins) {
      // Apply Laplace smoothing: add a small constant to avoid division by zero
      // and reduce extreme values in rare events
      double smoothed_pos = bin.count_pos + laplace_smoothing;
      double smoothed_neg = bin.count_neg + laplace_smoothing;
      
      double total_smoothed_pos = total_pos + bins.size() * laplace_smoothing;
      double total_smoothed_neg = total_neg + bins.size() * laplace_smoothing;
      
      double dist_pos = smoothed_pos / total_smoothed_pos;
      double dist_neg = smoothed_neg / total_smoothed_neg;
      
      // Calculate WoE with protection against edge cases
      if (dist_pos <= 0.0 && dist_neg <= 0.0) {
        bin.woe = 0.0;
      } else if (dist_pos <= 0.0) {
        bin.woe = -20.0; // Capped negative infinity for numerical stability
      } else if (dist_neg <= 0.0) {
        bin.woe = 20.0;  // Capped positive infinity for numerical stability
      } else {
        bin.woe = std::log(dist_pos / dist_neg);
      }
      
      // Calculate IV contribution for this bin
      if (std::isfinite(bin.woe)) {
        bin.iv = (dist_pos - dist_neg) * bin.woe;
        total_iv += bin.iv;
      } else {
        bin.iv = 0.0;
      }
    }
  }
  
  /**
   * @brief Check if bins have monotonic WoE values
   * 
   * Monotonicity means that WoE values consistently increase (or decrease) across bins.
   * This is often desired for interpretability and stability in credit scoring models.
   * 
   * @return bool True if WoE values are monotonically increasing
   */
  bool is_monotonic() const {
    if (bins.empty()) return true;
    
    double prev_woe = bins[0].woe;
    for (size_t i = 1; i < bins.size(); ++i) {
      if (bins[i].woe < prev_woe) {
        return false;
      }
      prev_woe = bins[i].woe;
    }
    return true;
  }
  
  /**
   * @brief Enforce monotonicity by merging bins
   * 
   * If WoE values are not monotonically increasing, this function merges bins
   * to achieve monotonicity, prioritizing minimal information loss.
   */
  void enforce_monotonicity() {
    if (bins.size() <= 1) return; // Nothing to enforce if 0 or 1 bin
    
    bool monotonic = false;
    while (!monotonic && iterations_run < max_iterations) {
      iterations_run++;
      monotonic = true;
      
      // Find and merge bin pairs that violate monotonicity
      for (size_t i = 1; i < bins.size(); ++i) {
        if (bins[i].woe < bins[i - 1].woe) {
          // Determine which merge causes less information loss
          double iv_loss_left = bins[i-1].iv + bins[i].iv;
          
          std::vector<NumericalBin> temp_bins_merge_left = bins;
          NumericalBin& left_in_temp = temp_bins_merge_left[i-1];
          NumericalBin& right_in_temp = temp_bins_merge_left[i];
          
          left_in_temp.upper_bound = right_in_temp.upper_bound;
          left_in_temp.count += right_in_temp.count;
          left_in_temp.count_pos += right_in_temp.count_pos;
          left_in_temp.count_neg += right_in_temp.count_neg;
          
          temp_bins_merge_left.erase(temp_bins_merge_left.begin() + i);
          
          // Recalculate WoE and IV for the merged bins
          int total_pos = 0, total_neg = 0;
          for (const auto& bin : temp_bins_merge_left) {
            total_pos += bin.count_pos;
            total_neg += bin.count_neg;
          }
          
          // Calculate IV after merge
          double smoothed_pos = left_in_temp.count_pos + laplace_smoothing;
          double smoothed_neg = left_in_temp.count_neg + laplace_smoothing;
          double total_smoothed_pos = total_pos + temp_bins_merge_left.size() * laplace_smoothing;
          double total_smoothed_neg = total_neg + temp_bins_merge_left.size() * laplace_smoothing;
          double dist_pos = smoothed_pos / total_smoothed_pos;
          double dist_neg = smoothed_neg / total_smoothed_neg;
          
          double new_woe;
          if (dist_pos <= 0.0 && dist_neg <= 0.0) {
            new_woe = 0.0;
          } else if (dist_pos <= 0.0) {
            new_woe = -20.0;
          } else if (dist_neg <= 0.0) {
            new_woe = 20.0;
          } else {
            new_woe = std::log(dist_pos / dist_neg);
          }
          
          double new_iv = std::isfinite(new_woe) ? (dist_pos - dist_neg) * new_woe : 0.0;
          double iv_loss_after_merge = new_iv;
          
          // If merging results in less information loss, merge bins
          if (iv_loss_after_merge >= iv_loss_left * 0.5) {
            merge_bins(i - 1);
          } else if (i < bins.size() - 1) {
            // Try merging the right bin with the next bin if possible
            merge_bins(i);
          } else {
            // If we're at the rightmost bin and can't merge right, merge left
            merge_bins(i - 1);
          }
          
          monotonic = false;
          if (bins.size() <= (size_t)min_bins) {
            return; // Stop if we've reached minimum number of bins
          }
          break;
        }
      }
    }
    
    if (iterations_run >= max_iterations) {
      converged = false;
      Rcpp::warning("Monotonicity enforcement did not converge after %d iterations", max_iterations);
    }
  }
  
  /**
   * @brief Validate the created bin structure
   * 
   * Performs sanity checks on the bins:
   * - Ensure bins are not empty
   * - Verify bins are ordered correctly
   * - Check for gaps between bins
   * - Validate boundary conditions (-Inf to +Inf)
   * 
   * @throws Rcpp::exception if validation fails
   */
  void validate_bins() const {
    // Check if bins exist
    if (bins.empty()) {
      Rcpp::stop("No bins available after binning.");
    }
    
    // Check bin ordering
    for (size_t i = 1; i < bins.size(); ++i) {
      if (bins[i - 1].upper_bound > bins[i].upper_bound) {
        Rcpp::stop("Bins not ordered correctly by upper_bound.");
      }
    }
    
    // Check boundary conditions
    if (bins.front().lower_bound != -std::numeric_limits<double>::infinity()) {
      Rcpp::stop("First bin doesn't start with -Inf.");
    }
    
    if (bins.back().upper_bound != std::numeric_limits<double>::infinity()) {
      Rcpp::stop("Last bin doesn't end with +Inf.");
    }
    
    // Check for gaps between bins
    for (size_t i = 1; i < bins.size(); ++i) {
      if (std::abs(bins[i].lower_bound - bins[i - 1].upper_bound) > 1e-10) {
        Rcpp::stop("Gap between bins " + std::to_string(i - 1) + " and " + std::to_string(i));
      }
    }
    
    // Check for empty bins
    for (size_t i = 0; i < bins.size(); ++i) {
      if (bins[i].count == 0) {
        Rcpp::warning("NumericalBin %d is empty", i);
      }
    }
  }
  
  /**
   * @brief Merge bins with frequency below the threshold
   * 
   * Identifies and merges bins that have a proportion of records below the bin_cutoff.
   * This improves statistical stability of the bins.
   */
  void merge_rare_bins() {
    double total = static_cast<double>(feature.size());
    bool merged = true;
    
    while (merged && iterations_run < max_iterations && bins.size() > (size_t)min_bins) {
      merged = false;
      
      // Find the bin with the lowest frequency
      double min_freq = std::numeric_limits<double>::infinity();
      size_t min_idx = bins.size();
      
      for (size_t i = 0; i < bins.size(); ++i) {
        double freq = static_cast<double>(bins[i].count) / total;
        if (freq < min_freq) {
          min_freq = freq;
          min_idx = i;
        }
      }
      
      // If the smallest bin is below the threshold, merge it
      if (min_freq < bin_cutoff && min_idx < bins.size()) {
        // Determine the optimal merge direction (left or right)
        if (min_idx == 0) {
          // Leftmost bin, can only merge right
          merge_bins(0);
        } else if (min_idx == bins.size() - 1) {
          // Rightmost bin, can only merge left
          merge_bins(min_idx - 1);
        } else {
          // Middle bin, choose direction based on minimizing information loss
          
          // Use entropy to decide merge direction (information before merge stored in bins)
          
          // Option 1: Merge with left bin
          std::vector<NumericalBin> temp_bins_left = bins;
          NumericalBin left_merged = temp_bins_left[min_idx-1];
          left_merged.upper_bound = temp_bins_left[min_idx].upper_bound;
          left_merged.count += temp_bins_left[min_idx].count;
          left_merged.count_pos += temp_bins_left[min_idx].count_pos;
          left_merged.count_neg += temp_bins_left[min_idx].count_neg;
          
          // Option 2: Merge with right bin
          std::vector<NumericalBin> temp_bins_right = bins;
          NumericalBin right_merged = temp_bins_right[min_idx];
          right_merged.upper_bound = temp_bins_right[min_idx+1].upper_bound;
          right_merged.count += temp_bins_right[min_idx+1].count;
          right_merged.count_pos += temp_bins_right[min_idx+1].count_pos;
          right_merged.count_neg += temp_bins_right[min_idx+1].count_neg;
          
          // Calculate entropy for merged bins
          double entropy_left_merged = calculate_entropy(left_merged.count_pos, left_merged.count_neg);
          double entropy_right_merged = calculate_entropy(right_merged.count_pos, right_merged.count_neg);
          
          // Choose direction with lower entropy (better information preservation)
          if (entropy_left_merged <= entropy_right_merged) {
            merge_bins(min_idx - 1);
          } else {
            merge_bins(min_idx);
          }
        }
        
        merged = true;
      } else {
        break; // No more rare bins to merge
      }
      
      iterations_run++;
    }
    
    if (iterations_run >= max_iterations && merged) {
      converged = false;
      Rcpp::warning("Rare bin merging did not complete within %d iterations", max_iterations);
    }
  }
  
public:
  /**
   * @brief Constructor for OBN_MDLP
   * 
   * @param feature Vector of numerical feature values
   * @param target Vector of binary target values (0/1)
   * @param min_bins Minimum number of bins allowed
   * @param max_bins Maximum number of bins allowed
   * @param bin_cutoff Minimum proportion of records required in a bin
   * @param max_n_prebins Maximum number of pre-bins before merging
   * @param convergence_threshold Threshold for convergence
   * @param max_iterations Maximum iterations for optimization
   * @param laplace_smoothing Smoothing parameter for WoE calculation
   */
  OBN_MDLP(
    const std::vector<double>& feature,
    const std::vector<int>& target,
    int min_bins = 3,
    int max_bins = 5,
    double bin_cutoff = 0.05,
    int max_n_prebins = 20,
    double convergence_threshold = 1e-6,
    int max_iterations = 1000,
    double laplace_smoothing = 0.5
  ) : feature(feature), target(target), min_bins(min_bins), max_bins(max_bins),
  bin_cutoff(bin_cutoff), max_n_prebins(max_n_prebins),
  convergence_threshold(convergence_threshold), max_iterations(max_iterations),
  laplace_smoothing(laplace_smoothing), iterations_run(0), converged(true), n_unique(0) {
    
    // Input validation
    validate_inputs();
    
    // Count unique values in feature
    std::set<double> uniq(feature.begin(), feature.end());
    n_unique = static_cast<int>(uniq.size());
    
    // Adjust min_bins and max_bins based on unique values
    if (n_unique < min_bins) {
      min_bins = n_unique > 1 ? n_unique : 1;
      if (max_bins < min_bins) {
        max_bins = min_bins;
      }
    }
    if (n_unique < max_bins) {
      max_bins = n_unique;
    }
  }
  
  /**
   * @brief Validate input data and parameters
   * 
   * Checks validity of feature and target vectors, parameter ranges, etc.
   * 
   * @throws Rcpp::exception if validation fails
   */
  void validate_inputs() {
    // Check vector sizes
    if (feature.size() != target.size()) {
      Rcpp::stop("feature and target must have the same size.");
    }
    
    // Check if vectors are empty
    if (feature.empty()) {
      Rcpp::stop("Input vectors cannot be empty.");
    }
    
    // Check parameters
    if (min_bins < 1) {
      Rcpp::stop("min_bins must be at least 1.");
    }
    if (max_bins < min_bins) {
      Rcpp::stop("max_bins must be >= min_bins.");
    }
    if (bin_cutoff <= 0.0 || bin_cutoff >= 1.0) {
      Rcpp::stop("bin_cutoff must be between 0 and 1.");
    }
    if (max_n_prebins < 2) {
      Rcpp::stop("max_n_prebins must be >= 2.");
    }
    if (laplace_smoothing < 0.0) {
      Rcpp::stop("laplace_smoothing must be >= 0.");
    }
    
    // Check target values (must be binary)
    for (int t : target) {
      if (t != 0 && t != 1) {
        Rcpp::stop("target must contain only 0 and 1.");
      }
    }
    
    // Count NaN/Inf in feature
    int nan_count = 0;
    int inf_count = 0;
    
    for (double f : feature) {
      if (std::isnan(f)) {
        nan_count++;
      } else if (std::isinf(f)) {
        inf_count++;
      }
    }
    
    if (nan_count > 0 || inf_count > 0) {
      Rcpp::warning("%d NaN and %d Inf values found in feature. These will be handled separately.", 
                    nan_count, inf_count);
    }
  }
  
  /**
   * @brief Perform the optimal binning
   * 
   * Main algorithm execution:
   * 1. Create initial pre-bins
   * 2. Apply MDL-based merging
   * 3. Handle rare bins
   * 4. Enforce monotonicity if needed
   * 5. Calculate final metrics
   */
  void fit() {
    // Handle special cases first
    if (handle_special_cases()) {
      return;
    }
    
    // Sort data for binning
    std::vector<std::pair<double, int>> sorted_data = prepare_sorted_data();
    
    // Handle case with very few unique values
    int actual_unique = count_unique_values(sorted_data);
    if (actual_unique <= 2) {
      handle_few_unique_values(sorted_data, actual_unique);
      return;
    }
    
    // Handle all identical values
    if (sorted_data.front().first == sorted_data.back().first) {
      handle_identical_values(sorted_data);
      return;
    }
    
    // Create initial pre-bins
    create_prebins(sorted_data);
    
    // Apply MDLP-based merging
    apply_mdl_merging();
    
    // Handle rare bins
    merge_rare_bins();
    
    // Calculate WoE and IV
    calculate_woe_iv();
    
    // Enforce monotonicity if needed
    if (bins.size() > (size_t)min_bins && !is_monotonic()) {
      enforce_monotonicity();
      // Recalculate after monotonicity enforcement
      calculate_woe_iv();
    }
    
    // Final validation
    validate_bins();
  }
  
  /**
   * @brief Handle special cases that don't require the full algorithm
   * 
   * @return bool True if a special case was handled and no further processing is needed
   */
  bool handle_special_cases() {
    // Case 1: Empty feature vector
    if (feature.empty()) {
      Rcpp::stop("Empty feature vector.");
      return true;
    }
    
    // Case 2: All the same target value
    bool all_same_target = true;
    int first_target = target[0];
    for (size_t i = 1; i < target.size(); ++i) {
      if (target[i] != first_target) {
        all_same_target = false;
        break;
      }
    }
    
    if (all_same_target) {
      // Create a single bin containing all data
      NumericalBin bin;
      bin.lower_bound = -std::numeric_limits<double>::infinity();
      bin.upper_bound = std::numeric_limits<double>::infinity();
      bin.count = static_cast<int>(feature.size());
      bin.count_pos = (first_target == 1) ? bin.count : 0;
      bin.count_neg = (first_target == 0) ? bin.count : 0;
      bin.woe = 0.0;  // WoE undefined when all same target
      bin.iv = 0.0;   // IV is zero when WoE is zero
      bins.push_back(bin);
      
      converged = true;
      iterations_run = 0;
      return true;
    }
    
    return false;
  }
  
  /**
   * @brief Prepare sorted data for binning
   * 
   * @return std::vector<std::pair<double, int>> Sorted pairs of (feature, target)
   */
  std::vector<std::pair<double, int>> prepare_sorted_data() {
    std::vector<std::pair<double, int>> sorted_data;
    sorted_data.reserve(feature.size());
    
    // Create pairs and filter out NaNs
    for (size_t i = 0; i < feature.size(); ++i) {
      if (!std::isnan(feature[i])) {
        sorted_data.emplace_back(feature[i], target[i]);
      }
    }
    
    // Sort by feature value
    std::sort(sorted_data.begin(), sorted_data.end(),
              [](const std::pair<double, int> &a, const std::pair<double, int> &b) {
                return a.first < b.first;
              });
    
    return sorted_data;
  }
  
  /**
   * @brief Count the number of unique values in sorted data
   * 
   * @param sorted_data Sorted vector of (feature, target) pairs
   * @return int Number of unique feature values
   */
  int count_unique_values(const std::vector<std::pair<double, int>>& sorted_data) {
    if (sorted_data.empty()) return 0;
    
    int actual_unique = 1;
    for (size_t i = 1; i < sorted_data.size(); ++i) {
      if (sorted_data[i].first != sorted_data[i-1].first) {
        actual_unique++;
      }
    }
    
    return actual_unique;
  }
  
  /**
   * @brief Handle cases with very few unique values
   * 
   * Creates optimal bins when there are only 1 or 2 unique values.
   * 
   * @param sorted_data Sorted vector of (feature, target) pairs
   * @param actual_unique Number of unique values
   */
  void handle_few_unique_values(const std::vector<std::pair<double, int>>& sorted_data, int actual_unique) {
    // One unique value - create a single bin
    if (actual_unique == 1) {
      NumericalBin bin;
      bin.lower_bound = -std::numeric_limits<double>::infinity();
      bin.upper_bound = std::numeric_limits<double>::infinity();
      bin.count = static_cast<int>(sorted_data.size());
      bin.count_pos = 0;
      bin.count_neg = 0;
      
      for (auto &p : sorted_data) {
        if (p.second == 1) bin.count_pos++; else bin.count_neg++;
      }
      
      bins.push_back(bin);
    } 
    // Two unique values - create two bins
    else if (actual_unique == 2) {
      std::vector<double> uniq_vals;
      uniq_vals.reserve(2);
      uniq_vals.push_back(sorted_data[0].first);
      
      for (size_t i = 1; i < sorted_data.size(); ++i) {
        if (sorted_data[i].first != uniq_vals[0]) {
          uniq_vals.push_back(sorted_data[i].first);
          break;
        }
      }
      
      // Create two bins
      NumericalBin bin1, bin2;
      bin1.lower_bound = -std::numeric_limits<double>::infinity();
      bin1.upper_bound = uniq_vals[0];
      bin1.count = 0; bin1.count_pos = 0; bin1.count_neg = 0;
      
      bin2.lower_bound = uniq_vals[0];
      bin2.upper_bound = std::numeric_limits<double>::infinity();
      bin2.count = 0; bin2.count_pos = 0; bin2.count_neg = 0;
      
      // Assign observations to bins
      for (auto &p : sorted_data) {
        if (p.first <= uniq_vals[0]) {
          bin1.count++;
          if (p.second == 1) bin1.count_pos++; else bin1.count_neg++;
        } else {
          bin2.count++;
          if (p.second == 1) bin2.count_pos++; else bin2.count_neg++;
        }
      }
      
      bins.push_back(bin1);
      bins.push_back(bin2);
    }
    
    calculate_woe_iv();
    converged = true;
    iterations_run = 0;
  }
  
  /**
   * @brief Handle case where all feature values are identical
   * 
   * @param sorted_data Sorted vector of (feature, target) pairs
   */
  void handle_identical_values(const std::vector<std::pair<double, int>>& sorted_data) {
    NumericalBin bin;
    bin.lower_bound = -std::numeric_limits<double>::infinity();
    bin.upper_bound = std::numeric_limits<double>::infinity();
    bin.count = static_cast<int>(sorted_data.size());
    bin.count_pos = 0;
    bin.count_neg = 0;
    
    for (auto &p : sorted_data) {
      if (p.second == 1) bin.count_pos++; else bin.count_neg++;
    }
    
    bins.push_back(bin);
    calculate_woe_iv();
  }
  
  /**
   * @brief Create initial pre-bins before optimization
   * 
   * @param sorted_data Sorted vector of (feature, target) pairs
   */
  void create_prebins(const std::vector<std::pair<double, int>>& sorted_data) {
    // Clear any existing bins
    bins.clear();
    
    // Determine number of records per bin for equal-frequency pre-binning
    int records_per_bin = std::max(1, static_cast<int>(sorted_data.size()) / max_n_prebins);
    
    // Create pre-bins by frequency
    for (size_t i = 0; i < sorted_data.size(); i += records_per_bin) {
      size_t end = std::min(i + records_per_bin, sorted_data.size());
      
      // Create new bin
      NumericalBin bin;
      
      // Set bin boundaries
      if (i == 0) {
        bin.lower_bound = -std::numeric_limits<double>::infinity();
      } else {
        bin.lower_bound = sorted_data[i].first;
      }
      
      if (end == sorted_data.size()) {
        bin.upper_bound = std::numeric_limits<double>::infinity();
      } else {
        bin.upper_bound = sorted_data[end].first;
      }
      
      // Set bin statistics
      bin.count = static_cast<int>(end - i);
      bin.count_pos = 0;
      bin.count_neg = 0;
      
      for (size_t j = i; j < end; ++j) {
        if (sorted_data[j].second == 1) {
          bin.count_pos++;
        } else {
          bin.count_neg++;
        }
      }
      
      // Calculate bin entropy
      bin.entropy = calculate_entropy(bin.count_pos, bin.count_neg);
      
      bins.push_back(bin);
    }
    
    // Fix potential issues with bin boundaries
    fix_bin_boundaries();
  }
  
  /**
   * @brief Fix potential issues with bin boundaries
   * 
   * Ensures bin boundaries are consistent and non-overlapping
   */
  void fix_bin_boundaries() {
    if (bins.size() <= 1) return;
    
    // Fix first bin lower bound
    bins.front().lower_bound = -std::numeric_limits<double>::infinity();
    
    // Fix last bin upper bound
    bins.back().upper_bound = std::numeric_limits<double>::infinity();
    
    // Fix any potential gaps or overlaps between bins
    for (size_t i = 1; i < bins.size(); ++i) {
      if (bins[i].lower_bound != bins[i-1].upper_bound) {
        bins[i].lower_bound = bins[i-1].upper_bound;
      }
    }
  }
  
  /**
   * @brief Apply MDLP-based merging
   * 
   * Iteratively merges bins to minimize the MDL cost
   */
  void apply_mdl_merging() {
    // Iterate until minimum bins reached or no more beneficial merges
    while (bins.size() > (size_t)min_bins && iterations_run < max_iterations) {
      double current_mdl = calculate_mdl_cost(bins);
      double best_mdl = current_mdl;
      size_t best_merge_index = bins.size();
      
      // Try merging each adjacent pair of bins
      for (size_t i = 0; i < bins.size() - 1; ++i) {
        // Create temporary bins for this merge scenario
        std::vector<NumericalBin> temp_bins = bins;
        
        // Merge bins i and i+1
        NumericalBin& left = temp_bins[i];
        NumericalBin& right = temp_bins[i + 1];
        
        left.upper_bound = right.upper_bound;
        left.count += right.count;
        left.count_pos += right.count_pos;
        left.count_neg += right.count_neg;
        
        temp_bins.erase(temp_bins.begin() + i + 1);
        
        // Calculate MDL cost after this merge
        double new_mdl = calculate_mdl_cost(temp_bins);
        
        // Keep track of best merge
        if (new_mdl < best_mdl) {
          best_mdl = new_mdl;
          best_merge_index = i;
        }
      }
      
      // If a beneficial merge was found, execute it
      if (best_merge_index < bins.size()) {
        merge_bins(best_merge_index);
      } else {
        // No beneficial merge found, stop
        break;
      }
      
      // Stop if we've reached the maximum number of bins
      if (bins.size() <= (size_t)max_bins) {
        break;
      }
      
      iterations_run++;
    }
    
    if (iterations_run >= max_iterations) {
      converged = false;
      Rcpp::warning("MDL merging did not complete within %d iterations", max_iterations);
    }
  }
  
  /**
   * @brief Get the results of binning
   * 
   * Returns a list with bin information including:
   * - NumericalBin identifiers and labels
   * - WoE and IV values
   * - Counts (total, positives, negatives)
   * - Cutpoints defining bin boundaries
   * - Algorithm convergence information
   * 
   * @return Rcpp::List Results of the binning process
   */
  Rcpp::List get_results() const {
    Rcpp::StringVector bin_labels;
    Rcpp::NumericVector woe_values;
    Rcpp::NumericVector iv_values;
    Rcpp::IntegerVector count_values;
    Rcpp::IntegerVector count_pos_values;
    Rcpp::IntegerVector count_neg_values;
    Rcpp::NumericVector cutpoints;
    
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(6);
    
    // Format bin information for output
    for (size_t i = 0; i < bins.size(); ++i) {
      oss.str("");
      oss.clear();
      
      // Format bin label as interval
      if (bins[i].lower_bound == -std::numeric_limits<double>::infinity()) {
        oss << "[-Inf";
      } else {
        oss << "[" << bins[i].lower_bound;
      }
      oss << ";";
      if (bins[i].upper_bound == std::numeric_limits<double>::infinity()) {
        oss << "+Inf)";
      } else {
        oss << bins[i].upper_bound << ")";
        cutpoints.push_back(bins[i].upper_bound);
      }
      
      // Add information to output vectors
      bin_labels.push_back(oss.str());
      woe_values.push_back(bins[i].woe);
      iv_values.push_back(bins[i].iv);
      count_values.push_back(bins[i].count);
      count_pos_values.push_back(bins[i].count_pos);
      count_neg_values.push_back(bins[i].count_neg);
    }
    
    // Create bin IDs (1-based indexing)
    Rcpp::NumericVector ids(bin_labels.size());
    for(int i = 0; i < bin_labels.size(); i++) {
      ids[i] = i + 1;
    }
    
    // Calculate total IV
    double total_iv = 0.0;
    for (double iv : iv_values) {
      total_iv += iv;
    }
    
    // Return results as a list
    return Rcpp::List::create(
      Named("id") = ids,
      Named("bin") = bin_labels,
      Named("woe") = woe_values,
      Named("iv") = iv_values,
      Named("count") = count_values,
      Named("count_pos") = count_pos_values,
      Named("count_neg") = count_neg_values,
      Named("cutpoints") = cutpoints,
      Named("total_iv") = total_iv,
      Named("converged") = converged,
      Named("iterations") = iterations_run
    );
  }
};


// [[Rcpp::export]]
Rcpp::List optimal_binning_numerical_mdlp(
   Rcpp::IntegerVector target,
   Rcpp::NumericVector feature,
   int min_bins = 3,
   int max_bins = 5,
   double bin_cutoff = 0.05,
   int max_n_prebins = 20,
   double convergence_threshold = 1e-6,
   int max_iterations = 1000,
   double laplace_smoothing = 0.5
) {
 std::vector<double> feature_vec = Rcpp::as<std::vector<double>>(feature);
 std::vector<int> target_vec = Rcpp::as<std::vector<int>>(target);
 OBN_MDLP binner(feature_vec, target_vec, min_bins, max_bins, bin_cutoff, max_n_prebins, convergence_threshold, max_iterations, laplace_smoothing);
 binner.fit();
 return binner.get_results();
}


