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
class OptimalBinningNumericalMDLP {
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
   * @brief Bin structure to store bin information
   * 
   * Each bin contains:
   * - Lower and upper bounds of the numeric range
   * - Count statistics (total, positives, negatives)
   * - Weight of Evidence (WoE) and Information Value (IV) metrics
   */
  struct Bin {
    double lower_bound;
    double upper_bound;
    int count;
    int count_pos;
    int count_neg;
    double woe;
    double iv;
    double entropy;
    
    // Constructor with initialization to avoid uninitialized values
    Bin() : lower_bound(0), upper_bound(0), count(0), count_pos(0), count_neg(0),
    woe(0), iv(0), entropy(0) {}
  };
  
  // Vector to store the bins created by the algorithm
  std::vector<Bin> bins;
  
  /**
   * @brief Calculate the entropy of a bin
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
    if (pos == 0 || neg == 0) return 0.0; // Pure bin has zero entropy
    double total = static_cast<double>(pos + neg);
    double p = static_cast<double>(pos) / total;
    double q = static_cast<double>(neg) / total;
    return -p * std::log2(p) - q * std::log2(q);
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
  double calculate_mdl_cost(const std::vector<Bin>& current_bins) const {
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
    
    Bin& left = bins[index];
    Bin& right = bins[index + 1];
    
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
          
          std::vector<Bin> temp_bins_merge_left = bins;
          Bin& left_in_temp = temp_bins_merge_left[i-1];
          Bin& right_in_temp = temp_bins_merge_left[i];
          
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
        Rcpp::warning("Bin %d is empty", i);
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
          
          // Information before merge
          double iv_left = bins[min_idx-1].iv;
          double iv_middle = bins[min_idx].iv;
          double iv_right = (min_idx+1 < bins.size()) ? bins[min_idx+1].iv : 0.0;
          
          // Option 1: Merge with left bin
          std::vector<Bin> temp_bins_left = bins;
          Bin left_merged = temp_bins_left[min_idx-1];
          left_merged.upper_bound = temp_bins_left[min_idx].upper_bound;
          left_merged.count += temp_bins_left[min_idx].count;
          left_merged.count_pos += temp_bins_left[min_idx].count_pos;
          left_merged.count_neg += temp_bins_left[min_idx].count_neg;
          
          // Option 2: Merge with right bin
          std::vector<Bin> temp_bins_right = bins;
          Bin right_merged = temp_bins_right[min_idx];
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
   * @brief Constructor for OptimalBinningNumericalMDLP
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
  OptimalBinningNumericalMDLP(
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
      Bin bin;
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
      Bin bin;
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
      Bin bin1, bin2;
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
    Bin bin;
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
      Bin bin;
      
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
        std::vector<Bin> temp_bins = bins;
        
        // Merge bins i and i+1
        Bin& left = temp_bins[i];
        Bin& right = temp_bins[i + 1];
        
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
   * - Bin identifiers and labels
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


//' @title Optimal Binning for Numerical Features using the Minimum Description Length Principle (MDLP)
//'
//' @description
//' This function performs optimal binning for numerical features using the Minimum Description Length Principle (MDLP).
//' It minimizes information loss by merging adjacent bins that reduce the MDL cost, while ensuring monotonicity in the Weight of Evidence (WoE).
//' The algorithm adjusts the number of bins between `min_bins` and `max_bins` and handles rare bins by merging them iteratively.
//' Designed for robust and numerically stable calculations, it incorporates protections for extreme cases and convergence controls.
//'
//' @details
//' ### Core Steps:
//' 1. **Input Validation**: Ensures feature and target are valid, numeric, and binary respectively. Validates consistency between `min_bins` and `max_bins`.
//' 2. **Pre-Binning**: Creates pre-bins based on equal frequencies or unique values if there are few observations.
//' 3. **MDL-Based Merging**: Iteratively merges bins to minimize the MDL cost, which combines model complexity and data fit quality.
//' 4. **Rare Bin Handling**: Merges bins with frequencies below the `bin_cutoff` threshold to ensure statistical stability.
//' 5. **Monotonicity Enforcement**: Adjusts bins to ensure that the WoE values are monotonically increasing or decreasing.
//' 6. **Validation**: Validates the final bin structure for consistency and correctness.
//'
//' ### Mathematical Framework:
//' **Entropy Calculation**: For a bin \( i \) with positive (\( p \)) and negative (\( n \)) counts:
//' \deqn{Entropy = -p \log_2(p) - n \log_2(n)}
//'
//' **MDL Cost**: Combines the cost of the model and data description:
//' \deqn{MDL\_Cost = Model\_Cost + Data\_Cost}
//' Where:
//' \deqn{Model\_Cost = \log_2(Number\_of\_bins - 1)}
//' \deqn{Data\_Cost = Total\_Entropy - \sum_{i} Count_i \times Entropy_i}
//'
//' **Weight of Evidence (WoE)**: For a bin \( i \) with Laplace smoothing parameter:
//' \deqn{WoE_i = \ln\left(\frac{n_{1i} + a}{n_{1} + ma} \cdot \frac{n_{0} + ma}{n_{0i} + a}\right)}
//' Where:
//' \itemize{
//'   \item \eqn{n_{1i}} is the count of positive cases in bin \(i\)
//'   \item \eqn{n_{0i}} is the count of negative cases in bin \(i\)
//'   \item \eqn{n_{1}} is the total count of positive cases
//'   \item \eqn{n_{0}} is the total count of negative cases
//'   \item \eqn{m} is the number of bins
//'   \item a is the Laplace smoothing parameter
//' }
//'
//' **Information Value (IV)**: Summarizes predictive power across all bins:
//' \deqn{IV = \sum_{i} (P(X|Y=1) - P(X|Y=0)) \times WoE_i}
//'
//' ### Features:
//' - Merges bins iteratively to minimize the MDL cost.
//' - Ensures monotonicity of WoE to improve model interpretability.
//' - Handles rare bins by merging categories with low frequencies.
//' - Stable against edge cases like all identical values or insufficient observations.
//' - Efficiently processes large datasets with iterative binning and convergence checks.
//' - Applies Laplace smoothing for robust WoE calculation in sparse bins.
//'
//' @param target An integer binary vector (0 or 1) representing the target variable.
//' @param feature A numeric vector representing the feature to bin.
//' @param min_bins Minimum number of bins (default: 3).
//' @param max_bins Maximum number of bins (default: 5).
//' @param bin_cutoff Minimum proportion of records per bin (default: 0.05).
//' @param max_n_prebins Maximum number of pre-bins before merging (default: 20).
//' @param convergence_threshold Convergence threshold for IV optimization (default: 1e-6).
//' @param max_iterations Maximum number of iterations allowed (default: 1000).
//' @param laplace_smoothing Smoothing parameter for WoE calculation (default: 0.5).
//'
//' @return A list with the following components:
//' \itemize{
//'   \item `id`: A numeric vector with bin identifiers (1-based).
//'   \item `bin`: A vector of bin names representing the intervals.
//'   \item `woe`: A numeric vector with the WoE values for each bin.
//'   \item `iv`: A numeric vector with the IV values for each bin.
//'   \item `count`: An integer vector with the total number of observations in each bin.
//'   \item `count_pos`: An integer vector with the count of positive cases in each bin.
//'   \item `count_neg`: An integer vector with the count of negative cases in each bin.
//'   \item `cutpoints`: A numeric vector of cut points defining the bins.
//'   \item `total_iv`: A numeric value representing the total information value of the binning.
//'   \item `converged`: A boolean indicating whether the algorithm converged.
//'   \item `iterations`: An integer with the number of iterations performed.
//' }
//'
//' @examples
//' \dontrun{
//' # Example usage
//' set.seed(123)
//' target <- sample(0:1, 100, replace = TRUE)
//' feature <- runif(100)
//' result <- optimal_binning_numerical_mdlp(target, feature, min_bins = 3, max_bins = 5)
//' print(result)
//' 
//' # With different parameters
//' result2 <- optimal_binning_numerical_mdlp(
//'   target, 
//'   feature, 
//'   min_bins = 2, 
//'   max_bins = 10,
//'   bin_cutoff = 0.03,
//'   laplace_smoothing = 0.1
//' )
//' 
//' # Print summary statistics
//' print(paste("Total Information Value:", round(result2$total_iv, 4)))
//' print(paste("Number of bins created:", length(result2$bin)))
//' }
//'
//' @references
//' \itemize{
//'   \item Fayyad, U. & Irani, K. (1993). "Multi-interval discretization of continuous-valued 
//'         attributes for classification learning." Proceedings of the International Joint 
//'         Conference on Artificial Intelligence, 1022-1027.
//'   \item Rissanen, J. (1978). "Modeling by shortest data description." Automatica, 14(5), 465-471.
//'   \item Good, I.J. (1952). "Rational Decisions." Journal of the Royal Statistical Society, 
//'         Series B, 14, 107-114. (Origin of Laplace smoothing/additive smoothing)
//' }
//'
//' @export
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
 OptimalBinningNumericalMDLP binner(feature_vec, target_vec, min_bins, max_bins, bin_cutoff, max_n_prebins, convergence_threshold, max_iterations, laplace_smoothing);
 binner.fit();
 return binner.get_results();
}









// // [[Rcpp::plugins(cpp11)]]
// // [[Rcpp::depends(Rcpp)]]
// 
// #include <Rcpp.h>
// #include <vector>
// #include <algorithm>
// #include <cmath>
// #include <limits>
// #include <string>
// #include <sstream>
// #include <iomanip>
// #include <set>
// 
// using namespace Rcpp;
// 
// class OptimalBinningNumericalMDLP {
// private:
//   std::vector<double> feature;
//   std::vector<int> target;
//   int min_bins;
//   int max_bins;
//   double bin_cutoff;
//   int max_n_prebins;
//   double convergence_threshold;
//   int max_iterations;
//   
//   int iterations_run;
//   bool converged;
//   
//   int n_unique; // Número de valores únicos da feature
//   
//   struct Bin {
//     double lower_bound;
//     double upper_bound;
//     int count;
//     int count_pos;
//     int count_neg;
//     double woe;
//     double iv;
//   };
//   
//   std::vector<Bin> bins;
//   
//   double calculate_entropy(int pos, int neg) const {
//     if (pos == 0 || neg == 0) return 0.0;
//     double total = static_cast<double>(pos + neg);
//     double p = pos / total;
//     double q = neg / total;
//     return -p * std::log2(p) - q * std::log2(q);
//   }
//   
//   double calculate_mdl_cost(const std::vector<Bin>& current_bins) const {
//     int total_count = 0;
//     int total_pos = 0;
//     for (const auto& bin : current_bins) {
//       total_count += bin.count;
//       total_pos += bin.count_pos;
//     }
//     int total_neg = total_count - total_pos;
//     
//     // Se todos positivos ou todos negativos, custo infinito (sem informação)
//     if (total_pos == 0 || total_neg == 0) {
//       return std::numeric_limits<double>::infinity();
//     }
//     
//     double model_cost = std::log2(static_cast<double>(current_bins.size()) - 1.0);
//     double data_cost = total_count * calculate_entropy(total_pos, total_neg);
//     
//     for (const auto& bin : current_bins) {
//       data_cost -= bin.count * calculate_entropy(bin.count_pos, bin.count_neg);
//     }
//     
//     return model_cost + data_cost;
//   }
//   
//   void merge_bins(size_t index) {
//     Bin& left = bins[index];
//     Bin& right = bins[index + 1];
//     
//     left.upper_bound = right.upper_bound;
//     left.count += right.count;
//     left.count_pos += right.count_pos;
//     left.count_neg += right.count_neg;
//     
//     bins.erase(bins.begin() + index + 1);
//   }
//   
//   void calculate_woe_iv() {
//     int total_pos = 0, total_neg = 0;
//     for (const auto& bin : bins) {
//       total_pos += bin.count_pos;
//       total_neg += bin.count_neg;
//     }
//     
//     if (total_pos == 0 || total_neg == 0) {
//       Rcpp::stop("All target values are the same. Cannot compute WoE/IV.");
//     }
//     
//     for (auto &bin : bins) {
//       double dist_pos = (double)bin.count_pos / total_pos;
//       double dist_neg = (double)bin.count_neg / total_neg;
//       
//       if (dist_pos <= 0 && dist_neg <= 0) {
//         bin.woe = 0.0;
//       } else if (dist_pos <= 0 && dist_neg > 0) {
//         bin.woe = -std::numeric_limits<double>::infinity();
//       } else if (dist_pos > 0 && dist_neg <= 0) {
//         bin.woe = std::numeric_limits<double>::infinity();
//       } else {
//         bin.woe = std::log(dist_pos / dist_neg);
//       }
//       
//       if ((dist_pos > 0 || dist_neg > 0) && std::isfinite(bin.woe)) {
//         bin.iv = (dist_pos - dist_neg) * bin.woe;
//       } else {
//         bin.iv = 0.0;
//       }
//     }
//   }
//   
//   bool is_monotonic() const {
//     if (bins.empty()) return true;
//     double prev_woe = bins[0].woe;
//     for (size_t i = 1; i < bins.size(); ++i) {
//       if (bins[i].woe < prev_woe) {
//         return false;
//       }
//       prev_woe = bins[i].woe;
//     }
//     return true;
//   }
//   
//   void enforce_monotonicity() {
//     bool monotonic = false;
//     while (!monotonic && iterations_run < max_iterations) {
//       iterations_run++;
//       monotonic = true;
//       for (size_t i = 1; i < bins.size(); ++i) {
//         if (bins[i].woe < bins[i - 1].woe) {
//           merge_bins(i - 1);
//           monotonic = false;
//           if (bins.size() <= (size_t)min_bins) {
//             return;
//           }
//           break;
//         }
//       }
//     }
//     if (iterations_run >= max_iterations) {
//       converged = false;
//     }
//   }
//   
//   void validate_bins() const {
//     if (bins.empty()) {
//       Rcpp::stop("No bins available after binning.");
//     }
//     
//     for (size_t i = 1; i < bins.size(); ++i) {
//       if (bins[i - 1].upper_bound > bins[i].upper_bound) {
//         Rcpp::stop("Bins not ordered correctly by upper_bound.");
//       }
//     }
//     
//     if (bins.front().lower_bound != -std::numeric_limits<double>::infinity()) {
//       Rcpp::stop("First bin doesn't start with -Inf.");
//     }
//     
//     if (bins.back().upper_bound != std::numeric_limits<double>::infinity()) {
//       Rcpp::stop("Last bin doesn't end with +Inf.");
//     }
//     
//     for (size_t i = 1; i < bins.size(); ++i) {
//       if (bins[i].lower_bound != bins[i - 1].upper_bound) {
//         Rcpp::stop("Gap between bins " + std::to_string(i - 1) + " and " + std::to_string(i));
//       }
//     }
//   }
//   
//   void merge_rare_bins() {
//     double total = (double)feature.size();
//     bool merged = true;
//     while (merged && iterations_run < max_iterations) {
//       merged = false;
//       for (size_t i = 0; i < bins.size(); ++i) {
//         double freq = (double)bins[i].count / total;
//         if (freq < bin_cutoff && bins.size() > (size_t)min_bins) {
//           if (i == 0) {
//             merge_bins(0);
//           } else {
//             merge_bins(i - 1);
//           }
//           merged = true;
//           break;
//         }
//       }
//       iterations_run++;
//     }
//     if (iterations_run >= max_iterations) {
//       converged = false;
//     }
//   }
//   
// public:
//   OptimalBinningNumericalMDLP(
//     const std::vector<double>& feature,
//     const std::vector<int>& target,
//     int min_bins = 3,
//     int max_bins = 5,
//     double bin_cutoff = 0.05,
//     int max_n_prebins = 20,
//     double convergence_threshold = 1e-6,
//     int max_iterations = 1000
//   ) : feature(feature), target(target), min_bins(min_bins), max_bins(max_bins),
//   bin_cutoff(bin_cutoff), max_n_prebins(max_n_prebins),
//   convergence_threshold(convergence_threshold), max_iterations(max_iterations),
//   iterations_run(0), converged(true), n_unique(0) {
//     
//     if (feature.size() != target.size()) {
//       Rcpp::stop("feature and target must have the same size.");
//     }
//     if (min_bins < 2) {
//       Rcpp::stop("min_bins must be at least 2.");
//     }
//     if (max_bins < min_bins) {
//       Rcpp::stop("max_bins must be >= min_bins.");
//     }
//     if (bin_cutoff <= 0.0 || bin_cutoff >= 1.0) {
//       Rcpp::stop("bin_cutoff must be between 0 and 1.");
//     }
//     if (max_n_prebins < 2) {
//       Rcpp::stop("max_n_prebins must be >= 2.");
//     }
//     for (int t : target) {
//       if (t != 0 && t != 1) {
//         Rcpp::stop("target must contain only 0 and 1.");
//       }
//     }
//     for (double f : feature) {
//       if (std::isnan(f) || std::isinf(f)) {
//         Rcpp::stop("feature contains NaN or Inf.");
//       }
//     }
//     
//     std::set<double> uniq(feature.begin(), feature.end());
//     n_unique = (int)uniq.size();
//     if (n_unique < min_bins) {
//       min_bins = n_unique;
//       if (max_bins < min_bins) {
//         max_bins = min_bins;
//       }
//     }
//     if (n_unique < max_bins) {
//       max_bins = n_unique;
//     }
//   }
//   
//   void fit() {
//     std::vector<std::pair<double,int>> sorted_data;
//     sorted_data.reserve(feature.size());
//     for (size_t i = 0; i < feature.size(); ++i) {
//       sorted_data.emplace_back(feature[i], target[i]);
//     }
//     std::sort(sorted_data.begin(), sorted_data.end(),
//               [](const std::pair<double,int> &a, const std::pair<double,int> &b) {
//                 return a.first < b.first;
//               });
//     
//     int actual_unique = 1;
//     for (size_t i = 1; i < sorted_data.size(); ++i) {
//       if (sorted_data[i].first != sorted_data[i-1].first) {
//         actual_unique++;
//       }
//     }
//     
//     if (actual_unique <= 2) {
//       // Handle trivial cases
//       if (actual_unique == 1) {
//         // One unique value
//         Bin bin;
//         bin.lower_bound = -std::numeric_limits<double>::infinity();
//         bin.upper_bound = std::numeric_limits<double>::infinity();
//         bin.count = (int)sorted_data.size();
//         bin.count_pos = 0;
//         bin.count_neg = 0;
//         for (auto &p : sorted_data) {
//           if (p.second == 1) bin.count_pos++; else bin.count_neg++;
//         }
//         bins.push_back(bin);
//       } else {
//         // Two unique values
//         std::vector<double> uniq_vals;
//         uniq_vals.reserve(2);
//         uniq_vals.push_back(sorted_data[0].first);
//         for (size_t i = 1; i < sorted_data.size(); ++i) {
//           if (sorted_data[i].first != uniq_vals[0]) {
//             uniq_vals.push_back(sorted_data[i].first);
//             break;
//           }
//         }
//         
//         Bin bin1, bin2;
//         bin1.lower_bound = -std::numeric_limits<double>::infinity();
//         bin1.upper_bound = uniq_vals[0];
//         bin1.count = 0; bin1.count_pos = 0; bin1.count_neg = 0;
//         
//         bin2.lower_bound = uniq_vals[0];
//         bin2.upper_bound = std::numeric_limits<double>::infinity();
//         bin2.count = 0; bin2.count_pos = 0; bin2.count_neg = 0;
//         
//         for (auto &p : sorted_data) {
//           if (p.first <= uniq_vals[0]) {
//             bin1.count++;
//             if (p.second == 1) bin1.count_pos++; else bin1.count_neg++;
//           } else {
//             bin2.count++;
//             if (p.second == 1) bin2.count_pos++; else bin2.count_neg++;
//           }
//         }
//         bins.push_back(bin1);
//         bins.push_back(bin2);
//       }
//       
//       calculate_woe_iv();
//       converged = true;
//       iterations_run = 0;
//       return;
//     }
//     
//     // Handle all identical values
//     if (sorted_data.front().first == sorted_data.back().first) {
//       Bin bin;
//       bin.lower_bound = -std::numeric_limits<double>::infinity();
//       bin.upper_bound = std::numeric_limits<double>::infinity();
//       bin.count = (int)sorted_data.size();
//       bin.count_pos = 0;
//       bin.count_neg = 0;
//       for (auto &p : sorted_data) {
//         if (p.second == 1) bin.count_pos++; else bin.count_neg++;
//       }
//       bins.push_back(bin);
//       calculate_woe_iv();
//       return;
//     }
//     
//     // Cria pré-bins por frequência
//     int records_per_bin = std::max(1, (int)sorted_data.size() / max_n_prebins);
//     for (size_t i = 0; i < sorted_data.size(); i += records_per_bin) {
//       size_t end = std::min(i + records_per_bin, sorted_data.size());
//       Bin bin;
//       if (i == 0) {
//         bin.lower_bound = -std::numeric_limits<double>::infinity();
//       } else {
//         bin.lower_bound = sorted_data[i].first;
//       }
//       if (end == sorted_data.size()) {
//         bin.upper_bound = std::numeric_limits<double>::infinity();
//       } else {
//         bin.upper_bound = sorted_data[end].first;
//       }
//       
//       bin.count = (int)(end - i);
//       bin.count_pos = 0;
//       bin.count_neg = 0;
//       for (size_t j = i; j < end; ++j) {
//         if (sorted_data[j].second == 1) bin.count_pos++; else bin.count_neg++;
//       }
//       bins.push_back(bin);
//     }
//     
//     // MDL merges
//     while (bins.size() > (size_t)min_bins && iterations_run < max_iterations) {
//       double current_mdl = calculate_mdl_cost(bins);
//       double best_mdl = current_mdl;
//       size_t best_merge_index = bins.size();
//       
//       for (size_t i = 0; i < bins.size() - 1; ++i) {
//         std::vector<Bin> temp_bins = bins;
//         Bin& left = temp_bins[i];
//         Bin& right = temp_bins[i + 1];
//         left.upper_bound = right.upper_bound;
//         left.count += right.count;
//         left.count_pos += right.count_pos;
//         left.count_neg += right.count_neg;
//         temp_bins.erase(temp_bins.begin() + i + 1);
//         
//         double new_mdl = calculate_mdl_cost(temp_bins);
//         if (new_mdl < best_mdl) {
//           best_mdl = new_mdl;
//           best_merge_index = i;
//         }
//       }
//       
//       if (best_merge_index < bins.size()) {
//         merge_bins(best_merge_index);
//       } else {
//         break;
//       }
//       
//       if (bins.size() <= (size_t)max_bins) {
//         break;
//       }
//       
//       iterations_run++;
//     }
//     
//     if (iterations_run >= max_iterations) {
//       converged = false;
//     }
//     
//     merge_rare_bins();
//     calculate_woe_iv();
//     
//     if (bins.size() > (size_t)min_bins && !is_monotonic()) {
//       enforce_monotonicity();
//     }
//     
//     validate_bins();
//   }
//   
//   Rcpp::List get_results() const {
//     Rcpp::StringVector bin_labels;
//     Rcpp::NumericVector woe_values;
//     Rcpp::NumericVector iv_values;
//     Rcpp::IntegerVector count_values;
//     Rcpp::IntegerVector count_pos_values;
//     Rcpp::IntegerVector count_neg_values;
//     Rcpp::NumericVector cutpoints;
//     
//     std::ostringstream oss;
//     oss << std::fixed << std::setprecision(6);
//     
//     for (size_t i = 0; i < bins.size(); ++i) {
//       oss.str("");
//       oss.clear();
//       
//       if (bins[i].lower_bound == -std::numeric_limits<double>::infinity()) {
//         oss << "[-Inf";
//       } else {
//         oss << "[" << bins[i].lower_bound;
//       }
//       oss << ";";
//       if (bins[i].upper_bound == std::numeric_limits<double>::infinity()) {
//         oss << "+Inf)";
//       } else {
//         oss << bins[i].upper_bound << ")";
//         cutpoints.push_back(bins[i].upper_bound);
//       }
//       
//       bin_labels.push_back(oss.str());
//       woe_values.push_back(bins[i].woe);
//       iv_values.push_back(bins[i].iv);
//       count_values.push_back(bins[i].count);
//       count_pos_values.push_back(bins[i].count_pos);
//       count_neg_values.push_back(bins[i].count_neg);
//     }
//     
//     Rcpp::NumericVector ids(bin_labels.size());
//     for(int i = 0; i < bin_labels.size(); i++) {
//       ids[i] = i + 1;
//     }
//     
//     return Rcpp::List::create(
//       Named("id") = ids,
//       Named("bin") = bin_labels,
//       Named("woe") = woe_values,
//       Named("iv") = iv_values,
//       Named("count") = count_values,
//       Named("count_pos") = count_pos_values,
//       Named("count_neg") = count_neg_values,
//       Named("cutpoints") = cutpoints,
//       Named("converged") = converged,
//       Named("iterations") = iterations_run
//     );
//   }
// };
// 
// 
// //' @title Optimal Binning for Numerical Features using the Minimum Description Length Principle (MDLP)
// //'
// //' @description
// //' This function performs optimal binning for numerical features using the Minimum Description Length Principle (MDLP).
// //' It minimizes information loss by merging adjacent bins that reduce the MDL cost, while ensuring monotonicity in the Weight of Evidence (WoE).
// //' The algorithm adjusts the number of bins between `min_bins` and `max_bins` and handles rare bins by merging them iteratively.
// //' Designed for robust and numerically stable calculations, it incorporates protections for extreme cases and convergence controls.
// //'
// //' @details
// //' ### Core Steps:
// //' 1. **Input Validation**: Ensures feature and target are valid, numeric, and binary respectively. Validates consistency between `min_bins` and `max_bins`.
// //' 2. **Pre-Binning**: Creates pre-bins based on equal frequencies or unique values if there are few observations.
// //' 3. **MDL-Based Merging**: Iteratively merges bins to minimize the MDL cost, which combines model complexity and data fit quality.
// //' 4. **Rare Bin Handling**: Merges bins with frequencies below the `bin_cutoff` threshold to ensure statistical stability.
// //' 5. **Monotonicity Enforcement**: Adjusts bins to ensure that the WoE values are monotonically increasing or decreasing.
// //' 6. **Validation**: Validates the final bin structure for consistency and correctness.
// //'
// //' ### Mathematical Framework:
// //' **Entropy Calculation**: For a bin \( i \) with positive (\( p \)) and negative (\( n \)) counts:
// //' \deqn{Entropy = -p \log_2(p) - n \log_2(n)}
// //'
// //' **MDL Cost**: Combines the cost of the model and data description. Lower MDL values indicate better binning.
// //'
// //' **Weight of Evidence (WoE)**: For a bin \( i \):
// //' \deqn{WoE_i = \ln\left(\frac{\text{Distribution of positives}_i}{\text{Distribution of negatives}_i}\right)}
// //'
// //' **Information Value (IV)**: Summarizes predictive power across all bins:
// //' \deqn{IV = \sum_{i} (P(X|Y=1) - P(X|Y=0)) \times WoE_i}
// //'
// //' ### Features:
// //' - Merges bins iteratively to minimize the MDL cost.
// //' - Ensures monotonicity of WoE to improve model interpretability.
// //' - Handles rare bins by merging categories with low frequencies.
// //' - Stable against edge cases like all identical values or insufficient observations.
// //' - Efficiently processes large datasets with iterative binning and convergence checks.
// //'
// //' ### Algorithm Parameters:
// //' - `min_bins`: Minimum number of bins (default: 3).
// //' - `max_bins`: Maximum number of bins (default: 5).
// //' - `bin_cutoff`: Minimum proportion of records required in a bin (default: 0.05).
// //' - `max_n_prebins`: Maximum number of pre-bins before merging (default: 20).
// //' - `convergence_threshold`: Threshold for convergence in terms of IV changes (default: 1e-6).
// //' - `max_iterations`: Maximum number of iterations for optimization (default: 1000).
// //'
// //' @param target An integer binary vector (0 or 1) representing the target variable.
// //' @param feature A numeric vector representing the feature to bin.
// //' @param min_bins Minimum number of bins (default: 3).
// //' @param max_bins Maximum number of bins (default: 5).
// //' @param bin_cutoff Minimum proportion of records per bin (default: 0.05).
// //' @param max_n_prebins Maximum number of pre-bins before merging (default: 20).
// //' @param convergence_threshold Convergence threshold for IV optimization (default: 1e-6).
// //' @param max_iterations Maximum number of iterations allowed (default: 1000).
// //'
// //' @return A list with the following components:
// //' \itemize{
// //'   \item `bin`: A vector of bin names representing the intervals.
// //'   \item `woe`: A numeric vector with the WoE values for each bin.
// //'   \item `iv`: A numeric vector with the IV values for each bin.
// //'   \item `count`: An integer vector with the total number of observations in each bin.
// //'   \item `count_pos`: An integer vector with the count of positive cases in each bin.
// //'   \item `count_neg`: An integer vector with the count of negative cases in each bin.
// //'   \item `cutpoints`: A numeric vector of cut points defining the bins.
// //'   \item `converged`: A boolean indicating whether the algorithm converged.
// //'   \item `iterations`: An integer with the number of iterations performed.
// //' }
// //'
// //' @examples
// //' \dontrun{
// //' # Example usage
// //' set.seed(123)
// //' target <- sample(0:1, 100, replace = TRUE)
// //' feature <- runif(100)
// //' result <- optimal_binning_numerical_mdlp(target, feature, min_bins = 3, max_bins = 5)
// //' print(result)
// //' }
// //'
// //' @export
// // [[Rcpp::export]]
// Rcpp::List optimal_binning_numerical_mdlp(
//    Rcpp::IntegerVector target,
//    Rcpp::NumericVector feature,
//    int min_bins = 3,
//    int max_bins = 5,
//    double bin_cutoff = 0.05,
//    int max_n_prebins = 20,
//    double convergence_threshold = 1e-6,
//    int max_iterations = 1000
// ) {
//  std::vector<double> feature_vec = Rcpp::as<std::vector<double>>(feature);
//  std::vector<int> target_vec = Rcpp::as<std::vector<int>>(target);
//  OptimalBinningNumericalMDLP binner(feature_vec, target_vec, min_bins, max_bins, bin_cutoff, max_n_prebins, convergence_threshold, max_iterations);
//  binner.fit();
//  return binner.get_results();
// }
