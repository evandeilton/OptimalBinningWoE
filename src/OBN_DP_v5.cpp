// [[Rcpp::plugins(cpp11)]]
// [[Rcpp::depends(Rcpp)]]

#include <Rcpp.h>
#include <vector>
#include <string>
#include <sstream>
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <limits>
#include <numeric>
#include <chrono>

using namespace Rcpp;

// Include shared headers
#include "common/optimal_binning_common.h"
#include "common/bin_structures.h"
#include "common/woe_iv_utils.h"

using namespace Rcpp;
using namespace OptimalBinning;


/**
 * @class OBN_DP
 * @brief Implements optimal binning for numerical variables using greedy merging
 *
 * IMPORTANT: Despite the "DP" name, this algorithm uses greedy heuristics, not true Dynamic
 * Programming with memoization. It performs pre-binning followed by iterative merging to
 * optimize Information Value while respecting monotonicity and bin count constraints.
 *
 * Algorithm Overview:
 * 1. Pre-binning: Create initial bins based on quantiles
 * 2. Greedy merging: Iteratively merge bins to satisfy constraints
 * 3. Monotonicity enforcement: Ensure WoE increases or decreases monotonically
 * 4. Constraint satisfaction: Respect min_bins, max_bins, bin_cutoff
 *
 * Complexity: O(n log n + k²) where n = sample size, k = number of bins
 *
 * Note: For true Dynamic Programming implementation, see research papers:
 * - Navas-Palencia (2020). "Optimal binning: mathematical programming formulation"
 */
class OBN_DP {
private:
  // Input data and parameters
  const std::vector<double>& feature;
  const std::vector<unsigned int>& target;
  int min_bins;
  int max_bins;
  double bin_cutoff;
  int max_n_prebins;
  double convergence_threshold;
  int max_iterations;
  std::string monotonic_trend;
  bool force_monotonic;
  
  // Result tracking
  bool converged;
  int iterations_run;
  long execution_time_ms;
  double total_iv;
  
  // Binning results
  std::vector<double> bin_edges;
  std::vector<double> woe_values;
  std::vector<double> iv_values;
  std::vector<double> event_rate_values;
  std::vector<std::string> bin_labels;
  std::vector<double> count_pos;
  std::vector<double> count_neg;
  std::vector<double> counts;
  
  // Dataset statistics
  double total_pos;
  double total_neg;
  double total_count;
  double overall_event_rate;
  
  // Constants
  // Constant removed (uses shared definition)
  // Local constant removed (uses shared definition)  // Cap for extreme WoE values
  
public:
  /**
   * @brief Constructor for the OBN_DP class
   * 
   * @param feature Vector of numerical feature values
   * @param target Vector of binary target values (0/1)
   * @param min_bins Minimum number of bins to create
   * @param max_bins Maximum number of bins to create
   * @param bin_cutoff Minimum proportion of observations for a bin
   * @param max_n_prebins Maximum number of pre-bins before optimization
   * @param convergence_threshold Threshold for algorithm convergence
   * @param max_iterations Maximum number of iterations
   * @param monotonic_trend Monotonicity direction ('auto', 'ascending', 'descending', 'none')
   */
  OBN_DP(
    const std::vector<double>& feature,
    const std::vector<unsigned int>& target,
    int min_bins,
    int max_bins,
    double bin_cutoff,
    int max_n_prebins,
    double convergence_threshold,
    int max_iterations,
    std::string monotonic_trend = "auto")
    : feature(feature),
      target(target),
      min_bins(min_bins),
      max_bins(max_bins),
      bin_cutoff(bin_cutoff),
      max_n_prebins(max_n_prebins),
      convergence_threshold(convergence_threshold),
      max_iterations(max_iterations),
      monotonic_trend(monotonic_trend),
      converged(true),
      iterations_run(0),
      execution_time_ms(0),
      total_iv(0.0) {
    
    // Calculate dataset statistics
    total_count = static_cast<double>(target.size());
    double sum_target = std::accumulate(target.begin(), target.end(), 0.0);
    total_pos = sum_target;
    total_neg = total_count - total_pos;
    overall_event_rate = total_pos / total_count;
    
    // Validate monotonic trend
    force_monotonic = true;
    if (monotonic_trend != "auto" && monotonic_trend != "ascending" && 
        monotonic_trend != "descending" && monotonic_trend != "none") {
      Rcpp::warning("Invalid monotonic_trend value. Using 'auto' instead.");
      this->monotonic_trend = "auto";
    }
    if (monotonic_trend == "none") {
      force_monotonic = false;
    }
    
    // Additional checks to prevent unexpected behavior
    if (max_bins < min_bins) {
      Rcpp::stop("max_bins must be >= min_bins");
    }
    if (max_n_prebins < 1) {
      max_n_prebins = 1;
    }
    if (bin_cutoff < 0.0) {
      bin_cutoff = 0.0;
    }
  }
  
  /**
   * @brief Performs the optimal binning algorithm
   * 
   * Main method that executes the complete binning process
   */
  void fit() {
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Validate inputs
    validate_inputs();
    
    // Adjust min_bins if necessary
    if (min_bins < 2) {
      min_bins = 2; // ensure minimum
    }
    if (min_bins > max_bins) {
      min_bins = max_bins;
    }
    
    // Check the number of unique feature values
    std::vector<double> unique_feature_values = get_unique_values();
    int num_unique_values = static_cast<int>(unique_feature_values.size());
    
    // If <=2 unique values, just create trivial bins without further optimization
    if (num_unique_values <= 2) {
      create_trivial_bins(unique_feature_values);
      
      auto end_time = std::chrono::high_resolution_clock::now();
      execution_time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        end_time - start_time).count();
      return;
    }
    
    // Initial pre-binning based on data distribution
    prebinning();
    
    // Calculate initial counts and WOE
    calculate_counts_woe_and_iv();
    
    // If total_pos or total_neg is too small, stop to avoid instability
    if (total_pos < EPSILON || total_neg < EPSILON) {
      // Already computed counts, WOE and IV, just finalize
      converged = true;
      
      auto end_time = std::chrono::high_resolution_clock::now();
      execution_time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        end_time - start_time).count();
      return;
    }
    
    // Determine monotonic direction if set to "auto"
    if (monotonic_trend == "auto") {
      determine_monotonic_direction();
    }
    
    // Enforce monotonicity if required
    if (force_monotonic) {
      enforce_monotonicity();
    }
    
    // Ensure bin constraints and handle rare bins
    ensure_bin_constraints();
    
    // Final IV calculation
    calculate_total_iv();
    
    auto end_time = std::chrono::high_resolution_clock::now();
    execution_time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
      end_time - start_time).count();
  }
  
  /**
   * @brief Returns the results of the binning process
   * 
   * @return Rcpp::List List containing the binning results
   */
  List get_results() const {
    // Exclude -Inf and +Inf from cutpoints
    std::vector<double> cutpoints;
    if (bin_edges.size() > 2) {
      cutpoints.reserve(bin_edges.size() - 2);
      for (size_t i = 1; i < bin_edges.size() - 1; ++i) {
        cutpoints.push_back(bin_edges[i]);
      }
    }
    
    // Create IDs for bins
    Rcpp::NumericVector ids(bin_labels.size());
    for(size_t i = 0; i < bin_labels.size(); i++) {
      ids[i] = i + 1;
    }
    
    return Rcpp::List::create(
      Named("id") = ids,
      Named("bin") = bin_labels,
      Named("woe") = woe_values,
      Named("iv") = iv_values,
      Named("count") = counts,
      Named("count_pos") = count_pos,
      Named("count_neg") = count_neg,
      Named("event_rate") = event_rate_values,
      Named("cutpoints") = cutpoints,
      Named("total_iv") = total_iv,
      Named("converged") = converged,
      Named("iterations") = iterations_run,
      Named("execution_time_ms") = execution_time_ms,
      Named("monotonic_trend") = monotonic_trend
    );
  }
  
private:
  /**
   * @brief Validates input data and parameters
   * 
   * Checks that the input data and parameters are valid and consistent
   */
  void validate_inputs() {
    // Check feature and target have same length
    if (feature.size() != target.size()) {
      Rcpp::stop("Feature and target vectors must have the same length");
    }
    
    // Check if target is binary (0/1)
    bool has_zero = false;
    bool has_one = false;
    
    for (auto t : target) {
      if (t == 0U) {
        has_zero = true;
      } else if (t == 1U) {
        has_one = true;
      } else {
        Rcpp::stop("Target must contain only values 0 and 1");
      }
      
      if (has_zero && has_one) {
        break;  // No need to check further
      }
    }
    
    if (!has_zero || !has_one) {
      Rcpp::stop("Target must contain both values 0 and 1");
    }
    
    // Check for NaN values in feature
    for (auto f : feature) {
      if (std::isnan(f)) {
        Rcpp::stop("Feature contains NaN values which are not supported");
      }
    }
  }
  
  /**
   * @brief Gets unique feature values
   * 
   * @return Vector of unique feature values, sorted
   */
  std::vector<double> get_unique_values() {
    std::vector<double> unique_values = feature;
    std::sort(unique_values.begin(), unique_values.end());
    unique_values.erase(
      std::unique(unique_values.begin(), unique_values.end()),
      unique_values.end());
    return unique_values;
  }
  
  /**
   * @brief Creates trivial bins when there are very few unique values
   * 
   * @param unique_values Vector of unique feature values
   */
  void create_trivial_bins(const std::vector<double>& unique_values) {
    bin_edges.clear();
    bin_edges.reserve(static_cast<size_t>(unique_values.size() + 1));
    bin_edges.push_back(-std::numeric_limits<double>::infinity());
    
    if (unique_values.size() == 2) {
      // For two unique values, use the midpoint as the split
      double midpoint = (unique_values[0] + unique_values[1]) / 2.0;
      bin_edges.push_back(midpoint);
    }
    
    bin_edges.push_back(std::numeric_limits<double>::infinity());
    
    // Calculate statistics for the bins
    calculate_counts_woe_and_iv();
    calculate_total_iv();
    
    converged = true;
    iterations_run = 0;
  }
  
  /**
   * @brief Performs initial pre-binning based on the data distribution
   * 
   * Creates initial bins using equal-frequency binning on the sorted feature
   */
  void prebinning() {
    // Sort indices by feature value
    std::vector<size_t> sorted_indices(feature.size());
    std::iota(sorted_indices.begin(), sorted_indices.end(), 0);
    std::sort(sorted_indices.begin(), sorted_indices.end(),
              [this](size_t i1, size_t i2) { return feature[i1] < feature[i2]; });
    
    // Extract sorted feature values
    std::vector<double> sorted_feature(feature.size());
    for (size_t i = 0; i < feature.size(); ++i) {
      sorted_feature[i] = feature[sorted_indices[i]];
    }
    
    // Find bin edges using equal-frequency method
    int n = static_cast<int>(feature.size());
    int bin_size = std::max(1, n / std::max(1, max_n_prebins));
    
    std::vector<double> edges;
    edges.reserve(static_cast<size_t>(std::max(1, max_n_prebins - 1)));
    
    for (int i = 1; i < max_n_prebins; ++i) {
      int idx = i * bin_size;
      if (idx < n) {
        edges.push_back(sorted_feature[static_cast<size_t>(idx)]);
      }
    }
    
    // Remove duplicates and ensure edges are sorted
    std::sort(edges.begin(), edges.end());
    edges.erase(std::unique(edges.begin(), edges.end()), edges.end());
    
    // Set bin edges with -Inf and +Inf as boundaries
    bin_edges.clear();
    bin_edges.reserve(edges.size() + 2);
    bin_edges.push_back(-std::numeric_limits<double>::infinity());
    for (const auto& edge : edges) {
      bin_edges.push_back(edge);
    }
    bin_edges.push_back(std::numeric_limits<double>::infinity());
  }
  
  /**
   * @brief Calculate counts, WoE, and IV for current bins
   * 
   * Counts observations in each bin and calculates statistics
   */
  void calculate_counts_woe_and_iv() {
    int num_bins = static_cast<int>(bin_edges.size()) - 1;
    count_pos.assign(static_cast<size_t>(num_bins), 0.0);
    count_neg.assign(static_cast<size_t>(num_bins), 0.0);
    counts.assign(static_cast<size_t>(num_bins), 0.0);
    
    // Count observations in each bin
    for (size_t i = 0; i < feature.size(); ++i) {
      int bin_idx = find_bin(feature[i]);
      if (bin_idx < 0 || bin_idx >= num_bins) {
        // Skip invalid bin indices - shouldn't happen with proper bin edges
        continue;
      }
      
      counts[static_cast<size_t>(bin_idx)] += 1.0;
      if (target[i] == 1U) {
        count_pos[static_cast<size_t>(bin_idx)] += 1.0;
      } else {
        count_neg[static_cast<size_t>(bin_idx)] += 1.0;
      }
    }
    
    // Calculate WoE, IV, and event rates
    calculate_woe();
    calculate_iv();
    update_bin_labels();
    calculate_event_rates();
  }
  
  /**
   * @brief Calculate Weight of Evidence (WoE) for each bin using shared utilities
   *
   * Uses compute_woe() from woe_iv_utils.h for consistent calculation across algorithms.
   * WoE = ln(% of events / % of non-events)
   */
  void calculate_woe() {
    int num_bins = static_cast<int>(counts.size());
    woe_values.resize(static_cast<size_t>(num_bins));

    for (int i = 0; i < num_bins; ++i) {
      int pos = static_cast<int>(count_pos[static_cast<size_t>(i)]);
      int neg = static_cast<int>(count_neg[static_cast<size_t>(i)]);
      int total_pos_int = static_cast<int>(total_pos);
      int total_neg_int = static_cast<int>(total_neg);

      // Use shared utility with Bayesian smoothing
      woe_values[static_cast<size_t>(i)] = compute_woe(
        pos, neg, total_pos_int, total_neg_int,
        SmoothingMethod::BAYESIAN, BAYESIAN_PRIOR_STRENGTH
      );
    }
  }
  
  /**
   * @brief Calculate Information Value (IV) for each bin
   * 
   * IV = (% of events - % of non-events) * WoE
   */
  void calculate_iv() {
    double safe_total_pos = std::max(total_pos, EPSILON);
    double safe_total_neg = std::max(total_neg, EPSILON);
    
    iv_values.resize(woe_values.size());
    
    for (size_t i = 0; i < woe_values.size(); ++i) {
      double p_rate = count_pos[i] / safe_total_pos;
      double n_rate = count_neg[i] / safe_total_neg;
      
      // IV formula
      iv_values[i] = (p_rate - n_rate) * woe_values[i];
    }
  }
  
  /**
   * @brief Calculate event rates for each bin
   * 
   * Event rate = count_pos / count
   */
  void calculate_event_rates() {
    event_rate_values.resize(counts.size());
    for (size_t i = 0; i < counts.size(); ++i) {
      if (counts[i] > EPSILON) {
        event_rate_values[i] = count_pos[i] / counts[i];
      } else {
        event_rate_values[i] = overall_event_rate;  // Use overall rate for empty bins
      }
    }
  }
  
  /**
   * @brief Update bin labels based on current bin edges
   * 
   * Creates human-readable labels for each bin
   */
  void update_bin_labels() {
    bin_labels.clear();
    bin_labels.reserve(bin_edges.size() - 1);
    
    for (size_t i = 0; i < bin_edges.size() - 1; ++i) {
      std::ostringstream oss;
      oss << std::fixed << std::setprecision(6);
      
      double left = bin_edges[i];
      double right = bin_edges[i+1];
      
      if (std::isinf(left) && left < 0) {
        oss << "(-Inf;" << right << "]";
      } else if (std::isinf(right) && right > 0) {
        oss << "(" << left << ";+Inf]";
      } else {
        oss << "(" << left << ";" << right << "]";
      }
      
      bin_labels.push_back(oss.str());
    }
  }
  
  /**
   * @brief Find which bin contains a given value
   * 
   * @param value The value to locate in bins
   * @return int The bin index that contains the value
   */
  int find_bin(double value) const {
    // Locate bin via upper_bound
    auto it = std::upper_bound(bin_edges.begin(), bin_edges.end(), value);
    int bin_idx = static_cast<int>(std::distance(bin_edges.begin(), it)) - 1;
    
    // Ensure bin_idx is within valid range
    if (bin_idx < 0) bin_idx = 0;
    if (bin_idx >= static_cast<int>(counts.size())) bin_idx = static_cast<int>(counts.size()) - 1;
    
    return bin_idx;
  }
  
  /**
   * @brief Determine the monotonic direction based on data
   * 
   * In 'auto' mode, analyzes the data to determine if WoE should increase or decrease
   * with the feature value
   */
  void determine_monotonic_direction() {
    if (woe_values.size() <= 2) {
      // Default to ascending with insufficient data
      monotonic_trend = "ascending";
      return;
    }
    
    // Compute correlation between feature values and target
    double feature_sum = 0.0;
    double feature_sq_sum = 0.0;
    double target_sum = 0.0;
    double target_sq_sum = 0.0;
    double cross_sum = 0.0;
    
    for (size_t i = 0; i < feature.size(); ++i) {
      feature_sum += feature[i];
      feature_sq_sum += feature[i] * feature[i];
      target_sum += static_cast<double>(target[i]);
      target_sq_sum += static_cast<double>(target[i]) * static_cast<double>(target[i]);
      cross_sum += feature[i] * static_cast<double>(target[i]);
    }
    
    double n = static_cast<double>(feature.size());
    double numerator = n * cross_sum - feature_sum * target_sum;
    double denom1 = std::sqrt(n * feature_sq_sum - feature_sum * feature_sum);
    double denom2 = std::sqrt(n * target_sq_sum - target_sum * target_sum);
    
    double correlation = 0.0;
    if (denom1 > EPSILON && denom2 > EPSILON) {
      correlation = numerator / (denom1 * denom2);
    }
    
    // Set monotonic trend based on correlation
    monotonic_trend = (correlation >= 0) ? "ascending" : "descending";
  }
  
  /**
   * @brief Enforce monotonicity in the WoE values
   * 
   * Merges bins that violate the monotonicity constraint
   */
  void enforce_monotonicity() {
    if (counts.size() <= 2) {
      // No enforcement needed for 2 or fewer bins
      return;
    }
    
    bool ascending = (monotonic_trend == "ascending");
    bool is_monotonic = false;
    int iterations = 0;
    
    // Merge bins until monotonicity is achieved or max iterations reached
    while (!is_monotonic && counts.size() > static_cast<size_t>(min_bins) && iterations < max_iterations) {
      is_monotonic = true;
      
      for (size_t i = 1; i < woe_values.size(); ++i) {
        bool violation = false;
        
        if (ascending && (woe_values[i] < woe_values[i - 1])) {
          violation = true;
        } else if (!ascending && (woe_values[i] > woe_values[i - 1])) {
          violation = true;
        }
        
        if (violation) {
          // Merge bin i-1 and i to fix violation
          merge_bins(static_cast<int>(i - 1));
          is_monotonic = false;
          break;
        }
      }
      
      iterations++;
      
      if (counts.size() == static_cast<size_t>(min_bins)) {
        // Stop if reached minimum number of bins
        break;
      }
    }
    
    iterations_run += iterations;
    if (iterations >= max_iterations) {
      converged = false;
    }
  }
  
  /**
   * @brief Merge two adjacent bins
   * 
   * @param idx Index of the first bin to merge
   */
  void merge_bins(int idx) {
    // Safety checks
    if (idx < 0 || idx >= static_cast<int>(counts.size()) - 1) {
      // Invalid index for merging
      return;
    }
    
    // Merge bin idx with idx+1
    bin_edges.erase(bin_edges.begin() + idx + 1);
    counts[idx] += counts[static_cast<size_t>(idx + 1)];
    counts.erase(counts.begin() + idx + 1);
    count_pos[idx] += count_pos[static_cast<size_t>(idx + 1)];
    count_pos.erase(count_pos.begin() + idx + 1);
    count_neg[idx] += count_neg[static_cast<size_t>(idx + 1)];
    count_neg.erase(count_neg.begin() + idx + 1);
    
    // Recalculate statistics after merging
    calculate_woe();
    calculate_iv();
    update_bin_labels();
    calculate_event_rates();
  }
  
  /**
   * @brief Ensure bin constraints are met
   * 
   * Handles bin count limits and rare bin merging
   */
  void ensure_bin_constraints() {
    // Ensure counts.size() <= max_bins by merging if needed
    int iterations = 0;
    while (counts.size() > static_cast<size_t>(max_bins) && iterations < max_iterations) {
      int idx = find_smallest_woe_diff();
      if (idx == -1) {
        // No valid merge index found, break to avoid infinite loop
        break;
      }
      merge_bins(idx);
      iterations++;
    }
    
    iterations_run += iterations;
    if (iterations >= max_iterations) {
      converged = false;
    }
    
    // Handle rare bins
    handle_rare_bins();
  }
  
  /**
   * @brief Find the pair of adjacent bins with smallest WoE difference
   * 
   * @return Index of the first bin in the pair with smallest WoE difference
   */
  int find_smallest_woe_diff() const {
    if (woe_values.size() <= 1) return -1;
    
    std::vector<double> woe_diffs(woe_values.size() - 1);
    for (size_t i = 0; i < woe_diffs.size(); ++i) {
      woe_diffs[i] = std::fabs(woe_values[i + 1] - woe_values[i]);
    }
    
    auto min_it = std::min_element(woe_diffs.begin(), woe_diffs.end());
    if (min_it == woe_diffs.end()) return -1;
    
    return static_cast<int>(std::distance(woe_diffs.begin(), min_it));
  }
  
  /**
   * @brief Handle rare bins by merging them with adjacent bins
   * 
   * Bins with fewer observations than bin_cutoff proportion are merged
   */
  void handle_rare_bins() {
    double total_count = std::accumulate(counts.begin(), counts.end(), 0.0);
    bool merged = true;
    int iterations = 0;
    
    // Merge bins that are too small until no rare bins or min_bins reached
    while (merged && counts.size() > static_cast<size_t>(min_bins) && iterations < max_iterations) {
      merged = false;
      
      for (size_t i = 0; i < counts.size(); ++i) {
        double proportion = total_count > 0.0 ? (counts[i] / total_count) : 0.0;
        
        if (proportion < bin_cutoff && counts.size() > static_cast<size_t>(min_bins)) {
          int merge_idx;
          
          if (i == 0) {
            merge_idx = 0;  // First bin, merge with next
          } else if (i == counts.size() - 1) {
            merge_idx = static_cast<int>(counts.size()) - 2;  // Last bin, merge with previous
          } else {
            // Middle bin, merge with more similar neighbor (by WoE)
            double diff_prev = std::fabs(woe_values[i] - woe_values[i - 1]);
            double diff_next = std::fabs(woe_values[i] - woe_values[i + 1]);
            
            merge_idx = (diff_prev <= diff_next) ? 
            static_cast<int>(i - 1) : static_cast<int>(i);
          }
          
          merge_bins(merge_idx);
          merged = true;
          iterations++;
          break;  // Restart loop after merge since indices changed
        }
      }
    }
    
    iterations_run += iterations;
    if (iterations >= max_iterations) {
      converged = false;
    }
  }
  
  /**
   * @brief Calculate total Information Value
   * 
   * Sum of IV contributions from all bins
   */
  void calculate_total_iv() {
    total_iv = std::accumulate(iv_values.begin(), iv_values.end(), 0.0);
  }
};

// [[Rcpp::export]]
Rcpp::List optimal_binning_numerical_dp(
   Rcpp::IntegerVector target,
   Rcpp::NumericVector feature,
   int min_bins = 3,
   int max_bins = 5,
   double bin_cutoff = 0.05,
   int max_n_prebins = 20,
   double convergence_threshold = 1e-6,
   int max_iterations = 1000,
   std::string monotonic_trend = "auto") {
 
 // Input validation
 if (feature.size() != target.size()) {
   Rcpp::stop("Feature and target must have the same length");
 }
 
 if (min_bins < 2) {
   Rcpp::warning("min_bins must be at least 2, setting to 2");
   min_bins = 2;
 }
 
 if (max_bins < min_bins) {
   Rcpp::warning("max_bins must be >= min_bins, setting max_bins = min_bins");
   max_bins = min_bins;
 }
 
 // Convert R vectors to C++
 std::vector<double> feature_vec = Rcpp::as<std::vector<double>>(feature);
 std::vector<unsigned int> target_vec; 
 target_vec.reserve(target.size());
 
 // Convert and validate target values
 bool has_invalid = false;
 for (int i = 0; i < target.size(); ++i) {
   if (IntegerVector::is_na(target[i])) {
     Rcpp::stop("Target cannot contain NA values");
   }
   
   int val = target[i];
   if (val != 0 && val != 1) {
     has_invalid = true;
   }
   
   target_vec.push_back(static_cast<unsigned int>(val));
 }
 
 if (has_invalid) {
   Rcpp::stop("Target must contain only values 0 and 1");
 }
 
 // Run the binning algorithm
 OBN_DP ob(
     feature_vec, target_vec, min_bins, max_bins, bin_cutoff, max_n_prebins,
     convergence_threshold, max_iterations, monotonic_trend);
 
 ob.fit();
 
 return ob.get_results();
}