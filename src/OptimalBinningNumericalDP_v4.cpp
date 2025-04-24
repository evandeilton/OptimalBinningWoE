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

/**
 * @class OptimalBinningNumericalDP
 * @brief Implements optimal binning for numerical variables using dynamic programming
 * 
 * This class performs optimal binning for a numerical feature with respect to a binary target.
 * It uses dynamic programming to find the optimal partitioning of the feature space into bins
 * that maximize the predictive power while respecting constraints on monotonicity and bin counts.
 */
class OptimalBinningNumericalDP {
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
  static constexpr double EPSILON = 1e-10;
  static constexpr double MAX_WOE = 20.0;  // Cap for extreme WoE values
  
public:
  /**
   * @brief Constructor for the OptimalBinningNumericalDP class
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
  OptimalBinningNumericalDP(
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
   * @brief Calculate Weight of Evidence (WoE) for each bin
   * 
   * WoE = ln(% of events / % of non-events)
   */
  void calculate_woe() {
    // Avoid division by zero
    double safe_total_pos = std::max(total_pos, EPSILON);
    double safe_total_neg = std::max(total_neg, EPSILON);
    
    int num_bins = static_cast<int>(counts.size());
    woe_values.resize(static_cast<size_t>(num_bins));
    
    for (int i = 0; i < num_bins; ++i) {
      // Ensure positive count is not zero
      double safe_pos = std::max(count_pos[static_cast<size_t>(i)], EPSILON);
      // Ensure negative count is not zero
      double safe_neg = std::max(count_neg[static_cast<size_t>(i)], EPSILON);
      
      // Calculate distribution ratios
      double rate_pos = safe_pos / safe_total_pos;
      double rate_neg = safe_neg / safe_total_neg;
      
      // Calculate WoE with limits to prevent extreme values
      double woe = std::log(rate_pos / rate_neg);
      woe = std::max(std::min(woe, MAX_WOE), -MAX_WOE);  // Cap extreme values
      
      woe_values[static_cast<size_t>(i)] = woe;
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

//' @title Optimal Binning for Numerical Variables using Dynamic Programming
//'
//' @description
//' Performs optimal binning for numerical variables using a Dynamic Programming approach.
//' It creates optimal bins for a numerical feature based on its relationship with a binary target variable, 
//' maximizing the predictive power while respecting user-defined constraints and enforcing monotonicity.
//'
//' @param target An integer vector of binary target values (0 or 1).
//' @param feature A numeric vector of feature values.
//' @param min_bins Minimum number of bins (default: 3).
//' @param max_bins Maximum number of bins (default: 5).
//' @param bin_cutoff Minimum proportion of total observations for a bin to avoid being merged (default: 0.05).
//' @param max_n_prebins Maximum number of pre-bins before the optimization process (default: 20).
//' @param convergence_threshold Convergence threshold for the algorithm (default: 1e-6).
//' @param max_iterations Maximum number of iterations allowed (default: 1000).
//' @param monotonic_trend Monotonicity direction. One of 'auto', 'ascending', 'descending', or 'none' (default: 'auto').
//'
//' @return A list containing the following elements:
//' \item{id}{Numeric vector of bin identifiers (1 to n).}
//' \item{bin}{Character vector of bin ranges.}
//' \item{woe}{Numeric vector of Weight of Evidence (WoE) values for each bin.}
//' \item{iv}{Numeric vector of Information Value (IV) for each bin.}
//' \item{count}{Numeric vector of total observations in each bin.}
//' \item{count_pos}{Numeric vector of positive target observations in each bin.}
//' \item{count_neg}{Numeric vector of negative target observations in each bin.}
//' \item{event_rate}{Numeric vector of event rates (proportion of positive events) in each bin.}
//' \item{cutpoints}{Numeric vector of cut points to generate the bins.}
//' \item{total_iv}{Total Information Value across all bins.}
//' \item{converged}{Logical indicating if the algorithm converged.}
//' \item{iterations}{Integer number of iterations run by the algorithm.}
//' \item{execution_time_ms}{Execution time in milliseconds.}
//' \item{monotonic_trend}{The monotonic trend used ('auto', 'ascending', 'descending', 'none').}
//'
//' @details
//' The Dynamic Programming algorithm for numerical variables works as follows:
//' 
//' \enumerate{
//'   \item Create initial pre-bins based on equal-frequency binning of the feature distribution
//'   \item Calculate bin statistics: counts, event rates, WoE, and IV
//'   \item If monotonicity is required, determine the appropriate trend:
//'     \itemize{
//'       \item In 'auto' mode: Calculate correlation between feature and target to choose direction
//'       \item In 'ascending'/'descending' mode: Use the specified direction
//'     }
//'   \item Enforce monotonicity by merging adjacent bins that violate the monotonic trend
//'   \item Ensure bin constraints are met:
//'     \itemize{
//'       \item If exceeding max_bins: Merge bins with the smallest WoE difference
//'       \item Handle rare bins: Merge bins with fewer than bin_cutoff proportion of observations
//'     }
//'   \item Calculate final statistics for the optimized bins
//' }
//'
//' The Weight of Evidence (WoE) measures the predictive power of each bin and is calculated as:
//' 
//' \deqn{WoE = \ln\left(\frac{\text{Distribution of Events}}{\text{Distribution of Non-Events}}\right)}
//'
//' The Information Value (IV) for each bin is calculated as:
//' 
//' \deqn{IV = (\text{Distribution of Events} - \text{Distribution of Non-Events}) \times WoE}
//'
//' The total IV is the sum of bin IVs and measures the overall predictive power of the feature.
//'
//' This implementation is based on the methodology described in:
//' 
//' \itemize{
//'   \item Navas-Palencia, G. (2022). "OptBinning: Mathematical Optimization for Optimal Binning". Journal of Open Source Software, 7(74), 4101.
//'   \item Siddiqi, N. (2017). "Intelligent Credit Scoring: Building and Implementing Better Credit Risk Scorecards". John Wiley & Sons, 2nd Edition.
//'   \item Thomas, L.C., Edelman, D.B., & Crook, J.N. (2017). "Credit Scoring and Its Applications". SIAM, 2nd Edition.
//'   \item Kotsiantis, S.B., & Kanellopoulos, D. (2006). "Discretization Techniques: A recent survey". GESTS International Transactions on Computer Science and Engineering, 32(1), 47-58.
//' }
//'
//' Monotonicity constraints are particularly important in credit scoring and risk modeling
//' applications, as they ensure that the model behaves in an intuitive and explainable way.
//'
//' @examples
//' # Create sample data
//' set.seed(123)
//' n <- 1000
//' target <- sample(0:1, n, replace = TRUE)
//' feature <- rnorm(n)
//'
//' # Run optimal binning
//' result <- optimal_binning_numerical_dp(target, feature, min_bins = 2, max_bins = 4)
//'
//' # Print results
//' print(result)
//'
//' @export
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
 OptimalBinningNumericalDP ob(
     feature_vec, target_vec, min_bins, max_bins, bin_cutoff, max_n_prebins,
     convergence_threshold, max_iterations, monotonic_trend);
 
 ob.fit();
 
 return ob.get_results();
}













// // [[Rcpp::plugins(cpp11)]]
// // [[Rcpp::depends(Rcpp)]]
// 
// #include <Rcpp.h>
// #include <vector>
// #include <string>
// #include <sstream>
// #include <algorithm>
// #include <cmath>
// #include <iomanip>
// #include <limits>
// #include <numeric>
// 
// using namespace Rcpp;
// 
// // Class for Optimal Binning using Dynamic Programming with Local Constraints
// class OptimalBinningNumericalDPLC {
// private:
//   const std::vector<double>& feature;
//   const std::vector<unsigned int>& target;
//   int min_bins;
//   int max_bins;
//   double bin_cutoff;
//   int max_n_prebins;
//   double convergence_threshold;
//   int max_iterations;
//   
//   bool converged;
//   int iterations_run;
//   
//   std::vector<double> bin_edges;
//   std::vector<double> woe_values;
//   std::vector<double> iv_values;
//   std::vector<std::string> bin_labels;
//   std::vector<double> count_pos;
//   std::vector<double> count_neg;
//   std::vector<double> counts;
//   
//   double total_pos;
//   double total_neg;
//   
//   static constexpr double EPSILON = 1e-10;
//   
// public:
//   OptimalBinningNumericalDPLC(const std::vector<double>& feature,
//                               const std::vector<unsigned int>& target,
//                               int min_bins,
//                               int max_bins,
//                               double bin_cutoff,
//                               int max_n_prebins,
//                               double convergence_threshold,
//                               int max_iterations)
//     : feature(feature),
//       target(target),
//       min_bins(min_bins),
//       max_bins(max_bins),
//       bin_cutoff(bin_cutoff),
//       max_n_prebins(max_n_prebins),
//       convergence_threshold(convergence_threshold),
//       max_iterations(max_iterations),
//       converged(true),
//       iterations_run(0) {
//     
//     double sum_target = std::accumulate(target.begin(), target.end(), 0.0);
//     total_pos = sum_target;
//     total_neg = static_cast<double>(target.size()) - total_pos;
//     
//     // Additional checks to prevent unexpected behavior
//     if (max_bins < min_bins) {
//       Rcpp::stop("max_bins must be >= min_bins");
//     }
//     if (max_n_prebins < 1) {
//       max_n_prebins = 1;
//     }
//     if (bin_cutoff < 0.0) {
//       bin_cutoff = 0.0;
//     }
//   }
//   
//   void fit() {
//     // Adjust min_bins if necessary
//     if (min_bins < 2) {
//       min_bins = 2; // ensure minimum
//     }
//     if (min_bins > max_bins) {
//       min_bins = max_bins;
//     }
//     
//     // Check the number of unique feature values
//     std::vector<double> unique_feature_values = feature;
//     std::sort(unique_feature_values.begin(), unique_feature_values.end());
//     unique_feature_values.erase(std::unique(unique_feature_values.begin(), unique_feature_values.end()), unique_feature_values.end());
//     int num_unique_values = static_cast<int>(unique_feature_values.size());
//     
//     // If <=2 unique values, just create trivial bins without further optimization
//     if (num_unique_values <= 2) {
//       bin_edges.clear();
//       bin_edges.reserve(static_cast<size_t>(num_unique_values + 1));
//       bin_edges.push_back(-std::numeric_limits<double>::infinity());
//       if (num_unique_values == 2) {
//         double midpoint = (unique_feature_values[0] + unique_feature_values[1]) / 2.0;
//         bin_edges.push_back(midpoint);
//       }
//       bin_edges.push_back(std::numeric_limits<double>::infinity());
//       calculate_counts_woe();
//       calculate_iv();
//       converged = true;
//       iterations_run = 0;
//       return;
//     }
//     
//     // Pre-binning
//     prebinning();
//     
//     // Calculate initial counts and WOE
//     calculate_counts_woe();
//     
//     // If total_pos or total_neg is zero, stop to avoid instability
//     if (total_pos < EPSILON || total_neg < EPSILON) {
//       // Already computed counts and WOE, just finalize IV
//       calculate_iv();
//       converged = true;
//       return;
//     }
//     
//     // Enforce monotonicity
//     enforce_monotonicity();
//     
//     // Ensure bin constraints and handle rare bins
//     ensure_bin_constraints();
//     
//     // Final IV calculation
//     calculate_iv();
//   }
//   
//   List get_results() const {
//     // Exclude -Inf and +Inf from cutpoints
//     std::vector<double> cutpoints;
//     if (bin_edges.size() > 2) {
//       cutpoints.reserve(bin_edges.size() - 2);
//       for (size_t i = 1; i < bin_edges.size() - 1; ++i) {
//         cutpoints.push_back(bin_edges[i]);
//       }
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
//       Named("count") = counts,
//       Named("count_pos") = count_pos,
//       Named("count_neg") = count_neg,
//       Named("cutpoints") = cutpoints,
//       Named("converged") = converged,
//       Named("iterations") = iterations_run
//     );
//   }
//   
// private:
//   void prebinning() {
//     // Initial pre-binning based on approximate quantiles
//     std::vector<size_t> sorted_indices(feature.size());
//     std::iota(sorted_indices.begin(), sorted_indices.end(), 0);
//     std::sort(sorted_indices.begin(), sorted_indices.end(),
//               [this](size_t i1, size_t i2) { return feature[i1] < feature[i2]; });
//     
//     std::vector<double> sorted_feature(feature.size());
//     for (size_t i = 0; i < feature.size(); ++i) {
//       sorted_feature[i] = feature[sorted_indices[i]];
//     }
//     
//     int n = static_cast<int>(feature.size());
//     int bin_size = std::max(1, n / std::max(1, max_n_prebins));
//     
//     std::vector<double> edges;
//     edges.reserve(static_cast<size_t>(std::max(1, max_n_prebins - 1)));
//     
//     for (int i = 1; i < max_n_prebins; ++i) {
//       int idx = i * bin_size;
//       if (idx < n) {
//         edges.push_back(sorted_feature[static_cast<size_t>(idx)]);
//       }
//     }
//     std::sort(edges.begin(), edges.end());
//     edges.erase(std::unique(edges.begin(), edges.end()), edges.end());
//     
//     bin_edges.clear();
//     bin_edges.reserve(edges.size() + 2);
//     bin_edges.push_back(-std::numeric_limits<double>::infinity());
//     for (size_t i = 0; i < edges.size(); ++i) {
//       bin_edges.push_back(edges[i]);
//     }
//     bin_edges.push_back(std::numeric_limits<double>::infinity());
//   }
//   
//   void calculate_counts_woe() {
//     int num_bins = static_cast<int>(bin_edges.size()) - 1;
//     count_pos.assign(static_cast<size_t>(num_bins), 0.0);
//     count_neg.assign(static_cast<size_t>(num_bins), 0.0);
//     counts.assign(static_cast<size_t>(num_bins), 0.0);
//     
//     for (size_t i = 0; i < feature.size(); ++i) {
//       int bin_idx = find_bin(feature[i]);
//       if (bin_idx < 0 || bin_idx >= num_bins) {
//         // Safety check, should not happen
//         continue;
//       }
//       counts[static_cast<size_t>(bin_idx)] += 1.0;
//       if (target[i] == 1U) {
//         count_pos[static_cast<size_t>(bin_idx)] += 1.0;
//       } else {
//         count_neg[static_cast<size_t>(bin_idx)] += 1.0;
//       }
//     }
//     
//     calculate_woe();
//     update_bin_labels();
//   }
//   
//   void calculate_woe() {
//     if (total_pos < EPSILON || total_neg < EPSILON) {
//       // Avoid division by zero
//       // If this happens, WOE might be unstable, just set small corrections
//       total_pos = std::max(total_pos, EPSILON);
//       total_neg = std::max(total_neg, EPSILON);
//     }
//     
//     int num_bins = static_cast<int>(counts.size());
//     woe_values.resize(static_cast<size_t>(num_bins));
//     
//     for (int i = 0; i < num_bins; ++i) {
//       double rate_pos = (count_pos[static_cast<size_t>(i)] > 0.0) ? (count_pos[static_cast<size_t>(i)] / total_pos) : (EPSILON / total_pos);
//       double rate_neg = (count_neg[static_cast<size_t>(i)] > 0.0) ? (count_neg[static_cast<size_t>(i)] / total_neg) : (EPSILON / total_neg);
//       
//       // Ensure rates are not zero
//       rate_pos = std::max(rate_pos, EPSILON);
//       rate_neg = std::max(rate_neg, EPSILON);
//       
//       woe_values[static_cast<size_t>(i)] = std::log(rate_pos / rate_neg);
//     }
//   }
//   
//   void update_bin_labels() {
//     bin_labels.clear();
//     bin_labels.reserve(bin_edges.size() - 1);
//     for (size_t i = 0; i < bin_edges.size() - 1; ++i) {
//       std::ostringstream oss;
//       oss << std::fixed << std::setprecision(6);
//       double left = bin_edges[i];
//       double right = bin_edges[i+1];
//       if (std::isinf(left) && left < 0) {
//         oss << "(-Inf;" << right << "]";
//       } else if (std::isinf(right) && right > 0) {
//         oss << "(" << left << ";+Inf]";
//       } else {
//         oss << "(" << left << ";" << right << "]";
//       }
//       bin_labels.push_back(oss.str());
//     }
//   }
//   
//   int find_bin(double value) const {
//     // Locate bin via upper_bound
//     auto it = std::upper_bound(bin_edges.begin(), bin_edges.end(), value);
//     int bin_idx = static_cast<int>(std::distance(bin_edges.begin(), it)) - 1;
//     // Ensure bin_idx is within valid range
//     if (bin_idx < 0) bin_idx = 0;
//     if (bin_idx >= static_cast<int>(counts.size())) bin_idx = static_cast<int>(counts.size()) - 1;
//     return bin_idx;
//   }
//   
//   void enforce_monotonicity() {
//     if (counts.size() <= 2) {
//       // If already small number of bins, no monotonic enforcement needed
//       return;
//     }
//     
//     // Determine monotonic direction from first two WOE values
//     bool increasing = true;
//     if (woe_values.size() >= 2) {
//       increasing = (woe_values[1] >= woe_values[0]);
//     }
//     
//     bool is_monotonic = false;
//     int iterations = 0;
//     
//     // Enforce monotonicity by merging bins that break monotonic order
//     while (!is_monotonic && counts.size() > static_cast<size_t>(min_bins) && iterations < max_iterations) {
//       is_monotonic = true;
//       for (size_t i = 1; i < woe_values.size(); ++i) {
//         if ((increasing && (woe_values[i] < woe_values[i - 1])) ||
//             (!increasing && (woe_values[i] > woe_values[i - 1]))) {
//           // Merge bin i-1 and i
//           merge_bins(static_cast<int>(i - 1));
//           is_monotonic = false;
//           break;
//         }
//       }
//       iterations++;
//       if (counts.size() == static_cast<size_t>(min_bins)) {
//         // Reached minimum number of bins, stop
//         break;
//       }
//     }
//     
//     iterations_run += iterations;
//     if (iterations >= max_iterations) {
//       converged = false;
//     }
//   }
//   
//   void merge_bins(int idx) {
//     // Safety checks on idx
//     if (idx < 0 || idx >= static_cast<int>(counts.size()) - 1) {
//       // Invalid index for merging
//       return;
//     }
//     
//     // Merge bin idx with idx+1
//     bin_edges.erase(bin_edges.begin() + idx + 1);
//     counts[idx] += counts[static_cast<size_t>(idx + 1)];
//     counts.erase(counts.begin() + idx + 1);
//     count_pos[idx] += count_pos[static_cast<size_t>(idx + 1)];
//     count_pos.erase(count_pos.begin() + idx + 1);
//     count_neg[idx] += count_neg[static_cast<size_t>(idx + 1)];
//     count_neg.erase(count_neg.begin() + idx + 1);
//     
//     // Recalculate WOE and labels after merging
//     calculate_woe();
//     update_bin_labels();
//   }
//   
//   void ensure_bin_constraints() {
//     // Ensure counts.size() <= max_bins by merging if needed
//     int iterations = 0;
//     while (counts.size() > static_cast<size_t>(max_bins) && iterations < max_iterations) {
//       int idx = find_smallest_woe_diff();
//       if (idx == -1) {
//         // No valid merge index found, break to avoid infinite loop
//         break;
//       }
//       merge_bins(idx);
//       iterations++;
//     }
//     iterations_run += iterations;
//     if (iterations >= max_iterations) {
//       converged = false;
//     }
//     
//     // Handle rare bins
//     handle_rare_bins();
//   }
//   
//   int find_smallest_woe_diff() const {
//     if (woe_values.size() <= 1) return -1;
//     std::vector<double> woe_diffs(woe_values.size() - 1);
//     for (size_t i = 0; i < woe_diffs.size(); ++i) {
//       woe_diffs[i] = std::fabs(woe_values[i + 1] - woe_values[i]);
//     }
//     auto min_it = std::min_element(woe_diffs.begin(), woe_diffs.end());
//     if (min_it == woe_diffs.end()) return -1;
//     int idx = static_cast<int>(std::distance(woe_diffs.begin(), min_it));
//     return idx; // This corresponds directly to the bin pair (idx, idx+1)
//   }
//   
//   void handle_rare_bins() {
//     double total_count = std::accumulate(counts.begin(), counts.end(), 0.0);
//     bool merged = true;
//     int iterations = 0;
//     
//     // Merge bins that are too small until no rare bins or min_bins reached
//     while (merged && counts.size() > static_cast<size_t>(min_bins) && iterations < max_iterations) {
//       merged = false;
//       for (size_t i = 0; i < counts.size(); ++i) {
//         double proportion = total_count > 0.0 ? (counts[i] / total_count) : 0.0;
//         if (proportion < bin_cutoff && counts.size() > static_cast<size_t>(min_bins)) {
//           int merge_idx;
//           if (i == 0) {
//             merge_idx = 0; 
//           } else if (i == counts.size() - 1) {
//             merge_idx = static_cast<int>(counts.size()) - 2;
//           } else {
//             double diff_prev = std::fabs(woe_values[i] - woe_values[i - 1]);
//             double diff_next = std::fabs(woe_values[i] - woe_values[i + 1]);
//             if (diff_prev <= diff_next) {
//               merge_idx = static_cast<int>(i - 1);
//             } else {
//               merge_idx = static_cast<int>(i);
//             }
//           }
//           merge_bins(merge_idx);
//           merged = true;
//           iterations++;
//           break; // Restart loop after merge
//         }
//       }
//     }
//     
//     iterations_run += iterations;
//     if (iterations >= max_iterations) {
//       converged = false;
//     }
//   }
//   
//   void calculate_iv() {
//     iv_values.resize(woe_values.size());
//     for (size_t i = 0; i < woe_values.size(); ++i) {
//       double p_rate = (total_pos > 0.0) ? (count_pos[i] / total_pos) : EPSILON;
//       double n_rate = (total_neg > 0.0) ? (count_neg[i] / total_neg) : EPSILON;
//       // Ensure p_rate, n_rate > 0
//       p_rate = std::max(p_rate, EPSILON);
//       n_rate = std::max(n_rate, EPSILON);
//       iv_values[i] = (p_rate - n_rate) * woe_values[i];
//     }
//   }
// };
// 
// 
// //' @title Optimal Binning for Numerical Variables using Dynamic Programming with Local Constraints (DPLC)
// //'
// //' @description
// //' Performs optimal binning for numerical variables using a Dynamic Programming with Local Constraints (DPLC) approach.
// //' It creates optimal bins for a numerical feature based on its relationship with a binary target variable, 
// //' maximizing the predictive power while respecting user-defined constraints and enforcing monotonicity.
// //'
// //' @param target An integer vector of binary target values (0 or 1).
// //' @param feature A numeric vector of feature values.
// //' @param min_bins Minimum number of bins (default: 3).
// //' @param max_bins Maximum number of bins (default: 5).
// //' @param bin_cutoff Minimum proportion of total observations for a bin to avoid being merged (default: 0.05).
// //' @param max_n_prebins Maximum number of pre-bins before the optimization process (default: 20).
// //' @param convergence_threshold Convergence threshold for the algorithm (default: 1e-6).
// //' @param max_iterations Maximum number of iterations allowed (default: 1000).
// //'
// //' @return A list containing the following elements:
// //' \item{bin}{Character vector of bin ranges.}
// //' \item{woe}{Numeric vector of WoE values for each bin.}
// //' \item{iv}{Numeric vector of Information Value (IV) for each bin.}
// //' \item{count}{Numeric vector of total observations in each bin.}
// //' \item{count_pos}{Numeric vector of positive target observations in each bin.}
// //' \item{count_neg}{Numeric vector of negative target observations in each bin.}
// //' \item{cutpoints}{Numeric vector of cut points to generate the bins.}
// //' \item{converged}{Logical indicating if the algorithm converged.}
// //' \item{iterations}{Integer number of iterations run by the algorithm.}
// //'
// //' @details
// //' The Dynamic Programming with Local Constraints (DPLC) algorithm for numerical variables works as follows:
// //' 1. Perform initial pre-binning based on quantiles of the feature distribution.
// //' 2. Calculate initial counts and Weight of Evidence (WoE) for each bin.
// //' 3. Enforce monotonicity of WoE values across bins by merging adjacent non-monotonic bins.
// //' 4. Ensure the number of bins is between \code{min_bins} and \code{max_bins}:
// //'   - Merge bins with the smallest WoE difference if above \code{max_bins}.
// //'   - Handle rare bins by merging those below the \code{bin_cutoff} threshold.
// //' 5. Calculate final Information Value (IV) for each bin.
// //'
// //' The algorithm aims to create bins that maximize the predictive power of the numerical variable while adhering to the specified constraints. It enforces monotonicity of WoE values, which is particularly useful for credit scoring and risk modeling applications.
// //'
// //' Weight of Evidence (WoE) is calculated as:
// //' \deqn{WoE = \ln\left(\frac{\text{Positive Rate}}{\text{Negative Rate}}\right)}
// //'
// //' Information Value (IV) is calculated as:
// //' \deqn{IV = (\text{Positive Rate} - \text{Negative Rate}) \times WoE}
// //'
// //' @examples
// //' # Create sample data
// //' set.seed(123)
// //' n <- 1000
// //' target <- sample(0:1, n, replace = TRUE)
// //' feature <- rnorm(n)
// //'
// //' # Run optimal binning
// //' result <- optimal_binning_numerical_dplc(target, feature, min_bins = 2, max_bins = 4)
// //'
// //' # Print results
// //' print(result)
// //'
// //' @export
// // [[Rcpp::export]]
// Rcpp::List optimal_binning_numerical_dplc(Rcpp::IntegerVector target,
//                                           Rcpp::NumericVector feature,
//                                           int min_bins = 3,
//                                           int max_bins = 5,
//                                           double bin_cutoff = 0.05,
//                                           int max_n_prebins = 20,
//                                           double convergence_threshold = 1e-6,
//                                           int max_iterations = 1000) {
//   if (min_bins < 2) {
//     Rcpp::stop("min_bins must be at least 2");
//   }
//   if (max_bins < min_bins) {
//     Rcpp::stop("max_bins must be greater than or equal to min_bins");
//   }
//   
//   std::vector<double> feature_vec = Rcpp::as<std::vector<double>>(feature);
//   std::vector<unsigned int> target_vec = Rcpp::as<std::vector<unsigned int>>(target);
//   
//   // Check that target contains only 0s and 1s
//   for (size_t i = 0; i < target_vec.size(); ++i) {
//     if (target_vec[i] != 0U && target_vec[i] != 1U) {
//       Rcpp::stop("Target variable must contain only 0 and 1");
//     }
//   }
//   
//   OptimalBinningNumericalDPLC ob(feature_vec, target_vec,
//                                  min_bins, max_bins, bin_cutoff, max_n_prebins,
//                                  convergence_threshold, max_iterations);
//   ob.fit();
//   
//   return ob.get_results();
// }
