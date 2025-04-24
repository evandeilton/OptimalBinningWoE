// [[Rcpp::plugins(cpp11)]]
// [[Rcpp::depends(Rcpp)]]

#include <Rcpp.h>
#include <algorithm>
#include <vector>
#include <string>
#include <cmath>
#include <numeric>
#include <sstream>
#include <limits>
#include <unordered_set>
#include <functional>

/**
 * @file OptimalBinningNumericalLDB.cpp
 * @brief Implementation of Local Density Binning (LDB) algorithm for optimal binning of numerical variables
 * 
 * This implementation provides methods for supervised discretization of numerical variables
 * using the Local Density Binning approach, which preserves local density structure
 * while maximizing predictive power.
 */

using namespace Rcpp;

/**
 * Class for Optimal Binning using Local Density Binning (LDB)
 * 
 * LDB is a supervised discretization method that adapts bin boundaries to the local
 * density structure of the data while maximizing the predictive relationship with
 * a binary target variable. The algorithm balances statistical stability, predictive
 * power, and interpretability constraints.
 */
class OptimalBinningNumericalLDB {
private:
  // Parameters
  int min_bins;
  int max_bins;
  double bin_cutoff;
  int max_n_prebins;
  double convergence_threshold;
  int max_iterations;
  bool enforce_monotonic;
  
  // Convergence status and iterations
  bool converged;
  int iterations_run;
  
  // Data
  std::vector<double> feature;
  std::vector<int> target;
  
  // Binning structures
  std::vector<double> bin_edges;
  std::vector<double> woe_values;
  std::vector<double> iv_values;
  std::vector<int> counts;
  std::vector<int> count_pos;
  std::vector<int> count_neg;
  std::vector<double> event_rates;
  std::vector<std::string> bin_labels;
  
  // Total Information Value
  double total_iv;
  
  // Monotonicity direction (1 for increasing, -1 for decreasing, 0 for undetermined)
  int monotonicity_direction;
  
  // Constant for numerical stability in comparisons
  static constexpr double EPSILON = 1e-10;
  
  // Smoothing parameter for Laplace correction
  static constexpr double ALPHA = 0.5;
  
  /**
   * Compute initial bins based on density analysis
   * Uses a density-sensitive approach to create initial bin boundaries
   */
  void compute_prebins();
  
  /**
   * Compute Weight of Evidence (WoE) and Information Value (IV) for each bin
   */
  void compute_woe_iv();
  
  /**
   * Enforce monotonicity in WoE values across bins
   */
  void enforce_monotonicity();
  
  /**
   * Merge bins based on frequency and information preservation
   */
  void merge_bins();
  
  /**
   * Create formatted bin labels for output
   */
  void create_bin_labels();
  
  /**
   * Calculate Weight of Evidence (WoE) with Laplace smoothing
   * 
   * @param pos Count of positive class observations
   * @param neg Count of negative class observations
   * @param total_pos Total positive class observations
   * @param total_neg Total negative class observations
   * @return WoE value
   */
  double calculateWOE(int pos, int neg, double total_pos, double total_neg) const {
    double good = static_cast<double>(pos);
    double bad = static_cast<double>(neg);
    
    // Number of bins (for smoothing)
    double num_bins = static_cast<double>(bin_edges.size() - 1);
    
    // Laplace smoothing
    good = (good + ALPHA) / (total_pos + num_bins * ALPHA);
    bad = (bad + ALPHA) / (total_neg + num_bins * ALPHA);
    
    // Numerical stability
    double epsilon = 1e-14;
    good = std::max(good, epsilon);
    bad = std::max(bad, epsilon);
    
    return std::log(good / bad);
  }
  
  /**
   * Calculate Information Value (IV) contribution for a bin
   * 
   * @param woe Weight of Evidence value
   * @param pos Count of positive class observations
   * @param neg Count of negative class observations
   * @param total_pos Total positive class observations
   * @param total_neg Total negative class observations
   * @return IV contribution
   */
  double calculateIV(double woe, int pos, int neg, double total_pos, double total_neg) const {
    double dist_good = (total_pos > 0) ? static_cast<double>(pos) / total_pos : 0.0;
    double dist_bad = (total_neg > 0) ? static_cast<double>(neg) / total_neg : 0.0;
    return (dist_good - dist_bad) * woe;
  }
  
  /**
   * Merge two adjacent bins
   * 
   * @param bin_idx1 Index of first bin
   * @param bin_idx2 Index of second bin (must be adjacent to bin_idx1)
   */
  void merge_adjacent_bins(size_t bin_idx1, size_t bin_idx2) {
    // Ensure indices are adjacent and in order
    if (bin_idx1 > bin_idx2) {
      std::swap(bin_idx1, bin_idx2);
    }
    if (bin_idx2 != bin_idx1 + 1) {
      Rcpp::stop("Cannot merge non-adjacent bins");
    }
    
    // Merge bin counts
    counts[bin_idx1] += counts[bin_idx2];
    count_pos[bin_idx1] += count_pos[bin_idx2];
    count_neg[bin_idx1] += count_neg[bin_idx2];
    
    // Update bin edge (the upper edge of bin_idx1 becomes the upper edge of bin_idx2)
    bin_edges[bin_idx1 + 1] = bin_edges[bin_idx2 + 1];
    
    // Remove bin_idx2 from all vectors
    bin_edges.erase(bin_edges.begin() + bin_idx2 + 1);
    counts.erase(counts.begin() + bin_idx2);
    count_pos.erase(count_pos.begin() + bin_idx2);
    count_neg.erase(count_neg.begin() + bin_idx2);
    woe_values.erase(woe_values.begin() + bin_idx2);
    iv_values.erase(iv_values.begin() + bin_idx2);
    event_rates.erase(event_rates.begin() + bin_idx2);
    
    // Update event rate for merged bin
    event_rates[bin_idx1] = (counts[bin_idx1] > 0) ? 
    static_cast<double>(count_pos[bin_idx1]) / counts[bin_idx1] : 0.0;
    
    // Recalculate WoE and IV for the merged bin
    double total_pos = std::accumulate(count_pos.begin(), count_pos.end(), 0.0);
    double total_neg = std::accumulate(count_neg.begin(), count_neg.end(), 0.0);
    
    woe_values[bin_idx1] = calculateWOE(count_pos[bin_idx1], count_neg[bin_idx1], total_pos, total_neg);
    iv_values[bin_idx1] = calculateIV(woe_values[bin_idx1], count_pos[bin_idx1], count_neg[bin_idx1], 
                                      total_pos, total_neg);
  }
  
  /**
   * Find the bin index for a given value using binary search
   * More efficient than linear search for large datasets
   * 
   * @param value The value to find the bin for
   * @return The index of the bin containing the value
   */
  size_t find_bin_index(double value) const {
    // Edge cases
    if (std::isnan(value)) {
      return SIZE_MAX; // Special value to indicate NaN
    }
    
    if (value <= bin_edges.front()) {
      return 0;
    }
    
    if (value > bin_edges.back()) {
      return bin_edges.size() - 2; // Last bin
    }
    
    // Binary search for the bin
    auto it = std::upper_bound(bin_edges.begin(), bin_edges.end(), value);
    size_t idx = std::distance(bin_edges.begin(), it) - 1;
    
    return idx;
  }
  
  /**
   * Validate input data and parameters
   * Throws exception if validation fails
   */
  void validate_inputs() const {
    if (feature.empty() || target.empty()) {
      Rcpp::stop("Feature and target vectors must not be empty.");
    }
    
    if (feature.size() != target.size()) {
      Rcpp::stop("Feature and target must have the same length.");
    }
    
    if (min_bins < 2) {
      Rcpp::stop("min_bins must be at least 2.");
    }
    
    if (max_bins < min_bins) {
      Rcpp::stop("max_bins must be greater than or equal to min_bins.");
    }
    
    if (bin_cutoff < 0.0 || bin_cutoff > 1.0) {
      Rcpp::stop("bin_cutoff must be between 0 and 1.");
    }
    
    if (max_n_prebins < min_bins) {
      Rcpp::stop("max_n_prebins must be greater than or equal to min_bins.");
    }
    
    // Check if target is binary and contains both 0 and 1
    std::unordered_set<int> unique_target;
    int pos_count = 0;
    
    for (int t : target) {
      if (t != 0 && t != 1) {
        Rcpp::stop("Target must contain only binary values (0 or 1).");
      }
      unique_target.insert(t);
      if (t == 1) pos_count++;
    }
    
    if (unique_target.size() != 2 || unique_target.find(0) == unique_target.end() || 
        unique_target.find(1) == unique_target.end()) {
      Rcpp::stop("Target must contain both classes (0 and 1).");
    }
    
    // Ensure we have enough valid values
    int valid_count = 0;
    for (double val : feature) {
      if (!std::isnan(val) && !std::isinf(val)) {
        valid_count++;
      }
    }
    
    if (valid_count < min_bins) {
      Rcpp::stop("Not enough valid feature values for minimum number of bins.");
    }
  }
  
  /**
   * Handle the special case of few unique values
   * 
   * @param unique_values Vector of unique feature values
   * @return True if special case was handled, False otherwise
   */
  bool handle_few_unique_values(const std::vector<double>& unique_values) {
    int unique_count = static_cast<int>(unique_values.size());
    
    // Case: <= 2 unique values
    if (unique_count <= 2) {
      bin_edges.clear();
      bin_edges.push_back(-std::numeric_limits<double>::infinity());
      
      if (unique_count == 1) {
        bin_edges.push_back(std::numeric_limits<double>::infinity());
      } else if (unique_count == 2) {
        // Use midpoint as bin edge
        double midpoint = (unique_values[0] + unique_values[1]) / 2.0;
        bin_edges.push_back(midpoint);
        bin_edges.push_back(std::numeric_limits<double>::infinity());
      }
      
      compute_woe_iv();
      create_bin_labels();
      converged = true;
      iterations_run = 0;
      return true;
    }
    
    // Case: <= min_bins unique values
    if (unique_count <= min_bins) {
      bin_edges.clear();
      bin_edges.push_back(-std::numeric_limits<double>::infinity());
      
      for (int i = 1; i < unique_count; ++i) {
        // Use midpoint between adjacent unique values
        double mid = (unique_values[i - 1] + unique_values[i]) / 2.0;
        bin_edges.push_back(mid);
      }
      
      bin_edges.push_back(std::numeric_limits<double>::infinity());
      
      compute_woe_iv();
      create_bin_labels();
      converged = true;
      iterations_run = 0;
      return true;
    }
    
    return false;
  }
  
  /**
   * Calculate local density estimation for a vector of values
   * Used to identify regions with higher density for bin boundary placement
   * 
   * @param sorted_values Sorted vector of feature values
   * @return Vector of density estimates at each point
   */
  std::vector<double> estimate_local_density(const std::vector<double>& sorted_values) const {
    size_t n = sorted_values.size();
    std::vector<double> density(n, 0.0);
    
    if (n <= 1) {
      return density;
    }
    
    // Estimate bandwidth using Silverman's rule of thumb
    double range = sorted_values.back() - sorted_values.front();
    double iqr = 0.0;
    
    if (n >= 4) {
      size_t q1_idx = n / 4;
      size_t q3_idx = 3 * n / 4;
      iqr = sorted_values[q3_idx] - sorted_values[q1_idx];
    }
    
    // Fallback if IQR is too small
    if (iqr < EPSILON) {
      iqr = range;
    }
    
    // Standard deviation estimation
    double sum = 0.0;
    double sum_sq = 0.0;
    
    for (double val : sorted_values) {
      sum += val;
      sum_sq += val * val;
    }
    
    double mean = sum / n;
    double variance = (sum_sq / n) - (mean * mean);
    double std_dev = std::sqrt(std::max(variance, EPSILON));
    
    // Bandwidth using Silverman's rule
    double h = 0.9 * std::min(std_dev, iqr / 1.34) * std::pow(n, -0.2);
    
    // Ensure minimum bandwidth
    h = std::max(h, range / 1000.0);
    
    // Kernel density estimation at each point
    // Using a simple Gaussian kernel
    for (size_t i = 0; i < n; ++i) {
      double xi = sorted_values[i];
      double local_sum = 0.0;
      
      for (size_t j = 0; j < n; ++j) {
        double xj = sorted_values[j];
        double z = (xi - xj) / h;
        local_sum += std::exp(-0.5 * z * z); // Gaussian kernel
      }
      
      density[i] = local_sum / (n * h * std::sqrt(2.0 * M_PI));
    }
    
    return density;
  }
  
public:
  /**
   * Constructor for OptimalBinningNumericalLDB
   * 
   * @param min_bins Minimum number of bins
   * @param max_bins Maximum number of bins
   * @param bin_cutoff Minimum frequency fraction for each bin
   * @param max_n_prebins Maximum number of pre-bins before optimization
   * @param enforce_monotonic Whether to enforce monotonicity in WoE
   * @param convergence_threshold Convergence threshold
   * @param max_iterations Maximum iterations allowed
   */
  OptimalBinningNumericalLDB(
    int min_bins = 3, 
    int max_bins = 5, 
    double bin_cutoff = 0.05,
    int max_n_prebins = 20, 
    bool enforce_monotonic = true,
    double convergence_threshold = 1e-6,
    int max_iterations = 1000)
    : min_bins(min_bins), 
      max_bins(max_bins),
      bin_cutoff(bin_cutoff), 
      max_n_prebins(max_n_prebins),
      enforce_monotonic(enforce_monotonic),
      convergence_threshold(convergence_threshold), 
      max_iterations(max_iterations),
      total_iv(0.0),
      converged(true), 
      iterations_run(0),
      monotonicity_direction(0) {}
  
  /**
   * Fit the binning model to data
   * 
   * @param feature_input Feature vector to be binned
   * @param target_input Binary target vector (0/1)
   */
  void fit(const std::vector<double>& feature_input, const std::vector<int>& target_input) {
    // Store and validate inputs
    this->feature = feature_input;
    this->target = target_input;
    
    validate_inputs();
    
    // Get valid feature values (non-NaN, non-Inf)
    std::vector<double> valid_feature;
    std::vector<int> valid_target;
    
    for (size_t i = 0; i < feature.size(); ++i) {
      if (!std::isnan(feature[i]) && !std::isinf(feature[i])) {
        valid_feature.push_back(feature[i]);
        valid_target.push_back(target[i]);
      }
    }
    
    // Extract unique sorted values
    std::vector<double> unique_feature = valid_feature;
    std::sort(unique_feature.begin(), unique_feature.end());
    unique_feature.erase(std::unique(unique_feature.begin(), unique_feature.end()), 
                         unique_feature.end());
    
    // Handle special cases of few unique values
    if (handle_few_unique_values(unique_feature)) {
      return;
    }
    
    // General case: proceed with density-based binning
    compute_prebins();
    compute_woe_iv();
    
    if (enforce_monotonic) {
      enforce_monotonicity();
    }
    
    merge_bins();
    create_bin_labels();
    
    // Check convergence
    converged = (iterations_run < max_iterations);
    
    // Calculate total IV
    total_iv = std::accumulate(iv_values.begin(), iv_values.end(), 0.0);
  }
  
  /**
   * Get the results of the binning process
   * 
   * @return List containing bin information and metrics
   */
  Rcpp::List transform() const {
    // Prepare cutpoints for output
    std::vector<double> cutpoints;
    if (bin_edges.size() > 2) {
      cutpoints.assign(bin_edges.begin() + 1, bin_edges.end() - 1);
    }
    
    // Create bin IDs (1-based indexing for R)
    Rcpp::NumericVector ids(bin_labels.size());
    for (int i = 0; i < static_cast<int>(bin_labels.size()); i++) {
      ids[i] = i + 1;
    }
    
    // Determine monotonicity status
    std::string monotonicity = "none";
    if (monotonicity_direction > 0) {
      monotonicity = "increasing";
    } else if (monotonicity_direction < 0) {
      monotonicity = "decreasing";
    }
    
    // Return results
    return Rcpp::List::create(
      Rcpp::Named("id") = ids,
      Rcpp::Named("bin") = bin_labels,
      Rcpp::Named("woe") = woe_values,
      Rcpp::Named("iv") = iv_values,
      Rcpp::Named("count") = counts,
      Rcpp::Named("count_pos") = count_pos,
      Rcpp::Named("count_neg") = count_neg,
      Rcpp::Named("event_rate") = event_rates,
      Rcpp::Named("cutpoints") = cutpoints,
      Rcpp::Named("converged") = converged,
      Rcpp::Named("iterations") = iterations_run,
      Rcpp::Named("total_iv") = total_iv,
      Rcpp::Named("monotonicity") = monotonicity
    );
  }
};

// Implementation of private methods

/**
 * Compute initial bins based on density analysis
 * This method uses local density estimates to place bin boundaries in optimal locations
 */
void OptimalBinningNumericalLDB::compute_prebins() {
  // Filter out NaN and Inf values
  std::vector<double> valid_feature;
  for (double val : feature) {
    if (!std::isnan(val) && !std::isinf(val)) {
      valid_feature.push_back(val);
    }
  }
  
  // Sort valid values
  std::sort(valid_feature.begin(), valid_feature.end());
  
  // Get density estimates
  std::vector<double> density = estimate_local_density(valid_feature);
  
  // Initialize bin edges
  bin_edges.clear();
  bin_edges.push_back(-std::numeric_limits<double>::infinity());
  
  size_t n = valid_feature.size();
  
  // Special case: very few values
  if (n < static_cast<size_t>(max_n_prebins)) {
    // Use all unique values if fewer than max_n_prebins
    std::vector<double> unique_values = valid_feature;
    unique_values.erase(std::unique(unique_values.begin(), unique_values.end()), unique_values.end());
    
    for (size_t i = 1; i < unique_values.size(); ++i) {
      bin_edges.push_back((unique_values[i-1] + unique_values[i]) / 2.0);
    }
  } else {
    // Density-based binning for larger datasets
    
    // Find local minima in density as potential cut points
    std::vector<size_t> min_indices;
    for (size_t i = 1; i < density.size() - 1; ++i) {
      if (density[i] < density[i-1] && density[i] < density[i+1]) {
        min_indices.push_back(i);
      }
    }
    
    // Sort minima by density (ascending)
    std::sort(min_indices.begin(), min_indices.end(),
              [&density](size_t i, size_t j) { return density[i] < density[j]; });
    
    // Use top n_cuts density minima for bin edges
    int n_cuts = std::min(max_n_prebins - 1, static_cast<int>(min_indices.size()));
    std::vector<size_t> selected_indices;
    
    if (n_cuts >= 1) {
      selected_indices.push_back(min_indices[0]);
      
      // Add more cut points, ensuring they're well-spaced
      for (size_t i = 1; i < min_indices.size() && selected_indices.size() < static_cast<size_t>(n_cuts); ++i) {
        bool too_close = false;
        for (size_t sel_idx : selected_indices) {
          if (std::abs(static_cast<int>(min_indices[i]) - static_cast<int>(sel_idx)) < static_cast<int>(n / (max_n_prebins * 2))) {
            too_close = true;
            break;
          }
        }
        
        if (!too_close) {
          selected_indices.push_back(min_indices[i]);
        }
      }
      
      // Sort selected indices
      std::sort(selected_indices.begin(), selected_indices.end());
      
      // Add bin edges at selected points
      for (size_t idx : selected_indices) {
        bin_edges.push_back(valid_feature[idx]);
      }
    }
    
    // If not enough density minima were found, supplement with quantile-based cuts
    if (bin_edges.size() < static_cast<size_t>(min_bins)) {
      int n_additional = min_bins - static_cast<int>(bin_edges.size());
      for (int i = 1; i <= n_additional; ++i) {
        double q = static_cast<double>(i) / (n_additional + 1);
        size_t idx = static_cast<size_t>(std::floor(q * n));
        if (idx >= n) idx = n - 1;
        bin_edges.push_back(valid_feature[idx]);
      }
      
      // Ensure bin edges are unique and sorted
      std::sort(bin_edges.begin(), bin_edges.end());
      bin_edges.erase(std::unique(bin_edges.begin(), bin_edges.end()), bin_edges.end());
    }
  }
  
  // Ensure last bin edge is +Inf
  if (bin_edges.back() != std::numeric_limits<double>::infinity()) {
    bin_edges.push_back(std::numeric_limits<double>::infinity());
  }
}

/**
 * Compute Weight of Evidence (WoE) and Information Value (IV) for each bin
 */
void OptimalBinningNumericalLDB::compute_woe_iv() {
  size_t num_bins = bin_edges.size() - 1;
  
  // Initialize counts and metrics vectors
  counts.assign(num_bins, 0);
  count_pos.assign(num_bins, 0);
  count_neg.assign(num_bins, 0);
  woe_values.assign(num_bins, 0.0);
  iv_values.assign(num_bins, 0.0);
  event_rates.assign(num_bins, 0.0);
  
  // Count total positives and negatives
  double total_pos = 0.0;
  double total_neg = 0.0;
  
  // First pass: count observations in each bin
  for (size_t i = 0; i < feature.size(); ++i) {
    // Skip NaN values
    if (std::isnan(feature[i])) {
      continue;
    }
    
    // Find bin for this observation using binary search
    size_t bin_idx = find_bin_index(feature[i]);
    
    if (bin_idx < counts.size()) {
      counts[bin_idx]++;
      if (target[i] == 1) {
        count_pos[bin_idx]++;
        total_pos += 1.0;
      } else {
        count_neg[bin_idx]++;
        total_neg += 1.0;
      }
    }
  }
  
  // Check for sufficient data
  if (total_pos <= 0.0 || total_neg <= 0.0) {
    Rcpp::stop("Target vector must contain both positive and negative cases after handling missing values.");
  }
  
  // Second pass: calculate WoE and IV for each bin
  for (size_t b = 0; b < num_bins; ++b) {
    event_rates[b] = (counts[b] > 0) ? static_cast<double>(count_pos[b]) / counts[b] : 0.0;
    woe_values[b] = calculateWOE(count_pos[b], count_neg[b], total_pos, total_neg);
    iv_values[b] = calculateIV(woe_values[b], count_pos[b], count_neg[b], total_pos, total_neg);
  }
}

/**
 * Enforce monotonicity in WoE values across bins
 */
void OptimalBinningNumericalLDB::enforce_monotonicity() {
  if (woe_values.size() <= 1) {
    return;
  }
  
  // Determine monotonicity direction
  // Use a more robust approach: look at correlation between bin index and WoE
  double sum_idx = 0.0;
  double sum_woe = 0.0;
  double sum_idx_sq = 0.0;
  double sum_woe_sq = 0.0;
  double sum_idx_woe = 0.0;
  
  for (size_t i = 0; i < woe_values.size(); ++i) {
    double idx = static_cast<double>(i);
    double woe = woe_values[i];
    
    sum_idx += idx;
    sum_woe += woe;
    sum_idx_sq += idx * idx;
    sum_woe_sq += woe * woe;
    sum_idx_woe += idx * woe;
  }
  
  double n = static_cast<double>(woe_values.size());
  double numerator = n * sum_idx_woe - sum_idx * sum_woe;
  double denominator_idx = n * sum_idx_sq - sum_idx * sum_idx;
  double denominator_woe = n * sum_woe_sq - sum_woe * sum_woe;
  
  // Correlation coefficient
  double correlation = 0.0;
  if (denominator_idx > EPSILON && denominator_woe > EPSILON) {
    correlation = numerator / std::sqrt(denominator_idx * denominator_woe);
  }
  
  // Set direction based on correlation
  monotonicity_direction = (correlation >= 0.0) ? 1 : -1;
  
  // Enforce monotonicity by merging bins
  bool monotonic = false;
  int iter = 0;
  
  while (!monotonic && counts.size() > static_cast<size_t>(min_bins) && (iterations_run + iter) < max_iterations) {
    monotonic = true;
    
    for (size_t b = 1; b < woe_values.size(); ++b) {
      double diff = woe_values[b] - woe_values[b - 1];
      
      // Check for monotonicity violation
      if ((monotonicity_direction > 0 && diff < 0) || (monotonicity_direction < 0 && diff > 0)) {
        // Merge bins b-1 and b
        merge_adjacent_bins(b - 1, b);
        
        monotonic = false;
        break;
      }
    }
    
    iter++;
  }
  
  iterations_run += iter;
}

/**
 * Merge bins based on frequency and information preservation
 */
void OptimalBinningNumericalLDB::merge_bins() {
  // Calculate minimum bin count
  size_t n = feature.size();
  double min_bin_count = bin_cutoff * static_cast<double>(n);
  
  // Step 1: Merge bins with frequency below threshold
  bool bins_merged = true;
  int iter = 0;
  
  while (bins_merged && counts.size() > static_cast<size_t>(min_bins) && iter < max_iterations) {
    bins_merged = false;
    
    for (size_t b = 0; b < counts.size(); ++b) {
      if (counts[b] < min_bin_count && counts.size() > static_cast<size_t>(min_bins)) {
        bins_merged = true;
        
        // Determine merge direction
        size_t merge_with;
        
        if (b == 0) {
          // First bin: merge with next
          merge_with = b + 1;
        } else if (b == counts.size() - 1) {
          // Last bin: merge with previous
          merge_with = b - 1;
        } else {
          // Middle bin: choose based on similarity in event rates
          double diff_prev = std::fabs(event_rates[b] - event_rates[b - 1]);
          double diff_next = std::fabs(event_rates[b] - event_rates[b + 1]);
          
          merge_with = (diff_prev <= diff_next) ? (b - 1) : (b + 1);
          
          // Consider IV preservation as secondary criterion
          if (std::fabs(diff_prev - diff_next) < EPSILON) {
            double iv_loss_prev = iv_values[b] + iv_values[b - 1];
            double iv_loss_next = iv_values[b] + iv_values[b + 1];
            
            // Prefer direction that preserves more IV
            merge_with = (iv_loss_prev >= iv_loss_next) ? (b - 1) : (b + 1);
          }
        }
        
        // Perform merge
        merge_adjacent_bins(b, merge_with);
        
        break;
      }
    }
    
    iter++;
  }
  
  iterations_run += iter;
  
  // Step 2: Ensure number of bins doesn't exceed max_bins
  while (counts.size() > static_cast<size_t>(max_bins) && iterations_run < max_iterations) {
    // Find pair of adjacent bins with smallest IV loss when merged
    size_t merge_idx1 = 0;
    size_t merge_idx2 = 1;
    double min_iv_loss = std::numeric_limits<double>::max();
    
    for (size_t b = 0; b < counts.size() - 1; ++b) {
      // Calculate combined IV for merged bin
      int combined_pos = count_pos[b] + count_pos[b + 1];
      int combined_neg = count_neg[b] + count_neg[b + 1];
      
      double total_pos = std::accumulate(count_pos.begin(), count_pos.end(), 0.0);
      double total_neg = std::accumulate(count_neg.begin(), count_neg.end(), 0.0);
      
      double combined_woe = calculateWOE(combined_pos, combined_neg, total_pos, total_neg);
      double combined_iv = calculateIV(combined_woe, combined_pos, combined_neg, total_pos, total_neg);
      
      // Current IV of the two bins
      double current_iv = iv_values[b] + iv_values[b + 1];
      
      // IV loss from merging
      double iv_loss = current_iv - combined_iv;
      
      if (iv_loss < min_iv_loss) {
        min_iv_loss = iv_loss;
        merge_idx1 = b;
        merge_idx2 = b + 1;
      }
    }
    
    // Perform merge
    merge_adjacent_bins(merge_idx1, merge_idx2);
    
    iterations_run++;
  }
}

/**
 * Create formatted bin labels for output
 */
void OptimalBinningNumericalLDB::create_bin_labels() {
  bin_labels.clear();
  size_t num_bins = bin_edges.size() - 1;
  bin_labels.reserve(num_bins);
  
  for (size_t b = 0; b < num_bins; ++b) {
    std::ostringstream oss;
    oss.precision(6);
    oss << std::fixed;
    
    oss << "(";
    
    if (std::isinf(bin_edges[b]) && bin_edges[b] < 0) {
      oss << "-Inf";
    } else {
      oss << bin_edges[b];
    }
    
    oss << ";";
    
    if (std::isinf(bin_edges[b + 1]) && bin_edges[b + 1] > 0) {
      oss << "+Inf";
    } else {
      oss << bin_edges[b + 1];
    }
    
    oss << "]";
    bin_labels.emplace_back(oss.str());
  }
}

//' @title Optimal Binning for Numerical Variables using Local Density Binning (LDB)
//'
//' @description
//' Implements the Local Density Binning (LDB) algorithm for optimal binning of numerical variables.
//' This method adapts bin boundaries based on the local density structure of the data while
//' maximizing the predictive relationship with a binary target variable. LDB is particularly
//' effective for features with non-uniform distributions or multiple modes.
//'
//' @details
//' ## Algorithm Overview
//' 
//' The Local Density Binning algorithm operates in several phases:
//' 
//' 1. **Density Analysis**: Analyzes the local density structure of the feature to identify
//'    regions of high and low density, placing bin boundaries preferentially at density minima.
//' 
//' 2. **Initial Binning**: Creates initial bins based on density minima and/or quantiles.
//' 
//' 3. **Statistical Optimization**:
//'    - Merges bins with frequencies below threshold for stability
//'    - Enforces monotonicity in Weight of Evidence (optional)
//'    - Adjusts to meet constraints on minimum and maximum bin count
//' 
//' 4. **Information Value Calculation**: Computes predictive metrics for each bin
//' 
//' ## Mathematical Foundation
//' 
//' The algorithm employs several statistical concepts:
//' 
//' ### 1. Kernel Density Estimation
//' 
//' To identify the local density structure:
//' 
//' \deqn{f_h(x) = \frac{1}{nh}\sum_{i=1}^{n}K\left(\frac{x-x_i}{h}\right)}
//' 
//' Where:
//' - \eqn{K} is a kernel function (Gaussian kernel in this implementation)
//' - \eqn{h} is the bandwidth parameter (selected using Silverman's rule of thumb)
//' - \eqn{n} is the number of observations
//' 
//' ### 2. Weight of Evidence (WoE)
//' 
//' For assessing the predictive power of each bin:
//' 
//' \deqn{WoE_i = \ln\left(\frac{(p_i + \alpha) / (P + k\alpha)}{(n_i + \alpha) / (N + k\alpha)}\right)}
//' 
//' Where:
//' - \eqn{p_i}: Number of positive cases in bin \eqn{i}
//' - \eqn{P}: Total number of positive cases
//' - \eqn{n_i}: Number of negative cases in bin \eqn{i}
//' - \eqn{N}: Total number of negative cases
//' - \eqn{\alpha}: Smoothing factor (0.5 in this implementation)
//' - \eqn{k}: Number of bins
//' 
//' ### 3. Information Value (IV)
//' 
//' For quantifying overall predictive power:
//' 
//' \deqn{IV_i = \left(\frac{p_i}{P} - \frac{n_i}{N}\right) \times WoE_i}
//' 
//' \deqn{IV_{total} = \sum_{i=1}^{k} IV_i}
//' 
//' ## Advantages of Local Density Binning
//' 
//' - **Respects Data Structure**: Places bin boundaries at natural gaps in the distribution
//' - **Adapts to Multimodality**: Handles features with multiple modes effectively
//' - **Maximizes Information**: Optimizes binning for predictive power
//' - **Statistical Stability**: Ensures sufficient observations in each bin
//' - **Interpretability**: Produces monotonic WoE patterns when requested
//'
//' @param target A binary integer vector (0 or 1) representing the target variable.
//' @param feature A numeric vector representing the feature to be binned.
//' @param min_bins Minimum number of bins (default: 3).
//' @param max_bins Maximum number of bins (default: 5).
//' @param bin_cutoff Minimum frequency fraction for each bin (default: 0.05).
//' @param max_n_prebins Maximum number of pre-bins before optimization (default: 20).
//' @param enforce_monotonic Whether to enforce monotonic WoE across bins (default: TRUE).
//' @param convergence_threshold Convergence threshold for optimization (default: 1e-6).
//' @param max_iterations Maximum iterations allowed (default: 1000).
//'
//' @return A list containing:
//' \item{id}{Numeric identifiers for each bin (1-based).}
//' \item{bin}{Character vector with bin intervals.}
//' \item{woe}{Numeric vector with Weight of Evidence values for each bin.}
//' \item{iv}{Numeric vector with Information Value contribution for each bin.}
//' \item{count}{Integer vector with the total number of observations in each bin.}
//' \item{count_pos}{Integer vector with the positive class count in each bin.}
//' \item{count_neg}{Integer vector with the negative class count in each bin.}
//' \item{event_rate}{Numeric vector with the event rate (proportion of positives) in each bin.}
//' \item{cutpoints}{Numeric vector with the bin boundaries (excluding infinities).}
//' \item{converged}{Logical indicating whether the algorithm converged.}
//' \item{iterations}{Integer count of iterations performed.}
//' \item{total_iv}{Numeric total Information Value of the binning solution.}
//' \item{monotonicity}{Character indicating the monotonicity direction ("increasing", "decreasing", or "none").}
//'
//' @examples
//' \dontrun{
//' # Generate synthetic data
//' set.seed(123)
//' target <- sample(0:1, 1000, replace = TRUE)
//' feature <- rnorm(1000)
//' 
//' # Basic usage
//' result <- optimal_binning_numerical_ldb(target, feature)
//' print(result)
//' 
//' # Custom parameters
//' result_custom <- optimal_binning_numerical_ldb(
//'   target = target,
//'   feature = feature,
//'   min_bins = 2,
//'   max_bins = 8,
//'   bin_cutoff = 0.03,
//'   enforce_monotonic = TRUE
//' )
//' 
//' # Access specific components
//' bins <- result$bin
//' woe_values <- result$woe
//' total_iv <- result$total_iv
//' monotonicity <- result$monotonicity
//' }
//'
//' @references
//' Bin, Y., Liang, S., Chen, Z., Yang, S., & Zhang, L. (2019). Density-based supervised discretization 
//' for continuous feature. *Knowledge-Based Systems*, 166, 1-17.
//' 
//' Belkin, M., & Niyogi, P. (2003). Laplacian eigenmaps for dimensionality reduction and data 
//' representation. *Neural Computation*, 15(6), 1373-1396.
//' 
//' Silverman, B. W. (1986). *Density Estimation for Statistics and Data Analysis*. Chapman and Hall/CRC.
//' 
//' Dougherty, J., Kohavi, R., & Sahami, M. (1995). Supervised and unsupervised discretization of 
//' continuous features. *Proceedings of the Twelfth International Conference on Machine Learning*, 194-202.
//' 
//' Siddiqi, N. (2006). *Credit Risk Scorecards: Developing and Implementing Intelligent Credit Scoring*. 
//' John Wiley & Sons.
//' 
//' Thomas, L. C. (2009). *Consumer Credit Models: Pricing, Profit and Portfolios*. Oxford University Press.
//'
//' @export
// [[Rcpp::export]]
Rcpp::List optimal_binning_numerical_ldb(
   Rcpp::IntegerVector target,
   Rcpp::NumericVector feature,
   int min_bins = 3,
   int max_bins = 5,
   double bin_cutoff = 0.05,
   int max_n_prebins = 20,
   bool enforce_monotonic = true,
   double convergence_threshold = 1e-6,
   int max_iterations = 1000) {
 
 try {
   // Convert R vectors to C++ vectors
   std::vector<double> feature_vec = Rcpp::as<std::vector<double>>(feature);
   std::vector<int> target_vec = Rcpp::as<std::vector<int>>(target);
   
   // Initialize and execute binning algorithm
   OptimalBinningNumericalLDB binner(
       min_bins, max_bins, 
       bin_cutoff, max_n_prebins,
       enforce_monotonic,
       convergence_threshold, max_iterations);
   
   binner.fit(feature_vec, target_vec);
   
   return binner.transform();
 } catch(std::exception &e) {
   forward_exception_to_r(e);
 } catch(...) {
   ::Rf_error("Unknown C++ exception in optimal_binning_numerical_ldb");
 }
 
 // Should never reach here
 return R_NilValue;
}










// // [[Rcpp::plugins(cpp11)]]
// // [[Rcpp::depends(Rcpp)]]
// 
// #include <Rcpp.h>
// #include <algorithm>
// #include <vector>
// #include <string>
// #include <cmath>
// #include <numeric>
// #include <sstream>
// #include <limits>
// #include <unordered_set>
// 
// using namespace Rcpp;
// 
// // Classe para Binning Ótimo Numérico usando Local Density Binning (LDB)
// class OptimalBinningNumericalLDB {
// private:
//   // Parâmetros
//   int min_bins;
//   int max_bins;
//   double bin_cutoff;
//   int max_n_prebins;
//   double convergence_threshold;
//   int max_iterations;
//   
//   // Status de convergência e iterações
//   bool converged;
//   int iterations_run;
//   
//   // Dados
//   std::vector<double> feature;
//   std::vector<int> target;
//   
//   // Estruturas de binning
//   std::vector<double> bin_edges;
//   std::vector<double> woe_values;
//   std::vector<double> iv_values;
//   std::vector<int> counts;
//   std::vector<int> count_pos;
//   std::vector<int> count_neg;
//   std::vector<std::string> bin_labels;
//   
//   // IV total
//   double total_iv;
//   
//   // Métodos privados
//   void compute_prebins();
//   void compute_woe_iv();
//   void enforce_monotonicity();
//   void merge_bins();
//   void create_bin_labels();
//   
//   // Métodos utilitários
//   double calculateWOE(int pos, int neg, double total_pos, double total_neg) const;
//   double calculateIV(double woe, int pos, int neg, double total_pos, double total_neg) const;
//   
// public:
//   // Construtor
//   OptimalBinningNumericalLDB(int min_bins = 3, int max_bins = 5, double bin_cutoff = 0.05,
//                              int max_n_prebins = 20, double convergence_threshold = 1e-6,
//                              int max_iterations = 1000);
//   
//   // Ajuste do modelo
//   void fit(const std::vector<double>& feature_input, const std::vector<int>& target_input);
//   
//   // Transform para obter resultados
//   Rcpp::List transform();
// };
// 
// // Construtor
// OptimalBinningNumericalLDB::OptimalBinningNumericalLDB(int min_bins, int max_bins, double bin_cutoff,
//                                                        int max_n_prebins, double convergence_threshold,
//                                                        int max_iterations) {
//   this->min_bins = min_bins;
//   this->max_bins = max_bins;
//   this->bin_cutoff = bin_cutoff;
//   this->max_n_prebins = max_n_prebins;
//   this->convergence_threshold = convergence_threshold;
//   this->max_iterations = max_iterations;
//   this->total_iv = 0.0;
//   this->converged = true;
//   this->iterations_run = 0;
// }
// 
// // Cálculo de WOE com suavização Laplace
// double OptimalBinningNumericalLDB::calculateWOE(int pos, int neg, double total_pos, double total_neg) const {
//   double good = static_cast<double>(pos);
//   double bad = static_cast<double>(neg);
//   
//   // Suavização Laplace
//   good = (good + 0.5) / (total_pos + 1.0);
//   bad = (bad + 0.5) / (total_neg + 1.0);
//   
//   // Evita log(0)
//   double epsilon = 1e-14;
//   good = std::max(good, epsilon);
//   bad = std::max(bad, epsilon);
//   
//   return std::log(good / bad);
// }
// 
// // Cálculo de IV
// double OptimalBinningNumericalLDB::calculateIV(double woe, int pos, int neg, double total_pos, double total_neg) const {
//   double dist_good = (pos > 0) ? static_cast<double>(pos) / total_pos : 0.0;
//   double dist_bad = (neg > 0) ? static_cast<double>(neg) / total_neg : 0.0;
//   return (dist_good - dist_bad) * woe;
// }
// 
// // Método fit
// void OptimalBinningNumericalLDB::fit(const std::vector<double>& feature_input, const std::vector<int>& target_input) {
//   // Validações de entrada
//   if (feature_input.empty() || target_input.empty()) {
//     Rcpp::stop("Feature and target vectors must not be empty.");
//   }
//   
//   if (feature_input.size() != target_input.size()) {
//     Rcpp::stop("Feature and target must have the same length.");
//   }
//   
//   if (min_bins < 2) {
//     Rcpp::stop("min_bins must be at least 2.");
//   }
//   
//   if (max_bins < min_bins) {
//     Rcpp::stop("max_bins must be greater than or equal to min_bins.");
//   }
//   
//   if (bin_cutoff < 0.0 || bin_cutoff > 1.0) {
//     Rcpp::stop("bin_cutoff must be between 0 and 1.");
//   }
//   
//   if (max_n_prebins < min_bins) {
//     Rcpp::stop("max_n_prebins must be greater than or equal to min_bins.");
//   }
//   
//   // Valida se target é binário e contém 0 e 1
//   std::unordered_set<int> target_set(target_input.begin(), target_input.end());
//   if (target_set.find(0) == target_set.end() || target_set.find(1) == target_set.end()) {
//     Rcpp::stop("Target must contain at least one 0 and one 1.");
//   }
//   
//   this->feature = feature_input;
//   this->target = target_input;
//   
//   // Número de valores únicos
//   std::vector<double> unique_feature = feature_input;
//   std::sort(unique_feature.begin(), unique_feature.end());
//   unique_feature.erase(std::unique(unique_feature.begin(), unique_feature.end()), unique_feature.end());
//   
//   int unique_count = static_cast<int>(unique_feature.size());
//   
//   // Caso <= 2 valores únicos
//   if (unique_count <= 2) {
//     bin_edges.clear();
//     bin_edges.push_back(-std::numeric_limits<double>::infinity());
//     
//     if (unique_count == 1) {
//       bin_edges.push_back(std::numeric_limits<double>::infinity());
//     } else if (unique_count == 2) {
//       double midpoint = (unique_feature[0] + unique_feature[1]) / 2.0;
//       bin_edges.push_back(midpoint);
//       bin_edges.push_back(std::numeric_limits<double>::infinity());
//     }
//     
//     compute_woe_iv();
//     create_bin_labels();
//     converged = true;
//     iterations_run = 0;
//     return;
//   }
//   
//   // Caso <= min_bins valores únicos
//   if (unique_count <= min_bins) {
//     bin_edges.clear();
//     bin_edges.push_back(-std::numeric_limits<double>::infinity());
//     for (int i = 1; i < unique_count; ++i) {
//       double mid = (unique_feature[i - 1] + unique_feature[i]) / 2.0;
//       bin_edges.push_back(mid);
//     }
//     bin_edges.push_back(std::numeric_limits<double>::infinity());
//     
//     compute_woe_iv();
//     create_bin_labels();
//     converged = true;
//     iterations_run = 0;
//     return;
//   }
//   
//   // Caso geral: mais do que min_bins valores
//   compute_prebins();
//   compute_woe_iv();
//   enforce_monotonicity();
//   merge_bins();
//   create_bin_labels();
//   
//   // Verifica se convergência foi atingida
//   converged = (iterations_run < max_iterations);
// }
// 
// // Computa pré-bins
// void OptimalBinningNumericalLDB::compute_prebins() {
//   size_t n = feature.size();
//   std::vector<double> sorted_feature = feature;
//   std::sort(sorted_feature.begin(), sorted_feature.end());
//   
//   bin_edges.clear();
//   bin_edges.push_back(-std::numeric_limits<double>::infinity());
//   
//   // Cria cortes baseados em quantis
//   for (int i = 1; i < max_n_prebins; ++i) {
//     size_t idx = static_cast<size_t>(std::floor(n * i / static_cast<double>(max_n_prebins)));
//     if (idx >= n) idx = n - 1;
//     double edge = sorted_feature[idx];
//     bin_edges.push_back(edge);
//   }
//   
//   bin_edges.push_back(std::numeric_limits<double>::infinity());
//   
//   // Remove duplicatas
//   bin_edges.erase(std::unique(bin_edges.begin(), bin_edges.end()), bin_edges.end());
// }
// 
// // Computa WoE e IV
// void OptimalBinningNumericalLDB::compute_woe_iv() {
//   size_t n = feature.size();
//   size_t num_bins = bin_edges.size() - 1;
//   
//   counts.assign(num_bins, 0);
//   count_pos.assign(num_bins, 0);
//   count_neg.assign(num_bins, 0);
//   woe_values.assign(num_bins, 0.0);
//   iv_values.assign(num_bins, 0.0);
//   
//   double total_pos = std::accumulate(target.begin(), target.end(), 0.0);
//   double total_neg = static_cast<double>(n) - total_pos;
//   
//   if (total_pos <= 0.0 || total_neg <= 0.0) {
//     Rcpp::stop("Target vector must contain both positive and negative cases.");
//   }
//   
//   // Atribui cada ponto ao bin
//   for (size_t i = 0; i < n; ++i) {
//     double x = feature[i];
//     int t = target[i];
//     int bin = -1;
//     
//     // Busca bin
//     // Otimização: poderia usar busca binária, mas mantendo por simplicidade
//     for (size_t b = 0; b < num_bins; ++b) {
//       if (x > bin_edges[b] && x <= bin_edges[b + 1]) {
//         bin = static_cast<int>(b);
//         break;
//       }
//     }
//     if (bin == -1) {
//       // Fallback: se não encontrou, pode ser caso extremo
//       if (x <= bin_edges.front()) {
//         bin = 0;
//       } else if (x > bin_edges.back()) {
//         bin = static_cast<int>(num_bins - 1);
//       } else {
//         Rcpp::stop("Error assigning data point to a bin.");
//       }
//     }
//     
//     counts[static_cast<size_t>(bin)]++;
//     if (t == 1) {
//       count_pos[static_cast<size_t>(bin)]++;
//     } else {
//       count_neg[static_cast<size_t>(bin)]++;
//     }
//   }
//   
//   for (size_t b = 0; b < num_bins; ++b) {
//     woe_values[b] = calculateWOE(count_pos[b], count_neg[b], total_pos, total_neg);
//     iv_values[b] = calculateIV(woe_values[b], count_pos[b], count_neg[b], total_pos, total_neg);
//   }
// }
// 
// // Impõe monotonicidade
// void OptimalBinningNumericalLDB::enforce_monotonicity() {
//   if (counts.size() <= static_cast<size_t>(min_bins)) {
//     return;
//   }
//   
//   // Determina direção da monotonicidade
//   int direction = 0;
//   // Tenta inferir a direção a partir dos primeiros bins
//   for (size_t b = 1; b < woe_values.size(); ++b) {
//     double diff = woe_values[b] - woe_values[b - 1];
//     if (diff > 0) {
//       direction++;
//     } else if (diff < 0) {
//       direction--;
//     }
//   }
//   
//   // direction > 0 sugere tendência crescente, < 0 decrescente
//   bool monotonic = false;
//   int iter = 0;
//   
//   while (!monotonic && counts.size() > static_cast<size_t>(min_bins) && (iterations_run + iter) < max_iterations) {
//     monotonic = true;
//     for (size_t b = 1; b < woe_values.size(); ++b) {
//       double diff = woe_values[b] - woe_values[b - 1];
//       if ((direction >= 0 && diff < 0) || (direction < 0 && diff > 0)) {
//         // Merge bins b-1 e b
//         int b1 = static_cast<int>(b - 1);
//         int b2 = static_cast<int>(b);
//         
//         counts[b1] += counts[b2];
//         count_pos[b1] += count_pos[b2];
//         count_neg[b1] += count_neg[b2];
//         
//         counts.erase(counts.begin() + b2);
//         count_pos.erase(count_pos.begin() + b2);
//         count_neg.erase(count_neg.begin() + b2);
//         bin_edges.erase(bin_edges.begin() + b2);
//         woe_values.erase(woe_values.begin() + b2);
//         iv_values.erase(iv_values.begin() + b2);
//         
//         double total_pos = std::accumulate(count_pos.begin(), count_pos.end(), 0.0);
//         double total_neg = std::accumulate(count_neg.begin(), count_neg.end(), 0.0);
//         
//         woe_values[b1] = calculateWOE(count_pos[b1], count_neg[b1], total_pos, total_neg);
//         iv_values[b1] = calculateIV(woe_values[b1], count_pos[b1], count_neg[b1], total_pos, total_neg);
//         
//         monotonic = false;
//         break;
//       }
//     }
//     iter++;
//   }
//   
//   iterations_run += iter;
// }
// 
// // Merge bins
// void OptimalBinningNumericalLDB::merge_bins() {
//   size_t n = feature.size();
//   double min_bin_count = bin_cutoff * static_cast<double>(n);
//   
//   bool bins_merged = true;
//   
//   // Merge bins com frequência baixa
//   while (bins_merged && counts.size() > static_cast<size_t>(min_bins) && iterations_run < max_iterations) {
//     bins_merged = false;
//     
//     for (size_t b = 0; b < counts.size(); ++b) {
//       if (counts[b] < min_bin_count && counts.size() > static_cast<size_t>(min_bins)) {
//         bins_merged = true;
//         size_t merge_with;
//         if (b == 0) {
//           merge_with = b + 1;
//         } else if (b == counts.size() - 1) {
//           merge_with = b - 1;
//         } else {
//           // Merge com vizinho com menor contagem
//           merge_with = (counts[b - 1] <= counts[b + 1]) ? (b - 1) : (b + 1);
//         }
//         
//         counts[merge_with] += counts[b];
//         count_pos[merge_with] += count_pos[b];
//         count_neg[merge_with] += count_neg[b];
//         
//         counts.erase(counts.begin() + b);
//         count_pos.erase(count_pos.begin() + b);
//         count_neg.erase(count_neg.begin() + b);
//         bin_edges.erase(bin_edges.begin() + b);
//         woe_values.erase(woe_values.begin() + b);
//         iv_values.erase(iv_values.begin() + b);
//         
//         double total_pos = std::accumulate(count_pos.begin(), count_pos.end(), 0.0);
//         double total_neg = std::accumulate(count_neg.begin(), count_neg.end(), 0.0);
//         
//         woe_values[merge_with] = calculateWOE(count_pos[merge_with], count_neg[merge_with], total_pos, total_neg);
//         iv_values[merge_with] = calculateIV(woe_values[merge_with], count_pos[merge_with], count_neg[merge_with], total_pos, total_neg);
//         
//         break;
//       }
//     }
//     iterations_run++;
//   }
//   
//   // Garante que número de bins não exceda max_bins
//   while (counts.size() > static_cast<size_t>(max_bins) && iterations_run < max_iterations) {
//     // Achar bin com menor IV para mesclar
//     size_t min_iv_idx = 0;
//     double min_iv = iv_values[0];
//     for (size_t b = 1; b < iv_values.size(); ++b) {
//       if (iv_values[b] < min_iv) {
//         min_iv = iv_values[b];
//         min_iv_idx = b;
//       }
//     }
//     
//     size_t merge_with = (min_iv_idx == 0) ? min_iv_idx + 1 : min_iv_idx - 1;
//     
//     counts[merge_with] += counts[min_iv_idx];
//     count_pos[merge_with] += count_pos[min_iv_idx];
//     count_neg[merge_with] += count_neg[min_iv_idx];
//     
//     counts.erase(counts.begin() + min_iv_idx);
//     count_pos.erase(count_pos.begin() + min_iv_idx);
//     count_neg.erase(count_neg.begin() + min_iv_idx);
//     bin_edges.erase(bin_edges.begin() + min_iv_idx);
//     woe_values.erase(woe_values.begin() + min_iv_idx);
//     iv_values.erase(iv_values.begin() + min_iv_idx);
//     
//     double total_pos = std::accumulate(count_pos.begin(), count_pos.end(), 0.0);
//     double total_neg = std::accumulate(count_neg.begin(), count_neg.end(), 0.0);
//     woe_values[merge_with] = calculateWOE(count_pos[merge_with], count_neg[merge_with], total_pos, total_neg);
//     iv_values[merge_with] = calculateIV(woe_values[merge_with], count_pos[merge_with], count_neg[merge_with], total_pos, total_neg);
//     
//     iterations_run++;
//   }
//   
//   // Calcula IV total
//   total_iv = std::accumulate(iv_values.begin(), iv_values.end(), 0.0);
// }
// 
// // Cria labels dos bins
// void OptimalBinningNumericalLDB::create_bin_labels() {
//   bin_labels.clear();
//   size_t num_bins = bin_edges.size() - 1;
//   bin_labels.reserve(num_bins);
//   
//   for (size_t b = 0; b < num_bins; ++b) {
//     std::ostringstream oss;
//     oss.precision(6);
//     oss << std::fixed;
//     oss << "(";
//     if (bin_edges[b] == -std::numeric_limits<double>::infinity()) {
//       oss << "-Inf";
//     } else {
//       oss << bin_edges[b];
//     }
//     oss << ";";
//     if (bin_edges[b + 1] == std::numeric_limits<double>::infinity()) {
//       oss << "+Inf";
//     } else {
//       oss << bin_edges[b + 1];
//     }
//     oss << "]";
//     bin_labels.emplace_back(oss.str());
//   }
// }
// 
// // Retorna resultados
// Rcpp::List OptimalBinningNumericalLDB::transform() {
//   std::vector<double> cutpoints;
//   if (bin_edges.size() > 2) {
//     cutpoints.assign(bin_edges.begin() + 1, bin_edges.end() - 1);
//   }
//   
//   Rcpp::NumericVector ids(bin_labels.size());
//   for(int i = 0; i < bin_labels.size(); i++) {
//     ids[i] = i + 1;
//   }
//   
//   return Rcpp::List::create(
//     Rcpp::Named("id") = ids,
//     Rcpp::Named("bins") = bin_labels,
//     Rcpp::Named("woe") = woe_values,
//     Rcpp::Named("iv") = iv_values,
//     Rcpp::Named("count") = counts,
//     Rcpp::Named("count_pos") = count_pos,
//     Rcpp::Named("count_neg") = count_neg,
//     Rcpp::Named("cutpoints") = cutpoints,
//     Rcpp::Named("converged") = converged,
//     Rcpp::Named("iterations") = iterations_run
//   );
// }
// 
// //' @title Optimal Binning for Numerical Variables using Local Density Binning (LDB)
// //'
// //' @description
// //' Implements the Local Density Binning (LDB) algorithm for optimal binning of numerical variables. 
// //' The method adjusts binning to maximize predictive power while maintaining monotonicity in Weight of Evidence (WoE),
// //' handling rare bins, and ensuring numerical stability.
// //'
// //' @details
// //' ### Key Features:
// //' - **Weight of Evidence (WoE)**: Ensures interpretability by calculating the WoE for each bin, useful for logistic regression and risk models.
// //' - **Information Value (IV)**: Evaluates the predictive power of the binned feature.
// //' - **Monotonicity**: Ensures WoE values are either strictly increasing or decreasing across bins.
// //' - **Rare Bin Handling**: Merges bins with low frequencies to maintain statistical reliability.
// //' - **Numerical Stability**: Prevents log(0) issues through smoothing (Laplace adjustment).
// //' - **Dynamic Adjustments**: Supports constraints on minimum and maximum bins, convergence thresholds, and iteration limits.
// //'
// //' ### Mathematical Framework:
// //' - **Weight of Evidence (WoE)**: For a bin \( i \):
// //'   \deqn{WoE_i = \ln\left(\frac{\text{Distribution of positives}_i}{\text{Distribution of negatives}_i}\right)}
// //'
// //' - **Information Value (IV)**: Aggregates predictive power across all bins:
// //'   \deqn{IV = \sum_{i=1}^{N} (\text{Distribution of positives}_i - \text{Distribution of negatives}_i) \times WoE_i}
// //'
// //' ### Algorithm Steps:
// //' 1. **Input Validation**: Ensures the feature and target vectors are valid and properly formatted.
// //' 2. **Pre-Binning**: Divides the feature into pre-bins based on quantile cuts or unique values.
// //' 3. **Rare Bin Merging**: Combines bins with frequencies below `bin_cutoff` to maintain statistical stability.
// //' 4. **WoE and IV Calculation**: Computes the WoE and IV values for each bin based on the target distribution.
// //' 5. **Monotonicity Enforcement**: Adjusts bins to ensure WoE values are monotonic (either increasing or decreasing).
// //' 6. **Bin Optimization**: Iteratively merges bins to respect constraints on `min_bins` and `max_bins`.
// //' 7. **Result Validation**: Ensures bins cover the entire range of the feature without overlap and adhere to constraints.
// //'
// //' ### Parameters:
// //' - `min_bins`: Minimum number of bins to be created (default: 3).
// //' - `max_bins`: Maximum number of bins allowed (default: 5).
// //' - `bin_cutoff`: Minimum proportion of total observations required for a bin to be retained as standalone (default: 0.05).
// //' - `max_n_prebins`: Maximum number of pre-bins before optimization (default: 20).
// //' - `convergence_threshold`: Threshold for determining convergence in terms of IV changes (default: 1e-6).
// //' - `max_iterations`: Maximum number of iterations allowed for optimization (default: 1000).
// //'
// //' @param target An integer binary vector (0 or 1) representing the response variable.
// //' @param feature A numeric vector representing the feature to be binned.
// //' @param min_bins Minimum number of bins to be created (default: 3).
// //' @param max_bins Maximum number of bins allowed (default: 5).
// //' @param bin_cutoff Minimum frequency proportion for retaining a bin (default: 0.05).
// //' @param max_n_prebins Maximum number of pre-bins before optimization (default: 20).
// //' @param convergence_threshold Convergence threshold for IV optimization (default: 1e-6).
// //' @param max_iterations Maximum number of iterations allowed for optimization (default: 1000).
// //'
// //' @return A list containing the following elements:
// //' \itemize{
// //'   \item `bins`: A vector of bin intervals in the format "[lower;upper)".
// //'   \item `woe`: A numeric vector of WoE values for each bin.
// //'   \item `iv`: A numeric vector of IV contributions for each bin.
// //'   \item `count`: An integer vector of the total number of observations per bin.
// //'   \item `count_pos`: An integer vector of the number of positive cases per bin.
// //'   \item `count_neg`: An integer vector of the number of negative cases per bin.
// //'   \item `cutpoints`: A numeric vector of the cutpoints defining the bin edges.
// //'   \item `converged`: A boolean indicating whether the algorithm converged.
// //'   \item `iterations`: An integer indicating the number of iterations executed.
// //' }
// //'
// //' @examples
// //' \dontrun{
// //' set.seed(123)
// //' target <- sample(0:1, 1000, replace = TRUE)
// //' feature <- rnorm(1000)
// //' result <- optimal_binning_numerical_ldb(target, feature, min_bins = 3, max_bins = 6)
// //' print(result$bins)
// //' print(result$woe)
// //' print(result$iv)
// //' }
// //'
// //' @export
// // [[Rcpp::export]]
// Rcpp::List optimal_binning_numerical_ldb(Rcpp::IntegerVector target,
//                                         Rcpp::NumericVector feature,
//                                         int min_bins = 3,
//                                         int max_bins = 5,
//                                         double bin_cutoff = 0.05,
//                                         int max_n_prebins = 20,
//                                         double convergence_threshold = 1e-6,
//                                         int max_iterations = 1000) {
//  std::vector<double> feature_vec = Rcpp::as<std::vector<double>>(feature);
//  std::vector<int> target_vec = Rcpp::as<std::vector<int>>(target);
//  
//  OptimalBinningNumericalLDB ob(min_bins, max_bins, bin_cutoff, max_n_prebins,
//                                convergence_threshold, max_iterations);
//  
//  ob.fit(feature_vec, target_vec);
//  
//  return ob.transform();
// }
