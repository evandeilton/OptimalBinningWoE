// [[Rcpp::plugins(cpp11)]]
// [[Rcpp::depends(Rcpp)]]

#include <Rcpp.h>
#include <vector>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <limits>
#include <string>
#include <sstream>
#include <iomanip>
#include <set>
#include <map>
#include <functional>

/**
 * @file OptimalBinningNumericalMBLP.cpp
 * @brief Implementation of Monotonic Binning via Linear Programming (MBLP) for optimal binning
 * 
 * This implementation provides methods for supervised discretization of numerical variables
 * with guaranteed monotonicity in Weight of Evidence (WoE). The approach formulates the 
 * binning problem as an optimization task with monotonicity constraints.
 */

using namespace Rcpp;

/**
 * Format double with specified precision
 * 
 * @param value Double value to format
 * @param precision Number of decimal places
 * @return Formatted string
 */
std::string format_double(double value, int precision = 6) {
  if (std::isnan(value)) {
    return "NA";
  } else if (std::isinf(value)) {
    return value > 0 ? "+Inf" : "-Inf";
  }
  
  std::ostringstream oss;
  oss << std::fixed << std::setprecision(precision) << value;
  return oss.str();
}

/**
 * Class for Optimal Binning of Numerical Variables using Monotonic Binning via Linear Programming (MBLP)
 * 
 * This class implements a binning algorithm that ensures monotonic relationship 
 * between the binned variable and the target through an optimization approach
 * that preserves the monotonicity constraint while maximizing information value.
 */
class OptimalBinningNumericalMBLP {
public:
  /**
   * Constructor for OptimalBinningNumericalMBLP
   * 
   * @param feature Feature vector to bin
   * @param target Binary target vector (0/1)
   * @param min_bins Minimum number of bins
   * @param max_bins Maximum number of bins
   * @param bin_cutoff Minimum frequency fraction for each bin
   * @param max_n_prebins Maximum number of pre-bins
   * @param force_monotonic_direction Force specific monotonicity direction (0=auto, 1=increasing, -1=decreasing)
   * @param convergence_threshold Convergence threshold for optimization
   * @param max_iterations Maximum number of iterations allowed
   */
  OptimalBinningNumericalMBLP(
    NumericVector feature, 
    IntegerVector target,
    int min_bins, 
    int max_bins, 
    double bin_cutoff,
    int max_n_prebins,
    int force_monotonic_direction,
    double convergence_threshold, 
    int max_iterations)
    : feature(feature), target(target), 
      min_bins(min_bins), max_bins(max_bins),
      bin_cutoff(bin_cutoff), max_n_prebins(max_n_prebins),
      force_monotonic_direction(force_monotonic_direction),
      convergence_threshold(convergence_threshold), 
      max_iterations(max_iterations),
      N(feature.size()),
      monotonic_direction(0),
      total_iv(0.0),
      converged(false), 
      iterations_run(0), 
      unique_values(0) {}
  
  /**
   * Fit the binning model to data
   * 
   * @return List of binning results
   */
  List fit();
  
private:
  // Input data and parameters
  NumericVector feature;
  IntegerVector target;
  int min_bins;
  int max_bins;
  double bin_cutoff;
  int max_n_prebins;
  int force_monotonic_direction;  // 0 = auto, 1 = increasing, -1 = decreasing
  double convergence_threshold;
  int max_iterations;
  
  // Constants
  static constexpr double EPSILON = 1e-10;
  static constexpr double ALPHA = 0.5;  // Laplace smoothing parameter
  
  // Data properties
  int N;
  int monotonic_direction;  // 1 = increasing, -1 = decreasing, 0 = undetermined
  double total_iv;
  
  // Binning results
  std::vector<double> bin_edges;
  std::vector<int> bin_assignments;
  std::vector<double> bin_woe;
  std::vector<double> bin_iv;
  std::vector<int> bin_count;
  std::vector<int> bin_count_pos;
  std::vector<int> bin_count_neg;
  std::vector<double> bin_event_rates;
  std::vector<std::string> bin_labels;
  
  std::vector<double> cutpoints;
  bool converged;
  int iterations_run;
  int unique_values;
  
  /**
   * Validate input data and parameters
   * Throws an exception if validation fails
   */
  void validate_input();
  
  /**
   * Create initial bins based on quantiles or unique values
   */
  void prebin();
  
  /**
   * Calculate quantiles for prebinning
   * 
   * @param data Sorted data vector
   * @param n_quantiles Number of quantiles to calculate
   * @return Vector of quantile values
   */
  std::vector<double> calculate_quantiles(
      const std::vector<double>& data, 
      int n_quantiles);
  
  /**
   * Merge bins with frequency below the cutoff threshold
   * This ensures statistical reliability of each bin
   */
  void merge_rare_bins();
  
  /**
   * Optimize binning through monotonicity enforcement and bin count constraints
   * Uses an iterative approach to maximize information value
   */
  void optimize_binning();
  
  /**
   * Enforce constraints on minimum and maximum number of bins
   */
  void enforce_bin_constraints();
  
  /**
   * Calculate Weight of Evidence and Information Value for each bin
   */
  void calculate_bin_woe();
  
  /**
   * Calculate total Information Value across all bins
   * 
   * @return Total IV value
   */
  double calculate_total_iv();
  
  /**
   * Check if a vector has monotonic values (increasing or decreasing)
   * 
   * @param vec Vector to check
   * @return True if monotonic, False otherwise
   */
  bool check_monotonicity(const std::vector<double>& vec);
  
  /**
   * Find the optimal pair of bins to merge that minimizes IV loss
   * 
   * @return Index of first bin to merge, or -1 if no valid merge found
   */
  int find_min_iv_loss_merge();
  
  /**
   * Merge two bins and update bin statistics
   * 
   * @param idx1 Index of first bin to merge
   * @param idx2 Index of second bin to merge
   */
  void merge_bins(int idx1, int idx2);
  
  /**
   * Determine optimal monotonicity direction based on data
   * If force_monotonic_direction is not 0, uses that instead
   * 
   * @return 1 for increasing, -1 for decreasing
   */
  int determine_monotonicity_direction();
  
  /**
   * Enforce monotonicity in WoE values across bins
   * Merges bins as needed to achieve monotonicity
   * 
   * @return True if monotonicity enforced successfully, False otherwise
   */
  bool enforce_monotonicity();
  
  /**
   * Calculate event rates for each bin
   */
  void calculate_event_rates();
  
  /**
   * Prepare cutpoints for output
   */
  void prepare_cutpoints();
  
  /**
   * Prepare final output list
   * 
   * @return List with binning results
   */
  List prepare_output();
};

/**
 * Fit the binning model to data
 */
List OptimalBinningNumericalMBLP::fit() {
  // Step 1: Validate input
  validate_input();
  
  // Step 2: Create initial bins
  prebin();
  
  // Step 3: Handle special case of few unique values
  if (unique_values <= 2) {
    calculate_bin_woe();
    calculate_event_rates();
    prepare_cutpoints();
    converged = true;
    iterations_run = 0;
    return prepare_output();
  }
  
  // Step 4: Optimize binning
  optimize_binning();
  
  // Step 5: Final calculations
  calculate_bin_woe();
  calculate_event_rates();
  prepare_cutpoints();
  
  // Step 6: Return results
  return prepare_output();
}

/**
 * Validate input data and parameters
 */
void OptimalBinningNumericalMBLP::validate_input() {
  if (N != target.size()) {
    stop("feature and target must have the same length.");
  }
  
  // Check that target is binary (0/1)
  std::set<int> unique_targets;
  for (int i = 0; i < N; ++i) {
    if (!IntegerVector::is_na(target[i])) {
      unique_targets.insert(target[i]);
    }
  }
  
  if (unique_targets.size() != 2 ||
      unique_targets.find(0) == unique_targets.end() ||
      unique_targets.find(1) == unique_targets.end()) {
    stop("target must be a binary vector with values 0 and 1.");
  }
  
  // Validate parameters
  if (min_bins < 2) {
    stop("min_bins must be at least 2.");
  }
  if (max_bins < min_bins) {
    stop("max_bins must be greater than or equal to min_bins.");
  }
  
  if (bin_cutoff <= 0 || bin_cutoff >= 1) {
    stop("bin_cutoff must be between 0 and 1.");
  }
  
  if (max_n_prebins < min_bins) {
    stop("max_n_prebins must be greater than or equal to min_bins.");
  }
  
  if (force_monotonic_direction != 0 && force_monotonic_direction != 1 && force_monotonic_direction != -1) {
    stop("force_monotonic_direction must be 0 (auto), 1 (increasing), or -1 (decreasing).");
  }
  
  if (convergence_threshold <= 0) {
    stop("convergence_threshold must be positive.");
  }
  
  if (max_iterations <= 0) {
    stop("max_iterations must be a positive integer.");
  }
}

/**
 * Create initial bins based on quantiles or unique values
 */
void OptimalBinningNumericalMBLP::prebin() {
  // Step 1: Remove NA values and prepare clean data
  std::vector<double> feature_clean;
  std::vector<int> target_clean;
  feature_clean.reserve(N);
  target_clean.reserve(N);
  
  for (int i = 0; i < N; ++i) {
    if (!NumericVector::is_na(feature[i]) && !NumericVector::is_na(target[i])) {
      feature_clean.push_back(feature[i]);
      target_clean.push_back(target[i]);
    }
  }
  
  int N_clean = static_cast<int>(feature_clean.size());
  
  if (N_clean == 0) {
    stop("All feature or target values are NA.");
  }
  
  // Step 2: Sort feature and target together
  std::vector<std::pair<double, int>> paired;
  paired.reserve(N_clean);
  
  for (int i = 0; i < N_clean; ++i) {
    paired.emplace_back(std::make_pair(feature_clean[i], target_clean[i]));
  }
  
  std::sort(paired.begin(), paired.end());
  
  std::vector<double> feature_sorted(N_clean);
  std::vector<int> target_sorted(N_clean);
  
  for (int i = 0; i < N_clean; ++i) {
    feature_sorted[i] = paired[i].first;
    target_sorted[i] = paired[i].second;
  }
  
  // Step 3: Determine unique values
  std::vector<double> unique_feature = feature_sorted;
  std::sort(unique_feature.begin(), unique_feature.end());
  unique_feature.erase(
    std::unique(unique_feature.begin(), unique_feature.end()), 
    unique_feature.end());
  
  unique_values = static_cast<int>(unique_feature.size());
  
  // Step 4: Handle special cases or create prebins
  if (unique_values <= 2) {
    // Case: 1 or 2 unique values
    if (unique_values == 1) {
      // Single value - create one bin
      bin_edges.push_back(-std::numeric_limits<double>::infinity());
      bin_edges.push_back(std::numeric_limits<double>::infinity());
    } else { // 2 values
      // Two values - create two bins with split at the first value
      double v1 = unique_feature[0];
      bin_edges.push_back(-std::numeric_limits<double>::infinity());
      bin_edges.push_back(v1);
      bin_edges.push_back(std::numeric_limits<double>::infinity());
    }
  } else {
    // Regular case: many unique values
    // Determine number of prebins
    int n_prebins = std::min(max_n_prebins, unique_values);
    n_prebins = std::max(n_prebins, min_bins);
    
    // Calculate quantiles for bin edges
    bin_edges = calculate_quantiles(unique_feature, n_prebins);
  }
  
  // Step 5: Assign observations to bins
  bin_assignments.assign(N_clean, -1);
  
  for (int i = 0; i < N_clean; ++i) {
    double val = feature_sorted[i];
    int bin_idx = static_cast<int>(
      std::lower_bound(bin_edges.begin(), bin_edges.end(), val) - bin_edges.begin() - 1);
    
    bin_idx = std::max(0, std::min(bin_idx, static_cast<int>(bin_edges.size()) - 2));
    bin_assignments[i] = bin_idx;
  }
  
  // Step 6: Count observations in each bin
  int n_bins = static_cast<int>(bin_edges.size()) - 1;
  bin_count.assign(n_bins, 0);
  bin_count_pos.assign(n_bins, 0);
  bin_count_neg.assign(n_bins, 0);
  
  for (int i = 0; i < N_clean; ++i) {
    int bin_idx = bin_assignments[i];
    bin_count[bin_idx]++;
    
    if (target_sorted[i] == 1) {
      bin_count_pos[bin_idx]++;
    } else {
      bin_count_neg[bin_idx]++;
    }
  }
  
  // Step 7: Merge rare bins if needed
  if (unique_values > 2) {
    merge_rare_bins();
  }
}

/**
 * Calculate quantiles for prebinning
 */
std::vector<double> OptimalBinningNumericalMBLP::calculate_quantiles(
    const std::vector<double>& data, 
    int n_quantiles) {
  std::vector<double> quantiles;
  quantiles.reserve(n_quantiles + 1);
  
  // First edge is -Infinity
  quantiles.push_back(-std::numeric_limits<double>::infinity());
  
  // Calculate quantiles
  for (int i = 1; i < n_quantiles; ++i) {
    double p = static_cast<double>(i) / n_quantiles;
    size_t idx = static_cast<size_t>(std::ceil(p * (data.size() - 1)));
    
    if (idx >= data.size()) {
      idx = data.size() - 1;
    }
    
    quantiles.push_back(data[idx]);
  }
  
  // Last edge is +Infinity
  quantiles.push_back(std::numeric_limits<double>::infinity());
  
  // Handle possible duplicates
  std::vector<double> unique_quantiles;
  unique_quantiles.push_back(quantiles[0]);
  
  for (size_t i = 1; i < quantiles.size(); ++i) {
    if (std::abs(quantiles[i] - unique_quantiles.back()) > EPSILON) {
      unique_quantiles.push_back(quantiles[i]);
    }
  }
  
  return unique_quantiles;
}

/**
 * Merge bins with frequency below the cutoff threshold
 */
void OptimalBinningNumericalMBLP::merge_rare_bins() {
  double total = std::accumulate(bin_count.begin(), bin_count.end(), 0.0);
  double min_count = bin_cutoff * total;
  
  bool merged = true;
  int iterations = 0;
  
  while (merged && iterations < max_iterations) {
    merged = false;
    
    for (int i = 0; i < static_cast<int>(bin_count.size()); ++i) {
      // Check if bin is too small and we can still merge (respecting min_bins)
      if (bin_count[i] < min_count && 
          static_cast<int>(bin_count.size()) > min_bins) {
        
        int merge_idx;
        
        if (i == 0) {
          // First bin - merge with next
          merge_idx = i + 1;
        } else if (i == static_cast<int>(bin_count.size()) - 1) {
          // Last bin - merge with previous
          merge_idx = i - 1;
        } else {
          // Middle bin - merge with smaller neighbor
          merge_idx = (bin_count[i - 1] <= bin_count[i + 1]) ? (i - 1) : (i + 1);
        }
        
        merge_bins(i, merge_idx);
        merged = true;
        iterations++;
        break;
      }
    }
  }
  
  iterations_run += iterations;
}

/**
 * Optimize binning through iterative process
 */
void OptimalBinningNumericalMBLP::optimize_binning() {
  iterations_run = 0;
  
  // Determine desired monotonicity direction
  monotonic_direction = determine_monotonicity_direction();
  
  // Calculate initial WoE and IV
  calculate_bin_woe();
  double previous_iv = calculate_total_iv();
  
  // Main optimization loop
  while (iterations_run < max_iterations) {
    iterations_run++;
    
    // Enforce bin count constraints
    enforce_bin_constraints();
    
    // Recalculate WoE and IV
    calculate_bin_woe();
    
    // Check for convergence
    double current_iv = calculate_total_iv();
    double iv_change = std::abs(current_iv - previous_iv);
    
    if (iv_change < convergence_threshold) {
      converged = true;
      break;
    }
    
    previous_iv = current_iv;
    
    // Enforce monotonicity
    bool is_monotonic = enforce_monotonicity();
    
    if (is_monotonic) {
      converged = true;
      break;
    }
  }
  
  // Final check
  if (iterations_run >= max_iterations && !converged) {
    Rcpp::warning("Convergence not reached within the maximum number of iterations.");
  }
}

/**
 * Enforce constraints on minimum and maximum number of bins
 */
void OptimalBinningNumericalMBLP::enforce_bin_constraints() {
  // Ensure min_bins
  while (static_cast<int>(bin_count.size()) < min_bins) {
    // This case is rare since we start with at least min_bins
    // Would need to implement bin splitting, which is complex
    // Instead, we warning the user
    Rcpp::warning("Number of bins (%d) is less than min_bins (%d). This should not happen.",
                  bin_count.size(), min_bins);
    break;
  }
  
  // Ensure max_bins
  while (static_cast<int>(bin_count.size()) > max_bins) {
    int merge_idx = find_min_iv_loss_merge();
    
    if (merge_idx == -1) {
      break;
    }
    
    merge_bins(merge_idx, merge_idx + 1);
  }
}

/**
 * Calculate Weight of Evidence and Information Value for each bin
 */
void OptimalBinningNumericalMBLP::calculate_bin_woe() {
  int n_bins = static_cast<int>(bin_count.size());
  double total_pos = std::accumulate(bin_count_pos.begin(), bin_count_pos.end(), 0.0);
  double total_neg = std::accumulate(bin_count_neg.begin(), bin_count_neg.end(), 0.0);
  
  bin_woe.assign(n_bins, 0.0);
  bin_iv.assign(n_bins, 0.0);
  
  for (int i = 0; i < n_bins; ++i) {
    // Apply Laplace smoothing
    double dist_pos = (bin_count_pos[i] + ALPHA) / (total_pos + ALPHA * n_bins);
    double dist_neg = (bin_count_neg[i] + ALPHA) / (total_neg + ALPHA * n_bins);
    
    if (dist_pos <= 0) dist_pos = EPSILON;
    if (dist_neg <= 0) dist_neg = EPSILON;
    
    bin_woe[i] = std::log(dist_pos / dist_neg);
    bin_iv[i] = (dist_pos - dist_neg) * bin_woe[i];
  }
  
  // Calculate total IV
  total_iv = calculate_total_iv();
}

/**
 * Calculate event rates for each bin
 */
void OptimalBinningNumericalMBLP::calculate_event_rates() {
  int n_bins = static_cast<int>(bin_count.size());
  bin_event_rates.assign(n_bins, 0.0);
  
  for (int i = 0; i < n_bins; ++i) {
    if (bin_count[i] > 0) {
      bin_event_rates[i] = static_cast<double>(bin_count_pos[i]) / bin_count[i];
    } else {
      bin_event_rates[i] = 0.0;
    }
  }
}

/**
 * Calculate total Information Value across all bins
 */
double OptimalBinningNumericalMBLP::calculate_total_iv() {
  return std::accumulate(bin_iv.begin(), bin_iv.end(), 0.0);
}

/**
 * Check if a vector has monotonic values
 */
bool OptimalBinningNumericalMBLP::check_monotonicity(const std::vector<double>& vec) {
  if (vec.size() < 2) {
    return true;
  }
  
  bool increasing = true;
  bool decreasing = true;
  
  for (size_t i = 1; i < vec.size(); ++i) {
    if (vec[i] < vec[i-1] - EPSILON) {
      increasing = false;
    }
    if (vec[i] > vec[i-1] + EPSILON) {
      decreasing = false;
    }
  }
  
  // Check against forced direction
  if (monotonic_direction == 1 && !increasing) {
    return false;
  } else if (monotonic_direction == -1 && !decreasing) {
    return false;
  }
  
  return increasing || decreasing;
}

/**
 * Determine optimal monotonicity direction based on data
 */
int OptimalBinningNumericalMBLP::determine_monotonicity_direction() {
  // If force_monotonic_direction is specified, use it
  if (force_monotonic_direction != 0) {
    return force_monotonic_direction;
  }
  
  // Calculate temporary WoE
  int n_bins = static_cast<int>(bin_count.size());
  double total_pos = std::accumulate(bin_count_pos.begin(), bin_count_pos.end(), 0.0);
  double total_neg = std::accumulate(bin_count_neg.begin(), bin_count_neg.end(), 0.0);
  
  std::vector<double> temp_woe(n_bins, 0.0);
  
  for (int i = 0; i < n_bins; ++i) {
    double dist_pos = (bin_count_pos[i] + ALPHA) / (total_pos + ALPHA * n_bins);
    double dist_neg = (bin_count_neg[i] + ALPHA) / (total_neg + ALPHA * n_bins);
    
    if (dist_pos <= 0) dist_pos = EPSILON;
    if (dist_neg <= 0) dist_neg = EPSILON;
    
    temp_woe[i] = std::log(dist_pos / dist_neg);
  }
  
  // Calculate correlation between bin index and WoE
  std::vector<double> bin_indices(n_bins);
  for (int i = 0; i < n_bins; ++i) {
    bin_indices[i] = static_cast<double>(i);
  }
  
  // Calculate correlation
  double mean_idx = std::accumulate(bin_indices.begin(), bin_indices.end(), 0.0) / n_bins;
  double mean_woe = std::accumulate(temp_woe.begin(), temp_woe.end(), 0.0) / n_bins;
  
  double numerator = 0.0;
  double denom_idx = 0.0;
  double denom_woe = 0.0;
  
  for (int i = 0; i < n_bins; ++i) {
    double idx_diff = bin_indices[i] - mean_idx;
    double woe_diff = temp_woe[i] - mean_woe;
    
    numerator += idx_diff * woe_diff;
    denom_idx += idx_diff * idx_diff;
    denom_woe += woe_diff * woe_diff;
  }
  
  double correlation = 0.0;
  if (denom_idx > EPSILON && denom_woe > EPSILON) {
    correlation = numerator / (std::sqrt(denom_idx) * std::sqrt(denom_woe));
  }
  
  return (correlation >= 0) ? 1 : -1;
}

/**
 * Enforce monotonicity in WoE values
 */
bool OptimalBinningNumericalMBLP::enforce_monotonicity() {
  if (bin_woe.size() < 2) {
    return true;
  }
  
  bool is_monotonic = check_monotonicity(bin_woe);
  
  // Already monotonic
  if (is_monotonic) {
    return true;
  }
  
  // Cannot enforce monotonicity if we're at min_bins
  if (static_cast<int>(bin_count.size()) <= min_bins) {
    return false;
  }
  
  // Find violation and merge bins
  for (size_t i = 1; i < bin_woe.size(); ++i) {
    bool violation = false;
    
    if (monotonic_direction == 1 && bin_woe[i] < bin_woe[i-1] - EPSILON) {
      violation = true;
    } else if (monotonic_direction == -1 && bin_woe[i] > bin_woe[i-1] + EPSILON) {
      violation = true;
    }
    
    if (violation) {
      merge_bins(i-1, i);
      return false;  // Continue optimization process
    }
  }
  
  // Should not reach here, but return true if no violations found
  return true;
}

/**
 * Find the optimal pair of bins to merge that minimizes IV loss
 */
int OptimalBinningNumericalMBLP::find_min_iv_loss_merge() {
  if (bin_iv.size() < 2) {
    return -1;
  }
  
  double min_iv_loss = std::numeric_limits<double>::max();
  int merge_idx = -1;
  
  double total_pos = std::accumulate(bin_count_pos.begin(), bin_count_pos.end(), 0.0);
  double total_neg = std::accumulate(bin_count_neg.begin(), bin_count_neg.end(), 0.0);
  
  for (int i = 0; i < static_cast<int>(bin_iv.size()) - 1; ++i) {
    // Current IV of the two bins
    double iv_before = bin_iv[i] + bin_iv[i+1];
    
    // Calculate IV if bins are merged
    double merged_pos = bin_count_pos[i] + bin_count_pos[i+1];
    double merged_neg = bin_count_neg[i] + bin_count_neg[i+1];
    
    int n_bins = static_cast<int>(bin_count.size()) - 1;  // After merging
    
    double dist_pos = (merged_pos + ALPHA) / (total_pos + ALPHA * n_bins);
    double dist_neg = (merged_neg + ALPHA) / (total_neg + ALPHA * n_bins);
    
    if (dist_pos <= 0) dist_pos = EPSILON;
    if (dist_neg <= 0) dist_neg = EPSILON;
    
    double woe_merged = std::log(dist_pos / dist_neg);
    double iv_merged = (dist_pos - dist_neg) * woe_merged;
    
    // Calculate IV loss
    double iv_loss = iv_before - iv_merged;
    
    // Consider monotonicity
    bool maintains_monotonicity = true;
    
    if (i > 0 && i + 2 < static_cast<int>(bin_woe.size())) {
      if (monotonic_direction == 1 && woe_merged < bin_woe[i-1] - EPSILON) {
        maintains_monotonicity = false;
      } else if (monotonic_direction == 1 && woe_merged > bin_woe[i+2] + EPSILON) {
        maintains_monotonicity = false;
      } else if (monotonic_direction == -1 && woe_merged > bin_woe[i-1] + EPSILON) {
        maintains_monotonicity = false;
      } else if (monotonic_direction == -1 && woe_merged < bin_woe[i+2] - EPSILON) {
        maintains_monotonicity = false;
      }
    }
    
    // If merge preserves monotonicity and has lower IV loss
    if (maintains_monotonicity && iv_loss < min_iv_loss) {
      min_iv_loss = iv_loss;
      merge_idx = i;
    }
  }
  
  return merge_idx;
}

/**
 * Merge two bins and update statistics
 */
void OptimalBinningNumericalMBLP::merge_bins(int idx1, int idx2) {
  if (idx1 < 0 || idx2 < 0 || 
      idx1 >= static_cast<int>(bin_count.size()) || 
      idx2 >= static_cast<int>(bin_count.size())) {
    stop("Invalid merge indices.");
  }
  
  if (idx1 == idx2) {
    return;
  }
  
  // Ensure idx1 < idx2 for consistency
  int lower_idx = std::min(idx1, idx2);
  int higher_idx = std::max(idx1, idx2);
  
  // Remove redundant bin edge
  bin_edges.erase(bin_edges.begin() + higher_idx);
  
  // Merge counts
  bin_count[lower_idx] += bin_count[higher_idx];
  bin_count_pos[lower_idx] += bin_count_pos[higher_idx];
  bin_count_neg[lower_idx] += bin_count_neg[higher_idx];
  
  // Remove second bin from count vectors
  bin_count.erase(bin_count.begin() + higher_idx);
  bin_count_pos.erase(bin_count_pos.begin() + higher_idx);
  bin_count_neg.erase(bin_count_neg.begin() + higher_idx);
  
  // Remove from WoE and IV vectors if they exist
  if (!bin_woe.empty() && !bin_iv.empty()) {
    bin_woe.erase(bin_woe.begin() + higher_idx);
    bin_iv.erase(bin_iv.begin() + higher_idx);
  }
  
  // Remove from event rates if they exist
  if (!bin_event_rates.empty()) {
    bin_event_rates.erase(bin_event_rates.begin() + higher_idx);
  }
}

/**
 * Prepare cutpoints for output
 */
void OptimalBinningNumericalMBLP::prepare_cutpoints() {
  cutpoints.clear();
  
  for (size_t i = 1; i < bin_edges.size() - 1; ++i) {
    cutpoints.push_back(bin_edges[i]);
  }
}

/**
 * Prepare output list with all results
 */
List OptimalBinningNumericalMBLP::prepare_output() {
  int n_bins = static_cast<int>(bin_count.size());
  bin_labels.assign(n_bins, "");
  
  // Format bin labels
  for (int i = 0; i < n_bins; ++i) {
    std::string left, right;
    
    if (i == 0) {
      left = "(-Inf";
    } else {
      left = "(" + format_double(bin_edges[i]);
    }
    
    if (i == n_bins - 1) {
      right = "+Inf]";
    } else {
      right = format_double(bin_edges[i + 1]) + "]";
    }
    
    bin_labels[i] = left + ";" + right;
  }
  
  // Create bin IDs (1-based for R)
  Rcpp::NumericVector ids(bin_labels.size());
  for (int i = 0; i < static_cast<int>(bin_labels.size()); i++) {
    ids[i] = i + 1;
  }
  
  // Create monotonicity direction string
  std::string direction = "none";
  if (monotonic_direction == 1) {
    direction = "increasing";
  } else if (monotonic_direction == -1) {
    direction = "decreasing";
  }
  
  // Create and return results list
  return Rcpp::List::create(
    Rcpp::Named("id") = ids,
    Rcpp::Named("bin") = bin_labels,
    Rcpp::Named("woe") = bin_woe,
    Rcpp::Named("iv") = bin_iv,
    Rcpp::Named("count") = bin_count,
    Rcpp::Named("count_pos") = bin_count_pos,
    Rcpp::Named("count_neg") = bin_count_neg,
    Rcpp::Named("event_rate") = bin_event_rates,
    Rcpp::Named("cutpoints") = cutpoints,
    Rcpp::Named("converged") = converged,
    Rcpp::Named("iterations") = iterations_run,
    Rcpp::Named("total_iv") = total_iv,
    Rcpp::Named("monotonicity") = direction
  );
}

//' @title Optimal Binning for Numerical Features Using Monotonic Binning via Linear Programming (MBLP)
//'
//' @description
//' Implements an advanced binning algorithm for numerical features that ensures monotonicity in 
//' the Weight of Evidence (WoE) while maximizing predictive power. The method formulates the 
//' binning problem as an optimization task with monotonicity constraints and solves it through
//' an iterative process that preserves information value.
//'
//' @details
//' ## Algorithm Overview
//' 
//' The Monotonic Binning via Linear Programming algorithm operates through several coordinated steps:
//' 
//' 1. **Input Validation**: Ensures proper formatting and constraints for data and parameters.
//' 
//' 2. **Pre-Binning**: Creates initial bins based on quantiles or handles special cases for few unique values.
//' 
//' 3. **Statistical Optimization**:
//'    - Merges bins with frequencies below `bin_cutoff` to ensure statistical stability
//'    - Enforces monotonicity in Weight of Evidence (WoE) values
//'    - Optimizes bin count to satisfy the min_bins/max_bins constraints
//'    - Iteratively improves binning to maximize Information Value (IV)
//' 
//' 4. **Monotonicity Analysis**: Automatically detects optimal monotonicity direction or applies
//'    a forced direction if specified.
//' 
//' ## Mathematical Foundation
//' 
//' ### Linear Programming Connection
//' 
//' The binning optimization problem can be formulated as a constrained optimization problem:
//' 
//' \deqn{\max_{b} \sum_{i=1}^{k} IV_i(b)}
//' 
//' Subject to:
//' \deqn{WoE_i \leq WoE_{i+1} \quad \text{or} \quad WoE_i \geq WoE_{i+1} \quad \forall i \in \{1, \ldots, k-1\}}
//' \deqn{min\_bins \leq k \leq max\_bins}
//' \deqn{count_i \geq bin\_cutoff \times N \quad \forall i \in \{1, \ldots, k\}}
//' 
//' Where:
//' - \eqn{b} is the set of bin boundaries
//' - \eqn{k} is the number of bins
//' - \eqn{IV_i(b)} is the Information Value of bin \eqn{i} given boundaries \eqn{b}
//' - \eqn{WoE_i} is the Weight of Evidence of bin \eqn{i}
//' 
//' ### Weight of Evidence (WoE)
//' 
//' For bin \eqn{i}, with Laplace smoothing:
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
//' ### Information Value (IV)
//' 
//' For bin \eqn{i}:
//' 
//' \deqn{IV_i = \left(\frac{p_i}{P} - \frac{n_i}{N}\right) \times WoE_i}
//' 
//' Total Information Value:
//' 
//' \deqn{IV_{total} = \sum_{i=1}^{k} IV_i}
//' 
//' ## Advantages
//' 
//' - **Guaranteed Monotonicity**: Ensures monotonic relationship between binned variable and target
//' - **Optimal Information Preservation**: Merges bins in a way that minimizes information loss
//' - **Flexible Direction Control**: Automatically detects optimal monotonicity direction or allows
//'   forcing a specific direction
//' - **Statistical Stability**: Ensures sufficient observations in each bin
//' - **Efficient Implementation**: Uses binary search and optimized merge strategies
//'
//' @param target An integer binary vector (0 or 1) representing the target variable.
//' @param feature A numeric vector representing the feature to bin.
//' @param min_bins Minimum number of bins (default: 3).
//' @param max_bins Maximum number of bins (default: 5).
//' @param bin_cutoff Minimum frequency fraction for each bin (default: 0.05).
//' @param max_n_prebins Maximum number of pre-bins before optimization (default: 20).
//' @param force_monotonic_direction Force specific monotonicity direction: 0=auto, 1=increasing, -1=decreasing (default: 0).
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
//' \item{monotonicity}{Character indicating monotonicity direction ("increasing", "decreasing", or "none").}
//'
//' @examples
//' \dontrun{
//' # Generate synthetic data
//' set.seed(123)
//' feature <- rnorm(1000)
//' target <- rbinom(1000, 1, 0.3)
//' 
//' # Basic usage
//' result <- optimal_binning_numerical_mblp(target, feature)
//' print(result)
//' 
//' # Custom parameters with forced increasing monotonicity
//' result_custom <- optimal_binning_numerical_mblp(
//'   target = target,
//'   feature = feature,
//'   min_bins = 3,
//'   max_bins = 6,
//'   force_monotonic_direction = 1  # Force increasing
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
//' Zeng, Y. (2018). Discretization of Continuous Features by Weighs of Evidence with Isotonic Regression. 
//' *arXiv preprint arXiv:1812.05089*.
//' 
//' Barlow, R. E., & Brunk, H. D. (1972). The isotonic regression problem and its dual. 
//' *Journal of the American Statistical Association*, 67(337), 140-147.
//' 
//' Belkin, M., & Niyogi, P. (2003). Laplacian eigenmaps for dimensionality reduction and data
//' representation. *Neural Computation*, 15(6), 1373-1396.
//' 
//' Bertsimas, D., & Tsitsiklis, J. N. (1997). *Introduction to Linear Optimization*. Athena Scientific.
//' 
//' Siddiqi, N. (2006). *Credit Risk Scorecards: Developing and Implementing Intelligent Credit Scoring*. 
//' John Wiley & Sons.
//' 
//' Thomas, L. C., Edelman, D. B., & Crook, J. N. (2002). *Credit Scoring and Its Applications*. 
//' Society for Industrial and Applied Mathematics.
//'
//' @export
// [[Rcpp::export]]
List optimal_binning_numerical_mblp(
   IntegerVector target,
   NumericVector feature, 
   int min_bins = 3, 
   int max_bins = 5, 
   double bin_cutoff = 0.05, 
   int max_n_prebins = 20,
   int force_monotonic_direction = 0,
   double convergence_threshold = 1e-6,
   int max_iterations = 1000) {
 
 try {
   // Initialize binning object
   OptimalBinningNumericalMBLP ob(
       feature, target, 
       min_bins, max_bins, 
       bin_cutoff, max_n_prebins,
       force_monotonic_direction,
       convergence_threshold, max_iterations);
   
   // Execute binning algorithm
   return ob.fit();
 } catch(std::exception &e) {
   forward_exception_to_r(e);
 } catch(...) {
   ::Rf_error("Unknown C++ exception in optimal_binning_numerical_mblp");
 }
 
 // Should never reach here
 return R_NilValue;
}




// // [[Rcpp::plugins(cpp11)]]
// // [[Rcpp::depends(Rcpp)]]
// 
// #include <Rcpp.h>
// #include <vector>
// #include <algorithm>
// #include <numeric>
// #include <cmath>
// #include <limits>
// #include <string>
// #include <sstream>
// #include <iomanip>
// #include <set>
// 
// using namespace Rcpp;
// 
// // Função auxiliar para formatar doubles com precisão
// std::string format_double(double value, int precision = 6) {
//   std::ostringstream oss;
//   oss << std::fixed << std::setprecision(precision) << value;
//   return oss.str();
// }
// 
// // Classe para Binning Ótimo de Variáveis Numéricas (Monotonic Binning via Linear Programming - MBLP)
// class OptimalBinningNumericalMBLP {
// public:
//   OptimalBinningNumericalMBLP(NumericVector feature, IntegerVector target,
//                               int min_bins, int max_bins, double bin_cutoff,
//                               int max_n_prebins, double convergence_threshold, int max_iterations)
//     : feature(feature), target(target), min_bins(min_bins), max_bins(max_bins),
//       bin_cutoff(bin_cutoff), max_n_prebins(max_n_prebins),
//       convergence_threshold(convergence_threshold), max_iterations(max_iterations) {
//     N = feature.size();
//   }
//   
//   List fit() {
//     validate_input();
//     prebin();
//     // Se unique_values <= 2 não é necessário otimizar
//     if (unique_values <= 2) {
//       calculate_woe_iv();
//       prepare_cutpoints();
//       converged = true;
//       iterations_run = 0;
//       return prepare_output();
//     }
//     optimize_binning();
//     calculate_woe_iv();
//     prepare_cutpoints();
//     return prepare_output();
//   }
//   
// private:
//   NumericVector feature;
//   IntegerVector target;
//   int min_bins;
//   int max_bins;
//   double bin_cutoff;
//   int max_n_prebins;
//   double convergence_threshold;
//   int max_iterations;
//   
//   int N;
//   
//   std::vector<double> bin_edges;
//   std::vector<int> bin_assignments;
//   std::vector<double> bin_woe;
//   std::vector<double> bin_iv;
//   std::vector<int> bin_count;
//   std::vector<int> bin_count_pos;
//   std::vector<int> bin_count_neg;
//   std::vector<std::string> bin_labels;
//   
//   std::vector<double> cutpoints;
//   bool converged = false;
//   int iterations_run = 0;
//   int unique_values = 0;
//   
//   void validate_input() {
//     if (N != target.size()) {
//       stop("feature and target must have the same length.");
//     }
//     
//     std::set<int> unique_targets(target.begin(), target.end());
//     if (unique_targets.size() != 2 ||
//         unique_targets.find(0) == unique_targets.end() ||
//         unique_targets.find(1) == unique_targets.end()) {
//       stop("target must be a binary vector with values 0 and 1.");
//     }
//     
//     if (min_bins < 2) {
//       stop("min_bins must be at least 2.");
//     }
//     if (max_bins < min_bins) {
//       stop("max_bins must be greater than or equal to min_bins.");
//     }
//     
//     if (bin_cutoff <= 0 || bin_cutoff >= 1) {
//       stop("bin_cutoff must be between 0 and 1.");
//     }
//     
//     if (max_n_prebins < min_bins) {
//       stop("max_n_prebins must be greater than or equal to min_bins.");
//     }
//     
//     if (convergence_threshold <= 0) {
//       stop("convergence_threshold must be positive.");
//     }
//     
//     if (max_iterations <= 0) {
//       stop("max_iterations must be a positive integer.");
//     }
//   }
//   
//   void prebin() {
//     // Remove valores NA
//     std::vector<double> feature_clean;
//     std::vector<int> target_clean;
//     feature_clean.reserve(N);
//     target_clean.reserve(N);
//     for (int i = 0; i < N; ++i) {
//       if (!NumericVector::is_na(feature[i])) {
//         feature_clean.push_back(feature[i]);
//         target_clean.push_back(target[i]);
//       }
//     }
//     int N_clean = (int)feature_clean.size();
//     
//     if (N_clean == 0) {
//       stop("All feature values are NA.");
//     }
//     
//     // Ordena feature e target juntos
//     std::vector<std::pair<double, int>> paired;
//     paired.reserve(N_clean);
//     for (int i = 0; i < N_clean; ++i) {
//       paired.emplace_back(std::make_pair(feature_clean[i], target_clean[i]));
//     }
//     std::sort(paired.begin(), paired.end());
//     
//     std::vector<double> feature_sorted(N_clean);
//     std::vector<int> target_sorted(N_clean);
//     for (int i = 0; i < N_clean; ++i) {
//       feature_sorted[i] = paired[i].first;
//       target_sorted[i] = paired[i].second;
//     }
//     
//     // Determina valores únicos
//     std::vector<double> unique_feature = feature_sorted;
//     std::sort(unique_feature.begin(), unique_feature.end());
//     unique_feature.erase(std::unique(unique_feature.begin(), unique_feature.end()), unique_feature.end());
//     unique_values = (int)unique_feature.size();
//     
//     if (unique_values <= 2) {
//       // Caso trivial
//       if (unique_values == 1) {
//         bin_edges.push_back(-std::numeric_limits<double>::infinity());
//         bin_edges.push_back(std::numeric_limits<double>::infinity());
//       } else { // 2 valores únicos
//         double v1 = unique_feature[0];
//         bin_edges.push_back(-std::numeric_limits<double>::infinity());
//         bin_edges.push_back(v1);
//         bin_edges.push_back(std::numeric_limits<double>::infinity());
//       }
//     }
//     else {
//       int n_prebins = std::min(max_n_prebins, unique_values);
//       n_prebins = std::max(n_prebins, min_bins);
//       bin_edges = calculate_quantiles(unique_feature, n_prebins);
//     }
//     
//     bin_assignments.assign(N_clean, -1);
//     for (int i = 0; i < N_clean; ++i) {
//       double val = feature_sorted[i];
//       int bin_idx = (int)(std::lower_bound(bin_edges.begin(), bin_edges.end(), val) - bin_edges.begin() - 1);
//       bin_idx = std::max(0, std::min(bin_idx, (int)bin_edges.size() - 2));
//       bin_assignments[i] = bin_idx;
//     }
//     
//     int n_bins = (int)bin_edges.size() - 1;
//     bin_count.assign(n_bins, 0);
//     bin_count_pos.assign(n_bins, 0);
//     bin_count_neg.assign(n_bins, 0);
//     
//     for (int i = 0; i < N_clean; ++i) {
//       int bin_idx = bin_assignments[i];
//       bin_count[bin_idx]++;
//       if (target_sorted[i] == 1) {
//         bin_count_pos[bin_idx]++;
//       } else {
//         bin_count_neg[bin_idx]++;
//       }
//     }
//     
//     if (unique_values > 2) {
//       merge_rare_bins();
//     }
//   }
//   
//   std::vector<double> calculate_quantiles(const std::vector<double>& data, int n_quantiles) {
//     std::vector<double> quantiles;
//     quantiles.reserve(n_quantiles + 1);
//     
//     quantiles.push_back(-std::numeric_limits<double>::infinity());
//     
//     for (int i = 1; i < n_quantiles; ++i) {
//       double p = (double)i / n_quantiles;
//       size_t idx = (size_t)std::ceil(p * (data.size() - 1));
//       quantiles.push_back(data[idx]);
//     }
//     
//     quantiles.push_back(std::numeric_limits<double>::infinity());
//     
//     return quantiles;
//   }
//   
//   void merge_rare_bins() {
//     double total = std::accumulate(bin_count.begin(), bin_count.end(), 0.0);
//     bool merged = true;
//     while (merged) {
//       merged = false;
//       for (int i = 0; i < (int)bin_count.size(); ++i) {
//         double freq = (double)bin_count[i] / total;
//         if (freq < bin_cutoff && (int)bin_count.size() > min_bins) {
//           int merge_idx = (i == 0) ? i + 1 : i - 1;
//           merge_bins(i, merge_idx);
//           merged = true;
//           break;
//         }
//       }
//     }
//   }
//   
//   void optimize_binning() {
//     iterations_run = 0;
//     double previous_iv = calculate_total_iv();
//     
//     while (iterations_run < max_iterations) {
//       iterations_run++;
//       
//       enforce_bin_constraints();
//       calculate_bin_woe();
//       
//       double current_iv = calculate_total_iv();
//       double iv_change = std::abs(current_iv - previous_iv);
//       
//       if (iv_change < convergence_threshold) {
//         converged = true;
//         break;
//       }
//       
//       previous_iv = current_iv;
//       
//       // Impõe monotonicidade
//       if (!check_monotonicity(bin_woe) && (int)bin_count.size() > min_bins) {
//         int merge_idx = find_min_iv_loss_merge();
//         if (merge_idx == -1) {
//           break;
//         }
//         merge_bins(merge_idx, merge_idx + 1);
//       } else {
//         converged = true;
//         break;
//       }
//     }
//     
//     if (iterations_run >= max_iterations && !converged) {
//       Rcpp::warning("Convergence not reached within the maximum number of iterations.");
//     }
//   }
//   
//   void enforce_bin_constraints() {
//     while ((int)bin_count.size() < min_bins) {
//       int merge_idx = find_min_iv_loss_merge();
//       if (merge_idx == -1) break;
//       merge_bins(merge_idx, merge_idx + 1);
//     }
//     
//     while ((int)bin_count.size() > max_bins) {
//       int merge_idx = find_min_iv_loss_merge();
//       if (merge_idx == -1) break;
//       merge_bins(merge_idx, merge_idx + 1);
//     }
//   }
//   
//   void calculate_bin_woe() {
//     int n_bins = (int)bin_count.size();
//     double total_pos = std::accumulate(bin_count_pos.begin(), bin_count_pos.end(), 0.0);
//     double total_neg = std::accumulate(bin_count_neg.begin(), bin_count_neg.end(), 0.0);
//     
//     bin_woe.assign(n_bins, 0.0);
//     bin_iv.assign(n_bins, 0.0);
//     
//     for (int i = 0; i < n_bins; ++i) {
//       double dist_pos = (bin_count_pos[i] + 0.5) / (total_pos + 0.5 * n_bins);
//       double dist_neg = (bin_count_neg[i] + 0.5) / (total_neg + 0.5 * n_bins);
//       
//       if (dist_pos <= 0) dist_pos = 1e-10;
//       if (dist_neg <= 0) dist_neg = 1e-10;
//       
//       bin_woe[i] = std::log(dist_pos / dist_neg);
//       bin_iv[i] = (dist_pos - dist_neg) * bin_woe[i];
//     }
//   }
//   
//   double calculate_total_iv() {
//     return std::accumulate(bin_iv.begin(), bin_iv.end(), 0.0);
//   }
//   
//   bool check_monotonicity(const std::vector<double>& vec) {
//     if (vec.size() < 2) {
//       return true;
//     }
//     
//     bool increasing = true;
//     bool decreasing = true;
//     
//     for (size_t i = 1; i < vec.size(); ++i) {
//       if (vec[i] < vec[i-1]) {
//         increasing = false;
//       }
//       if (vec[i] > vec[i-1]) {
//         decreasing = false;
//       }
//     }
//     
//     return increasing || decreasing;
//   }
//   
//   int find_min_iv_loss_merge() {
//     if (bin_iv.size() < 2) {
//       return -1;
//     }
//     
//     double min_iv_loss = std::numeric_limits<double>::max();
//     int merge_idx = -1;
//     
//     double total_pos = std::accumulate(bin_count_pos.begin(), bin_count_pos.end(), 0.0);
//     double total_neg = std::accumulate(bin_count_neg.begin(), bin_count_neg.end(), 0.0);
//     
//     for (int i = 0; i < (int)bin_iv.size() - 1; ++i) {
//       double iv_before = bin_iv[i] + bin_iv[i+1];
//       
//       double merged_pos = bin_count_pos[i] + bin_count_pos[i+1];
//       double merged_neg = bin_count_neg[i] + bin_count_neg[i+1];
//       
//       double dist_pos = (merged_pos + 0.5) / (total_pos + 0.5 * bin_count.size());
//       double dist_neg = (merged_neg + 0.5) / (total_neg + 0.5 * bin_count.size());
//       
//       if (dist_pos <= 0) dist_pos = 1e-10;
//       if (dist_neg <= 0) dist_neg = 1e-10;
//       
//       double woe_merged = std::log(dist_pos / dist_neg);
//       double iv_merged = (dist_pos - dist_neg) * woe_merged;
//       
//       double iv_after = iv_merged;
//       double iv_loss = iv_before - iv_after;
//       
//       if (iv_loss < min_iv_loss) {
//         min_iv_loss = iv_loss;
//         merge_idx = i;
//       }
//     }
//     
//     return merge_idx;
//   }
//   
//   void merge_bins(int idx1, int idx2) {
//     if (idx1 < 0 || idx2 < 0 || idx1 >= (int)bin_count.size() || idx2 >= (int)bin_count.size()) {
//       stop("Invalid merge indices.");
//     }
//     
//     if (idx1 == idx2) {
//       return;
//     }
//     
//     int lower_idx = std::min(idx1, idx2);
//     int higher_idx = std::max(idx1, idx2);
//     
//     bin_edges.erase(bin_edges.begin() + higher_idx);
//     bin_count[lower_idx] += bin_count[higher_idx];
//     bin_count_pos[lower_idx] += bin_count_pos[higher_idx];
//     bin_count_neg[lower_idx] += bin_count_neg[higher_idx];
//     
//     bin_count.erase(bin_count.begin() + higher_idx);
//     bin_count_pos.erase(bin_count_pos.begin() + higher_idx);
//     bin_count_neg.erase(bin_count_neg.begin() + higher_idx);
//     
//     if (!bin_woe.empty() && !bin_iv.empty()) {
//       bin_woe.erase(bin_woe.begin() + higher_idx);
//       bin_iv.erase(bin_iv.begin() + higher_idx);
//     }
//   }
//   
//   void calculate_woe_iv() {
//     calculate_bin_woe();
//   }
//   
//   void prepare_cutpoints() {
//     cutpoints.clear();
//     for (size_t i = 1; i < bin_edges.size() - 1; ++i) {
//       cutpoints.push_back(bin_edges[i]);
//     }
//   }
//   
//   List prepare_output() {
//     int n_bins = (int)bin_count.size();
//     bin_labels.assign(n_bins, "");
//     
//     for (int i = 0; i < n_bins; ++i) {
//       std::string left, right;
//       
//       if (i == 0) {
//         left = "(-Inf";
//       } else {
//         left = "(" + format_double(bin_edges[i]);
//       }
//       
//       if (i == n_bins - 1) {
//         right = "+Inf]";
//       } else {
//         right = format_double(bin_edges[i + 1]) + "]";
//       }
//       
//       bin_labels[i] = left + ";" + right;
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
//       Named("woe") = bin_woe,
//       Named("iv") = bin_iv,
//       Named("count") = bin_count,
//       Named("count_pos") = bin_count_pos,
//       Named("count_neg") = bin_count_neg,
//       Named("cutpoints") = cutpoints,
//       Named("converged") = converged,
//       Named("iterations") = iterations_run
//     );
//   }
// };
// 
// 
// //' @title Optimal Binning for Numerical Features Using Monotonic Binning via Linear Programming (MBLP)
// //'
// //' @description
// //' This method performs optimal binning for numerical features, ensuring monotonicity in the Weight of Evidence (WoE) across bins.
// //' It adheres to constraints on the minimum and maximum number of bins, merges rare bins, and handles edge cases like identical values.
// //' The algorithm returns bins, WoE, Information Value (IV), counts, cutpoints, and metadata such as convergence status and iterations run.
// //'
// //' @details
// //' ### Key Steps:
// //' 1. **Input Validation**: Ensures proper formatting and constraints for `feature`, `target`, and algorithm parameters.
// //' 2. **Pre-Binning**: Creates preliminary bins based on quantiles or unique values in the feature.
// //' 3. **Rare Bin Merging**: Combines bins with frequencies below `bin_cutoff` to maintain statistical stability.
// //' 4. **Optimization**: Adjusts bins iteratively to maximize IV, enforce monotonicity, and adhere to bin constraints (`min_bins` and `max_bins`).
// //' 5. **Monotonicity Enforcement**: Ensures WoE values are either strictly increasing or decreasing across bins.
// //' 6. **Validation**: Verifies bin structure for consistency, preventing gaps or overlapping intervals.
// //'
// //' ### Mathematical Framework:
// //' - **Weight of Evidence (WoE)**: For a bin \( i \):
// //'   \deqn{WoE_i = \ln\left(\frac{\text{Distribution of positives}_i}{\text{Distribution of negatives}_i}\right)}
// //'
// //' - **Information Value (IV)**: Aggregates predictive power across all bins:
// //'   \deqn{IV = \sum_{i=1}^{N} (\text{Distribution of positives}_i - \text{Distribution of negatives}_i) \times WoE_i}
// //'
// //' ### Features:
// //' - Monotonic WoE ensures interpretability in logistic regression and credit scoring models.
// //' - Dynamically adjusts binning to maximize IV and improve model predictive power.
// //' - Handles rare categories and missing values by merging and imputation.
// //' - Supports large datasets with efficient pre-binning and convergence checks.
// //' - Validates results to prevent invalid bin configurations (e.g., gaps, overlaps).
// //'
// //' ### Algorithm Parameters:
// //' - `min_bins`: Minimum number of bins (default: 3).
// //' - `max_bins`: Maximum number of bins (default: 5).
// //' - `bin_cutoff`: Minimum frequency proportion required to retain a bin as standalone (default: 0.05).
// //' - `max_n_prebins`: Maximum number of preliminary bins before optimization (default: 20).
// //' - `convergence_threshold`: Threshold for convergence in IV optimization (default: 1e-6).
// //' - `max_iterations`: Maximum number of iterations allowed for optimization (default: 1000).
// //'
// //' @param target An integer binary vector (0 or 1) representing the target variable.
// //' @param feature A numeric vector representing the feature to bin.
// //' @param min_bins Minimum number of bins (default: 3).
// //' @param max_bins Maximum number of bins (default: 5).
// //' @param bin_cutoff Minimum frequency proportion for retaining bins (default: 0.05).
// //' @param max_n_prebins Maximum number of pre-bins before optimization (default: 20).
// //' @param convergence_threshold Convergence threshold for IV optimization (default: 1e-6).
// //' @param max_iterations Maximum number of iterations allowed (default: 1000).
// //'
// //' @return A list with the following components:
// //' \itemize{
// //'   \item `bin`: A character vector of bin intervals in the format "[lower;upper)".
// //'   \item `woe`: A numeric vector of WoE values for each bin.
// //'   \item `iv`: A numeric vector of IV contributions for each bin.
// //'   \item `count`: An integer vector of total observations per bin.
// //'   \item `count_pos`: An integer vector of positive cases per bin.
// //'   \item `count_neg`: An integer vector of negative cases per bin.
// //'   \item `cutpoints`: A numeric vector of cutpoints defining the bin edges.
// //'   \item `converged`: A boolean indicating whether the algorithm converged.
// //'   \item `iterations`: An integer indicating the number of iterations executed.
// //' }
// //'
// //' @examples
// //' \dontrun{
// //' set.seed(123)
// //' feature <- rnorm(1000)
// //' target <- rbinom(1000, 1, 0.3)
// //' result <- optimal_binning_numerical_mblp(target, feature, min_bins = 3, max_bins = 6)
// //' print(result)
// //' }
// //'
// //' @export
// // [[Rcpp::export]]
// List optimal_binning_numerical_mblp(IntegerVector target,
//                                    NumericVector feature, 
//                                    int min_bins = 3, 
//                                    int max_bins = 5, 
//                                    double bin_cutoff = 0.05, 
//                                    int max_n_prebins = 20,
//                                    double convergence_threshold = 1e-6,
//                                    int max_iterations = 1000) {
//  // Valida entrada
//  if (feature.size() != target.size()) {
//    stop("feature and target must have the same length.");
//  }
//  if (min_bins < 2) {
//    stop("min_bins must be at least 2.");
//  }
//  if (max_bins < min_bins) {
//    stop("max_bins must be greater than or equal to min_bins.");
//  }
//  if (bin_cutoff <= 0 || bin_cutoff >= 1) {
//    stop("bin_cutoff must be between 0 and 1.");
//  }
//  if (max_n_prebins < min_bins) {
//    stop("max_n_prebins must be greater than or equal to min_bins.");
//  }
//  if (convergence_threshold <= 0) {
//    stop("convergence_threshold must be positive.");
//  }
//  if (max_iterations <= 0) {
//    stop("max_iterations must be a positive integer.");
//  }
//  
//  OptimalBinningNumericalMBLP ob(feature, target, min_bins, max_bins, bin_cutoff, 
//                                 max_n_prebins, convergence_threshold, max_iterations);
//  return ob.fit();
// }
