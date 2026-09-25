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
#include <cstdlib>

/**
 * @file OBN_LDB.cpp
 * @brief Implementation of Local Density Binning (LDB) algorithm for optimal binning of numerical variables
 * 
 * This implementation provides methods for supervised discretization of numerical variables
 * using the Local Density Binning approach, which preserves local density structure
 * while maximizing predictive power.
 */


// Include shared headers
#include "common/optimal_binning_common.h"
#include "common/bin_structures.h"

using namespace Rcpp;
using namespace OptimalBinning;


/**
 * Class for Optimal Binning using Local Density Binning (LDB)
 * 
 * LDB is a supervised discretization method that adapts bin boundaries to the local
 * density structure of the data while maximizing the predictive relationship with
 * a binary target variable. The algorithm balances statistical stability, predictive
 * power, and interpretability constraints.
 */
class OBN_LDB {
private:
  // Parameters
  int min_bins;
  int max_bins;
  double bin_cutoff;
  int max_n_prebins;
  // (convergence_threshold is accepted by the constructor for API
  // compatibility; no step of this algorithm uses it.)
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
  // Constant removed (uses shared definition)
  
  // Smoothing parameter for Laplace correction
  static constexpr double ALPHA = 0.5;
  
  /**
   * Compute initial bins based on density analysis
   * Uses a density-sensitive approach to create initial bin boundaries
   */
  void compute_prebins(const std::vector<double>& valid_feature);
  
  /**
   * Compute Weight of Evidence (WoE) and Information Value (IV) for each bin
   */
  void compute_woe_iv();
  
  /**
   * Enforce monotonicity in WoE values across bins
   */
  void enforce_monotonicity();

  /**
   * Merge adjacent bins that violate monotonicity_direction until none is
   * left, min_bins is reached, or max_iterations is exhausted
   */
  void merge_monotonicity_violations();
  
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

    // Recalculate WoE and IV for EVERY bin, not just the merged one. The
    // Laplace denominators depend on the bin count K, so after a merge the
    // untouched bins still carried the smoothing of the previous K: the
    // returned WoE mixed several K values (none of them the final one), and
    // the monotonicity test compared WoE values computed on different scales.
    refresh_woe_iv();
  }

  /**
   * Recompute WoE and IV of all bins from their counts, with the current
   * number of bins as the smoothing K.
   */
  void refresh_woe_iv() {
    double total_pos = std::accumulate(count_pos.begin(), count_pos.end(), 0.0);
    double total_neg = std::accumulate(count_neg.begin(), count_neg.end(), 0.0);
    for (size_t b = 0; b < counts.size(); ++b) {
      woe_values[b] = calculateWOE(count_pos[b], count_neg[b], total_pos, total_neg);
      iv_values[b] = calculateIV(woe_values[b], count_pos[b], count_neg[b],
                                 total_pos, total_neg);
    }
  }
  
  /**
   * Find the bin index for a given value using binary search
   * More efficient than linear search for large datasets
   * 
   * @param value The value to find the bin for
   * @return The index of the bin containing the value
   */
  size_t find_bin_index(double value) const {
    // Callers never pass NaN. Bins are right-closed (a, b], as the emitted
    // labels state, so a value sitting exactly on an edge belongs to the bin
    // BELOW it: the bin index is the position of the first upper edge
    // (edges[1..], the last of which is +Inf) that is >= value. -Inf lands in
    // bin 0 and +Inf in the last bin without special cases.
    auto first_upper = bin_edges.begin() + 1;
    auto it = std::lower_bound(first_upper, bin_edges.end(), value);
    return static_cast<size_t>(std::distance(first_upper, it));
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

    for (int t : target) {
      if (t != 0 && t != 1) {
        Rcpp::stop("Target must contain only binary values (0 or 1).");
      }
      unique_target.insert(t);
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
   * Midpoint of two finite values that does not overflow: (a + b) / 2 is
   * exact-halving of the rounded sum, so it is kept whenever it is finite
   * (bit-identical to the former expression) and a/2 + b/2 is used only when
   * a + b overflows, e.g. for values near +/-1e308.
   */
  static double safe_midpoint(double a, double b) {
    double m = (a + b) / 2.0;
    if (!std::isfinite(m)) {
      m = a / 2.0 + b / 2.0;
    }
    return m;
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
        double midpoint = safe_midpoint(unique_values[0], unique_values[1]);
        bin_edges.push_back(midpoint);
        bin_edges.push_back(std::numeric_limits<double>::infinity());
      }
      
      compute_woe_iv();
      create_bin_labels();
      converged = true;
      iterations_run = 0;
      // This path returned total_iv = 0 whatever the bins' IV was.
      total_iv = std::accumulate(iv_values.begin(), iv_values.end(), 0.0);
      return true;
    }
    
    // Case: <= min_bins unique values
    if (unique_count <= min_bins) {
      bin_edges.clear();
      bin_edges.push_back(-std::numeric_limits<double>::infinity());
      
      for (int i = 1; i < unique_count; ++i) {
        // Use midpoint between adjacent unique values
        double mid = safe_midpoint(unique_values[i - 1], unique_values[i]);
        bin_edges.push_back(mid);
      }
      
      bin_edges.push_back(std::numeric_limits<double>::infinity());
      
      compute_woe_iv();
      create_bin_labels();
      converged = true;
      iterations_run = 0;
      // This path returned total_iv = 0 whatever the bins' IV was.
      total_iv = std::accumulate(iv_values.begin(), iv_values.end(), 0.0);
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
    // Only called from compute_prebins(), with more distinct values than
    // min_bins (>= 2), so n >= 3 here.
    size_t n = sorted_values.size();
    std::vector<double> density(n, 0.0);

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
    
    const double nd = static_cast<double>(n);
    double mean = sum / nd;
    double variance = (sum_sq / nd) - (mean * mean);
    double std_dev = std::sqrt(std::max(variance, EPSILON));

    // Bandwidth using Silverman's rule
    double h = 0.9 * std::min(std_dev, iqr / 1.34) * std::pow(nd, -0.2);

    // Ensure minimum bandwidth
    h = std::max(h, range / 1000.0);

    // Values near +/-DBL_MAX overflow the range or the moments. The grid
    // estimator then divides by an infinite spacing and casts NaN to an
    // index, which is undefined behaviour. There is no usable density in
    // that case; a flat one makes the caller fall back to quantile cuts.
    if (!std::isfinite(range) || !std::isfinite(h) || !(h > 0.0)) {
      return density;
    }
    
    // Kernel density estimation, by linear binning on a grid rather than by
    // evaluating every point against every other one. The double loop this
    // replaces measured n^2.00 and was the entire cost of the algorithm.
    density = gaussian_kde_sorted(sorted_values, h);
    
    return density;
  }
  
public:
  /**
   * Constructor for OBN_LDB
   * 
   * @param min_bins Minimum number of bins
   * @param max_bins Maximum number of bins
   * @param bin_cutoff Minimum frequency fraction for each bin
   * @param max_n_prebins Maximum number of pre-bins before optimization
   * @param enforce_monotonic Whether to enforce monotonicity in WoE
   * @param convergence_threshold Convergence threshold
   * @param max_iterations Maximum iterations allowed
   */
  OBN_LDB(
    int min_bins_ = 3,
    int max_bins_ = 5,
    double bin_cutoff_ = 0.05,
    int max_n_prebins_ = 20,
    bool enforce_monotonic_ = true,
    double /* convergence_threshold (unused) */ = 1e-6,
    int max_iterations_ = 1000)
    : min_bins(min_bins_),
      max_bins(max_bins_),
      bin_cutoff(bin_cutoff_),
      max_n_prebins(max_n_prebins_),
      max_iterations(max_iterations_),
      enforce_monotonic(enforce_monotonic_),
      converged(true), 
      iterations_run(0),
      total_iv(0.0),
      monotonicity_direction(0) {}
  
  /**
   * Fit the binning model to data
   * 
   * @param feature_input Feature vector to be binned
   * @param target_input Binary target vector (0/1)
   */
  void fit(std::vector<double> feature_input, std::vector<int> target_input) {
    // Store and validate inputs (taken by value and moved: no extra copy)
    this->feature = std::move(feature_input);
    this->target = std::move(target_input);

    validate_inputs();

    // Finite feature values, sorted once. The pre-binning used to filter and
    // sort the whole feature a second time, and a matching target vector was
    // built here and never read.
    std::vector<double> sorted_feature;
    sorted_feature.reserve(feature.size());
    for (size_t i = 0; i < feature.size(); ++i) {
      if (!std::isnan(feature[i]) && !std::isinf(feature[i])) {
        sorted_feature.push_back(feature[i]);
      }
    }
    std::sort(sorted_feature.begin(), sorted_feature.end());

    // Extract unique sorted values
    std::vector<double> unique_feature(sorted_feature);
    unique_feature.erase(std::unique(unique_feature.begin(), unique_feature.end()),
                         unique_feature.end());

    // Handle special cases of few unique values
    if (handle_few_unique_values(unique_feature)) {
      return;
    }

    // General case: proceed with density-based binning
    compute_prebins(sorted_feature);
    compute_woe_iv();

    if (enforce_monotonic) {
      enforce_monotonicity();
    }

    merge_bins();

    // The frequency and max_bins merges can reintroduce a violation: with
    // Laplace smoothing the WoE of a pooled bin need not lie between the
    // WoE of its halves (two bins at (0+, 5-) give a pooled bin below both).
    // Re-run the violation merge, in the direction already chosen, so the
    // documented monotonic result holds (it still stops at min_bins).
    if (enforce_monotonic) {
      merge_monotonicity_violations();
    }

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
void OBN_LDB::compute_prebins(const std::vector<double>& valid_feature) {
  // valid_feature: the finite feature values, sorted ascending (by fit()).

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
      bin_edges.push_back(safe_midpoint(unique_values[i-1], unique_values[i]));
    }
  } else {
    // Density-based binning for larger datasets
    
    // Find local minima in density as potential cut points.
    //
    // The search runs over DISTINCT values (first occurrence of each run of
    // ties). Tied observations share one density value, so on the raw sorted
    // vector a strict minimum could only ever sit on a value that occurs
    // once: any feature with repeated values (integers, rounded amounts) got
    // no density-based cut at all and fell through to the min_bins quantile
    // cuts. Without ties the distinct values are the observations and this
    // is the former loop exactly.
    std::vector<size_t> first_idx;
    first_idx.reserve(n);
    for (size_t i = 0; i < n; ++i) {
      if (i == 0 || valid_feature[i] != valid_feature[i - 1]) {
        first_idx.push_back(i);
      }
    }

    std::vector<size_t> min_indices;
    for (size_t j = 1; j + 1 < first_idx.size(); ++j) {
      const double d = density[first_idx[j]];
      if (d < density[first_idx[j - 1]] && d < density[first_idx[j + 1]]) {
        min_indices.push_back(first_idx[j]);
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
      const long min_gap = static_cast<long>(n / static_cast<size_t>(max_n_prebins * 2));
      for (size_t i = 1; i < min_indices.size() && selected_indices.size() < static_cast<size_t>(n_cuts); ++i) {
        bool too_close = false;
        for (size_t sel_idx : selected_indices) {
          if (std::labs(static_cast<long>(min_indices[i]) - static_cast<long>(sel_idx)) < min_gap) {
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
        size_t idx = static_cast<size_t>(std::floor(q * static_cast<double>(n)));
        if (idx >= n) idx = n - 1;
        bin_edges.push_back(valid_feature[idx]);
      }
      
      // Ensure bin edges are unique and sorted
      std::sort(bin_edges.begin(), bin_edges.end());
      bin_edges.erase(std::unique(bin_edges.begin(), bin_edges.end()), bin_edges.end());

      // A quantile that lands on the largest value (heavy ties at the top)
      // would open an empty (max, +Inf] bin, which the rare-bin merge cannot
      // remove once min_bins is reached; such a cut separates nothing.
      bool dropped = false;
      while (bin_edges.size() > 1 && bin_edges.back() >= valid_feature.back()) {
        bin_edges.pop_back();
        dropped = true;
      }

      // If that left too few bins, the observation quantiles sat inside the
      // run of ties at the top: take the missing cuts from the distinct
      // values below the maximum instead, spread evenly over them.
      if (dropped && bin_edges.size() < static_cast<size_t>(min_bins)) {
        std::vector<double> distinct;
        for (size_t j = 0; j < first_idx.size(); ++j) {
          distinct.push_back(valid_feature[first_idx[j]]);
        }
        distinct.pop_back();  // the maximum itself is never a cut
        const int m = min_bins - static_cast<int>(bin_edges.size());
        for (int j = 1; j <= m; ++j) {
          const size_t idx = std::min(
            static_cast<size_t>(j) * distinct.size() / static_cast<size_t>(m + 1),
            distinct.size() - 1);
          bin_edges.push_back(distinct[idx]);
        }
        std::sort(bin_edges.begin(), bin_edges.end());
        bin_edges.erase(std::unique(bin_edges.begin(), bin_edges.end()), bin_edges.end());
      }
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
void OBN_LDB::compute_woe_iv() {
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
void OBN_LDB::enforce_monotonicity() {
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

  merge_monotonicity_violations();
}

/**
 * Merge adjacent bins violating the chosen monotonicity direction
 */
void OBN_LDB::merge_monotonicity_violations() {
  if (woe_values.size() <= 1 || monotonicity_direction == 0) {
    return;
  }

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
void OBN_LDB::merge_bins() {
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
  
  // Step 2: Ensure number of bins doesn't exceed max_bins.
  //
  // max_bins is a hard post-condition: each pass removes one bin, so the loop
  // is finite without an iteration cap. It used to stop at max_iterations as
  // well, which returned more than max_bins bins whenever the cap was small;
  // a pass beyond the cap is still counted, so `converged` reports it.
  // Class totals are invariant under merging and are summed once per pass,
  // not once per candidate pair.
  while (counts.size() > static_cast<size_t>(max_bins)) {
    // Find pair of adjacent bins with smallest IV loss when merged
    size_t merge_idx1 = 0;
    size_t merge_idx2 = 1;
    double min_iv_loss = std::numeric_limits<double>::max();

    const double total_pos = std::accumulate(count_pos.begin(), count_pos.end(), 0.0);
    const double total_neg = std::accumulate(count_neg.begin(), count_neg.end(), 0.0);

    for (size_t b = 0; b < counts.size() - 1; ++b) {
      // Calculate combined IV for merged bin
      int combined_pos = count_pos[b] + count_pos[b + 1];
      int combined_neg = count_neg[b] + count_neg[b + 1];

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
void OBN_LDB::create_bin_labels() {
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
    OBN_LDB binner(
        min_bins, max_bins, 
        bin_cutoff, max_n_prebins,
        enforce_monotonic,
        convergence_threshold, max_iterations);
    
    binner.fit(std::move(feature_vec), std::move(target_vec));
    
    return binner.transform();
  } catch(std::exception &e) {
    forward_exception_to_r(e);
  } catch(...) {
    Rcpp::stop("Unknown C++ exception in optimal_binning_numerical_ldb");
  }
  
  // Should never reach here
  return R_NilValue;
}
