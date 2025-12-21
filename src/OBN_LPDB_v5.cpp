// [[Rcpp::plugins(cpp11)]]
// [[Rcpp::depends(Rcpp)]]

#include <Rcpp.h>
#include <vector>
#include <algorithm>
#include <cmath>
#include <map>
#include <string>
#include <sstream>
#include <numeric>
#include <limits>
#include <unordered_set>
#include <functional>

/**
 * @file OBN_LPDB.cpp
 * @brief Implementation of Local Polynomial Density Binning (LPDB) algorithm for optimal binning
 * 
 * This implementation provides methods for supervised discretization of numerical variables
 * using local polynomial regression for density estimation combined with information-theoretic
 * metrics to maximize predictive power.
 */

using namespace Rcpp;

// Include shared headers
#include "common/optimal_binning_common.h"
#include "common/bin_structures.h"

using namespace Rcpp;
using namespace OptimalBinning;


/**
 * Calculate Pearson correlation coefficient between two vectors
 * 
 * @param x First vector of values
 * @param y Second vector of values
 * @return Correlation coefficient in range [-1, 1]
 */
inline double compute_correlation(const std::vector<double> &x, const std::vector<double> &y) {
  if (x.size() != y.size() || x.empty()) {
    Rcpp::stop("Vectors must be of the same non-zero length for correlation.");
  }
  
  double mean_x = std::accumulate(x.begin(), x.end(), 0.0) / x.size();
  double mean_y = std::accumulate(y.begin(), y.end(), 0.0) / y.size();
  
  double numerator = 0.0;
  double denom_x = 0.0;
  double denom_y = 0.0;
  
  for (size_t i = 0; i < x.size(); ++i) {
    double dx = x[i] - mean_x;
    double dy = y[i] - mean_y;
    numerator += dx * dy;
    denom_x += dx * dx;
    denom_y += dy * dy;
  }
  
  // Handle division by zero
  if (denom_x < 1e-10 || denom_y < 1e-10) {
    Rcpp::warning("Standard deviation is near zero. Returning correlation as 0.");
    return 0.0;
  }
  
  return numerator / std::sqrt(denom_x * denom_y);
}

/**
 * Class for Optimal Binning using Local Polynomial Density Binning (LPDB)
 * 
 * LPDB uses polynomial regression to estimate local density of the feature distribution
 * and places bin boundaries at points of interest in the density function (e.g., inflection 
 * points, local minima). This enhances the binning's ability to capture natural patterns
 * in the data while optimizing predictive power.
 */
class OBN_LPDB {
public:
  /**
   * Constructor for OBN_LPDB
   * 
   * @param min_bins Minimum number of bins
   * @param max_bins Maximum number of bins
   * @param bin_cutoff Minimum fraction of observations required for each bin
   * @param max_n_prebins Maximum number of pre-bins before optimization
   * @param polynomial_degree Degree of polynomial used for density estimation
   * @param enforce_monotonic Whether to enforce monotonic relationship in WoE
   * @param convergence_threshold Convergence threshold for optimization
   * @param max_iterations Maximum number of iterations allowed
   */
  OBN_LPDB(
    int min_bins = 3, 
    int max_bins = 5, 
    double bin_cutoff = 0.05, 
    int max_n_prebins = 20,
    int polynomial_degree = 3,
    bool enforce_monotonic = true,
    double convergence_threshold = 1e-6, 
    int max_iterations = 1000)
    : min_bins(min_bins), 
      max_bins(max_bins), 
      bin_cutoff(bin_cutoff), 
      max_n_prebins(max_n_prebins),
      polynomial_degree(polynomial_degree),
      enforce_monotonic(enforce_monotonic),
      convergence_threshold(convergence_threshold), 
      max_iterations(max_iterations),
      converged(true), 
      iterations_run(0),
      monotonicity_direction(0),
      total_iv(0.0) {
    
    // Validate parameters
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
    if (polynomial_degree < 1 || polynomial_degree > 10) {
      Rcpp::stop("polynomial_degree must be between 1 and 10.");
    }
  }
  
  /**
   * Fit the binning model to data
   * 
   * @param feature Feature vector to bin
   * @param target Binary target vector (0/1)
   * @return List with binning results
   */
  Rcpp::List fit(Rcpp::NumericVector feature, Rcpp::IntegerVector target);
  
private:
  // Parameters
  int min_bins;
  int max_bins;
  double bin_cutoff;
  int max_n_prebins;
  int polynomial_degree;
  bool enforce_monotonic;
  double convergence_threshold;
  int max_iterations;
  
  // State variables
  bool converged;
  int iterations_run;
  int monotonicity_direction;  // 1 for increasing, -1 for decreasing, 0 for undetermined
  double total_iv;
  
  // Constants
  // Constant removed (uses shared definition)
  static constexpr double ALPHA = 0.5;  // Laplace smoothing parameter
  
  /**
   * Structure representing a bin and its statistics
   */
  // Local NumericalBin definition removed

  
  /**
   * Perform polynomial-based density estimation prebinning
   * Uses local polynomial regression for density estimation and places
   * bin boundaries at interest points of the density function
   * 
   * @param feature Feature vector
   * @param target Target vector
   * @return Vector of bins
   */
  std::vector<NumericalBin> polynomial_density_prebinning(
      const std::vector<double> &feature, 
      const std::vector<int> &target);
  
  /**
   * Calculate Weight of Evidence (WoE) and Information Value (IV) for each bin
   * Uses Laplace smoothing for robustness
   * 
   * @param bins Vector of bins
   * @param total_pos Total count of positive observations
   * @param total_neg Total count of negative observations
   */
  void calculate_woe_iv(
      std::vector<NumericalBin> &bins, 
      int total_pos, 
      int total_neg);
  
  /**
   * Enforce monotonicity in WoE values across bins
   * Determines optimal monotonicity direction and merges bins to achieve it
   * 
   * @param bins Vector of bins
   * @param total_pos Total count of positive observations
   * @param total_neg Total count of negative observations
   */
  void enforce_monotonicity(
      std::vector<NumericalBin> &bins, 
      int total_pos, 
      int total_neg);
  
  /**
   * Merge bins with low frequencies and ensure max_bins constraint
   * Prioritizes merging bins that preserve information value
   * 
   * @param bins Vector of bins
   * @param total_pos Total count of positive observations
   * @param total_neg Total count of negative observations
   * @param min_count Minimum count required for each bin
   */
  void optimize_bins(
      std::vector<NumericalBin> &bins, 
      int total_pos, 
      int total_neg,
      int min_count);
  
  /**
   * Calculate local polynomial regression for density estimation
   * Fits polynomials locally for robust nonparametric density estimation
   * 
   * @param x Sorted feature values
   * @param bandwidth Smoothing bandwidth
   * @return Vector of density estimates at each point
   */
  std::vector<double> local_polynomial_density(
      const std::vector<double> &x, 
      double bandwidth);
  
  /**
   * Find critical points in the density curve
   * Identifies local minima, maxima, and inflection points
   * 
   * @param x Sorted feature values
   * @param density Density estimates at each point
   * @param max_points Maximum number of critical points to find
   * @return Vector of x values at critical points
   */
  std::vector<double> find_density_critical_points(
      const std::vector<double> &x, 
      const std::vector<double> &density,
      int max_points);
  
  /**
   * Merge two adjacent bins
   * Updates statistics for the merged bin
   * 
   * @param bins Vector of bins
   * @param idx1 Index of first bin to merge
   * @param idx2 Index of second bin to merge
   * @param total_pos Total count of positive observations
   * @param total_neg Total count of negative observations
   */
  void merge_adjacent_bins(
      std::vector<NumericalBin> &bins, 
      size_t idx1, 
      size_t idx2, 
      int total_pos, 
      int total_neg);
  
  /**
   * Calculate Weight of Evidence with Laplace smoothing
   * 
   * @param pos Count of positives
   * @param neg Count of negatives
   * @param total_pos Total count of positives
   * @param total_neg Total count of negatives
   * @param num_bins Number of bins (for smoothing)
   * @return WoE value
   */
  double calculate_woe(
      int pos, 
      int neg, 
      int total_pos, 
      int total_neg,
      int num_bins) const;
  
  /**
   * Calculate Information Value contribution
   * 
   * @param woe Weight of Evidence value
   * @param pos Count of positives
   * @param neg Count of negatives
   * @param total_pos Total count of positives
   * @param total_neg Total count of negatives
   * @return IV contribution
   */
  double calculate_iv(
      double woe, 
      int pos, 
      int neg, 
      int total_pos, 
      int total_neg) const;
  
  /**
   * Format bin interval as string for output
   * 
   * @param lower Lower boundary
   * @param upper Upper boundary
   * @param first Whether this is the first bin
   * @param last Whether this is the last bin
   * @return Formatted string
   */
  std::string format_bin_interval(
      double lower, 
      double upper, 
      bool first = false, 
      bool last = false) const;
  
  /**
   * Handle the special case of few unique values
   * 
   * @param feature Feature vector
   * @param target Target vector
   * @param unique_values Vector of unique feature values
   * @param total_pos Total count of positives
   * @param total_neg Total count of negatives
   * @return List with results if handled specially, or empty list otherwise
   */
  Rcpp::List handle_special_cases(
      const std::vector<double> &feature,
      const std::vector<int> &target,
      const std::vector<double> &unique_values,
      int total_pos,
      int total_neg);
  
  /**
   * Find the optimal bin for a value using binary search
   * 
   * @param value The value to find bin for
   * @param bins Vector of bins
   * @return Index of the bin containing the value
   */
  size_t find_bin_index(
      double value, 
      const std::vector<NumericalBin> &bins) const;
};

/**
 * Fit the binning model to data
 */
Rcpp::List OBN_LPDB::fit(Rcpp::NumericVector feature, Rcpp::IntegerVector target) {
  int n = feature.size();
  if (n != target.size()) {
    Rcpp::stop("feature and target must have the same length.");
  }
  
  // Ensure target is binary (0/1)
  IntegerVector unique_targets = unique(target);
  if (unique_targets.size() != 2 || 
      (std::find(unique_targets.begin(), unique_targets.end(), 0) == unique_targets.end()) ||
      (std::find(unique_targets.begin(), unique_targets.end(), 1) == unique_targets.end())) {
    Rcpp::stop("Target must be binary (0 and 1) and contain both classes.");
  }
  
  // Remove NA values
  LogicalVector not_na = (!is_na(feature)) & (!is_na(target));
  NumericVector clean_feature = feature[not_na];
  IntegerVector clean_target = target[not_na];
  
  if (clean_feature.size() == 0) {
    Rcpp::stop("No valid observations after removing missing values.");
  }
  
  // Count positives and negatives
  int total_pos = std::accumulate(clean_target.begin(), clean_target.end(), 0);
  int total_neg = clean_target.size() - total_pos;
  
  if (total_pos == 0 || total_neg == 0) {
    Rcpp::stop("Target must have both positive and negative classes.");
  }
  
  // Get unique values
  NumericVector unique_vec = unique(clean_feature);
  std::vector<double> unique_feature = Rcpp::as<std::vector<double>>(unique_vec);
  std::sort(unique_feature.begin(), unique_feature.end());
  
  // Convert to C++ vectors
  std::vector<double> feature_vec = Rcpp::as<std::vector<double>>(clean_feature);
  std::vector<int> target_vec = Rcpp::as<std::vector<int>>(clean_target);
  
  // Check for special cases (few unique values)
  Rcpp::List special_case_result = handle_special_cases(
    feature_vec, target_vec, unique_feature, total_pos, total_neg);
  
  if (special_case_result.size() > 0) {
    return special_case_result;
  }
  
  // Regular binning process
  std::vector<NumericalBin> bins = polynomial_density_prebinning(feature_vec, target_vec);
  
  // Calculate WoE and IV
  calculate_woe_iv(bins, total_pos, total_neg);
  
  // Apply monotonicity if requested
  if (enforce_monotonic) {
    enforce_monotonicity(bins, total_pos, total_neg);
  }
  
  // Optimize bins (merge rare bins, ensure max_bins)
  int min_count = static_cast<int>(std::ceil(bin_cutoff * feature_vec.size()));
  optimize_bins(bins, total_pos, total_neg, min_count);
  
  // Calculate total IV
  total_iv = 0.0;
  for (const auto& bin : bins) {
    total_iv += bin.iv;
  }
  
  // Create output
  std::vector<std::string> bin_labels;
  std::vector<double> woe_values;
  std::vector<double> iv_values;
  std::vector<int> counts;
  std::vector<int> counts_pos;
  std::vector<int> counts_neg;
  std::vector<double> event_rates;
  std::vector<double> cutpoints;
  std::vector<double> centroids;
  
  for (size_t i = 0; i < bins.size(); ++i) {
    const auto& bin = bins[i];
    
    bin_labels.push_back(format_bin_interval(
        bin.lower_bound, bin.upper_bound, i == 0, i == bins.size() - 1));
    
    woe_values.push_back(bin.woe);
    iv_values.push_back(bin.iv);
    counts.push_back(bin.count);
    counts_pos.push_back(bin.count_pos);
    counts_neg.push_back(bin.count_neg);
    event_rates.push_back(bin.event_rate());
    centroids.push_back(bin.centroid);
    
    if (i < bins.size() - 1) {
      cutpoints.push_back(bin.upper_bound);
    }
  }
  
  // Create bin IDs (1-based for R)
  Rcpp::NumericVector ids(bin_labels.size());
  for (size_t i = 0; i < bin_labels.size(); i++) {
    ids[i] = static_cast<double>(i + 1);
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
    Rcpp::Named("count_pos") = counts_pos,
    Rcpp::Named("count_neg") = counts_neg,
    Rcpp::Named("event_rate") = event_rates,
    Rcpp::Named("centroids") = centroids,
    Rcpp::Named("cutpoints") = cutpoints,
    Rcpp::Named("converged") = converged,
    Rcpp::Named("iterations") = iterations_run,
    Rcpp::Named("total_iv") = total_iv,
    Rcpp::Named("monotonicity") = monotonicity
  );
}

/**
 * Handle special cases like few unique values
 */
Rcpp::List OBN_LPDB::handle_special_cases(
    const std::vector<double> &feature,
    const std::vector<int> &target,
    const std::vector<double> &unique_values,
    int total_pos,
    int total_neg) {
  
  // Case: Single unique value
  if (unique_values.size() == 1) {
    // Create a single bin
    NumericalBin bin;
    bin.lower_bound = -std::numeric_limits<double>::infinity();
    bin.upper_bound = std::numeric_limits<double>::infinity();
    bin.count = feature.size();
    bin.count_pos = total_pos;
    bin.count_neg = total_neg;
    bin.centroid = unique_values[0];
    // bin.event_rate() assignment removed (calculated dynamically)
    
    // Calculate WoE and IV
    int num_bins = 1;
    bin.woe = calculate_woe(bin.count_pos, bin.count_neg, total_pos, total_neg, num_bins);
    bin.iv = calculate_iv(bin.woe, bin.count_pos, bin.count_neg, total_pos, total_neg);
    
    // Create output
    std::vector<std::string> bin_labels;
    bin_labels.emplace_back(format_bin_interval(bin.lower_bound, bin.upper_bound, true, true));
    
    NumericVector woe_values(1, bin.woe);
    NumericVector iv_values(1, bin.iv);
    IntegerVector counts(1, bin.count);
    IntegerVector counts_pos(1, bin.count_pos);
    IntegerVector counts_neg(1, bin.count_neg);
    NumericVector event_rates(1, bin.event_rate());
    NumericVector centroids(1, bin.centroid);
    NumericVector cutpoints; // Empty, only one bin
    
    // Create bin IDs (1-based for R)
    Rcpp::NumericVector ids(1, 1.0);
    
    // Total IV
    total_iv = bin.iv;
    
    return Rcpp::List::create(
      Rcpp::Named("id") = ids,
      Rcpp::Named("bin") = bin_labels,
      Rcpp::Named("woe") = woe_values,
      Rcpp::Named("iv") = iv_values,
      Rcpp::Named("count") = counts,
      Rcpp::Named("count_pos") = counts_pos,
      Rcpp::Named("count_neg") = counts_neg,
      Rcpp::Named("event_rate") = event_rates,
      Rcpp::Named("centroids") = centroids,
      Rcpp::Named("cutpoints") = cutpoints,
      Rcpp::Named("converged") = true,
      Rcpp::Named("iterations") = 0,
      Rcpp::Named("total_iv") = total_iv,
      Rcpp::Named("monotonicity") = "none"
    );
  }
  
  // Case: Two unique values
  if (unique_values.size() == 2) {
    // Create two bins
    std::vector<NumericalBin> bins(2);
    
    // Define boundaries
    double midpoint = (unique_values[0] + unique_values[1]) / 2.0;
    
    bins[0].lower_bound = -std::numeric_limits<double>::infinity();
    bins[0].upper_bound = midpoint;
    bins[0].centroid = unique_values[0];
    
    bins[1].lower_bound = midpoint;
    bins[1].upper_bound = std::numeric_limits<double>::infinity();
    bins[1].centroid = unique_values[1];
    
    // Assign observations to bins
    for (size_t i = 0; i < feature.size(); ++i) {
      double val = feature[i];
      int t = target[i];
      
      size_t bin_idx = (val <= midpoint) ? 0 : 1;
      
      bins[bin_idx].count++;
      if (t == 1) {
        bins[bin_idx].count_pos++;
      } else {
        bins[bin_idx].count_neg++;
      }
    }
    
    // Calculate WoE, IV, and event rates
    int num_bins = 2;
    for (auto& bin : bins) {
      // bin.event_rate() assignment removed (calculated dynamically)
      bin.woe = calculate_woe(bin.count_pos, bin.count_neg, total_pos, total_neg, num_bins);
      bin.iv = calculate_iv(bin.woe, bin.count_pos, bin.count_neg, total_pos, total_neg);
    }
    
    // Determine monotonicity
    monotonicity_direction = (bins[1].woe >= bins[0].woe) ? 1 : -1;
    std::string monotonicity = (monotonicity_direction > 0) ? "increasing" : "decreasing";
    
    // Create output
    std::vector<std::string> bin_labels;
    std::vector<double> woe_values;
    std::vector<double> iv_values;
    std::vector<int> counts;
    std::vector<int> counts_pos;
    std::vector<int> counts_neg;
    std::vector<double> event_rates;
    std::vector<double> centroids;
    
    for (size_t i = 0; i < bins.size(); ++i) {
      const auto& bin = bins[i];
      
      bin_labels.push_back(format_bin_interval(
          bin.lower_bound, bin.upper_bound, i == 0, i == bins.size() - 1));
      
      woe_values.push_back(bin.woe);
      iv_values.push_back(bin.iv);
      counts.push_back(bin.count);
      counts_pos.push_back(bin.count_pos);
      counts_neg.push_back(bin.count_neg);
      event_rates.push_back(bin.event_rate());
      centroids.push_back(bin.centroid);
    }
    
    // Create cutpoints
    NumericVector cutpoints(1, midpoint);
    
    // Create bin IDs (1-based for R)
    Rcpp::NumericVector ids(2);
    ids[0] = 1;
    ids[1] = 2;
    
    // Calculate total IV
    total_iv = 0.0;
    for (const auto& bin : bins) {
      total_iv += bin.iv;
    }
    
    return Rcpp::List::create(
      Rcpp::Named("id") = ids,
      Rcpp::Named("bin") = bin_labels,
      Rcpp::Named("woe") = woe_values,
      Rcpp::Named("iv") = iv_values,
      Rcpp::Named("count") = counts,
      Rcpp::Named("count_pos") = counts_pos,
      Rcpp::Named("count_neg") = counts_neg,
      Rcpp::Named("event_rate") = event_rates,
      Rcpp::Named("centroids") = centroids,
      Rcpp::Named("cutpoints") = cutpoints,
      Rcpp::Named("converged") = true,
      Rcpp::Named("iterations") = 0,
      Rcpp::Named("total_iv") = total_iv,
      Rcpp::Named("monotonicity") = monotonicity
    );
  }
  
  // If unique values less than or equal to min_bins, create one bin per unique value
  if (unique_values.size() <= static_cast<size_t>(min_bins)) {
    // Create bins based on unique values
    std::vector<NumericalBin> bins(unique_values.size());
    
    // Define boundaries
    for (size_t i = 0; i < unique_values.size(); ++i) {
      double lower = (i == 0) ? -std::numeric_limits<double>::infinity() : 
      (unique_values[i-1] + unique_values[i]) / 2.0;
      
      double upper = (i == unique_values.size()-1) ? std::numeric_limits<double>::infinity() : 
        (unique_values[i] + unique_values[i+1]) / 2.0;
      
      bins[i].lower_bound = lower;
      bins[i].upper_bound = upper;
      bins[i].centroid = unique_values[i];
    }
    
    // Assign observations to bins
    for (size_t i = 0; i < feature.size(); ++i) {
      double val = feature[i];
      int t = target[i];
      
      // Find appropriate bin using binary search
      size_t bin_idx = 0;
      for (size_t j = 0; j < bins.size(); ++j) {
        if (val > bins[j].lower_bound && val <= bins[j].upper_bound) {
          bin_idx = j;
          break;
        }
      }
      
      bins[bin_idx].count++;
      if (t == 1) {
        bins[bin_idx].count_pos++;
      } else {
        bins[bin_idx].count_neg++;
      }
    }
    
    // Calculate WoE, IV, and event rates
    int num_bins = static_cast<int>(bins.size());
    for (auto& bin : bins) {
      // bin.event_rate() assignment removed (calculated dynamically)
      bin.woe = calculate_woe(bin.count_pos, bin.count_neg, total_pos, total_neg, num_bins);
      bin.iv = calculate_iv(bin.woe, bin.count_pos, bin.count_neg, total_pos, total_neg);
    }
    
    // Apply monotonicity if requested and possible
    if (enforce_monotonic && bins.size() >= 2) {
      enforce_monotonicity(bins, total_pos, total_neg);
    }
    
    // Create output
    std::vector<std::string> bin_labels;
    std::vector<double> woe_values;
    std::vector<double> iv_values;
    std::vector<int> counts;
    std::vector<int> counts_pos;
    std::vector<int> counts_neg;
    std::vector<double> event_rates;
    std::vector<double> centroids;
    std::vector<double> cutpoints;
    
    for (size_t i = 0; i < bins.size(); ++i) {
      const auto& bin = bins[i];
      
      bin_labels.push_back(format_bin_interval(
          bin.lower_bound, bin.upper_bound, i == 0, i == bins.size() - 1));
      
      woe_values.push_back(bin.woe);
      iv_values.push_back(bin.iv);
      counts.push_back(bin.count);
      counts_pos.push_back(bin.count_pos);
      counts_neg.push_back(bin.count_neg);
      event_rates.push_back(bin.event_rate());
      centroids.push_back(bin.centroid);
      
      if (i < bins.size() - 1) {
        cutpoints.push_back(bin.upper_bound);
      }
    }
    
    // Create bin IDs (1-based for R)
    Rcpp::NumericVector ids(bins.size());
    for (size_t i = 0; i < bins.size(); i++) {
      ids[i] = i + 1;
    }
    
    // Calculate total IV
    total_iv = 0.0;
    for (const auto& bin : bins) {
      total_iv += bin.iv;
    }
    
    // Determine monotonicity
    std::string monotonicity = "none";
    if (monotonicity_direction > 0) {
      monotonicity = "increasing";
    } else if (monotonicity_direction < 0) {
      monotonicity = "decreasing";
    }
    
    return Rcpp::List::create(
      Rcpp::Named("id") = ids,
      Rcpp::Named("bin") = bin_labels,
      Rcpp::Named("woe") = woe_values,
      Rcpp::Named("iv") = iv_values,
      Rcpp::Named("count") = counts,
      Rcpp::Named("count_pos") = counts_pos,
      Rcpp::Named("count_neg") = counts_neg,
      Rcpp::Named("event_rate") = event_rates,
      Rcpp::Named("centroids") = centroids,
      Rcpp::Named("cutpoints") = cutpoints,
      Rcpp::Named("converged") = true,
      Rcpp::Named("iterations") = iterations_run,
      Rcpp::Named("total_iv") = total_iv,
      Rcpp::Named("monotonicity") = monotonicity
    );
  }
  
  // If no special case, return empty list
  return Rcpp::List();
}

/**
 * Perform polynomial-based density estimation and prebinning
 */
std::vector<NumericalBin> OBN_LPDB::polynomial_density_prebinning(
    const std::vector<double> &feature, 
    const std::vector<int> &target) {
  
  // Create working copy and sort
  std::vector<double> sorted_feature = feature;
  std::sort(sorted_feature.begin(), sorted_feature.end());
  
  // Estimate bandwidth for density estimation
  // Silverman's rule of thumb
  double n = static_cast<double>(sorted_feature.size());
  double std_dev = 0.0;
  double mean = 0.0;
  
  // Calculate mean and standard deviation
  for (double val : sorted_feature) {
    mean += val;
  }
  mean /= n;
  
  for (double val : sorted_feature) {
    std_dev += (val - mean) * (val - mean);
  }
  std_dev = std::sqrt(std_dev / n);
  
  // Calculate bandwidth
  double bandwidth = 0.9 * std_dev * std::pow(n, -0.2);
  
  // Estimate density using local polynomial regression
  std::vector<double> density = local_polynomial_density(sorted_feature, bandwidth);
  
  // Find critical points in density curve
  int max_critical_points = std::min(max_n_prebins - 1, static_cast<int>(sorted_feature.size() / 10));
  std::vector<double> critical_points = find_density_critical_points(
    sorted_feature, density, max_critical_points);
  
  // Add quantile-based points if not enough critical points found
  if (critical_points.size() < static_cast<size_t>(min_bins - 1)) {
    int n_additional = min_bins - 1 - static_cast<int>(critical_points.size());
    
    for (int i = 1; i <= n_additional; ++i) {
      double q = static_cast<double>(i) / (n_additional + 1);
      size_t idx = static_cast<size_t>(q * sorted_feature.size());
      
      if (idx >= sorted_feature.size()) {
        idx = sorted_feature.size() - 1;
      }
      
      critical_points.push_back(sorted_feature[idx]);
    }
  }
  
  // Ensure critical points are unique and sorted
  std::sort(critical_points.begin(), critical_points.end());
  critical_points.erase(
    std::unique(critical_points.begin(), critical_points.end()), 
    critical_points.end());
  
  // Create bins using critical points as boundaries
  std::vector<NumericalBin> bins;
  bins.reserve(critical_points.size() + 1);
  
  // First bin
  NumericalBin first_bin;
  first_bin.lower_bound = -std::numeric_limits<double>::infinity();
  first_bin.upper_bound = critical_points.empty() ? 
  std::numeric_limits<double>::infinity() : 
    critical_points[0];
  bins.push_back(first_bin);
  
  // Middle bins
  for (size_t i = 0; i < critical_points.size() - 1; ++i) {
    NumericalBin bin;
    bin.lower_bound = critical_points[i];
    bin.upper_bound = critical_points[i + 1];
    bins.push_back(bin);
  }
  
  // Last bin
  if (!critical_points.empty()) {
    NumericalBin last_bin;
    last_bin.lower_bound = critical_points.back();
    last_bin.upper_bound = std::numeric_limits<double>::infinity();
    bins.push_back(last_bin);
  }
  
  // Assign observations to bins
  for (size_t i = 0; i < feature.size(); ++i) {
    double val = feature[i];
    int t = target[i];
    
    // Skip NaN values
    if (std::isnan(val)) {
      continue;
    }
    
    // Find bin using binary search
    size_t bin_idx = find_bin_index(val, bins);
    
    if (bin_idx < bins.size()) {
      bins[bin_idx].count++;
      
      // Update centroid using online mean calculation
      bins[bin_idx].centroid = bins[bin_idx].centroid + 
        (val - bins[bin_idx].centroid) / bins[bin_idx].count;
      
      if (t == 1) {
        bins[bin_idx].count_pos++;
      } else {
        bins[bin_idx].count_neg++;
      }
    }
  }
  
  // Event rates are calculated dynamically via event_rate() method
  // Observations assigned above; bins ready to return
  
  return bins;
}

/**
 * Find the bin index for a value using binary search
 */
size_t OBN_LPDB::find_bin_index(
    double value, 
    const std::vector<NumericalBin> &bins) const {
  
  // Handle NaN
  if (std::isnan(value)) {
    return bins.size(); // Invalid index
  }
  
  // Edge cases
  if (value <= bins.front().upper_bound) {
    return 0;
  }
  
  if (value > bins.back().lower_bound) {
    return bins.size() - 1;
  }
  
  // Binary search
  int low = 0;
  int high = static_cast<int>(bins.size()) - 1;
  
  while (low <= high) {
    int mid = low + (high - low) / 2;
    
    if (value > bins[mid].lower_bound && value <= bins[mid].upper_bound) {
      return mid;
    }
    
    if (value <= bins[mid].lower_bound) {
      high = mid - 1;
    } else {
      low = mid + 1;
    }
  }
  
  // Fallback to linear search in case of numerical issues
  for (size_t i = 0; i < bins.size(); ++i) {
    if (value > bins[i].lower_bound && value <= bins[i].upper_bound) {
      return i;
    }
  }
  
  // If all else fails, put in the last bin
  return bins.size() - 1;
}

/**
 * Calculate local polynomial density estimation
 */
std::vector<double> OBN_LPDB::local_polynomial_density(
    const std::vector<double> &x, 
    double bandwidth) {
  
  size_t n = x.size();
  std::vector<double> density(n, 0.0);
  
  // Quick check for trivial case
  if (n <= 1) {
    return density;
  }
  
  // Ensure positive bandwidth
  if (bandwidth <= EPSILON) {
    bandwidth = std::max(
      (x.back() - x.front()) / static_cast<double>(n), 
      EPSILON * 10.0);
  }
  
  // Simple kernel density estimation with Gaussian kernel
  // For more complex datasets, a true local polynomial regression would be better
  for (size_t i = 0; i < n; ++i) {
    double xi = x[i];
    double sum = 0.0;
    
    for (size_t j = 0; j < n; ++j) {
      double xj = x[j];
      double z = (xi - xj) / bandwidth;
      sum += std::exp(-0.5 * z * z);
    }
    
    density[i] = sum / (n * bandwidth * std::sqrt(2.0 * M_PI));
  }
  
  return density;
}

/**
 * Find critical points in density curve
 */
std::vector<double> OBN_LPDB::find_density_critical_points(
    const std::vector<double> &x, 
    const std::vector<double> &density,
    int max_points) {
  
  size_t n = density.size();
  std::vector<double> critical_points;
  
  // Need at least 3 points for derivatives
  if (n < 3) {
    return critical_points;
  }
  
  // Calculate first differences (approximate first derivative)
  std::vector<double> first_deriv(n - 1);
  for (size_t i = 0; i < n - 1; ++i) {
    first_deriv[i] = (density[i+1] - density[i]) / (x[i+1] - x[i]);
  }
  
  // Find sign changes in first derivative (local extrema)
  std::vector<size_t> extrema_indices;
  for (size_t i = 0; i < first_deriv.size() - 1; ++i) {
    if ((first_deriv[i] > 0 && first_deriv[i+1] < 0) ||  // Local maximum
        (first_deriv[i] < 0 && first_deriv[i+1] > 0)) {  // Local minimum
      extrema_indices.push_back(i + 1);
    }
  }
  
  // Calculate second differences (approximate second derivative)
  std::vector<double> second_deriv(n - 2);
  for (size_t i = 0; i < n - 2; ++i) {
    second_deriv[i] = (first_deriv[i+1] - first_deriv[i]) / (x[i+2] - x[i]);
  }
  
  // Find sign changes in second derivative (inflection points)
  std::vector<size_t> inflection_indices;
  for (size_t i = 0; i < second_deriv.size() - 1; ++i) {
    if ((second_deriv[i] > 0 && second_deriv[i+1] < 0) ||
        (second_deriv[i] < 0 && second_deriv[i+1] > 0)) {
      inflection_indices.push_back(i + 1);
    }
  }
  
  // Prioritize local minima for bin boundaries
  for (size_t i = 0; i < extrema_indices.size() && critical_points.size() < static_cast<size_t>(max_points); ++i) {
    size_t idx = extrema_indices[i];
    
    // Check if it's a local minimum (first derivative changes from negative to positive)
    if (idx > 0 && idx < first_deriv.size() &&
        first_deriv[idx-1] < 0 && first_deriv[idx] > 0) {
      critical_points.push_back(x[idx]);
    }
  }
  
  // Add inflection points if needed
  for (size_t i = 0; i < inflection_indices.size() && critical_points.size() < static_cast<size_t>(max_points); ++i) {
    size_t idx = inflection_indices[i];
    if (idx < x.size()) {
      critical_points.push_back(x[idx]);
    }
  }
  
  // Add other extrema if still need more points
  for (size_t i = 0; i < extrema_indices.size() && critical_points.size() < static_cast<size_t>(max_points); ++i) {
    size_t idx = extrema_indices[i];
    if (idx < x.size() && 
        std::find(critical_points.begin(), critical_points.end(), x[idx]) == critical_points.end()) {
      critical_points.push_back(x[idx]);
    }
  }
  
  return critical_points;
}

/**
 * Calculate WoE and IV for bins
 */
void OBN_LPDB::calculate_woe_iv(
    std::vector<NumericalBin> &bins, 
    int total_pos, 
    int total_neg) {
  
  int num_bins = static_cast<int>(bins.size());
  
  for (auto& bin : bins) {
    bin.woe = calculate_woe(bin.count_pos, bin.count_neg, total_pos, total_neg, num_bins);
    bin.iv = calculate_iv(bin.woe, bin.count_pos, bin.count_neg, total_pos, total_neg);
  }
}

/**
 * Calculate Weight of Evidence with Laplace smoothing
 */
double OBN_LPDB::calculate_woe(
    int pos, 
    int neg, 
    int total_pos, 
    int total_neg,
    int num_bins) const {
  
  // Apply Laplace smoothing
  double smoothed_pos = (static_cast<double>(pos) + ALPHA) / 
    (static_cast<double>(total_pos) + num_bins * ALPHA);
  
  double smoothed_neg = (static_cast<double>(neg) + ALPHA) / 
    (static_cast<double>(total_neg) + num_bins * ALPHA);
  
  // Handle potential numerical issues
  if (smoothed_pos < EPSILON) smoothed_pos = EPSILON;
  if (smoothed_neg < EPSILON) smoothed_neg = EPSILON;
  
  return std::log(smoothed_pos / smoothed_neg);
}

/**
 * Calculate Information Value contribution
 */
double OBN_LPDB::calculate_iv(
    double woe, 
    int pos, 
    int neg, 
    int total_pos, 
    int total_neg) const {
  
  double dist_pos = static_cast<double>(pos) / static_cast<double>(total_pos);
  double dist_neg = static_cast<double>(neg) / static_cast<double>(total_neg);
  
  return (dist_pos - dist_neg) * woe;
}

/**
 * Enforce monotonicity in WoE values
 */
void OBN_LPDB::enforce_monotonicity(
    std::vector<NumericalBin> &bins, 
    int total_pos, 
    int total_neg) {
  
  if (bins.size() <= 1) {
    return;
  }
  
  // Determine monotonicity direction
  // Calculate correlation between centroids and WoE
  std::vector<double> centroids;
  std::vector<double> woe_values;
  
  for (const auto& bin : bins) {
    centroids.push_back(bin.centroid);
    woe_values.push_back(bin.woe);
  }
  
  double correlation = compute_correlation(centroids, woe_values);
  monotonicity_direction = (correlation >= 0) ? 1 : -1;
  
  // Enforce monotonicity through bin merging
  bool monotonic = false;
  int iterations = 0;
  
  while (!monotonic && bins.size() > static_cast<size_t>(min_bins) && iterations < max_iterations) {
    monotonic = true;
    
    for (size_t i = 1; i < bins.size(); ++i) {
      bool violation = (monotonicity_direction > 0 && bins[i].woe < bins[i-1].woe) || 
        (monotonicity_direction < 0 && bins[i].woe > bins[i-1].woe);
      
      if (violation) {
        // Merge bins i-1 and i
        merge_adjacent_bins(bins, i-1, i, total_pos, total_neg);
        
        monotonic = false;
        iterations++;
        break;
      }
    }
    
    if (monotonic || bins.size() <= static_cast<size_t>(min_bins)) {
      break;
    }
  }
  
  if (iterations >= max_iterations) {
    converged = false;
  }
  
  iterations_run += iterations;
}

/**
 * Optimize bins through merging
 */
void OBN_LPDB::optimize_bins(
    std::vector<NumericalBin> &bins, 
    int total_pos, 
    int total_neg,
    int min_count) {
  
  if (bins.empty()) {
    return;
  }
  
  // Step 1: Merge rare bins
  int iterations = 0;
  bool merged = true;
  
  while (merged && bins.size() > static_cast<size_t>(min_bins) && iterations < max_iterations) {
    merged = false;
    
    for (size_t i = 0; i < bins.size(); ++i) {
      if (bins[i].count < min_count) {
        // Determine merge direction
        size_t merge_idx;
        
        if (i == 0) {
          // First bin - merge with next
          merge_idx = 1;
        } else if (i == bins.size() - 1) {
          // Last bin - merge with previous
          merge_idx = i - 1;
        } else {
          // Middle bin - pick neighbor with more similar event rate
          double diff_prev = std::fabs(bins[i].event_rate() - bins[i-1].event_rate());
          double diff_next = std::fabs(bins[i].event_rate() - bins[i+1].event_rate());
          
          merge_idx = (diff_prev <= diff_next) ? i - 1 : i + 1;
        }
        
        // Merge bins
        if (i < merge_idx) {
          merge_adjacent_bins(bins, i, merge_idx, total_pos, total_neg);
        } else {
          merge_adjacent_bins(bins, merge_idx, i, total_pos, total_neg);
        }
        
        merged = true;
        iterations++;
        break;
      }
    }
  }
  
  // Step 2: Ensure max_bins constraint
  while (bins.size() > static_cast<size_t>(max_bins) && iterations < max_iterations) {
    // Find pair of adjacent bins with smallest IV loss when merged
    double min_iv_loss = std::numeric_limits<double>::max();
    size_t merge_idx = 0;
    
    for (size_t i = 0; i < bins.size() - 1; ++i) {
      // Calculate current IV
      double current_iv = bins[i].iv + bins[i+1].iv;
      
      // Calculate IV if merged
      int merged_pos = bins[i].count_pos + bins[i+1].count_pos;
      int merged_neg = bins[i].count_neg + bins[i+1].count_neg;
      
      double merged_woe = calculate_woe(
        merged_pos, merged_neg, total_pos, total_neg, static_cast<int>(bins.size()) - 1);
      
      double merged_iv = calculate_iv(
        merged_woe, merged_pos, merged_neg, total_pos, total_neg);
      
      // Calculate IV loss
      double iv_loss = current_iv - merged_iv;
      
      if (iv_loss < min_iv_loss) {
        min_iv_loss = iv_loss;
        merge_idx = i;
      }
    }
    
    // Merge bins
    merge_adjacent_bins(bins, merge_idx, merge_idx + 1, total_pos, total_neg);
    iterations++;
  }
  
  if (iterations >= max_iterations) {
    converged = false;
  }
  
  iterations_run += iterations;
}

/**
 * Merge two adjacent bins
 */
void OBN_LPDB::merge_adjacent_bins(
    std::vector<NumericalBin> &bins, 
    size_t idx1, 
    size_t idx2, 
    int total_pos, 
    int total_neg) {
  
  if (idx1 > idx2) {
    std::swap(idx1, idx2);
  }
  
  if (idx2 != idx1 + 1 || idx2 >= bins.size()) {
    Rcpp::stop("Can only merge adjacent bins within bounds.");
  }
  
  // Calculate merged bin centroid
  double total_count = bins[idx1].count + bins[idx2].count;
  if (total_count > 0) {
    bins[idx1].centroid = (bins[idx1].centroid * bins[idx1].count + 
      bins[idx2].centroid * bins[idx2].count) / total_count;
  }
  
  // Update bin boundaries
  bins[idx1].upper_bound = bins[idx2].upper_bound;
  
  // Update counts
  bins[idx1].count += bins[idx2].count;
  bins[idx1].count_pos += bins[idx2].count_pos;
  bins[idx1].count_neg += bins[idx2].count_neg;
  
  // Update event rate
  // bins[idx1].event_rate() assignment removed (calculated dynamically)
  
  // Update WoE and IV
  int num_bins = static_cast<int>(bins.size()) - 1;
  bins[idx1].woe = calculate_woe(
    bins[idx1].count_pos, bins[idx1].count_neg, total_pos, total_neg, num_bins);
  
  bins[idx1].iv = calculate_iv(
    bins[idx1].woe, bins[idx1].count_pos, bins[idx1].count_neg, total_pos, total_neg);
  
  // Remove the second bin
  bins.erase(bins.begin() + idx2);
}

/**
 * Format bin interval for display
 */
std::string OBN_LPDB::format_bin_interval(
    double lower, 
    double upper, 
    bool first, 
    bool last) const {
  
  std::ostringstream oss;
  oss.precision(6);
  oss << std::fixed;
  
  oss << "(";
  
  if (first || (std::isinf(lower) && lower < 0)) {
    oss << "-Inf";
  } else {
    oss << lower;
  }
  
  oss << "; ";
  
  if (last || (std::isinf(upper) && upper > 0)) {
    oss << "+Inf";
  } else {
    oss << upper;
  }
  
  oss << "]";
  return oss.str();
}

// [[Rcpp::export]]
Rcpp::List optimal_binning_numerical_lpdb(
   Rcpp::IntegerVector target,
   Rcpp::NumericVector feature,
   int min_bins = 3,
   int max_bins = 5,
   double bin_cutoff = 0.05,
   int max_n_prebins = 20,
   int polynomial_degree = 3,
   bool enforce_monotonic = true,
   double convergence_threshold = 1e-6,
   int max_iterations = 1000) {
 
 try {
   // Initialize binning algorithm
   OBN_LPDB binning(
       min_bins, max_bins, 
       bin_cutoff, max_n_prebins,
       polynomial_degree, enforce_monotonic,
       convergence_threshold, max_iterations);
   
   // Execute binning
   return binning.fit(feature, target);
 } catch(std::exception &e) {
   forward_exception_to_r(e);
 } catch(...) {
   ::Rf_error("Unknown C++ exception in optimal_binning_numerical_lpdb");
 }
 
 // Should never reach here
 return R_NilValue;
}
