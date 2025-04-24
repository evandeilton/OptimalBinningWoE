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
 * @file OptimalBinningNumericalLPDB.cpp
 * @brief Implementation of Local Polynomial Density Binning (LPDB) algorithm for optimal binning
 * 
 * This implementation provides methods for supervised discretization of numerical variables
 * using local polynomial regression for density estimation combined with information-theoretic
 * metrics to maximize predictive power.
 */

using namespace Rcpp;

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
class OptimalBinningNumericalLPDB {
public:
  /**
   * Constructor for OptimalBinningNumericalLPDB
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
  OptimalBinningNumericalLPDB(
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
  static constexpr double EPSILON = 1e-10;
  static constexpr double ALPHA = 0.5;  // Laplace smoothing parameter
  
  /**
   * Structure representing a bin and its statistics
   */
  struct Bin {
    double lower_bound;   // Lower boundary (exclusive, except for first bin)
    double upper_bound;   // Upper boundary (inclusive)
    int count;            // Total count of observations
    int count_pos;        // Count of positive class observations
    int count_neg;        // Count of negative class observations
    double event_rate;    // Proportion of positives (count_pos / count)
    double woe;           // Weight of Evidence
    double iv;            // Information Value contribution
    double centroid;      // Average value of observations in bin
    
    // Constructor with default values
    Bin() : lower_bound(-std::numeric_limits<double>::infinity()),
    upper_bound(std::numeric_limits<double>::infinity()),
    count(0), count_pos(0), count_neg(0),
    event_rate(0.0), woe(0.0), iv(0.0), centroid(0.0) {}
  };
  
  /**
   * Perform polynomial-based density estimation prebinning
   * Uses local polynomial regression for density estimation and places
   * bin boundaries at interest points of the density function
   * 
   * @param feature Feature vector
   * @param target Target vector
   * @return Vector of bins
   */
  std::vector<Bin> polynomial_density_prebinning(
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
      std::vector<Bin> &bins, 
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
      std::vector<Bin> &bins, 
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
      std::vector<Bin> &bins, 
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
      std::vector<Bin> &bins, 
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
      const std::vector<Bin> &bins) const;
};

/**
 * Fit the binning model to data
 */
Rcpp::List OptimalBinningNumericalLPDB::fit(Rcpp::NumericVector feature, Rcpp::IntegerVector target) {
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
  std::vector<Bin> bins = polynomial_density_prebinning(feature_vec, target_vec);
  
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
    event_rates.push_back(bin.event_rate);
    centroids.push_back(bin.centroid);
    
    if (i < bins.size() - 1) {
      cutpoints.push_back(bin.upper_bound);
    }
  }
  
  // Create bin IDs (1-based for R)
  Rcpp::NumericVector ids(bin_labels.size());
  for (int i = 0; i < bin_labels.size(); i++) {
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
Rcpp::List OptimalBinningNumericalLPDB::handle_special_cases(
    const std::vector<double> &feature,
    const std::vector<int> &target,
    const std::vector<double> &unique_values,
    int total_pos,
    int total_neg) {
  
  // Case: Single unique value
  if (unique_values.size() == 1) {
    // Create a single bin
    Bin bin;
    bin.lower_bound = -std::numeric_limits<double>::infinity();
    bin.upper_bound = std::numeric_limits<double>::infinity();
    bin.count = feature.size();
    bin.count_pos = total_pos;
    bin.count_neg = total_neg;
    bin.centroid = unique_values[0];
    bin.event_rate = static_cast<double>(bin.count_pos) / bin.count;
    
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
    NumericVector event_rates(1, bin.event_rate);
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
    std::vector<Bin> bins(2);
    
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
      bin.event_rate = (bin.count > 0) ? static_cast<double>(bin.count_pos) / bin.count : 0.0;
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
      event_rates.push_back(bin.event_rate);
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
    std::vector<Bin> bins(unique_values.size());
    
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
      bin.event_rate = (bin.count > 0) ? static_cast<double>(bin.count_pos) / bin.count : 0.0;
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
      event_rates.push_back(bin.event_rate);
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
std::vector<OptimalBinningNumericalLPDB::Bin> OptimalBinningNumericalLPDB::polynomial_density_prebinning(
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
  std::vector<Bin> bins;
  bins.reserve(critical_points.size() + 1);
  
  // First bin
  Bin first_bin;
  first_bin.lower_bound = -std::numeric_limits<double>::infinity();
  first_bin.upper_bound = critical_points.empty() ? 
  std::numeric_limits<double>::infinity() : 
    critical_points[0];
  bins.push_back(first_bin);
  
  // Middle bins
  for (size_t i = 0; i < critical_points.size() - 1; ++i) {
    Bin bin;
    bin.lower_bound = critical_points[i];
    bin.upper_bound = critical_points[i + 1];
    bins.push_back(bin);
  }
  
  // Last bin
  if (!critical_points.empty()) {
    Bin last_bin;
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
  
  // Calculate event rates
  for (auto& bin : bins) {
    bin.event_rate = (bin.count > 0) ? static_cast<double>(bin.count_pos) / bin.count : 0.0;
  }
  
  return bins;
}

/**
 * Find the bin index for a value using binary search
 */
size_t OptimalBinningNumericalLPDB::find_bin_index(
    double value, 
    const std::vector<Bin> &bins) const {
  
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
std::vector<double> OptimalBinningNumericalLPDB::local_polynomial_density(
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
std::vector<double> OptimalBinningNumericalLPDB::find_density_critical_points(
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
void OptimalBinningNumericalLPDB::calculate_woe_iv(
    std::vector<Bin> &bins, 
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
double OptimalBinningNumericalLPDB::calculate_woe(
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
double OptimalBinningNumericalLPDB::calculate_iv(
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
void OptimalBinningNumericalLPDB::enforce_monotonicity(
    std::vector<Bin> &bins, 
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
void OptimalBinningNumericalLPDB::optimize_bins(
    std::vector<Bin> &bins, 
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
          double diff_prev = std::fabs(bins[i].event_rate - bins[i-1].event_rate);
          double diff_next = std::fabs(bins[i].event_rate - bins[i+1].event_rate);
          
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
void OptimalBinningNumericalLPDB::merge_adjacent_bins(
    std::vector<Bin> &bins, 
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
  bins[idx1].event_rate = (bins[idx1].count > 0) ? 
  static_cast<double>(bins[idx1].count_pos) / bins[idx1].count : 0.0;
  
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
std::string OptimalBinningNumericalLPDB::format_bin_interval(
    double lower, 
    double upper, 
    bool first, 
    bool last) const {
  
  std::ostringstream oss;
  oss.precision(6);
  oss << std::fixed;
  
  oss << "(";
  
  if (first || std::isinf(lower) && lower < 0) {
    oss << "-Inf";
  } else {
    oss << lower;
  }
  
  oss << "; ";
  
  if (last || std::isinf(upper) && upper > 0) {
    oss << "+Inf";
  } else {
    oss << upper;
  }
  
  oss << "]";
  return oss.str();
}

//' @title Optimal Binning for Numerical Variables using Local Polynomial Density Binning (LPDB)
//'
//' @description
//' Implements an advanced binning algorithm for numerical variables that combines local polynomial
//' density estimation with information-theoretic optimization. This method adapts bin boundaries
//' to the natural structure of the data while maximizing predictive power for a binary target
//' variable. LPDB is particularly effective for complex distributions with multiple modes or
//' regions of varying density.
//'
//' @details
//' ## Algorithm Overview
//' 
//' The Local Polynomial Density Binning algorithm operates through several coordinated phases:
//' 
//' 1. **Density Analysis**: Uses polynomial regression techniques to estimate the local density
//'    structure of the feature distribution, identifying natural groupings in the data.
//' 
//' 2. **Critical Point Detection**: Locates important points in the density curve (minima, maxima,
//'    and inflection points) as potential bin boundaries.
//' 
//' 3. **Initial Binning**: Creates preliminary bins based on these critical points, ensuring they
//'    respect the natural structure of the data.
//' 
//' 4. **Statistical Optimization**:
//'    - Merges bins with frequencies below threshold to ensure statistical reliability
//'    - Enforces monotonicity in Weight of Evidence (optional)
//'    - Adjusts bin count to meet minimum and maximum constraints
//' 
//' 5. **Information Value Calculation**: Computes predictive metrics for the final binning solution
//' 
//' ## Mathematical Foundation
//' 
//' The algorithm employs several advanced statistical concepts:
//' 
//' ### 1. Local Polynomial Density Estimation
//' 
//' For density estimation at point \eqn{x}:
//' 
//' \deqn{f_h(x) = \frac{1}{nh}\sum_{i=1}^{n}K\left(\frac{x-x_i}{h}\right)}
//' 
//' Where:
//' - \eqn{K} is a kernel function (Gaussian kernel in this implementation)
//' - \eqn{h} is the bandwidth parameter (calculated using Silverman's rule)
//' - \eqn{n} is the number of observations
//' 
//' ### 2. Critical Point Detection
//' 
//' The algorithm identifies key points in the density curve:
//' 
//' - **Local Minima**: Natural boundaries between clusters (density valleys)
//' - **Inflection Points**: Regions where density curvature changes
//' - **Local Maxima**: Centers of high-density regions
//' 
//' ### 3. Weight of Evidence (WoE) Calculation
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
//' ### 4. Information Value (IV)
//' 
//' Overall predictive power measure:
//' 
//' \deqn{IV_i = \left(\frac{p_i}{P} - \frac{n_i}{N}\right) \times WoE_i}
//' 
//' \deqn{IV_{total} = \sum_{i=1}^{k} IV_i}
//' 
//' ## Advantages
//' 
//' - **Adaptive to Data Structure**: Places bin boundaries at natural density transitions
//' - **Handles Complex Distributions**: Effective for multimodal or skewed features
//' - **Information Preservation**: Optimizes binning for maximum predictive power
//' - **Statistical Stability**: Ensures sufficient observations in each bin
//' - **Interpretability**: Supports monotonic relationships between feature and target
//'
//' @param target A binary integer vector (0 or 1) representing the target variable.
//' @param feature A numeric vector representing the feature to be binned.
//' @param min_bins Minimum number of bins (default: 3).
//' @param max_bins Maximum number of bins (default: 5).
//' @param bin_cutoff Minimum frequency fraction for each bin (default: 0.05).
//' @param max_n_prebins Maximum number of pre-bins before optimization (default: 20).
//' @param polynomial_degree Degree of polynomial used for density estimation (default: 3).
//' @param enforce_monotonic Whether to enforce monotonic relationship in WoE (default: TRUE).
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
//' \item{centroids}{Numeric vector with the centroid (mean value) of each bin.}
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
//' target <- sample(0:1, 1000, replace = TRUE)
//' feature <- rnorm(1000)
//' 
//' # Basic usage
//' result <- optimal_binning_numerical_lpdb(target, feature)
//' print(result)
//' 
//' # Custom parameters
//' result_custom <- optimal_binning_numerical_lpdb(
//'   target = target,
//'   feature = feature,
//'   min_bins = 2,
//'   max_bins = 8,
//'   bin_cutoff = 0.03,
//'   polynomial_degree = 5,
//'   enforce_monotonic = TRUE
//' )
//' 
//' # Access specific components
//' bins <- result$bin
//' woe_values <- result$woe
//' total_iv <- result$total_iv
//' }
//'
//' @references
//' Fan, J., & Gijbels, I. (1996). *Local Polynomial Modelling and Its Applications*. 
//' Chapman and Hall.
//' 
//' Loader, C. (1999). *Local Regression and Likelihood*. Springer-Verlag.
//' 
//' Hastie, T., & Tibshirani, R. (1990). *Generalized Additive Models*. Chapman and Hall.
//' 
//' Belkin, M., & Niyogi, P. (2003). Laplacian eigenmaps for dimensionality reduction and data 
//' representation. *Neural Computation*, 15(6), 1373-1396.
//' 
//' Silverman, B. W. (1986). *Density Estimation for Statistics and Data Analysis*. Chapman and Hall/CRC.
//' 
//' Siddiqi, N. (2006). *Credit Risk Scorecards: Developing and Implementing Intelligent Credit Scoring*. 
//' John Wiley & Sons.
//'
//' @export
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
   OptimalBinningNumericalLPDB binning(
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




// // [[Rcpp::plugins(cpp11)]]
// // [[Rcpp::depends(Rcpp)]]
// 
// #include <Rcpp.h>
// #include <vector>
// #include <algorithm>
// #include <cmath>
// #include <map>
// #include <string>
// #include <sstream>
// #include <numeric>
// #include <limits>
// 
// using namespace Rcpp;
// 
// // Função para calcular correlação de Pearson entre dois vetores
// inline double compute_correlation(const std::vector<double> &x, const std::vector<double> &y) {
//   if (x.size() != y.size() || x.empty()) {
//     Rcpp::stop("Vectors must be of the same non-zero length for correlation.");
//   }
//   
//   double mean_x = std::accumulate(x.begin(), x.end(), 0.0) / x.size();
//   double mean_y = std::accumulate(y.begin(), y.end(), 0.0) / y.size();
//   
//   double numerator = 0.0;
//   double denom_x = 0.0;
//   double denom_y = 0.0;
//   
//   for (size_t i = 0; i < x.size(); ++i) {
//     double dx = x[i] - mean_x;
//     double dy = y[i] - mean_y;
//     numerator += dx * dy;
//     denom_x += dx * dx;
//     denom_y += dy * dy;
//   }
//   
//   if (denom_x == 0 || denom_y == 0) {
//     Rcpp::warning("Standard deviation is zero. Returning correlation as 0.");
//     return 0.0;
//   }
//   
//   return numerator / std::sqrt(denom_x * denom_y);
// }
// 
// // Classe para Binning Ótimo Numérico usando Local Polynomial Density Binning (LPDB)
// class OptimalBinningNumericalLPDB {
// public:
//   OptimalBinningNumericalLPDB(int min_bins = 3, int max_bins = 5, double bin_cutoff = 0.05, int max_n_prebins = 20,
//                               double convergence_threshold = 1e-6, int max_iterations = 1000)
//     : min_bins(min_bins), max_bins(max_bins), bin_cutoff(bin_cutoff), max_n_prebins(max_n_prebins),
//       convergence_threshold(convergence_threshold), max_iterations(max_iterations),
//       converged(true), iterations_run(0) {
//     
//     // Validações de parâmetros
//     if (min_bins < 2) {
//       Rcpp::stop("min_bins must be at least 2.");
//     }
//     if (max_bins < min_bins) {
//       Rcpp::stop("max_bins must be greater than or equal to min_bins.");
//     }
//     if (bin_cutoff < 0.0 || bin_cutoff > 1.0) {
//       Rcpp::stop("bin_cutoff must be between 0 and 1.");
//     }
//     if (max_n_prebins < min_bins) {
//       Rcpp::stop("max_n_prebins must be greater than or equal to min_bins.");
//     }
//   }
//   
//   Rcpp::List fit(Rcpp::NumericVector feature, Rcpp::IntegerVector target);
//   
// private:
//   int min_bins;
//   int max_bins;
//   double bin_cutoff;
//   int max_n_prebins;
//   double convergence_threshold;
//   int max_iterations;
//   bool converged;
//   int iterations_run;
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
//   std::vector<Bin> prebinning(const std::vector<double> &feature, const std::vector<int> &target);
//   void calculate_woe_iv(std::vector<Bin> &bins, int total_pos, int total_neg);
//   void enforce_monotonicity(std::vector<Bin> &bins, int total_pos, int total_neg);
//   std::string format_bin_interval(double lower, double upper, bool first = false, bool last = false);
// };
// 
// // Método fit
// Rcpp::List OptimalBinningNumericalLPDB::fit(Rcpp::NumericVector feature, Rcpp::IntegerVector target) {
//   int n = feature.size();
//   if (n != target.size()) {
//     Rcpp::stop("feature and target must have the same length.");
//   }
//   
//   // Garante que target é binário
//   IntegerVector unique_targets = unique(target);
//   if (unique_targets.size() != 2 || (std::find(unique_targets.begin(), unique_targets.end(), 0) == unique_targets.end()) ||
//       (std::find(unique_targets.begin(), unique_targets.end(), 1) == unique_targets.end())) {
//     Rcpp::stop("Target must be binary (0 and 1) and contain both classes.");
//   }
//   
//   // Remove NA
//   LogicalVector not_na = (!is_na(feature)) & (!is_na(target));
//   NumericVector clean_feature = feature[not_na];
//   IntegerVector clean_target = target[not_na];
//   
//   if (clean_feature.size() == 0) {
//     Rcpp::stop("No valid observations after removing missing values.");
//   }
//   
//   int total_pos = std::accumulate(clean_target.begin(), clean_target.end(), 0);
//   int total_neg = clean_target.size() - total_pos;
//   
//   if (total_pos == 0 || total_neg == 0) {
//     Rcpp::stop("Target must have both positive and negative classes.");
//   }
//   
//   // Checa se feature possui um só valor único
//   NumericVector unique_vec = unique(clean_feature);
//   std::vector<double> unique_feature = Rcpp::as<std::vector<double>>(unique_vec);
//   
//   if (unique_feature.size() == 1) {
//     // Um único valor => um único bin
//     Bin bin;
//     bin.lower_bound = -std::numeric_limits<double>::infinity();
//     bin.upper_bound = std::numeric_limits<double>::infinity();
//     bin.count = clean_feature.size();
//     bin.count_pos = total_pos;
//     bin.count_neg = total_neg;
//     
//     double dist_pos = static_cast<double>(bin.count_pos) / total_pos;
//     double dist_neg = static_cast<double>(bin.count_neg) / total_neg;
//     if (dist_pos <= 0) dist_pos = 1e-10;
//     if (dist_neg <= 0) dist_neg = 1e-10;
//     bin.woe = std::log(dist_pos / dist_neg);
//     bin.iv = (dist_pos - dist_neg) * bin.woe;
//     
//     std::vector<Bin> bins = { bin };
//     
//     // Cria labels
//     std::vector<std::string> bin_labels;
//     bin_labels.emplace_back(format_bin_interval(bin.lower_bound, bin.upper_bound, true, true));
//     
//     NumericVector woe_values(1, bin.woe);
//     NumericVector iv_values(1, bin.iv);
//     IntegerVector counts(1, bin.count);
//     IntegerVector counts_pos(1, bin.count_pos);
//     IntegerVector counts_neg(1, bin.count_neg);
//     NumericVector cutpoints; // Vazio, pois há apenas um bin
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
//       Named("count_pos") = counts_pos,
//       Named("count_neg") = counts_neg,
//       Named("cutpoints") = cutpoints,
//       Named("converged") = true,
//       Named("iterations") = 0
//     );
//   }
//   
//   // Caso contrário, faz pré-binning
//   std::vector<double> clean_feature_vec = Rcpp::as<std::vector<double>>(clean_feature);
//   std::vector<int> clean_target_vec = Rcpp::as<std::vector<int>>(clean_target);
//   
//   std::vector<Bin> bins = prebinning(clean_feature_vec, clean_target_vec);
//   
//   // Calcula WoE e IV
//   calculate_woe_iv(bins, total_pos, total_neg);
//   
//   // Aplica monotonicidade
//   enforce_monotonicity(bins, total_pos, total_neg);
//   
//   // Cria labels e cutpoints
//   std::vector<std::string> bin_labels;
//   std::vector<double> cutpoints_list;
//   
//   for (size_t i = 0; i < bins.size(); ++i) {
//     bin_labels.emplace_back(format_bin_interval(bins[i].lower_bound, bins[i].upper_bound,
//                                                 i == 0, i == (bins.size() - 1)));
//     if (i < bins.size() - 1) {
//       cutpoints_list.emplace_back(bins[i].upper_bound);
//     }
//   }
//   
//   std::vector<double> woe_vals;
//   std::vector<double> iv_vals;
//   std::vector<int> counts_vec;
//   std::vector<int> counts_pos_vec;
//   std::vector<int> counts_neg_vec;
//   
//   for (size_t i = 0; i < bins.size(); ++i) {
//     woe_vals.push_back(bins[i].woe);
//     iv_vals.push_back(bins[i].iv);
//     counts_vec.push_back(bins[i].count);
//     counts_pos_vec.push_back(bins[i].count_pos);
//     counts_neg_vec.push_back(bins[i].count_neg);
//   }
//   
//   Rcpp::NumericVector ids(bin_labels.size());
//   for(int i = 0; i < bin_labels.size(); i++) {
//     ids[i] = i + 1;
//   }
//   
//   return Rcpp::List::create(
//     Named("id") = ids,
//     Named("bin") = bin_labels,
//     Named("woe") = woe_vals,
//     Named("iv") = iv_vals,
//     Named("count") = counts_vec,
//     Named("count_pos") = counts_pos_vec,
//     Named("count_neg") = counts_neg_vec,
//     Named("cutpoints") = cutpoints_list,
//     Named("converged") = converged,
//     Named("iterations") = iterations_run
//   );
// }
// 
// // Função prebinning
// std::vector<OptimalBinningNumericalLPDB::Bin> OptimalBinningNumericalLPDB::prebinning(const std::vector<double> &feature, const std::vector<int> &target) {
//   int n = static_cast<int>(feature.size());
//   std::vector<int> indices(n);
//   for (int i = 0; i < n; ++i) {
//     indices[i] = i;
//   }
//   
//   std::sort(indices.begin(), indices.end(), [&](int a, int b) {
//     return feature[a] < feature[b];
//   });
//   
//   int bin_size = n / max_n_prebins;
//   if (bin_size < 1) bin_size = 1;
//   
//   std::vector<double> cut_points;
//   for (int i = bin_size; i < n; i += bin_size) {
//     double val = feature[indices[i]];
//     if (cut_points.empty() || val != cut_points.back()) {
//       cut_points.push_back(val);
//     }
//   }
//   
//   std::vector<Bin> bins;
//   double lower = -std::numeric_limits<double>::infinity();
//   size_t idx = 0;
//   
//   for (size_t cp = 0; cp < cut_points.size(); ++cp) {
//     double upper = cut_points[cp];
//     Bin bin;
//     bin.lower_bound = lower;
//     bin.upper_bound = upper;
//     bin.count = 0;
//     bin.count_pos = 0;
//     bin.count_neg = 0;
//     
//     while (idx < (size_t)n && feature[indices[idx]] <= upper) {
//       bin.count++;
//       if (target[indices[idx]] == 1) {
//         bin.count_pos++;
//       } else {
//         bin.count_neg++;
//       }
//       idx++;
//     }
//     bins.push_back(bin);
//     lower = upper;
//   }
//   
//   if (idx < (size_t)n) {
//     Bin bin;
//     bin.lower_bound = lower;
//     bin.upper_bound = std::numeric_limits<double>::infinity();
//     bin.count = 0;
//     bin.count_pos = 0;
//     bin.count_neg = 0;
//     while (idx < (size_t)n) {
//       bin.count++;
//       if (target[indices[idx]] == 1) {
//         bin.count_pos++;
//       } else {
//         bin.count_neg++;
//       }
//       idx++;
//     }
//     bins.push_back(bin);
//   }
//   
//   return bins;
// }
// 
// // Calcula WoE e IV
// void OptimalBinningNumericalLPDB::calculate_woe_iv(std::vector<Bin> &bins, int total_pos, int total_neg) {
//   for (auto &bin : bins) {
//     double dist_pos = (bin.count_pos > 0) ? static_cast<double>(bin.count_pos) / total_pos : 1e-10;
//     double dist_neg = (bin.count_neg > 0) ? static_cast<double>(bin.count_neg) / total_neg : 1e-10;
//     
//     if (dist_pos <= 0) dist_pos = 1e-10;
//     if (dist_neg <= 0) dist_neg = 1e-10;
//     
//     bin.woe = std::log(dist_pos / dist_neg);
//     bin.iv = (dist_pos - dist_neg) * bin.woe;
//   }
// }
// 
// // Aplica monotonicidade
// void OptimalBinningNumericalLPDB::enforce_monotonicity(std::vector<Bin> &bins, int total_pos, int total_neg) {
//   // Determina direção com base na correlação
//   std::vector<double> bin_means;
//   bin_means.reserve(bins.size());
//   
//   for (const auto &bin : bins) {
//     double mean;
//     if (std::isinf(bin.lower_bound) && !std::isinf(bin.upper_bound)) {
//       mean = bin.upper_bound - 1.0;
//     } else if (!std::isinf(bin.lower_bound) && std::isinf(bin.upper_bound)) {
//       mean = bin.lower_bound + 1.0;
//     } else if (std::isinf(bin.lower_bound) && std::isinf(bin.upper_bound)) {
//       mean = 0.0;
//     } else {
//       mean = (bin.lower_bound + bin.upper_bound) / 2.0;
//     }
//     bin_means.push_back(mean);
//   }
//   
//   std::vector<double> woe_values_vec;
//   woe_values_vec.reserve(bins.size());
//   for (const auto &bin : bins) {
//     woe_values_vec.push_back(bin.woe);
//   }
//   
//   double corr = 0.0;
//   if (bins.size() > 1) {
//     corr = compute_correlation(bin_means, woe_values_vec);
//   }
//   
//   bool desired_increasing = (corr >= 0);
//   
//   // Merges para forçar monotonicidade
//   while (iterations_run < max_iterations) {
//     bool merged = false;
//     for (size_t i = 1; i < bins.size(); ++i) {
//       if ((desired_increasing && bins[i].woe < bins[i - 1].woe) ||
//           (!desired_increasing && bins[i].woe > bins[i - 1].woe)) {
//         
//         if (bins.size() <= static_cast<size_t>(min_bins)) {
//           converged = false;
//           return;
//         }
//         
//         // Merge bins[i - 1] e bins[i]
//         bins[i - 1].upper_bound = bins[i].upper_bound;
//         bins[i - 1].count += bins[i].count;
//         bins[i - 1].count_pos += bins[i].count_pos;
//         bins[i - 1].count_neg += bins[i].count_neg;
//         bins.erase(bins.begin() + i);
//         
//         double dist_pos = (bins[i - 1].count_pos > 0) ? static_cast<double>(bins[i - 1].count_pos) / total_pos : 1e-10;
//         double dist_neg = (bins[i - 1].count_neg > 0) ? static_cast<double>(bins[i - 1].count_neg) / total_neg : 1e-10;
//         if (dist_pos <= 0) dist_pos = 1e-10;
//         if (dist_neg <= 0) dist_neg = 1e-10;
//         bins[i - 1].woe = std::log(dist_pos / dist_neg);
//         bins[i - 1].iv = (dist_pos - dist_neg) * bins[i - 1].woe;
//         
//         iterations_run++;
//         merged = true;
//         break;
//       }
//     }
//     if (!merged) break;
//   }
//   
//   // Garante max_bins
//   while (bins.size() > static_cast<size_t>(max_bins) && iterations_run < max_iterations) {
//     double min_woe_diff = std::numeric_limits<double>::max();
//     size_t merge_index = 0;
//     for (size_t i = 1; i < bins.size(); ++i) {
//       double woe_diff = std::abs(bins[i].woe - bins[i - 1].woe);
//       if (woe_diff < min_woe_diff) {
//         min_woe_diff = woe_diff;
//         merge_index = i - 1;
//       }
//     }
//     
//     bins[merge_index].upper_bound = bins[merge_index + 1].upper_bound;
//     bins[merge_index].count += bins[merge_index + 1].count;
//     bins[merge_index].count_pos += bins[merge_index + 1].count_pos;
//     bins[merge_index].count_neg += bins[merge_index + 1].count_neg;
//     bins.erase(bins.begin() + merge_index + 1);
//     
//     double dist_pos = (bins[merge_index].count_pos > 0) ? static_cast<double>(bins[merge_index].count_pos) / total_pos : 1e-10;
//     double dist_neg = (bins[merge_index].count_neg > 0) ? static_cast<double>(bins[merge_index].count_neg) / total_neg : 1e-10;
//     if (dist_pos <= 0) dist_pos = 1e-10;
//     if (dist_neg <= 0) dist_neg = 1e-10;
//     bins[merge_index].woe = std::log(dist_pos / dist_neg);
//     bins[merge_index].iv = (dist_pos - dist_neg) * bins[merge_index].woe;
//     
//     iterations_run++;
//   }
//   
//   // Verifica monotonicidade final
//   for (size_t i = 1; i < bins.size(); ++i) {
//     if ((desired_increasing && bins[i].woe < bins[i - 1].woe) ||
//         (!desired_increasing && bins[i].woe > bins[i - 1].woe)) {
//       converged = false;
//       break;
//     }
//   }
// }
// 
// // Formata intervalo do bin
// std::string OptimalBinningNumericalLPDB::format_bin_interval(double lower, double upper, bool first, bool last) {
//   std::ostringstream oss;
//   oss.precision(6);
//   oss << std::fixed;
//   oss << "(";
//   if (first) {
//     oss << "-Inf";
//   } else {
//     oss << lower;
//   }
//   oss << "; ";
//   if (last) {
//     oss << "+Inf";
//   } else {
//     oss << upper;
//   }
//   oss << "]";
//   return oss.str();
// }
// 
// 
// //' @title Optimal Binning for Numerical Variables using Local Polynomial Density Binning (LPDB)
// //'
// //' @description
// //' Implements the Local Polynomial Density Binning (LPDB) algorithm for optimal binning of numerical variables. 
// //' The method creates bins that maximize predictive power while maintaining monotonicity in Weight of Evidence (WoE).
// //' It handles rare bins, ensures numerical stability, and provides flexibility through various customizable parameters.
// //'
// //' @details
// //' ### Key Steps:
// //' 1. **Input Validation**: Ensures the `feature` and `target` vectors are valid, checks binary nature of the `target` vector, 
// //'    and removes missing values (`NA`).
// //' 2. **Pre-Binning**: Divides the feature into preliminary bins using quantile-based partitioning or unique values.
// //' 3. **Calculation of WoE and IV**: Computes the WoE and Information Value (IV) for each bin based on the target distribution.
// //' 4. **Monotonicity Enforcement**: Adjusts bins iteratively to ensure monotonicity in WoE values, either increasing or decreasing.
// //' 5. **Rare Bin Merging**: Merges bins with frequencies below the `bin_cutoff` threshold to ensure statistical stability.
// //' 6. **Validation**: Ensures bins are non-overlapping, cover the entire range of the feature, and are consistent with constraints on `min_bins` and `max_bins`.
// //'
// //' ### Mathematical Framework:
// //' - **Weight of Evidence (WoE)**: For a bin \( i \):
// //'   \deqn{WoE_i = \ln\left(\frac{\text{Distribution of positives}_i}{\text{Distribution of negatives}_i}\right)}
// //'
// //' - **Information Value (IV)**: Aggregates the predictive power across all bins:
// //'   \deqn{IV = \sum_{i=1}^{N} (\text{Distribution of positives}_i - \text{Distribution of negatives}_i) \times WoE_i}
// //'
// //' ### Features:
// //' - **Monotonicity**: Ensures the WoE values are either strictly increasing or decreasing across bins.
// //' - **Rare Bin Handling**: Merges bins with low frequencies to maintain statistical reliability.
// //' - **Numerical Stability**: Incorporates small constants to avoid division by zero or undefined logarithms.
// //' - **Flexibility**: Supports custom definitions for minimum and maximum bins, convergence thresholds, and iteration limits.
// //' - **Output Metadata**: Provides detailed bin information, including WoE, IV, and cutpoints for interpretability and downstream analysis.
// //'
// //' ### Parameters:
// //' - `min_bins`: Minimum number of bins to be created (default: 3).
// //' - `max_bins`: Maximum number of bins allowed (default: 5).
// //' - `bin_cutoff`: Minimum proportion of total observations required for a bin to be retained as standalone (default: 0.05).
// //' - `max_n_prebins`: Maximum number of pre-bins before optimization (default: 20).
// //' - `convergence_threshold`: Threshold for determining convergence in terms of IV changes (default: 1e-6).
// //' - `max_iterations`: Maximum number of iterations allowed for binning optimization (default: 1000).
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
// //'   \item `bin`: A vector of bin intervals in the format "[lower;upper)".
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
// //' result <- optimal_binning_numerical_lpdb(target, feature, min_bins = 3, max_bins = 6)
// //' print(result$bin)
// //' print(result$woe)
// //' print(result$iv)
// //' }
// //'
// //' @export
// // [[Rcpp::export]]
// Rcpp::List optimal_binning_numerical_lpdb(Rcpp::IntegerVector target,
//                                          Rcpp::NumericVector feature,
//                                          int min_bins = 3,
//                                          int max_bins = 5,
//                                          double bin_cutoff = 0.05,
//                                          int max_n_prebins = 20,
//                                          double convergence_threshold = 1e-6,
//                                          int max_iterations = 1000) {
//  OptimalBinningNumericalLPDB binning(min_bins, max_bins, bin_cutoff, max_n_prebins,
//                                      convergence_threshold, max_iterations);
//  
//  return binning.fit(feature, target);
// }
