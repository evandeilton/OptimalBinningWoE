// [[Rcpp::plugins(cpp11)]]
// [[Rcpp::depends(Rcpp)]]

#include <Rcpp.h>
#include <vector>
#include <algorithm>
#include <cmath>
#include <limits>
#include <sstream>
#include <iomanip>
#include <unordered_set>
#include <numeric>
#include <functional>

/**
 * @file OptimalBinningNumericalEWB.cpp
 * @brief Implementation of Optimal Binning for numerical variables using Equal-Width Binning
 * 
 * This implementation provides methods for supervised discretization of numerical variables
 * using equal-width binning with subsequent optimization for predictive modeling.
 */

using namespace Rcpp;

// Class for Optimal Binning using Equal-Width Binning
class OptimalBinningNumericalEWB {
private:
  std::vector<double> feature;
  std::vector<int> target;
  int min_bins;
  int max_bins;
  double bin_cutoff;
  int max_n_prebins;
  double convergence_threshold;
  int max_iterations;
  bool is_monotonic;
  
  bool converged;
  int iterations_run;
  
  // Structure representing a bin with its statistical properties
  struct Bin {
    double lower;      // Lower bound of the bin (inclusive)
    double upper;      // Upper bound of the bin (inclusive)
    int count;         // Total observations in the bin
    int count_pos;     // Positive class observations
    int count_neg;     // Negative class observations
    double woe;        // Weight of Evidence
    double iv;         // Information Value contribution
    
    Bin(double lb = -std::numeric_limits<double>::infinity(),
        double ub = std::numeric_limits<double>::infinity(),
        int c = 0, int cp = 0, int cn = 0)
      : lower(lb), upper(ub), count(c), count_pos(cp), count_neg(cn), woe(0.0), iv(0.0) {}
  };
  
  std::vector<Bin> bins;
  
  // Total statistics
  int total_pos;        // Total positive observations
  int total_neg;        // Total negative observations
  int unique_count;     // Number of unique feature values
  
  // Constants
  static constexpr double EPSILON = 1e-10;
  static constexpr double LAPLACE_SMOOTHING = 0.5;
  
  /**
   * Convert double to formatted string with proper handling of infinity
   * @param value Double value to convert
   * @return Formatted string representation
   */
  std::string double_to_string(double value) const {
    if (std::isinf(value)) {
      return value > 0 ? "+Inf" : "-Inf";
    }
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(6) << value;
    return ss.str();
  }
  
  /**
   * Calculate Weight of Evidence for a bin
   * @param pos Count of positive class observations
   * @param neg Count of negative class observations
   * @return Weight of Evidence value
   */
  double calculate_woe(int pos, int neg) const {
    // Apply Laplace smoothing to avoid division by zero
    double pos_rate = (pos + LAPLACE_SMOOTHING) / (total_pos + bins.size() * LAPLACE_SMOOTHING);
    double neg_rate = (neg + LAPLACE_SMOOTHING) / (total_neg + bins.size() * LAPLACE_SMOOTHING);
    
    // WoE = ln(% of positive / % of negative)
    return std::log(pos_rate / neg_rate);
  }
  
  /**
   * Calculate Information Value contribution for a bin
   * @param woe Weight of Evidence value
   * @param pos Count of positive class observations
   * @param neg Count of negative class observations
   * @return Information Value
   */
  double calculate_iv(double woe, int pos, int neg) const {
    // Apply smoothing to distribution rates
    double p_rate = (pos + LAPLACE_SMOOTHING) / (total_pos + bins.size() * LAPLACE_SMOOTHING);
    double n_rate = (neg + LAPLACE_SMOOTHING) / (total_neg + bins.size() * LAPLACE_SMOOTHING);
    
    // IV = (% of positive - % of negative) * WoE
    return (p_rate - n_rate) * woe;
  }
  
  /**
   * Validate input parameters and data
   * Throws an exception if inputs are invalid
   */
  void validate_inputs() {
    if (feature.empty()) {
      Rcpp::stop("Feature vector is empty.");
    }
    
    if (feature.size() != target.size()) {
      Rcpp::stop("Feature and target vectors must have the same length.");
    }
    
    for (const auto& t : target) {
      if (t != 0 && t != 1) {
        Rcpp::stop("Target vector must contain only 0 and 1.");
      }
    }
    
    // Handle NaN values by filtering them out for unique count
    std::vector<double> clean_feature;
    for (const auto& val : feature) {
      if (!std::isnan(val)) {
        clean_feature.push_back(val);
      }
    }
    
    if (clean_feature.empty()) {
      Rcpp::stop("Feature vector contains only NaN values.");
    }
    
    std::unordered_set<double> unique_values(clean_feature.begin(), clean_feature.end());
    unique_count = static_cast<int>(unique_values.size());
    
    if (unique_count <= 1) {
      Rcpp::stop("Feature vector must contain at least two unique non-NaN values.");
    }
    
    if (min_bins < 2) {
      Rcpp::stop("min_bins must be at least 2.");
    }
    
    if (max_bins < min_bins) {
      Rcpp::stop("max_bins must be greater than or equal to min_bins.");
    }
    
    if (bin_cutoff <= 0 || bin_cutoff >= 1) {
      Rcpp::stop("bin_cutoff must be between 0 and 1.");
    }
    
    if (max_n_prebins <= 0) {
      Rcpp::stop("max_n_prebins must be positive.");
    }
    
    if (max_iterations <= 0) {
      Rcpp::stop("max_iterations must be positive.");
    }
  }
  
  /**
   * Create bins for the case of few unique values
   * This is a special case where we create one bin per unique value
   */
  void create_unique_bins() {
    bins.clear();
    
    // Filter out NaN values
    std::vector<double> clean_feature;
    std::vector<int> clean_target;
    
    for (size_t i = 0; i < feature.size(); ++i) {
      if (!std::isnan(feature[i])) {
        clean_feature.push_back(feature[i]);
        clean_target.push_back(target[i]);
      }
    }
    
    // Get unique values and sort them
    std::vector<double> unique_values = clean_feature;
    std::sort(unique_values.begin(), unique_values.end());
    unique_values.erase(std::unique(unique_values.begin(), unique_values.end()), unique_values.end());
    
    // Create initial bins structure
    for (size_t i = 0; i < unique_values.size(); ++i) {
      double lower = (i == 0) ? -std::numeric_limits<double>::infinity() : unique_values[i-1];
      double upper = unique_values[i];
      bins.emplace_back(lower, upper, 0, 0, 0);
    }
    
    // Assign observations to bins
    for (size_t i = 0; i < clean_feature.size(); ++i) {
      double value = clean_feature[i];
      int target_value = clean_target[i];
      
      // Find appropriate bin using binary search
      auto it = std::upper_bound(unique_values.begin(), unique_values.end(), value);
      size_t bin_idx = std::distance(unique_values.begin(), it);
      
      if (bin_idx > 0) { // Adjust index since we want the bin where value falls
        bin_idx -= 1;
      }
      
      // Update bin counts
      bins[bin_idx].count++;
      if (target_value == 1) {
        bins[bin_idx].count_pos++;
      } else {
        bins[bin_idx].count_neg++;
      }
    }
  }
  
  /**
   * Create initial pre-bins using equal width strategy
   * This divides the feature range into equal intervals
   */
  void create_prebins() {
    // Filter out NaN values
    std::vector<double> clean_feature;
    for (const auto& val : feature) {
      if (!std::isnan(val)) {
        clean_feature.push_back(val);
      }
    }
    
    if (clean_feature.empty()) {
      Rcpp::stop("No valid non-NaN values in feature.");
    }
    
    double min_value = *std::min_element(clean_feature.begin(), clean_feature.end());
    double max_value = *std::max_element(clean_feature.begin(), clean_feature.end());
    
    // Handle the special case where min and max are very close
    if (std::fabs(min_value - max_value) < EPSILON) {
      int total_c = static_cast<int>(clean_feature.size());
      int pos_c = 0;
      for (size_t i = 0; i < feature.size(); ++i) {
        if (!std::isnan(feature[i]) && target[i] == 1) {
          pos_c++;
        }
      }
      int neg_c = total_c - pos_c;
      
      bins.clear();
      bins.emplace_back(min_value - EPSILON, max_value + EPSILON, total_c, pos_c, neg_c);
      return;
    }
    
    // Determine optimal number of prebins based on unique count and max_n_prebins
    int n_prebins = std::min({max_n_prebins, unique_count, static_cast<int>(clean_feature.size())});
    n_prebins = std::max(n_prebins, min_bins); // Ensure we have at least min_bins
    
    // Calculate bin width
    double range = max_value - min_value;
    if (range < EPSILON) {
      range = EPSILON;
    }
    double bin_width = range / n_prebins;
    
    // Create bins with equal width
    bins.clear();
    bins.reserve(static_cast<size_t>(n_prebins));
    
    for (int i = 0; i < n_prebins; ++i) {
      double lower = (i == 0) ? 
      -std::numeric_limits<double>::infinity() : 
      min_value + i * bin_width;
      double upper = (i == n_prebins - 1) ? 
      std::numeric_limits<double>::infinity() : 
        min_value + (i + 1) * bin_width;
      
      // Ensure upper >= lower (could happen due to floating point precision)
      if (upper < lower) {
        upper = lower;
      }
      
      bins.emplace_back(lower, upper, 0, 0, 0);
    }
  }
  
  /**
   * Assign data observations to appropriate bins
   * Uses binary search for efficiency
   */
  void assign_data_to_bins() {
    // Prepare vector of upper bounds for binary search
    std::vector<double> upper_bounds;
    upper_bounds.reserve(bins.size());
    for (const auto& bin : bins) {
      upper_bounds.push_back(bin.upper);
    }
    
    // Assign each observation to a bin
    for (size_t i = 0; i < feature.size(); ++i) {
      // Skip NaN values
      if (std::isnan(feature[i])) continue;
      
      double value = feature[i];
      int target_value = target[i];
      
      // Binary search to find the appropriate bin
      auto it = std::upper_bound(upper_bounds.begin(), upper_bounds.end(), value);
      size_t bin_idx = (it == upper_bounds.end()) ? 
      bins.size() - 1 : 
        std::distance(upper_bounds.begin(), it);
      
      // Verify bin boundaries (should always be valid with proper bin creation)
      if (bin_idx >= bins.size() || value < bins[bin_idx].lower) {
        // Handle edge cases
        if (value < bins.front().lower) {
          bin_idx = 0;
        } else if (value > bins.back().upper) {
          bin_idx = bins.size() - 1;
        } else {
          // Fallback to linear search if binary search has issues
          bin_idx = 0;
          for (size_t b = 0; b < bins.size(); ++b) {
            if (value >= bins[b].lower && value <= bins[b].upper) {
              bin_idx = b;
              break;
            }
          }
        }
      }
      
      // Update bin statistics
      bins[bin_idx].count++;
      if (target_value == 1) {
        bins[bin_idx].count_pos++;
      } else {
        bins[bin_idx].count_neg++;
      }
    }
  }
  
  /**
   * Merge bins with frequencies below the specified cutoff
   * Ensures statistical reliability of each bin
   */
  void merge_rare_bins() {
    int total_count = std::accumulate(bins.begin(), bins.end(), 0,
                                      [](int sum, const Bin& bin) { return sum + bin.count; });
    double cutoff_count = bin_cutoff * total_count;
    int iterations = 0;
    
    while (iterations < max_iterations) {
      // Stop if we've reached minimum bins
      if (static_cast<int>(bins.size()) <= min_bins) {
        break;
      }
      
      bool merged = false;
      
      // Find and merge rare bins
      for (size_t i = 0; i < bins.size(); ++i) {
        if (bins[i].count < cutoff_count && static_cast<int>(bins.size()) > min_bins) {
          // Strategy: prefer to merge with adjacent bin having similar class distribution
          if (i == 0 && bins.size() > 1) {
            // If first bin, merge with next
            bins[0].upper = bins[1].upper;
            bins[0].count += bins[1].count;
            bins[0].count_pos += bins[1].count_pos;
            bins[0].count_neg += bins[1].count_neg;
            bins.erase(bins.begin() + 1);
          } else if (i == bins.size() - 1) {
            // If last bin, merge with previous
            bins[i-1].upper = bins[i].upper;
            bins[i-1].count += bins[i].count;
            bins[i-1].count_pos += bins[i].count_pos;
            bins[i-1].count_neg += bins[i].count_neg;
            bins.erase(bins.begin() + i);
          } else {
            // For middle bins, choose merge direction based on similarity of distributions
            double ratio_curr = (bins[i].count_pos > 0) ? 
            static_cast<double>(bins[i].count_pos) / bins[i].count : 0.0;
            double ratio_prev = (bins[i-1].count_pos > 0) ? 
            static_cast<double>(bins[i-1].count_pos) / bins[i-1].count : 0.0;
            double ratio_next = (bins[i+1].count_pos > 0) ? 
            static_cast<double>(bins[i+1].count_pos) / bins[i+1].count : 0.0;
            
            // Merge with more similar bin
            if (std::fabs(ratio_curr - ratio_prev) <= std::fabs(ratio_curr - ratio_next)) {
              // Merge with previous
              bins[i-1].upper = bins[i].upper;
              bins[i-1].count += bins[i].count;
              bins[i-1].count_pos += bins[i].count_pos;
              bins[i-1].count_neg += bins[i].count_neg;
              bins.erase(bins.begin() + i);
            } else {
              // Merge with next
              bins[i].upper = bins[i+1].upper;
              bins[i].count += bins[i+1].count;
              bins[i].count_pos += bins[i+1].count_pos;
              bins[i].count_neg += bins[i+1].count_neg;
              bins.erase(bins.begin() + (i+1));
            }
          }
          
          merged = true;
          break;
        }
      }
      
      iterations++;
      if (!merged) {
        break;
      }
    }
    
    iterations_run += iterations;
    if (iterations >= max_iterations) {
      converged = false;
    }
  }
  
  /**
   * Ensure number of bins doesn't exceed max_bins by merging
   * bins with the smallest impact on information value
   */
  void ensure_max_bins() {
    int iterations = 0;
    
    // Merge until bins.size() <= max_bins
    while (static_cast<int>(bins.size()) > max_bins && iterations < max_iterations) {
      // Calculate WoE and IV for current bins to make informed merging decisions
      calculate_woe_iv();
      
      if (bins.size() <= 1) break;
      
      // Find pair of adjacent bins with smallest combined IV
      size_t min_index = 0;
      double min_iv_loss = std::numeric_limits<double>::max();
      
      for (size_t i = 0; i < bins.size() - 1; ++i) {
        // Estimate IV loss if bins i and i+1 are merged
        Bin merged_bin = bins[i];
        merged_bin.upper = bins[i+1].upper;
        merged_bin.count += bins[i+1].count;
        merged_bin.count_pos += bins[i+1].count_pos;
        merged_bin.count_neg += bins[i+1].count_neg;
        
        double merged_woe = calculate_woe(merged_bin.count_pos, merged_bin.count_neg);
        double merged_iv = calculate_iv(merged_woe, merged_bin.count_pos, merged_bin.count_neg);
        
        // IV loss = sum of individual IVs - merged IV
        double iv_loss = (bins[i].iv + bins[i+1].iv) - merged_iv;
        
        if (iv_loss < min_iv_loss) {
          min_iv_loss = iv_loss;
          min_index = i;
        }
      }
      
      // Merge bins[min_index] and bins[min_index + 1]
      bins[min_index].upper = bins[min_index + 1].upper;
      bins[min_index].count += bins[min_index + 1].count;
      bins[min_index].count_pos += bins[min_index + 1].count_pos;
      bins[min_index].count_neg += bins[min_index + 1].count_neg;
      bins.erase(bins.begin() + (min_index + 1));
      
      iterations++;
    }
    
    iterations_run += iterations;
    if (iterations >= max_iterations) {
      converged = false;
    }
  }
  
  /**
   * Enforce monotonicity in Weight of Evidence across bins
   * This improves interpretability and model stability
   */
  void enforce_monotonicity() {
    if (!is_monotonic || bins.size() <= 2) {
      return;
    }
    
    // Calculate WoE for current bins
    calculate_woe_iv();
    
    int iterations = 0;
    bool is_monotonic_bins = false;
    
    // Determine preferred monotonicity direction (increasing or decreasing)
    // Based on the first two bins
    bool prefer_increasing = true;
    if (bins.size() >= 2) {
      prefer_increasing = (bins[1].woe >= bins[0].woe);
    }
    
    // Continue merging until monotonicity is achieved or min_bins reached
    while (!is_monotonic_bins && static_cast<int>(bins.size()) > min_bins && iterations < max_iterations) {
      is_monotonic_bins = true;
      
      for (size_t i = 1; i < bins.size(); ++i) {
        // Check for monotonicity violation
        if ((prefer_increasing && bins[i].woe < bins[i-1].woe - EPSILON) ||
            (!prefer_increasing && bins[i].woe > bins[i-1].woe + EPSILON)) {
          
          // IV-preserving merge strategy
          double iv_loss_prev = bins[i-1].iv;
          double iv_loss_next = bins[i].iv;
          
          // Estimate merged IVs
          Bin merged_bin = bins[i-1];
          merged_bin.upper = bins[i].upper;
          merged_bin.count += bins[i].count;
          merged_bin.count_pos += bins[i].count_pos;
          merged_bin.count_neg += bins[i].count_neg;
          
          double merged_woe = calculate_woe(merged_bin.count_pos, merged_bin.count_neg);
          double merged_iv = calculate_iv(merged_woe, merged_bin.count_pos, merged_bin.count_neg);
          
          double iv_loss = (bins[i-1].iv + bins[i].iv) - merged_iv;
          
          // Perform the merge
          bins[i-1].upper = bins[i].upper;
          bins[i-1].count += bins[i].count;
          bins[i-1].count_pos += bins[i].count_pos;
          bins[i-1].count_neg += bins[i].count_neg;
          bins.erase(bins.begin() + i);
          
          // Recalculate WoE and IV
          calculate_woe_iv();
          
          // Mark as non-monotonic to continue checking
          is_monotonic_bins = false;
          break;
        }
      }
      
      iterations++;
      
      // Stop if minimum bins reached
      if (static_cast<int>(bins.size()) <= min_bins) {
        break;
      }
    }
    
    iterations_run += iterations;
    if (iterations >= max_iterations) {
      converged = false;
    }
  }
  
  /**
   * Calculate Weight of Evidence and Information Value for all bins
   */
  void calculate_woe_iv() {
    for (auto& bin : bins) {
      bin.woe = calculate_woe(bin.count_pos, bin.count_neg);
      bin.iv = calculate_iv(bin.woe, bin.count_pos, bin.count_neg);
    }
  }
  
public:
  /**
   * Constructor for OptimalBinningNumericalEWB
   * 
   * @param feature_ Numeric vector with feature values to be binned
   * @param target_ Binary vector (0/1) with target values
   * @param min_bins_ Minimum number of bins
   * @param max_bins_ Maximum number of bins
   * @param bin_cutoff_ Minimum frequency fraction for each bin
   * @param max_n_prebins_ Maximum number of pre-bins before optimization
   * @param is_monotonic_ Whether to enforce monotonicity in WoE
   * @param convergence_threshold_ Convergence threshold for algorithm
   * @param max_iterations_ Maximum iterations for optimization
   */
  OptimalBinningNumericalEWB(
    const std::vector<double>& feature_, 
    const std::vector<int>& target_,
    int min_bins_ = 3, 
    int max_bins_ = 5, 
    double bin_cutoff_ = 0.05, 
    int max_n_prebins_ = 20,
    bool is_monotonic_ = true,
    double convergence_threshold_ = 1e-6, 
    int max_iterations_ = 1000)
    : feature(feature_), target(target_), 
      min_bins(min_bins_), max_bins(max_bins_),
      bin_cutoff(bin_cutoff_), max_n_prebins(max_n_prebins_),
      convergence_threshold(convergence_threshold_), 
      max_iterations(max_iterations_),
      is_monotonic(is_monotonic_),
      converged(true), iterations_run(0), 
      total_pos(0), total_neg(0), unique_count(0) {}
  
  /**
   * Execute the binning algorithm
   */
  void fit() {
    // Step 1: Validate inputs
    validate_inputs();
    
    // Step 2: Compute class totals
    total_pos = std::accumulate(target.begin(), target.end(), 0);
    total_neg = static_cast<int>(target.size()) - total_pos;
    
    if (total_pos == 0 || total_neg == 0) {
      Rcpp::stop("Target must contain at least one positive and one negative case.");
    }
    
    // Step 3: Pre-binning strategy selection
    if (unique_count <= 2) {
      // For very few unique values, create bins directly
      create_unique_bins();
      calculate_woe_iv();
      converged = true;
      iterations_run = 0;
      return;
    }
    
    // Step 4: Create initial equal-width bins
    create_prebins();
    
    // Step 5: Assign data points to bins
    assign_data_to_bins();
    
    // Step 6: Merge rare bins for statistical stability
    merge_rare_bins();
    
    // Step 7: Calculate initial WoE/IV metrics
    calculate_woe_iv();
    
    // Step 8: Enforce monotonicity if requested
    if (is_monotonic) {
      enforce_monotonicity();
    }
    
    // Step 9: Ensure max_bins constraint
    ensure_max_bins();
    
    // Step 10: Final calculation of metrics
    calculate_woe_iv();
  }
  
  /**
   * Get results of the binning process
   * @return List with bin information and metrics
   */
  List get_results() const {
    std::vector<std::string> bin_labels;
    std::vector<double> woe_values;
    std::vector<double> iv_values;
    std::vector<int> counts;
    std::vector<int> counts_pos;
    std::vector<int> counts_neg;
    std::vector<double> cutpoints;
    
    bin_labels.reserve(bins.size());
    woe_values.reserve(bins.size());
    iv_values.reserve(bins.size());
    counts.reserve(bins.size());
    counts_pos.reserve(bins.size());
    counts_neg.reserve(bins.size());
    cutpoints.reserve(bins.size() > 0 ? bins.size() - 1 : 0);
    
    double total_iv = 0.0;
    
    for (size_t i = 0; i < bins.size(); ++i) {
      const auto& bin = bins[i];
      std::string label = "(" + double_to_string(bin.lower) + ";" + double_to_string(bin.upper) + "]";
      bin_labels.push_back(label);
      woe_values.push_back(bin.woe);
      iv_values.push_back(bin.iv);
      counts.push_back(bin.count);
      counts_pos.push_back(bin.count_pos);
      counts_neg.push_back(bin.count_neg);
      total_iv += bin.iv;
      
      // Cutpoints exclude last bin upper
      if (i < bins.size() - 1) {
        cutpoints.push_back(bin.upper);
      }
    }
    
    // Create bin IDs (1-based indexing for R)
    Rcpp::NumericVector ids(bin_labels.size());
    for(int i = 0; i < bin_labels.size(); i++) {
      ids[i] = i + 1;
    }
    
    return Rcpp::List::create(
      Named("id") = ids,
      Named("bin") = bin_labels,
      Named("woe") = woe_values,
      Named("iv") = iv_values,
      Named("count") = counts,
      Named("count_pos") = counts_pos,
      Named("count_neg") = counts_neg,
      Named("cutpoints") = cutpoints,
      Named("converged") = converged,
      Named("iterations") = iterations_run,
      Named("total_iv") = total_iv
    );
  }
};

//' @title Optimal Binning for Numerical Variables using Equal-Width Binning
//'
//' @description
//' Performs optimal binning for numerical variables using equal-width intervals as a starting point, 
//' followed by a suite of optimization steps. This method balances predictive power and interpretability
//' by creating statistically stable bins with a strong relationship to the target variable.
//' The algorithm is particularly useful for risk modeling, credit scoring, and feature engineering in 
//' classification tasks.
//'
//' @param target Integer binary vector (0 or 1) representing the target variable.
//' @param feature Numeric vector with the values of the feature to be binned.
//' @param min_bins Minimum number of bins (default: 3).
//' @param max_bins Maximum number of bins (default: 5).
//' @param bin_cutoff Minimum fraction of observations each bin must contain (default: 0.05).
//' @param max_n_prebins Maximum number of pre-bins before optimization (default: 20).
//' @param is_monotonic Logical indicating whether to enforce monotonicity in WoE (default: TRUE).
//' @param convergence_threshold Convergence threshold for optimization process (default: 1e-6).
//' @param max_iterations Maximum number of iterations allowed (default: 1000).
//'
//' @return A list containing:
//' \item{id}{Numeric identifiers for each bin (1-based indexing).}
//' \item{bin}{Character vector with the interval specification of each bin (e.g., "(-Inf;0.5]").}
//' \item{woe}{Numeric vector with the Weight of Evidence values for each bin.}
//' \item{iv}{Numeric vector with the Information Value contribution for each bin.}
//' \item{count}{Integer vector with the total number of observations in each bin.}
//' \item{count_pos}{Integer vector with the number of positive observations in each bin.}
//' \item{count_neg}{Integer vector with the number of negative observations in each bin.}
//' \item{cutpoints}{Numeric vector with the cut points between bins (excluding infinity).}
//' \item{converged}{Logical value indicating whether the algorithm converged.}
//' \item{iterations}{Number of iterations performed by the algorithm.}
//' \item{total_iv}{Total Information Value of the binning solution.}
//'
//' @details
//' ## Algorithm Overview
//' 
//' The implementation follows a multi-stage approach:
//' 
//' 1. **Pre-processing**:
//'    - Validation of inputs and handling of missing values
//'    - Special processing for features with few unique values
//' 
//' 2. **Equal-Width Binning**:
//'    - Division of the feature range into intervals of equal width
//'    - Initial assignment of observations to bins
//' 
//' 3. **Statistical Optimization**:
//'    - Merging of rare bins with frequencies below threshold
//'    - WoE monotonicity enforcement (optional)
//'    - Optimization to meet maximum bins constraint
//' 
//' 4. **Metric Calculation**:
//'    - Weight of Evidence (WoE) and Information Value (IV) computation
//' 
//' ## Mathematical Foundation
//' 
//' The algorithm uses two key metrics from information theory:
//' 
//' 1. **Weight of Evidence (WoE)** for bin \eqn{i}:
//'    \deqn{WoE_i = \ln\left(\frac{p_i/P}{n_i/N}\right)}
//'    
//'    Where:
//'    - \eqn{p_i}: Number of positive cases in bin \eqn{i}
//'    - \eqn{P}: Total number of positive cases
//'    - \eqn{n_i}: Number of negative cases in bin \eqn{i}
//'    - \eqn{N}: Total number of negative cases
//'    
//' 2. **Information Value (IV)** for bin \eqn{i}:
//'    \deqn{IV_i = \left(\frac{p_i}{P} - \frac{n_i}{N}\right) \times WoE_i}
//'    
//'    The total Information Value is the sum across all bins:
//'    \deqn{IV_{total} = \sum_{i=1}^{k} IV_i}
//'    
//' 3. **Laplace Smoothing**:
//'    To handle zero counts, the algorithm employs Laplace smoothing:
//'    \deqn{\frac{p_i + \alpha}{P + k\alpha}, \frac{n_i + \alpha}{N + k\alpha}}
//'    
//'    Where:
//'    - \eqn{\alpha}: Smoothing factor (0.5 in this implementation)
//'    - \eqn{k}: Number of bins
//' 
//' ## Monotonicity Enforcement
//' 
//' When `is_monotonic = TRUE`, the algorithm ensures that WoE values either consistently 
//' increase or decrease across bins. This property is desirable for:
//' 
//' - Interpretability: Monotonic relationships are easier to explain
//' - Robustness: Reduces overfitting and improves stability
//' - Business logic: Aligns with domain knowledge expectations
//' 
//' The algorithm determines the preferred monotonicity direction (increasing or decreasing) 
//' based on the initial bins and proceeds to merge bins that violate this pattern while 
//' minimizing information loss.
//' 
//' ## Handling Edge Cases
//' 
//' The algorithm includes special handling for:
//' 
//' - Missing values (NaN)
//' - Features with few unique values
//' - Nearly constant features
//' - Highly imbalanced target distributions
//'
//' @examples
//' \dontrun{
//' # Generate synthetic data
//' set.seed(123)
//' target <- sample(0:1, 1000, replace = TRUE)
//' feature <- rnorm(1000)
//' 
//' # Basic usage
//' result <- optimal_binning_numerical_ewb(target, feature)
//' print(result)
//' 
//' # Custom parameters
//' result_custom <- optimal_binning_numerical_ewb(
//'   target = target,
//'   feature = feature,
//'   min_bins = 2,
//'   max_bins = 8,
//'   bin_cutoff = 0.03,
//'   is_monotonic = TRUE
//' )
//' 
//' # Extract cutpoints for use in prediction
//' cutpoints <- result$cutpoints
//' 
//' # Calculate total information value
//' total_iv <- result$total_iv
//' }
//'
//' @references
//' Dougherty, J., Kohavi, R., & Sahami, M. (1995). Supervised and Unsupervised Discretization of 
//' Continuous Features. *Proceedings of the Twelfth International Conference on Machine Learning*, 194-202.
//' 
//' García, S., Luengo, J., Sáez, J. A., López, V., & Herrera, F. (2013). A survey of discretization 
//' techniques: Taxonomy and empirical analysis in supervised learning. *IEEE Transactions on Knowledge 
//' and Data Engineering*, 25(4), 734-750.
//' 
//' Kotsiantis, S., & Kanellopoulos, D. (2006). Discretization Techniques: A Recent Survey. 
//' *GESTS International Transactions on Computer Science and Engineering*, 32(1), 47-58.
//' 
//' Siddiqi, N. (2006). *Credit Risk Scorecards: Developing and Implementing Intelligent Credit Scoring*. 
//' John Wiley & Sons.
//' 
//' Thomas, L. C. (2009). *Consumer Credit Models: Pricing, Profit and Portfolios*. Oxford University Press.
//' 
//' Zeng, Y. (2014). Univariate feature selection and binner. *arXiv preprint arXiv:1410.5420*.
//'
//' @export
// [[Rcpp::export]]
Rcpp::List optimal_binning_numerical_ewb(
 Rcpp::IntegerVector target, 
 Rcpp::NumericVector feature,
 int min_bins = 3, 
 int max_bins = 5, 
 double bin_cutoff = 0.05,
 int max_n_prebins = 20,
 bool is_monotonic = true,
 double convergence_threshold = 1e-6, 
 int max_iterations = 1000) {

try {
 // Convert R vectors to STL containers
 std::vector<double> feature_vec = Rcpp::as<std::vector<double>>(feature);
 std::vector<int> target_vec = Rcpp::as<std::vector<int>>(target);
 
 // Initialize and execute binning algorithm
 OptimalBinningNumericalEWB binner(
     feature_vec, target_vec, 
     min_bins, max_bins, 
     bin_cutoff, max_n_prebins,
     is_monotonic,
     convergence_threshold, max_iterations);
 
 binner.fit();
 return binner.get_results();
} catch(std::exception &ex) {
 forward_exception_to_r(ex);
} catch(...) {
 ::Rf_error("Unknown C++ exception in optimal_binning_numerical_ewb");
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
// #include <limits>
// #include <sstream>
// #include <iomanip>
// #include <unordered_set>
// #include <numeric>
// 
// using namespace Rcpp;
// 
// // Class for Optimal Binning using Equal-Width Binning
// class OptimalBinningNumericalEWB {
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
//   bool converged;
//   int iterations_run;
//   
//   struct Bin {
//     double lower;
//     double upper;
//     int count;
//     int count_pos;
//     int count_neg;
//     double woe;
//     double iv;
//     
//     Bin(double lb = -std::numeric_limits<double>::infinity(),
//         double ub = std::numeric_limits<double>::infinity(),
//         int c = 0, int cp = 0, int cn = 0)
//       : lower(lb), upper(ub), count(c), count_pos(cp), count_neg(cn), woe(0.0), iv(0.0) {}
//   };
//   
//   std::vector<Bin> bins;
//   
//   // Total positives and negatives
//   int total_pos;
//   int total_neg;
//   
//   // Number of unique feature values
//   int unique_count;
//   
//   static constexpr double EPSILON = 1e-10;
//   
//   std::string double_to_string(double value) const {
//     if (std::isinf(value)) {
//       return value > 0 ? "+Inf" : "-Inf";
//     }
//     std::ostringstream ss;
//     ss << std::fixed << std::setprecision(6) << value;
//     return ss.str();
//   }
//   
//   double calculate_woe(int pos, int neg) const {
//     double pos_rate = (pos > 0) ? (static_cast<double>(pos) / total_pos) : (EPSILON / (total_pos + EPSILON));
//     double neg_rate = (neg > 0) ? (static_cast<double>(neg) / total_neg) : (EPSILON / (total_neg + EPSILON));
//     // Avoid division by zero or log of zero by adding EPSILON
//     pos_rate = std::max(pos_rate, EPSILON);
//     neg_rate = std::max(neg_rate, EPSILON);
//     return std::log(pos_rate / neg_rate);
//   }
//   
//   double calculate_iv(double woe, int pos, int neg) const {
//     double p_rate = (total_pos > 0) ? (static_cast<double>(pos) / total_pos) : EPSILON;
//     double n_rate = (total_neg > 0) ? (static_cast<double>(neg) / total_neg) : EPSILON;
//     return (p_rate - n_rate) * woe;
//   }
//   
//   void validate_inputs() {
//     if (feature.empty()) {
//       Rcpp::stop("Feature vector is empty.");
//     }
//     if (feature.size() != target.size()) {
//       Rcpp::stop("Feature and target vectors must have the same length.");
//     }
//     
//     for (const auto& t : target) {
//       if (t != 0 && t != 1) {
//         Rcpp::stop("Target vector must contain only 0 and 1.");
//       }
//     }
//     
//     std::unordered_set<double> unique_values(feature.begin(), feature.end());
//     unique_count = static_cast<int>(unique_values.size());
//     
//     if (unique_count <= 1) {
//       Rcpp::stop("Feature vector must contain at least two unique values.");
//     }
//     
//     if (min_bins < 2) {
//       Rcpp::stop("min_bins must be at least 2.");
//     }
//     if (max_bins < min_bins) {
//       Rcpp::stop("max_bins must be greater than or equal to min_bins.");
//     }
//     if (bin_cutoff <= 0 || bin_cutoff >= 1) {
//       Rcpp::stop("bin_cutoff must be between 0 and 1.");
//     }
//     if (max_n_prebins <= 0) {
//       Rcpp::stop("max_n_prebins must be positive.");
//     }
//     if (max_iterations <= 0) {
//       Rcpp::stop("max_iterations must be positive.");
//     }
//   }
//   
//   void create_unique_bins() {
//     bins.clear();
//     bins.reserve(static_cast<size_t>(unique_count));
//     
//     std::vector<double> unique_values(feature);
//     std::sort(unique_values.begin(), unique_values.end());
//     unique_values.erase(std::unique(unique_values.begin(), unique_values.end()), unique_values.end());
//     
//     for (const auto& val : unique_values) {
//       bins.emplace_back(val, val, 0, 0, 0);
//     }
//     
//     for (size_t i = 0; i < feature.size(); ++i) {
//       double value = feature[i];
//       int target_value = target[i];
//       // Exact bin match since lower=upper for unique bins
//       for (auto& bin : bins) {
//         if (std::fabs(value - bin.lower) < EPSILON) {
//           bin.count++;
//           if (target_value == 1) {
//             bin.count_pos++;
//           } else {
//             bin.count_neg++;
//           }
//           break;
//         }
//       }
//     }
//   }
//   
//   void create_prebins() {
//     double min_value = *std::min_element(feature.begin(), feature.end());
//     double max_value = *std::max_element(feature.begin(), feature.end());
//     
//     if (std::fabs(min_value - max_value) < EPSILON) {
//       // All values are approximately the same, single bin
//       int total_c = static_cast<int>(feature.size());
//       int pos_c = std::count(target.begin(), target.end(), 1);
//       int neg_c = total_c - pos_c;
//       bins.emplace_back(min_value, max_value, total_c, pos_c, neg_c);
//       return;
//     }
//     
//     int n_prebins = std::min(max_n_prebins, static_cast<int>(feature.size()));
//     // Avoid division by zero
//     double range = max_value - min_value;
//     if (range < EPSILON) {
//       range = EPSILON;
//     }
//     double bin_width = range / n_prebins;
//     
//     bins.clear();
//     bins.reserve(static_cast<size_t>(n_prebins));
//     
//     for (int i = 0; i < n_prebins; ++i) {
//       double lower = min_value + i * bin_width;
//       double upper = (i == n_prebins - 1) ? max_value : (min_value + (i + 1) * bin_width);
//       // Ensure upper >= lower
//       if (upper < lower) {
//         upper = lower;
//       }
//       bins.emplace_back(lower, upper, 0, 0, 0);
//     }
//   }
//   
//   void assign_data_to_bins() {
//     // Assign each observation to a bin
//     for (size_t i = 0; i < feature.size(); ++i) {
//       double value = feature[i];
//       int target_value = target[i];
//       int bin_index = find_bin(value);
//       if (bin_index < 0) {
//         // Should not happen if bins cover the range correctly
//         // Fall back to closest bin if needed
//         if (value < bins.front().lower) {
//           bin_index = 0;
//         } else if (value > bins.back().upper) {
//           bin_index = static_cast<int>(bins.size()) - 1;
//         } else {
//           Rcpp::stop("Value does not fit into any bin.");
//         }
//       }
//       bins[static_cast<size_t>(bin_index)].count++;
//       if (target_value == 1) {
//         bins[static_cast<size_t>(bin_index)].count_pos++;
//       } else {
//         bins[static_cast<size_t>(bin_index)].count_neg++;
//       }
//     }
//   }
//   
//   int find_bin(double value) const {
//     // Since bins are in ascending order, we can find via linear search or binary search.
//     // Here linear is used. For large data, binary search could be considered.
//     for (size_t b = 0; b < bins.size(); ++b) {
//       if (value >= bins[b].lower && value <= bins[b].upper) {
//         return static_cast<int>(b);
//       }
//     }
//     return -1;
//   }
//   
//   void merge_rare_bins() {
//     int total_count = std::accumulate(bins.begin(), bins.end(), 0,
//                                       [](int sum, const Bin& bin) { return sum + bin.count; });
//     double cutoff_count = bin_cutoff * total_count;
//     int iterations = 0;
//     
//     while (iterations < max_iterations) {
//       bool merged = false;
//       // Avoid merging below min_bins
//       if (static_cast<int>(bins.size()) <= min_bins) {
//         break;
//       }
//       for (size_t i = 0; i < bins.size(); ++i) {
//         if (bins[i].count < cutoff_count && static_cast<int>(bins.size()) > min_bins) {
//           if (i == 0 && bins.size() > 1) {
//             // Merge with next bin
//             bins[0].upper = bins[1].upper;
//             bins[0].count += bins[1].count;
//             bins[0].count_pos += bins[1].count_pos;
//             bins[0].count_neg += bins[1].count_neg;
//             bins.erase(bins.begin() + 1);
//           } else if (i > 0) {
//             // Merge with previous bin
//             bins[i - 1].upper = bins[i].upper;
//             bins[i - 1].count += bins[i].count;
//             bins[i - 1].count_pos += bins[i].count_pos;
//             bins[i - 1].count_neg += bins[i].count_neg;
//             bins.erase(bins.begin() + i);
//           }
//           merged = true;
//           break;
//         }
//       }
//       iterations++;
//       if (!merged) {
//         break;
//       }
//     }
//     iterations_run += iterations;
//     if (iterations >= max_iterations) {
//       converged = false;
//     }
//   }
//   
//   void ensure_max_bins() {
//     int iterations = 0;
//     // Merge until bins.size() <= max_bins
//     while (static_cast<int>(bins.size()) > max_bins && iterations < max_iterations) {
//       // Find pair of adjacent bins with smallest combined count
//       if (bins.size() <= 1) break;
//       size_t min_index = 0;
//       int min_count = bins[0].count + bins[1].count;
//       for (size_t i = 1; i < bins.size() - 1; ++i) {
//         int combined_count = bins[i].count + bins[i + 1].count;
//         if (combined_count < min_count) {
//           min_count = combined_count;
//           min_index = i;
//         }
//       }
//       // Merge bins[min_index] and bins[min_index + 1]
//       if (min_index < bins.size() - 1) {
//         bins[min_index].upper = bins[min_index + 1].upper;
//         bins[min_index].count += bins[min_index + 1].count;
//         bins[min_index].count_pos += bins[min_index + 1].count_pos;
//         bins[min_index].count_neg += bins[min_index + 1].count_neg;
//         bins.erase(bins.begin() + (min_index + 1));
//       }
//       iterations++;
//     }
//     iterations_run += iterations;
//     if (iterations >= max_iterations) {
//       converged = false;
//     }
//   }
//   
//   void enforce_monotonicity() {
//     if (bins.size() <= 2) {
//       return;
//     }
//     
//     int iterations = 0;
//     bool is_monotonic = false;
//     bool increasing = true;
//     if (bins.size() >= 2) {
//       increasing = (bins[1].woe >= bins[0].woe);
//     }
//     
//     while (!is_monotonic && static_cast<int>(bins.size()) > min_bins && iterations < max_iterations) {
//       is_monotonic = true;
//       for (size_t i = 1; i < bins.size(); ++i) {
//         if ((increasing && bins[i].woe < bins[i - 1].woe) ||
//             (!increasing && bins[i].woe > bins[i - 1].woe)) {
//           // Merge bins[i - 1] and bins[i]
//           if (i < bins.size()) {
//             bins[i - 1].upper = bins[i].upper;
//             bins[i - 1].count += bins[i].count;
//             bins[i - 1].count_pos += bins[i].count_pos;
//             bins[i - 1].count_neg += bins[i].count_neg;
//             bins.erase(bins.begin() + i);
//             calculate_woe_iv();
//             is_monotonic = false;
//             break;
//           }
//         }
//       }
//       iterations++;
//       if (static_cast<int>(bins.size()) == min_bins) {
//         break;
//       }
//     }
//     iterations_run += iterations;
//     if (iterations >= max_iterations) {
//       converged = false;
//     }
//   }
//   
//   void calculate_woe_iv() {
//     for (auto& bin : bins) {
//       bin.woe = calculate_woe(bin.count_pos, bin.count_neg);
//       bin.iv = calculate_iv(bin.woe, bin.count_pos, bin.count_neg);
//     }
//   }
//   
// public:
//   OptimalBinningNumericalEWB(const std::vector<double>& feature_, const std::vector<int>& target_,
//                              int min_bins_ = 3, int max_bins_ = 5, double bin_cutoff_ = 0.05, int max_n_prebins_ = 20,
//                              double convergence_threshold_ = 1e-6, int max_iterations_ = 1000)
//     : feature(feature_), target(target_), min_bins(min_bins_), max_bins(max_bins_),
//       bin_cutoff(bin_cutoff_), max_n_prebins(max_n_prebins_),
//       convergence_threshold(convergence_threshold_), max_iterations(max_iterations_),
//       converged(true), iterations_run(0), total_pos(0), total_neg(0), unique_count(0) {}
//   
//   void fit() {
//     validate_inputs();
//     
//     // Compute totals
//     total_pos = std::accumulate(target.begin(), target.end(), 0);
//     total_neg = static_cast<int>(target.size()) - total_pos;
//     
//     if (total_pos == 0 || total_neg == 0) {
//       Rcpp::stop("Target must contain at least one positive and one negative case.");
//     }
//     
//     if (unique_count <= 2) {
//       // If 2 or fewer unique values, simple bin creation
//       create_unique_bins();
//       calculate_woe_iv();
//       converged = true;
//       iterations_run = 0;
//       return;
//     }
//     
//     // Create and assign to prebins
//     create_prebins();
//     assign_data_to_bins();
//     
//     // Merge rare bins
//     merge_rare_bins();
//     
//     // Calculate initial WoE/IV
//     calculate_woe_iv();
//     
//     // Enforce monotonicity
//     enforce_monotonicity();
//     
//     // Ensure max_bins
//     ensure_max_bins();
//     
//     // Recalculate WoE/IV
//     calculate_woe_iv();
//   }
//   
//   List get_results() const {
//     std::vector<std::string> bin_labels;
//     std::vector<double> woe_values;
//     std::vector<double> iv_values;
//     std::vector<int> counts;
//     std::vector<int> counts_pos;
//     std::vector<int> counts_neg;
//     std::vector<double> cutpoints;
//     
//     bin_labels.reserve(bins.size());
//     woe_values.reserve(bins.size());
//     iv_values.reserve(bins.size());
//     counts.reserve(bins.size());
//     counts_pos.reserve(bins.size());
//     counts_neg.reserve(bins.size());
//     cutpoints.reserve(bins.size() > 0 ? bins.size() - 1 : 0);
//     
//     for (size_t i = 0; i < bins.size(); ++i) {
//       const auto& bin = bins[i];
//       std::string label = "(" + double_to_string(bin.lower) + ";" + double_to_string(bin.upper) + "]";
//       bin_labels.push_back(label);
//       woe_values.push_back(bin.woe);
//       iv_values.push_back(bin.iv);
//       counts.push_back(bin.count);
//       counts_pos.push_back(bin.count_pos);
//       counts_neg.push_back(bin.count_neg);
//       // Cutpoints exclude last bin upper
//       if (i < bins.size() - 1) {
//         cutpoints.push_back(bin.upper);
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
//       Named("count_pos") = counts_pos,
//       Named("count_neg") = counts_neg,
//       Named("cutpoints") = cutpoints,
//       Named("converged") = converged,
//       Named("iterations") = iterations_run
//     );
//   }
// };
// 
// 
// //' @title Optimal Binning for Numerical Variables using Equal-Width Binning
// //'
// //' @description
// //' Performs optimal binning for numerical variables using equal-width intervals (Equal-Width Binning) 
// //' with subsequent merging and adjustment steps. This procedure aims to create an interpretable binning 
// //' strategy with good predictive power, taking into account monotonicity and minimum splits within the bins.
// //'
// //' @param target Integer binary vector (0 or 1) representing the target variable.
// //' @param feature Numeric vector with the values of the feature to be binned.
// //' @param min_bins Minimum number of bins (default: 3).
// //' @param max_bins Maximum number of bins (default: 5).
// //' @param bin_cutoff Minimum fraction of observations each bin must contain (default: 0.05).
// //' @param max_n_prebins Maximum number of pre-bins before optimization (default: 20).
// //' @param convergence_threshold Convergence threshold (default: 1e-6).
// //' @param max_iterations Maximum number of iterations allowed (default: 1000).
// //'
// //' @return A list containing:
// //' \item{bins}{Character vector with the interval of each bin.}
// //' \item{woe}{Numeric vector with the WoE values for each bin.}
// //' \item{iv}{Numeric vector with the IV value for each bin.}
// //' \item{count}{Numeric vector with the total number of observations in each bin.}
// //' \item{count_pos}{Numeric vector with the total number of positive observations in each bin.}
// //' \item{count_neg}{Numeric vector with the total number of negative observations in each bin.}
// //' \item{cutpoints}{Numeric vector with the cut points.}
// //' \item{converged}{Logical value indicating whether the algorithm converged.}
// //' \item{iterations}{Number of iterations performed by the algorithm.}
// //'
// //' @details
// //' The algorithm consists of the following steps:
// //' 1. Creation of equal-width pre-bins.
// //' 2. Assignment of data to these pre-bins.
// //' 3. Merging of rare bins (with few observations).
// //' 4. Calculation of initial WoE and IV.
// //' 5. Ensuring WoE monotonicity by merging non-monotonic bins.
// //' 6. Adjustment to ensure the maximum number of bins does not exceed max_bins.
// //' 7. Recalculating WoE and IV at the end.
// //'
// //' This method aims to provide bins that balance interpretability, monotonicity, and predictive power, useful in risk modeling and credit scoring.
// //'
// //' @examples
// //' set.seed(123)
// //' target <- sample(0:1, 1000, replace = TRUE)
// //' feature <- rnorm(1000)
// //' result <- optimal_binning_numerical_ewb(target, feature)
// //' print(result)
// //'
// //' @export
// // [[Rcpp::export]]
// Rcpp::List optimal_binning_numerical_ewb(Rcpp::IntegerVector target, Rcpp::NumericVector feature,
//                                          int min_bins = 3, int max_bins = 5, double bin_cutoff = 0.05,
//                                          int max_n_prebins = 20,
//                                          double convergence_threshold = 1e-6, int max_iterations = 1000) {
//   std::vector<double> feature_vec = Rcpp::as<std::vector<double>>(feature);
//   std::vector<int> target_vec = Rcpp::as<std::vector<int>>(target);
//   
//   OptimalBinningNumericalEWB binner(feature_vec, target_vec, min_bins, max_bins, bin_cutoff, max_n_prebins,
//                                     convergence_threshold, max_iterations);
//   binner.fit();
//   return binner.get_results();
// }
