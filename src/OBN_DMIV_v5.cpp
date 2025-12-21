#include <Rcpp.h>
#include <algorithm>
#include <vector>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <numeric>

using namespace Rcpp;

// Include shared headers
#include "common/optimal_binning_common.h"
#include "common/bin_structures.h"

using namespace Rcpp;
using namespace OptimalBinning;


/**
 * Core class implementing Optimal Binning for numerical variables using various Divergence Measures.
 * Based on the theoretical framework from Zeng (2013) "Metric Divergence Measures and Information Value in Credit Scoring".
 * 
 * The algorithm transforms continuous numerical variables into optimal discrete bins 
 * based on their relationship with a binary target variable, maximizing the selected
 * divergence measure between distributions of positive and negative cases.
 */
class OBN_DMIV {
private:
  std::vector<double> feature;
  std::vector<int> target;
  int min_bins;
  int max_bins;
  double bin_cutoff;
  int max_n_prebins;
  bool is_monotonic;
  double convergence_threshold;
  int max_iterations;
  std::string bin_method;        // 'woe' or 'woe1'
  std::string divergence_method; // 'he', 'kl', 'tr', 'klj', 'sc', 'js', 'l1', 'l2', 'ln'
  
  // Small constant to handle floating-point comparison
  // Constant removed (uses shared definition)
  
  // Structure to represent a bin with its properties
  // Local NumericalBin struct definition removed in favor of Shared NumericalBin
  
  std::vector<NumericalBin> bins;     // Container for all bins
  bool converged;            // Flag indicating algorithm convergence
  int iterations_run;        // Count of iterations performed
  
  /**
   * Validate input arguments for correctness and consistency
   * Throws std::invalid_argument if validation fails
   */
  void validate_inputs() {
    if (feature.size() != target.size()) {
      throw std::invalid_argument("Feature and target vectors must have the same length.");
    }
    
    // Check for valid binary target values (0 or 1)
    for (auto& val : target) {
      if (val != 0 && val != 1) {
        throw std::invalid_argument("Target values must be binary (0 or 1).");
      }
    }
    
    if (min_bins < 2) {
      throw std::invalid_argument("min_bins must be at least 2.");
    }
    if (max_bins < min_bins) {
      throw std::invalid_argument("max_bins must be >= min_bins.");
    }
    if (bin_cutoff < 0 || bin_cutoff > 1) {
      throw std::invalid_argument("bin_cutoff must be between 0 and 1.");
    }
    if (max_n_prebins < min_bins) {
      throw std::invalid_argument("max_n_prebins must be >= min_bins.");
    }
    if (convergence_threshold <= 0) {
      throw std::invalid_argument("convergence_threshold must be > 0.");
    }
    if (max_iterations <= 0) {
      throw std::invalid_argument("max_iterations must be > 0.");
    }
    
    // Validate bin_method and divergence_method
    if (bin_method != "woe" && bin_method != "woe1") {
      throw std::invalid_argument("bin_method must be either 'woe' or 'woe1'.");
    }
    
    std::vector<std::string> valid_divergence_methods = {
      "he", "kl", "tr", "klj", "sc", "js", "l1", "l2", "ln"
    };
    
    if (std::find(valid_divergence_methods.begin(), valid_divergence_methods.end(), 
                  divergence_method) == valid_divergence_methods.end()) {
      throw std::invalid_argument("Invalid divergence_method. Must be one of: 'he', 'kl', 'tr', 'klj', 'sc', 'js', 'l1', 'l2', 'ln'.");
    }
  }
  
  /**
   * Handle the special case where feature has <= 2 unique values
   * This creates bins directly without optimization
   * 
   * @param unique_values Vector of unique values in the feature
   */
  void handle_two_or_fewer_unique_values(const std::vector<double>& unique_values) {
    bins.clear();
    // Construct bins based on unique values
    for (size_t i = 0; i < unique_values.size(); ++i) {
      NumericalBin bin;
      bin.lower_bound = (i == 0) ? -std::numeric_limits<double>::infinity() : unique_values[i - 1];
      bin.upper_bound = (i == unique_values.size() - 1) ? std::numeric_limits<double>::infinity() : unique_values[i];
      bin.count_pos = 0;
      bin.count_neg = 0;
      bin.woe = 0.0;
      bin.divergence = 0.0;
      bins.push_back(bin);
    }
    
    // Assign observations to bins
    for (size_t i = 0; i < feature.size(); ++i) {
      double val = feature[i];
      int tgt = target[i];
      
      // With only 1 or 2 bins, linear scan is efficient enough
      for (auto &bin : bins) {
        if (val > bin.lower_bound - EPSILON && val <= bin.upper_bound + EPSILON) {
          if (tgt == 1) bin.count_pos++;
          else bin.count_neg++;
          break;
        }
      }
    }
    
    compute_bin_metrics();
    converged = true;
    iterations_run = 0;
  }
  
  /**
   * Compute a quantile from a sorted vector
   * 
   * @param data Vector of values
   * @param q Quantile value between 0 and 1
   * @return The q-th quantile value
   */
  double quantile(const std::vector<double>& data, double q) {
    if (data.empty()) return 0.0;
    
    std::vector<double> temp = data;
    std::sort(temp.begin(), temp.end());
    
    if (q <= 0.0) return temp.front();
    if (q >= 1.0) return temp.back();
    
    // Calculate index with interpolation
    double idx_exact = q * (temp.size() - 1);
    size_t idx_lower = static_cast<size_t>(std::floor(idx_exact));
    size_t idx_upper = static_cast<size_t>(std::ceil(idx_exact));
    
    // Handle edge cases
    if (idx_lower == idx_upper) return temp[idx_lower];
    
    // Linear interpolation
    double weight_upper = idx_exact - idx_lower;
    double weight_lower = 1.0 - weight_upper;
    
    return weight_lower * temp[idx_lower] + weight_upper * temp[idx_upper];
  }
  
  /**
   * Initial binning step using quantiles to create starting bins
   * This provides a good starting point for optimization
   */
  void prebinning() {
    // Handle missing values by excluding them
    std::vector<double> clean_feature;
    std::vector<int> clean_target;
    
    for (size_t i = 0; i < feature.size(); ++i) {
      if (!std::isnan(feature[i])) {
        clean_feature.push_back(feature[i]);
        clean_target.push_back(target[i]);
      }
    }
    
    if (clean_feature.empty()) {
      throw std::invalid_argument("No valid non-NA values in feature.");
    }
    
    // Get unique values
    std::vector<double> unique_values = clean_feature;
    std::sort(unique_values.begin(), unique_values.end());
    unique_values.erase(std::unique(unique_values.begin(), unique_values.end()), unique_values.end());
    
    // Special case for few unique values
    if (unique_values.size() <= 2) {
      handle_two_or_fewer_unique_values(unique_values);
      return;
    }
    
    // If limited unique values, create a bin for each value
    if (unique_values.size() <= static_cast<size_t>(min_bins)) {
      bins.clear();
      for (size_t i = 0; i < unique_values.size(); ++i) {
        NumericalBin bin;
        bin.lower_bound = (i == 0) ? -std::numeric_limits<double>::infinity() : unique_values[i - 1];
        bin.upper_bound = (i == unique_values.size() - 1) ? std::numeric_limits<double>::infinity() : unique_values[i];
        bin.count_pos = 0;
        bin.count_neg = 0;
        bin.woe = 0.0;
        bin.divergence = 0.0;
        bins.push_back(bin);
      }
    } else {
      // Use quantile-based initial binning for better distribution
      int n_prebins = std::min(static_cast<int>(unique_values.size()), max_n_prebins);
      
      std::vector<double> quantiles;
      for (int i = 1; i < n_prebins; ++i) {
        double q = static_cast<double>(i) / n_prebins;
        double qval = quantile(clean_feature, q);
        quantiles.push_back(qval);
      }
      
      // Remove duplicate quantiles (can happen with skewed distributions)
      std::sort(quantiles.begin(), quantiles.end());
      quantiles.erase(std::unique(quantiles.begin(), quantiles.end()), quantiles.end());
      
      // Create bins based on quantiles
      bins.clear();
      bins.resize(quantiles.size() + 1);
      
      for (size_t i = 0; i < bins.size(); ++i) {
        if (i == 0) {
          bins[i].lower_bound = -std::numeric_limits<double>::infinity();
          bins[i].upper_bound = quantiles[i];
        } else if (i == bins.size() - 1) {
          bins[i].lower_bound = quantiles[i - 1];
          bins[i].upper_bound = std::numeric_limits<double>::infinity();
        } else {
          bins[i].lower_bound = quantiles[i - 1];
          bins[i].upper_bound = quantiles[i];
        }
        
        bins[i].count_pos = 0;
        bins[i].count_neg = 0;
        bins[i].woe = 0.0;
        bins[i].divergence = 0.0;
      }
    }
    
    // Extract upper boundaries for binary search
    std::vector<double> uppers;
    uppers.reserve(bins.size());
    for (auto &b : bins) {
      uppers.push_back(b.upper_bound);
    }
    
    // Assign observations to bins using binary search for performance
    for (size_t i = 0; i < clean_feature.size(); ++i) {
      double val = clean_feature[i];
      int tgt = clean_target[i];
      
      // Binary search for the correct bin:
      // Find the first bin with upper boundary >= value
      auto it = std::lower_bound(uppers.begin(), uppers.end(), val + EPSILON);
      size_t idx = it - uppers.begin();
      
      // Count by target value
      if (idx < bins.size()) {  // Safety check
        if (tgt == 1) {
          bins[idx].count_pos++;
        } else {
          bins[idx].count_neg++;
        }
      }
    }
  }
  
  /**
   * Merge bins with frequency below the specified cutoff
   * This ensures each bin has sufficient statistical reliability
   */
  void merge_rare_bins() {
    // Calculate total count across all bins
    int total_count = 0;
    for (auto &bin : bins) {
      total_count += (bin.count_pos + bin.count_neg);
    }
    
    double cutoff_count = bin_cutoff * total_count;
    
    // Iteratively merge rare bins with neighbors
    for (auto it = bins.begin(); it != bins.end();) {
      int bin_count = it->count_pos + it->count_neg;
      
      // Check if bin is too small and we can still merge (respecting min_bins)
      if (bin_count < cutoff_count && bins.size() > static_cast<size_t>(min_bins)) {
        if (it != bins.begin()) {
          // Merge with previous bin (preferred)
          auto prev = std::prev(it);
          prev->upper_bound = it->upper_bound;
          prev->count_pos += it->count_pos;
          prev->count_neg += it->count_neg;
          it = bins.erase(it);
        } else if (std::next(it) != bins.end()) {
          // Merge with next bin if this is the first bin
          auto nxt = std::next(it);
          nxt->lower_bound = it->lower_bound;
          nxt->count_pos += it->count_pos;
          nxt->count_neg += it->count_neg;
          it = bins.erase(it);
        } else {
          // Edge case: only one bin or no valid merge candidate
          ++it;
        }
      } else {
        ++it;
      }
    }
  }
  
  /**
   * Compute bin metrics including Weight of Evidence (WoE) and chosen divergence measure
   * WoE can be traditional or Zeng's WOE1 depending on bin_method setting
   * Also computes the chosen divergence measure for each bin
   */
  void compute_bin_metrics() {
    // Calculate totals
    int total_pos = 0;
    int total_neg = 0;
    
    for (auto &bin : bins) {
      total_pos += bin.count_pos;
      total_neg += bin.count_neg;
    }
    
    // Apply Laplace smoothing to avoid division by zero
    // Using a Bayesian-inspired approach with small pseudo-counts
    double pos_denom = total_pos + bins.size() * 0.5;
    double neg_denom = total_neg + bins.size() * 0.5;
    
    // Special calculation for L2 and L_infinity metrics
    std::vector<double> abs_diffs;  // For L_infinity
    std::vector<double> squared_diffs;  // For L2
    
    for (auto &bin : bins) {
      // Calculate proportions with smoothing
      double dist_pos = (bin.count_pos + 0.5) / pos_denom;
      double dist_neg = (bin.count_neg + 0.5) / neg_denom;
      
      // Calculate WoE with based on the selected method
      if (bin_method == "woe") {
        // Traditional WoE: ln((p_i/P)/(n_i/N))
        bin.woe = std::log(dist_pos / dist_neg);
      } else {  // bin_method == "woe1"
        // Zeng's WOE1: ln(g_i/b_i) - direct log odds ratio
        bin.woe = std::log((bin.count_pos + 0.5) / (bin.count_neg + 0.5));
      }
      
      // Calculate individual bin contribution to the divergence measure
      // For most measures, we calculate per-bin contributions here
      if (divergence_method == "he") {
        // Hellinger Discrimination
        bin.divergence = 0.5 * std::pow(std::sqrt(dist_pos) - std::sqrt(dist_neg), 2);
        
      } else if (divergence_method == "kl") {
        // Kullback-Leibler
        bin.divergence = dist_pos * std::log(dist_pos / dist_neg);
        
      } else if (divergence_method == "tr") {
        // Triangular Discrimination
        bin.divergence = std::pow(dist_pos - dist_neg, 2) / (dist_pos + dist_neg);
        
      } else if (divergence_method == "klj") {
        // J-Divergence (symmetric KL)
        bin.divergence = (dist_pos - dist_neg) * std::log(dist_pos / dist_neg);
        
      } else if (divergence_method == "sc") {
        // Symmetric Chi-Square Divergence
        bin.divergence = std::pow(dist_pos - dist_neg, 2) * (dist_pos + dist_neg) / (dist_pos * dist_neg);
        
      } else if (divergence_method == "js") {
        // Jensen-Shannon Divergence
        double m = (dist_pos + dist_neg) / 2.0;
        bin.divergence = 0.5 * (dist_pos * std::log(dist_pos / m) + dist_neg * std::log(dist_neg / m));
        
      } else if (divergence_method == "l1") {
        // L1 metric (Manhattan distance)
        bin.divergence = std::abs(dist_pos - dist_neg);
        
      } else if (divergence_method == "l2") {
        // For L2, store squared differences to compute the square root of sum later
        double sq_diff = std::pow(dist_pos - dist_neg, 2);
        squared_diffs.push_back(sq_diff);
        bin.divergence = sq_diff;  // Temporarily store squared difference
        
      } else if (divergence_method == "ln") {
        // For L_infinity, track absolute differences to find maximum later
        double abs_diff = std::abs(dist_pos - dist_neg);
        abs_diffs.push_back(abs_diff);
        bin.divergence = abs_diff;  // Temporarily store absolute difference
      }
    }
    
    // For L2 metric, calculate the square root of sum of squares
    if (divergence_method == "l2") {
      double sum_squared = std::accumulate(squared_diffs.begin(), squared_diffs.end(), 0.0);
      double l2_divergence = std::sqrt(sum_squared);
      
      // Adjust bin contributions proportionally to maintain total
      if (sum_squared > EPSILON) {  // Avoid division by zero
        for (size_t i = 0; i < bins.size(); ++i) {
          bins[i].divergence = squared_diffs[i] * l2_divergence / sum_squared;
        }
      }
    }
    
    // For L_infinity metric, find the maximum absolute difference
    if (divergence_method == "ln") {
      if (!abs_diffs.empty()) {
        double max_diff = *std::max_element(abs_diffs.begin(), abs_diffs.end());
        
        // Only the bin(s) with the maximum difference contribute to the total
        for (size_t i = 0; i < bins.size(); ++i) {
          bins[i].divergence = (std::abs(abs_diffs[i] - max_diff) < EPSILON) ? max_diff : 0.0;
        }
      }
    }
  }
  
  /**
   * Enforce monotonicity in Weight of Evidence across bins
   * This ensures the relationship between feature and target is monotonic,
   * improving interpretability and stability
   */
  void enforce_monotonicity() {
    if (bins.size() <= 1) return;
    
    // Check if already monotonic (either increasing or decreasing)
    bool is_increasing = true;
    bool is_decreasing = true;
    
    for (size_t i = 1; i < bins.size(); ++i) {
      if (bins[i].woe < bins[i-1].woe - EPSILON) is_increasing = false;
      if (bins[i].woe > bins[i-1].woe + EPSILON) is_decreasing = false;
    }
    
    // If already monotonic, do nothing
    if (is_increasing || is_decreasing) return;
    
    // Determine preferred direction (maximize total divergence)
    double div_increase = 0.0;
    double div_decrease = 0.0;
    
    // Test both directions by simulating merges
    std::vector<NumericalBin> temp_bins_inc = bins;
    std::vector<NumericalBin> temp_bins_dec = bins;
    
    // Simulate increasing monotonicity
    for (size_t i = 1; i < temp_bins_inc.size(); ) {
      if (temp_bins_inc[i].woe < temp_bins_inc[i-1].woe - EPSILON) {
        // Merge current bin into previous
        temp_bins_inc[i-1].upper_bound = temp_bins_inc[i].upper_bound;
        temp_bins_inc[i-1].count_pos += temp_bins_inc[i].count_pos;
        temp_bins_inc[i-1].count_neg += temp_bins_inc[i].count_neg;
        temp_bins_inc.erase(temp_bins_inc.begin() + i);
      } else {
        ++i;
      }
    }
    
    // Simulate decreasing monotonicity
    for (size_t i = 1; i < temp_bins_dec.size(); ) {
      if (temp_bins_dec[i].woe > temp_bins_dec[i-1].woe + EPSILON) {
        // Merge current bin into previous
        temp_bins_dec[i-1].upper_bound = temp_bins_dec[i].upper_bound;
        temp_bins_dec[i-1].count_pos += temp_bins_dec[i].count_pos;
        temp_bins_dec[i-1].count_neg += temp_bins_dec[i].count_neg;
        temp_bins_dec.erase(temp_bins_dec.begin() + i);
      } else {
        ++i;
      }
    }
    
    // Recalculate metrics for both scenarios
    // Save original bins
    std::vector<NumericalBin> original_bins = bins;
    
    // Evaluate increasing scenario
    bins = temp_bins_inc;
    compute_bin_metrics();
    for (auto &bin : bins) {
      div_increase += bin.divergence;
    }
    
    // Evaluate decreasing scenario
    bins = temp_bins_dec;
    compute_bin_metrics();
    for (auto &bin : bins) {
      div_decrease += bin.divergence;
    }
    
    // Choose direction with higher total divergence
    if (div_increase >= div_decrease) {
      bins = temp_bins_inc;
    } else {
      bins = temp_bins_dec;
    }
    
    // Make sure we still meet min_bins requirement
    if (bins.size() < static_cast<size_t>(min_bins)) {
      // Revert to original bins if we can't meet min_bins
      bins = original_bins;
      
      // Try a more conservative approach: merge only when necessary
      std::vector<NumericalBin> best_bins = bins;
      double best_divergence = 0.0;
      
      compute_bin_metrics();
      for (auto &bin : bins) {
        best_divergence += bin.divergence;
      }
      
      // Try merging in increasing direction until monotonic
      for (size_t i = 1; i < bins.size() && bins.size() > static_cast<size_t>(min_bins); ) {
        if (bins[i].woe < bins[i-1].woe - EPSILON) {
          // Merge current bin into previous
          bins[i-1].upper_bound = bins[i].upper_bound;
          bins[i-1].count_pos += bins[i].count_pos;
          bins[i-1].count_neg += bins[i].count_neg;
          bins.erase(bins.begin() + i);
          compute_bin_metrics();
        } else {
          ++i;
        }
      }
      
      // Calculate total divergence for increasing direction
      double inc_divergence = 0.0;
      for (auto &bin : bins) {
        inc_divergence += bin.divergence;
      }
      
      // Update best bins if better
      if (inc_divergence > best_divergence) {
        best_bins = bins;
        best_divergence = inc_divergence;
      }
      
      // Reset and try decreasing direction
      bins = original_bins;
      
      // Try merging in decreasing direction until monotonic
      for (size_t i = 1; i < bins.size() && bins.size() > static_cast<size_t>(min_bins); ) {
        if (bins[i].woe > bins[i-1].woe + EPSILON) {
          // Merge current bin into previous
          bins[i-1].upper_bound = bins[i].upper_bound;
          bins[i-1].count_pos += bins[i].count_pos;
          bins[i-1].count_neg += bins[i].count_neg;
          bins.erase(bins.begin() + i);
          compute_bin_metrics();
        } else {
          ++i;
        }
      }
      
      // Calculate total divergence for decreasing direction
      double dec_divergence = 0.0;
      for (auto &bin : bins) {
        dec_divergence += bin.divergence;
      }
      
      // Use best result
      if (dec_divergence > best_divergence) {
        best_bins = bins;
      } else if (best_divergence > 0.0) {
        bins = best_bins;
      } else {
        // If neither approach worked well, revert to original
        bins = original_bins;
      }
    }
    
    // Ensure metrics are up to date
    compute_bin_metrics();
  }
  
public:
  /**
   * Constructor for the OBN_DMIV algorithm
   * 
   * @param feature_ Numeric vector of feature values to be binned
   * @param target_ Binary vector (0/1) representing the target variable
   * @param min_bins_ Minimum number of bins to generate
   * @param max_bins_ Maximum number of bins to generate
   * @param bin_cutoff_ Minimum frequency fraction for each bin
   * @param max_n_prebins_ Maximum number of pre-bins before optimization
   * @param is_monotonic_ Whether to enforce monotonicity in WoE
   * @param convergence_threshold_ Convergence threshold for divergence change
   * @param max_iterations_ Maximum number of iterations allowed
   * @param bin_method_ Method for WoE calculation ('woe' or 'woe1')
   * @param divergence_method_ Divergence measure to optimize ('he', 'kl', etc.)
   */
  OBN_DMIV(
    const std::vector<double> &feature_,
    const std::vector<int> &target_,
    int min_bins_ = 3,
    int max_bins_ = 5,
    double bin_cutoff_ = 0.05,
    int max_n_prebins_ = 20,
    bool is_monotonic_ = true,
    double convergence_threshold_ = 1e-6,
    int max_iterations_ = 1000,
    std::string bin_method_ = "woe1",
    std::string divergence_method_ = "l2")
    : feature(feature_), target(target_), min_bins(min_bins_), max_bins(max_bins_),
      bin_cutoff(bin_cutoff_), max_n_prebins(max_n_prebins_), is_monotonic(is_monotonic_),
      convergence_threshold(convergence_threshold_), max_iterations(max_iterations_),
      bin_method(bin_method_), divergence_method(divergence_method_),
      converged(false), iterations_run(0) {
    validate_inputs();
  }
  
  /**
   * Execute the optimal binning algorithm and return results
   * 
   * @return Rcpp::List containing bin information, WoE, divergence measures, and other metrics
   */
  Rcpp::List fit() {
    // Step 1: Initial prebinning
    prebinning();
    
    if (!converged) {  // Only proceed if not already converged (e.g., few unique values)
      // Step 2: Merge rare bins
      merge_rare_bins();
      compute_bin_metrics();
      
      // Step 3: Enforce monotonicity if requested
      if (is_monotonic) {
        enforce_monotonicity();
      }
      
      double prev_total_div = -std::numeric_limits<double>::infinity();
      iterations_run = 0;
      
      // Step 4: Branch and Bound optimization
      // Iteratively merge bins with smallest divergence until max_bins constraint is met
      while (bins.size() > static_cast<size_t>(max_bins) && iterations_run < max_iterations) {
        // Find bin with minimum divergence contribution
        auto min_div_it = std::min_element(bins.begin(), bins.end(),
                                           [](const NumericalBin &a, const NumericalBin &b) { 
                                             return std::abs(a.divergence) < std::abs(b.divergence); 
                                           });
        
        // Merge the minimum divergence bin with an adjacent bin
        if (min_div_it != bins.begin()) {
          // Prefer merging with previous bin
          auto prev = std::prev(min_div_it);
          prev->upper_bound = min_div_it->upper_bound;
          prev->count_pos += min_div_it->count_pos;
          prev->count_neg += min_div_it->count_neg;
          bins.erase(min_div_it);
        } else if (std::next(min_div_it) != bins.end()) {
          // If it's the first bin, merge with the next one
          auto nxt = std::next(min_div_it);
          nxt->lower_bound = min_div_it->lower_bound;
          nxt->count_pos += min_div_it->count_pos;
          nxt->count_neg += min_div_it->count_neg;
          bins.erase(min_div_it);
        } else {
          // Edge case: only one bin left
          break;
        }
        
        // Recompute metrics after merge
        compute_bin_metrics();
        
        // Re-enforce monotonicity if needed
        if (is_monotonic) {
          enforce_monotonicity();
        }
        
        // Calculate total divergence for convergence check
        double total_div = 0.0;
        for (const auto& bin : bins) {
          total_div += bin.divergence;
        }
        
        // Check convergence based on divergence change
        if (std::fabs(total_div - prev_total_div) < convergence_threshold) {
          converged = true;
          break;
        }
        
        prev_total_div = total_div;
        iterations_run++;
      }
    }
    
    // Step 5: Prepare output
    Rcpp::StringVector bin_labels;
    Rcpp::NumericVector woe_values;
    Rcpp::NumericVector divergence_values;
    Rcpp::IntegerVector counts;
    Rcpp::IntegerVector counts_pos;
    Rcpp::IntegerVector counts_neg;
    Rcpp::NumericVector cutpoints;
    
    for (const auto &bin : bins) {
      // Create readable bin labels with interval notation
      std::string lower_str = std::isinf(bin.lower_bound) ? "-Inf" : std::to_string(bin.lower_bound);
      std::string upper_str = std::isinf(bin.upper_bound) ? "+Inf" : std::to_string(bin.upper_bound);
      std::string bin_label = "(" + lower_str + ";" + upper_str + "]";
      
      bin_labels.push_back(bin_label);
      woe_values.push_back(bin.woe);
      divergence_values.push_back(bin.divergence);
      counts.push_back(bin.count_pos + bin.count_neg);
      counts_pos.push_back(bin.count_pos);
      counts_neg.push_back(bin.count_neg);
      
      // Store cutpoints (excluding infinity)
      if (!std::isinf(bin.upper_bound)) {
        cutpoints.push_back(bin.upper_bound);
      }
    }
    
    // Create bin IDs (1-based indexing for R)
    Rcpp::IntegerVector ids(bin_labels.size());
    for(R_xlen_t i = 0; i < bin_labels.size(); i++) {
      ids[i] = static_cast<int>(i + 1);
    }
    
    // Calculate total divergence
    double total_divergence = 0.0;
    for (R_xlen_t i = 0; i < divergence_values.size(); i++) {
      total_divergence += divergence_values[i];
    }
    
    // Return comprehensive results
    return Rcpp::List::create(
      Rcpp::Named("id") = ids,
      Rcpp::Named("bin") = bin_labels,
      Rcpp::Named("woe") = woe_values,
      Rcpp::Named("divergence") = divergence_values,
      Rcpp::Named("count") = counts,
      Rcpp::Named("count_pos") = counts_pos,
      Rcpp::Named("count_neg") = counts_neg,
      Rcpp::Named("cutpoints") = cutpoints,
      Rcpp::Named("converged") = converged,
      Rcpp::Named("iterations") = iterations_run,
      Rcpp::Named("total_divergence") = total_divergence,
      Rcpp::Named("bin_method") = bin_method,
      Rcpp::Named("divergence_method") = divergence_method
    );
  }
};

// [[Rcpp::export]]
Rcpp::List optimal_binning_numerical_dmiv(
   Rcpp::IntegerVector target,
   Rcpp::NumericVector feature,
   int min_bins = 3,
   int max_bins = 5,
   double bin_cutoff = 0.05,
   int max_n_prebins = 20,
   bool is_monotonic = true,
   double convergence_threshold = 1e-6,
   int max_iterations = 1000,
   std::string bin_method = "woe1",
   std::string divergence_method = "l2"
) {
 try {
   // Convert R vectors to STL containers
   std::vector<double> feature_vec = Rcpp::as<std::vector<double>>(feature);
   std::vector<int> target_vec = Rcpp::as<std::vector<int>>(target);
   
   // Create and execute the algorithm
   OBN_DMIV ob(
       feature_vec,
       target_vec,
       min_bins,
       max_bins,
       bin_cutoff,
       max_n_prebins,
       is_monotonic,
       convergence_threshold,
       max_iterations,
       bin_method,
       divergence_method
   );
   
   // Return results
   return ob.fit();
 } catch (const std::exception& e) {
   Rcpp::stop("Error in optimal binning: " + std::string(e.what()));
 }
}