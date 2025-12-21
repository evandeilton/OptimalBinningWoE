// [[Rcpp::plugins(cpp11)]]

#include <Rcpp.h>
#include <algorithm>
#include <vector>
#include <cmath>
#include <limits>
#include <numeric>
#include <sstream>
#include <iomanip>
#include <stdexcept>
#include <set>
#include <unordered_map>

using namespace Rcpp;

// Include shared headers
#include "common/optimal_binning_common.h"
#include "common/bin_structures.h"

using namespace Rcpp;
using namespace OptimalBinning;


/**
 * @brief Optimal Binning for Numerical Variables using Optimal Supervised Learning Partitioning (OSLP)
 *
 * IMPORTANT: Despite "LP" (Linear Programming) in the name, this algorithm uses greedy
 * heuristics with information-theoretic optimization, not formal LP with simplex/interior-point.
 *
 * Algorithm Overview:
 * 1. Information-theoretic pre-binning based on quantiles
 * 2. Greedy bin merging using IV preservation strategy
 * 3. Monotonicity enforcement in WoE values
 * 4. Iterative refinement until convergence
 *
 * Complexity: O(n log n + k² * iterations) where n = sample size, k = bins
 * Space: O(n + k)
 *
 * Key Features:
 * - Supervised learning approach (uses target variable from start)
 * - Information preservation strategy in merging
 * - Laplace smoothing for robust WoE calculation
 * - Comprehensive edge case handling
 *
 * References:
 * - Belcastro, L., et al. (2020). "Optimal Binning: Mathematical Programming Formulation"
 * - Mironchyk, P., & Tchistiakov, V. (2017). "Monotone Optimal Binning Algorithm"
 * - Good, I.J. (1952). "Rational Decisions", JRSS
 */
class OBN_OSLP {
private:
  // Input data
  std::vector<double> feature;
  std::vector<double> target;
  
  // Algorithm parameters
  int min_bins;
  int max_bins;
  double bin_cutoff;
  int max_n_prebins;
  double convergence_threshold;
  int max_iterations;
  double laplace_smoothing;
  
  // NumericalBin structure
  std::vector<double> bin_edges;
  std::vector<std::string> bin_labels;
  std::vector<double> woe_values;
  std::vector<double> iv_values;
  std::vector<int> count_values;
  std::vector<int> count_pos_values;
  std::vector<int> count_neg_values;
  std::vector<double> event_rate_values;
  
  // Algorithm state
  double total_iv;
  bool converged;
  int iterations_run;
  
public:
  /**
   * @brief Constructor for OBN_OSLP
   * 
   * @param feature Vector of feature values to bin
   * @param target Vector of binary target values (0/1)
   * @param min_bins Minimum number of bins to create
   * @param max_bins Maximum number of bins to create
   * @param bin_cutoff Minimum proportion of observations in a bin
   * @param max_n_prebins Maximum number of pre-bins
   * @param convergence_threshold Threshold for convergence
   * @param max_iterations Maximum number of iterations
   * @param laplace_smoothing Smoothing parameter for WoE calculation
   */
  OBN_OSLP(
    const std::vector<double>& feature,
    const std::vector<double>& target,
    int min_bins = 3,
    int max_bins = 5,
    double bin_cutoff = 0.05,
    int max_n_prebins = 20,
    double convergence_threshold = 1e-6,
    int max_iterations = 1000,
    double laplace_smoothing = 0.5
  ) : feature(feature), target(target),
  min_bins(std::max(min_bins, 2)),
  max_bins(std::max(max_bins, min_bins)),
  bin_cutoff(bin_cutoff),
  max_n_prebins(std::max(max_n_prebins, min_bins)),
  convergence_threshold(convergence_threshold),
  max_iterations(max_iterations),
  laplace_smoothing(laplace_smoothing),
  total_iv(0.0),
  converged(false),
  iterations_run(0) {
    
    // Validate inputs
    validateInputs();
  }
  
  /**
   * @brief Fit the optimal binning model
   * 
   * Main entry point that executes the binning algorithm and returns the results
   * 
   * @return Rcpp::List Results of the binning process
   */
  Rcpp::List fit() {
    // Handle missing values
    handleMissingValues();
    
    // Determine unique values
    std::vector<double> unique_vals = getUniqueValues();
    
    if (unique_vals.size() <= 2) {
      // Handle special case: very few unique values
      handleLowUniqueValues(unique_vals);
      converged = true;
      iterations_run = 0;
      return createOutput();
    }
    
    // Main algorithm steps
    createInitialBins(unique_vals);
    mergeSmallBins();
    enforceMonotonicity();
    
    // Calculate final bin statistics
    calculateBins();
    
    // Validate the final binning solution
    validateBinning();
    
    return createOutput();
  }
  
private:
  /**
   * @brief Validate input parameters and data
   * 
   * Checks for valid inputs and throws exceptions if necessary
   */
  void validateInputs() {
    if (feature.size() != target.size()) {
      throw std::invalid_argument("Feature and target must have the same length.");
    }
    
    if (feature.empty()) {
      throw std::invalid_argument("Feature and target cannot be empty.");
    }
    
    if (bin_cutoff <= 0 || bin_cutoff >= 1) {
      throw std::invalid_argument("bin_cutoff must be between 0 and 1.");
    }
    
    if (convergence_threshold <= 0) {
      throw std::invalid_argument("convergence_threshold must be positive.");
    }
    
    if (max_iterations <= 0) {
      throw std::invalid_argument("max_iterations must be positive.");
    }
    
    if (laplace_smoothing < 0) {
      throw std::invalid_argument("laplace_smoothing must be non-negative.");
    }
    
    // Check that target is binary (0/1)
    bool has_zero = false, has_one = false;
    for (double t : target) {
      if (t == 0) has_zero = true;
      else if (t == 1) has_one = true;
      else throw std::invalid_argument("Target must contain only 0 and 1.");
      
      if (has_zero && has_one) break;
    }
    
    if (!has_zero || !has_one) {
      throw std::invalid_argument("Target must contain both classes (0 and 1).");
    }
  }
  
  /**
   * @brief Handle missing or extreme values in the feature
   * 
   * Checks for NaN or Inf values and throws an exception if found
   */
  void handleMissingValues() {
    // Count NaN/Inf values
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
      throw std::invalid_argument(
          "Feature contains " + std::to_string(nan_count) + " NaN and " +
            std::to_string(inf_count) + " Inf values. Please handle these values before binning."
      );
    }
  }
  
  /**
   * @brief Get unique values in the feature vector
   * 
   * @return std::vector<double> Sorted vector of unique values
   */
  std::vector<double> getUniqueValues() {
    std::set<double> unique_set(feature.begin(), feature.end());
    std::vector<double> unique_vals(unique_set.begin(), unique_set.end());
    return unique_vals;
  }
  
  /**
   * @brief Handle cases with very few unique values
   * 
   * Creates bins when there are only 1 or 2 unique values
   * 
   * @param unique_vals Vector of unique feature values
   */
  void handleLowUniqueValues(const std::vector<double>& unique_vals) {
    bin_edges.clear();
    
    if (unique_vals.size() == 1) {
      // One unique value - create a single bin
      bin_edges.push_back(-std::numeric_limits<double>::infinity());
      bin_edges.push_back(std::numeric_limits<double>::infinity());
    } else { 
      // Two unique values - create two bins with a boundary in between
      double midpoint = (unique_vals[0] + unique_vals[1]) / 2.0;
      bin_edges.push_back(-std::numeric_limits<double>::infinity());
      bin_edges.push_back(midpoint);
      bin_edges.push_back(std::numeric_limits<double>::infinity());
    }
    
    // Calculate statistics for these bins
    calculateBins();
  }
  
  /**
   * @brief Create initial bins using quantile-based approach
   * 
   * Creates a set of initial bins based on the distribution of unique values
   * 
   * @param unique_vals Vector of unique feature values
   */
  void createInitialBins(const std::vector<double>& unique_vals) {
    int n_unique = static_cast<int>(unique_vals.size());
    
    // Determine number of pre-bins
    int n_pre = std::min(max_n_prebins, n_unique);
    n_pre = std::max(n_pre, min_bins);
    
    // Reset bin edges and start with -Infinity
    bin_edges.clear();
    bin_edges.push_back(-std::numeric_limits<double>::infinity());
    
    // Calculate quantile-based edges
    for (int i = 1; i < n_pre; ++i) {
      int idx = static_cast<int>((i / static_cast<double>(n_pre)) * (n_unique - 1));
      double edge = unique_vals[idx];
      
      // Ensure no duplicate edges
      if (edge > bin_edges.back()) {
        bin_edges.push_back(edge);
      }
    }
    
    // Add +Infinity as the last edge
    bin_edges.push_back(std::numeric_limits<double>::infinity());
    
    // Calculate initial bin statistics
    calculateBins();
  }
  
  /**
   * @brief Calculate bin statistics
   * 
   * Computes counts, WoE, IV, and event rates for all bins
   */
  void calculateBins() {
    int n_bins = static_cast<int>(bin_edges.size() - 1);
    
    // Initialize vectors for bin statistics
    woe_values.assign(n_bins, 0.0);
    iv_values.assign(n_bins, 0.0);
    count_values.assign(n_bins, 0);
    count_pos_values.assign(n_bins, 0);
    count_neg_values.assign(n_bins, 0);
    event_rate_values.assign(n_bins, 0.0);
    
    // Count observations in each bin
    for (size_t i = 0; i < feature.size(); ++i) {
      double fval = feature[i];
      double tval = target[i];
      int idx = findBin(fval);
      
      if (idx >= 0 && idx < n_bins) {
        count_values[idx]++;
        if (tval == 1) {
          count_pos_values[idx]++;
        } else {
          count_neg_values[idx]++;
        }
      }
    }
    
    // Calculate totals across all bins
    double total_pos = std::accumulate(count_pos_values.begin(), count_pos_values.end(), 0.0);
    double total_neg = std::accumulate(count_neg_values.begin(), count_neg_values.end(), 0.0);
    
    // Calculate event rates, WoE, and IV with Laplace smoothing
    total_iv = 0.0;
    
    for (int i = 0; i < n_bins; ++i) {
      // Calculate event rate
      event_rate_values[i] = (count_values[i] > 0) ? 
      static_cast<double>(count_pos_values[i]) / count_values[i] : 0.0;
      
      // Apply Laplace smoothing for WoE calculation
      double smoothed_pos = count_pos_values[i] + laplace_smoothing;
      double smoothed_neg = count_neg_values[i] + laplace_smoothing;
      
      double total_smoothed_pos = total_pos + n_bins * laplace_smoothing;
      double total_smoothed_neg = total_neg + n_bins * laplace_smoothing;
      
      double dist_pos = smoothed_pos / total_smoothed_pos;
      double dist_neg = smoothed_neg / total_smoothed_neg;
      
      // Calculate WoE with protection against extreme values
      if (dist_pos <= 0.0 && dist_neg <= 0.0) {
        woe_values[i] = 0.0;
      } else if (dist_pos <= 0.0) {
        woe_values[i] = -20.0;  // Cap for numerical stability
      } else if (dist_neg <= 0.0) {
        woe_values[i] = 20.0;   // Cap for numerical stability
      } else {
        woe_values[i] = std::log(dist_pos / dist_neg);
      }
      
      // Calculate IV with protection against non-finite values
      if (std::isfinite(woe_values[i])) {
        iv_values[i] = (dist_pos - dist_neg) * woe_values[i];
      } else {
        iv_values[i] = 0.0;
      }
      
      // Add to total IV
      total_iv += iv_values[i];
    }
    
    // Update bin labels
    updateBinLabels();
  }
  
  /**
   * @brief Merge bins with frequency below the threshold
   * 
   * Identifies and merges small bins, preserving information value
   */
  void mergeSmallBins() {
    bool merged = true;
    int total_count = static_cast<int>(feature.size());
    iterations_run = 0;
    
    while (merged && static_cast<int>(bin_edges.size()) - 1 > min_bins && iterations_run < max_iterations) {
      merged = false;
      
      // Find the bin with the smallest proportion
      size_t smallest_idx = 0;
      double smallest_prop = std::numeric_limits<double>::max();
      
      for (size_t i = 0; i < count_values.size(); i++) {
        double prop = static_cast<double>(count_values[i]) / total_count;
        if (prop < smallest_prop) {
          smallest_prop = prop;
          smallest_idx = i;
        }
      }
      
      // If the smallest bin is below the cutoff, merge it
      if (smallest_prop < bin_cutoff && count_values.size() > static_cast<size_t>(min_bins)) {
        // Determine which neighbor to merge with
        if (smallest_idx == 0) {
          // Leftmost bin, merge with right
          mergeBins(0, 1);
        } else if (smallest_idx == count_values.size() - 1) {
          // Rightmost bin, merge with left
          mergeBins(count_values.size() - 2, count_values.size() - 1);
        } else {
          // Middle bin, decide based on information preservation
          double iv_loss_left = computeIVLoss(smallest_idx - 1, smallest_idx);
          double iv_loss_right = computeIVLoss(smallest_idx, smallest_idx + 1);
          
          if (iv_loss_left <= iv_loss_right) {
            mergeBins(smallest_idx - 1, smallest_idx);
          } else {
            mergeBins(smallest_idx, smallest_idx + 1);
          }
        }
        
        calculateBins();
        merged = true;
      }
      
      iterations_run++;
    }
  }
  
  /**
   * @brief Enforce monotonicity in WoE values
   * 
   * Merges bins to ensure monotonically increasing or decreasing WoE
   */
  void enforceMonotonicity() {
    // Determine if WoE should be increasing or decreasing
    bool increasing = guessTrend();
    
    double prev_iv = total_iv;
    
    // Iteratively merge bins until monotonicity is achieved
    while (!isMonotonic(increasing) && 
           static_cast<int>(bin_edges.size()) - 1 > min_bins && 
           iterations_run < max_iterations) {
      
      // Find first monotonicity violation
      for (size_t i = 1; i < woe_values.size(); i++) {
        if ((increasing && woe_values[i] < woe_values[i-1]) ||
            (!increasing && woe_values[i] > woe_values[i-1])) {
          
          // Find the merge that would best preserve information
          
          // If merging i-1 and i would cause a new violation with i-2,
          // consider other merge options
          if (i > 1) {
            double test_woe = estimateMergedWoE(i-1, i);
            if ((increasing && test_woe < woe_values[i-2]) ||
                (!increasing && test_woe > woe_values[i-2])) {
              
              // Try merging i and i+1 instead, if possible
              if (i < woe_values.size() - 1) {
                mergeBins(i, i+1);
              } else {
                // Last resort: merge i-1 and i despite the new violation
                mergeBins(i-1, i);
              }
            } else {
              // Standard case: merge i-1 and i
              mergeBins(i-1, i);
            }
          } else {
            // If i=1, simply merge the first two bins
            mergeBins(0, 1);
          }
          
          break;
        }
      }
      
      calculateBins();
      
      // Check for convergence
      double diff = std::fabs(total_iv - prev_iv);
      if (diff < convergence_threshold) {
        converged = true;
        break;
      }
      
      prev_iv = total_iv;
      iterations_run++;
    }
    
    // If too many bins, merge based on minimal IV difference
    while (static_cast<int>(bin_edges.size()) - 1 > max_bins && iterations_run < max_iterations) {
      size_t idx = findMinIVMerge();
      mergeBins(idx, idx + 1);
      calculateBins();
      iterations_run++;
    }
    
    // Mark as converged or not
    if (iterations_run >= max_iterations) {
      converged = false;
    } else {
      converged = true;
    }
  }
  
  /**
   * @brief Determine if WoE values should be increasing or decreasing
   * 
   * @return bool True if WoE should be increasing, false if decreasing
   */
  bool guessTrend() const {
    int inc = 0, dec = 0;
    
    for (size_t i = 1; i < woe_values.size(); i++) {
      if (woe_values[i] > woe_values[i-1]) inc++;
      else if (woe_values[i] < woe_values[i-1]) dec++;
    }
    
    return inc >= dec;
  }
  
  /**
   * @brief Check if WoE values are monotonic
   * 
   * @param increasing True to check for monotonically increasing, false for decreasing
   * @return bool True if WoE values are monotonic in the specified direction
   */
  bool isMonotonic(bool increasing) const {
    for (size_t i = 1; i < woe_values.size(); i++) {
      if (increasing && woe_values[i] < woe_values[i-1]) return false;
      if (!increasing && woe_values[i] > woe_values[i-1]) return false;
    }
    return true;
  }
  
  /**
   * @brief Find the pair of adjacent bins with minimal IV value
   * 
   * @return size_t Index of the left bin in the pair with minimal IV
   */
  size_t findMinIVMerge() const {
    double min_iv_sum = std::numeric_limits<double>::max();
    size_t idx = 0;
    
    for (size_t i = 0; i < iv_values.size() - 1; i++) {
      double iv_sum = iv_values[i] + iv_values[i+1];
      if (iv_sum < min_iv_sum) {
        min_iv_sum = iv_sum;
        idx = i;
      }
    }
    
    return idx;
  }
  
  /**
   * @brief Calculate the information loss from merging two bins
   * 
   * @param i Index of the left bin
   * @param j Index of the right bin
   * @return double Information loss (difference in IV)
   */
  double computeIVLoss(size_t i, size_t j) const {
    if (i >= iv_values.size() || j >= iv_values.size()) {
      return std::numeric_limits<double>::max();
    }
    
    // Original IV
    double original_iv = iv_values[i] + iv_values[j];
    
    // Estimate IV after merge
    double merged_pos = count_pos_values[i] + count_pos_values[j];
    double merged_neg = count_neg_values[i] + count_neg_values[j];
    
    double total_pos = std::accumulate(count_pos_values.begin(), count_pos_values.end(), 0.0);
    double total_neg = std::accumulate(count_neg_values.begin(), count_neg_values.end(), 0.0);
    
    // Apply Laplace smoothing
    double smoothed_pos = merged_pos + laplace_smoothing;
    double smoothed_neg = merged_neg + laplace_smoothing;
    
    double total_smoothed_pos = total_pos + (bin_edges.size() - 2) * laplace_smoothing;
    double total_smoothed_neg = total_neg + (bin_edges.size() - 2) * laplace_smoothing;
    
    double dist_pos = smoothed_pos / total_smoothed_pos;
    double dist_neg = smoothed_neg / total_smoothed_neg;
    
    // Estimate WoE
    double woe = std::log(dist_pos / dist_neg);
    
    // Estimate IV
    double merged_iv = (dist_pos - dist_neg) * woe;
    
    // Return IV loss
    return original_iv - merged_iv;
  }
  
  /**
   * @brief Estimate WoE value after merging two bins
   * 
   * @param i Index of the left bin
   * @param j Index of the right bin
   * @return double Estimated WoE after merge
   */
  double estimateMergedWoE(size_t i, size_t j) const {
    if (i >= count_pos_values.size() || j >= count_pos_values.size()) {
      return 0.0;
    }
    
    // Merged counts
    double merged_pos = count_pos_values[i] + count_pos_values[j];
    double merged_neg = count_neg_values[i] + count_neg_values[j];
    
    // Total counts
    double total_pos = std::accumulate(count_pos_values.begin(), count_pos_values.end(), 0.0);
    double total_neg = std::accumulate(count_neg_values.begin(), count_neg_values.end(), 0.0);
    
    // Apply Laplace smoothing
    double smoothed_pos = merged_pos + laplace_smoothing;
    double smoothed_neg = merged_neg + laplace_smoothing;
    
    double total_smoothed_pos = total_pos + (bin_edges.size() - 2) * laplace_smoothing;
    double total_smoothed_neg = total_neg + (bin_edges.size() - 2) * laplace_smoothing;
    
    double dist_pos = smoothed_pos / total_smoothed_pos;
    double dist_neg = smoothed_neg / total_smoothed_neg;
    
    // Return estimated WoE
    if (dist_pos <= 0.0 || dist_neg <= 0.0) {
      return 0.0;
    } else {
      return std::log(dist_pos / dist_neg);
    }
  }
  
  /**
   * @brief Merge two bins
   * 
   * @param i Index of the left bin
   * @param j Index of the right bin
   */
  void mergeBins(size_t i, size_t j) {
    if (i > j) std::swap(i, j);
    if (j >= bin_edges.size() - 1) return;
    
    // Remove the edge between bins i and j
    bin_edges.erase(bin_edges.begin() + j);
  }
  
  /**
   * @brief Find which bin a value belongs to
   * 
   * @param val Value to find bin for
   * @return int Index of the bin
   */
  int findBin(double val) const {
    auto it = std::upper_bound(bin_edges.begin(), bin_edges.end(), val);
    int idx = static_cast<int>(std::distance(bin_edges.begin(), it)) - 1;
    
    // Handle edge cases
    if (idx < 0) idx = 0;
    if (idx >= static_cast<int>(bin_edges.size()) - 1) idx = static_cast<int>(bin_edges.size()) - 2;
    
    return idx;
  }
  
  /**
   * @brief Update bin labels based on bin edges
   */
  void updateBinLabels() {
    bin_labels.clear();
    
    for (size_t i = 0; i < bin_edges.size() - 1; i++) {
      std::ostringstream oss;
      oss << std::fixed << std::setprecision(6);
      
      if (std::isinf(bin_edges[i]) && bin_edges[i] < 0) {
        oss << "[-Inf;";
      } else {
        oss << "[" << bin_edges[i] << ";";
      }
      
      if (std::isinf(bin_edges[i+1])) {
        oss << "+Inf)";
      } else {
        oss << bin_edges[i+1] << ")";
      }
      
      bin_labels.push_back(oss.str());
    }
  }
  
  /**
   * @brief Validate the final binning solution
   * 
   * Performs various checks to ensure the binning is valid
   */
  void validateBinning() const {
    // Check if bins exist
    if (bin_edges.empty() || bin_edges.size() < 2) {
      throw std::runtime_error("No valid bins created during binning process.");
    }
    
    // Check bin boundaries
    if (bin_edges.front() != -std::numeric_limits<double>::infinity()) {
      throw std::runtime_error("First bin must start at -Infinity.");
    }
    
    if (bin_edges.back() != std::numeric_limits<double>::infinity()) {
      throw std::runtime_error("Last bin must end at +Infinity.");
    }
    
    // Check for ordering
    for (size_t i = 1; i < bin_edges.size(); i++) {
      if (bin_edges[i] <= bin_edges[i-1] && std::isfinite(bin_edges[i]) && std::isfinite(bin_edges[i-1])) {
        throw std::runtime_error("NumericalBin edges must be in ascending order.");
      }
    }
    
    // Check for empty bins
    for (size_t i = 0; i < count_values.size(); i++) {
      if (count_values[i] == 0) {
        throw std::runtime_error("Empty bin detected at index " + std::to_string(i));
      }
    }
  }
  
  /**
   * @brief Create the output list with all binning results
   * 
   * @return Rcpp::List Results of the binning process
   */
  Rcpp::List createOutput() const {
    // Extract cutpoints (all edges except -Inf and +Inf)
    std::vector<double> cutpoints;
    for (size_t i = 1; i < bin_edges.size() - 1; i++) {
      if (std::isfinite(bin_edges[i])) {
        cutpoints.push_back(bin_edges[i]);
      }
    }
    
    // Create bin IDs (1-based indexing)
    Rcpp::NumericVector ids(bin_labels.size());
    for(size_t i = 0; i < bin_labels.size(); i++) {
      ids[i] = static_cast<double>(i + 1);
    }
    
    // Return a list with all binning information
    return Rcpp::List::create(
      Named("id") = ids,
      Named("bin") = bin_labels,
      Named("woe") = woe_values,
      Named("iv") = iv_values,
      Named("count") = count_values,
      Named("count_pos") = count_pos_values,
      Named("count_neg") = count_neg_values,
      Named("event_rate") = event_rate_values,
      Named("cutpoints") = cutpoints,
      Named("total_iv") = total_iv,
      Named("converged") = converged,
      Named("iterations") = iterations_run
    );
  }
};

// [[Rcpp::export]]
Rcpp::List optimal_binning_numerical_oslp(
   Rcpp::NumericVector target,
   Rcpp::NumericVector feature,
   int min_bins = 3,
   int max_bins = 5,
   double bin_cutoff = 0.05,
   int max_n_prebins = 20,
   double convergence_threshold = 1e-6,
   int max_iterations = 1000,
   double laplace_smoothing = 0.5
) {
 // Validate basic inputs
 if (feature.size() != target.size()) {
   Rcpp::stop("Feature and target must have the same length.");
 }
 if (feature.size() == 0) {
   Rcpp::stop("Feature and target cannot be empty.");
 }
 if (min_bins < 2) {
   Rcpp::stop("min_bins must be at least 2.");
 }
 if (max_bins < min_bins) {
   Rcpp::stop("max_bins must be greater or equal to min_bins.");
 }
 if (bin_cutoff <= 0 || bin_cutoff >= 1) {
   Rcpp::stop("bin_cutoff must be between 0 and 1.");
 }
 if (max_n_prebins < min_bins) {
   Rcpp::stop("max_n_prebins must be at least min_bins.");
 }
 if (convergence_threshold <= 0) {
   Rcpp::stop("convergence_threshold must be positive.");
 }
 if (max_iterations <= 0) {
   Rcpp::stop("max_iterations must be positive.");
 }
 if (laplace_smoothing < 0) {
   Rcpp::stop("laplace_smoothing must be non-negative.");
 }
 
 // Convert R vectors to C++ vectors
 std::vector<double> feature_vec(feature.begin(), feature.end());
 std::vector<double> target_vec(target.begin(), target.end());
 
 try {
   // Create and run the binning algorithm
   OBN_OSLP binning(
       feature_vec, target_vec,
       min_bins, max_bins,
       bin_cutoff, max_n_prebins,
       convergence_threshold, max_iterations,
       laplace_smoothing
   );
   
   return binning.fit();
 } catch (const std::exception &e) {
   Rcpp::stop("Error in optimal_binning_numerical_oslp: " + std::string(e.what()));
 }
}
