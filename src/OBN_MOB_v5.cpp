// [[Rcpp::plugins(cpp11)]]
// [[Rcpp::depends(Rcpp)]]

#include <Rcpp.h>
#include <vector>
#include <algorithm>
#include <numeric>
#include <string>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <set>
#include <sstream>
#include <iomanip>
#include <unordered_map>

using namespace Rcpp;

// Include shared headers
#include "common/optimal_binning_common.h"
#include "common/bin_structures.h"

using namespace Rcpp;
using namespace OptimalBinning;


/**
 * @brief NumericalBin metrics structure
 * 
 * Contains all information about a single bin, including
 * boundaries, counts, and calculated metrics.
 */
// struct NumericalBin {
//   double lower;          // Lower bound of the bin
//   double upper;          // Upper bound of the bin
//   double woe;            // Weight of Evidence
//   double iv;             // Information Value
//   int count;             // Total count in the bin
//   int count_pos;         // Count of positives in the bin
//   int count_neg;         // Count of negatives in the bin
//   double event_rate;     // Proportion of positives in the bin
//   
//   // Constructor with default initialization
//   NumericalBin() 
//     : lower(-std::numeric_limits<double>::infinity()),
//       upper(std::numeric_limits<double>::infinity()),
//       woe(0.0),
//       iv(0.0),
//       count(0),
//       count_pos(0),
//       count_neg(0),
//       event_rate(0.0) {}
// };

/**
 * @brief Monotonic Optimal Binning (MOB) for Numerical Features
 * 
 * This class implements the Monotonic Optimal Binning algorithm for numerical features
 * in credit scoring and risk modeling applications. The algorithm ensures monotonicity
 * in the Weight of Evidence (WoE) values, which is often a desirable property for
 * interpretability and stability in production models.
 * 
 * Key features:
 * 1. Creates initial pre-bins based on frequency distribution
 * 2. Merges rare bins to ensure statistical stability
 * 3. Enforces monotonicity in WoE values
 * 4. Optimizes bin boundaries to maximize information value
 * 5. Handles special cases (few unique values, extreme values)
 * 6. Applies Laplace smoothing for robust WoE calculation
 */
class OBN_MOB {
public:
  /**
   * @brief Constructor for the MOB algorithm
   * 
   * @param min_bins_ Minimum number of bins to create
   * @param max_bins_ Maximum number of bins to create
   * @param bin_cutoff_ Minimum proportion of records required in a bin
   * @param max_n_prebins_ Maximum number of pre-bins to create initially
   * @param convergence_threshold_ Threshold for convergence in the iterative process
   * @param max_iterations_ Maximum number of iterations for optimization
   * @param laplace_smoothing_ Smoothing parameter for WoE calculation (0 = no smoothing)
   */
  OBN_MOB(int min_bins_ = 3, 
                             int max_bins_ = 5, 
                             double bin_cutoff_ = 0.05,
                             int max_n_prebins_ = 20, 
                             double convergence_threshold_ = 1e-6,
                             int max_iterations_ = 1000,
                             double laplace_smoothing_ = 0.5)
    : min_bins(min_bins_),
      max_bins(std::max(max_bins_, min_bins_)),
      bin_cutoff(bin_cutoff_),
      max_n_prebins(std::max(max_n_prebins_, min_bins_)),
      convergence_threshold(convergence_threshold_),
      max_iterations(max_iterations_),
      laplace_smoothing(laplace_smoothing_),
      converged(false),
      iterations(0),
      total_count(0),
      total_pos(0),
      total_neg(0) {
    
    // Validate input parameters
    validate_parameters();
  }
  
  /**
   * @brief Fit the MOB model to the provided data
   * 
   * This is the main entry point for the algorithm. It processes the feature and target
   * vectors, creates the optimal bins, and calculates metrics.
   * 
   * @param feature_ Numeric vector of feature values
   * @param target_ Binary vector of target values (0/1)
   */
  void fit(const std::vector<double>& feature_, const std::vector<int>& target_) {
    // Validate input data
    validate_input_data(feature_, target_);
    
    // Copy data
    feature = feature_;
    target = target_;
    
    // Handle missing values if present
    handle_missing_values();
    
    // Calculate totals
    total_count = static_cast<int>(feature.size());
    total_pos = std::accumulate(target.begin(), target.end(), 0);
    total_neg = total_count - total_pos;
    
    // Count unique values
    std::set<double> unique_vals;
    for (const auto& val : feature) {
      if (std::isfinite(val)) {
        unique_vals.insert(val);
      }
    }
    int n_unique = static_cast<int>(unique_vals.size());
    
    // Adjust min_bins and max_bins based on unique values
    adjust_bin_parameters(n_unique);
    
    // Prepare sorted data for binning
    std::vector<std::pair<double, int>> sorted_data = prepare_sorted_data();
    
    // Handle special cases
    if (handle_special_cases(sorted_data, n_unique)) {
      return;
    }
    
    // Create initial pre-bins
    create_prebins(sorted_data, n_unique);
    
    // Optimize bins
    optimize_bins();
    
    // Final WoE/IV calculation
    calculate_woe_iv();
    
    // Validate final bins
    validate_bins();
  }
  
  /**
   * @brief Get the metrics for each bin
   * 
   * @return Vector of NumericalBin containing all bin information
   */
  std::vector<NumericalBin> get_bin_metrics() const {
    return bins;
  }
  
  /**
   * @brief Get cutpoints that define the bin boundaries
   * 
   * @return Vector of numeric cutpoints (excludes -Inf and +Inf)
   */
  std::vector<double> get_cutpoints() const {
    std::vector<double> cp;
    cp.reserve(bins.size() - 1);  // There are n-1 cutpoints for n bins
    
    for (size_t i = 0; i < bins.size() - 1; i++) {
      if (std::isfinite(bins[i].upper_bound)) {
        cp.push_back(bins[i].upper_bound);
      }
    }
    
    return cp;
  }
  
  /**
   * @brief Check if the algorithm converged within max_iterations
   * 
   * @return Boolean indicating convergence status
   */
  bool has_converged() const { 
    return converged; 
  }
  
  /**
   * @brief Get the number of iterations performed
   * 
   * @return Integer count of iterations
   */
  int get_iterations() const { 
    return iterations; 
  }
  
  /**
   * @brief Get the total Information Value of the binning
   * 
   * @return Double value representing the sum of all bin IVs
   */
  double get_total_iv() const {
    double total_iv = 0.0;
    for (const auto& bin : bins) {
      total_iv += bin.iv;
    }
    return total_iv;
  }
  
private:
  // Algorithm parameters
  int min_bins;
  int max_bins;
  double bin_cutoff;
  int max_n_prebins;
  double convergence_threshold;
  int max_iterations;
  double laplace_smoothing;
  
  // State variables
  bool converged;
  int iterations;
  
  // Data storage
  std::vector<double> feature;
  std::vector<int> target;
  int total_count;
  int total_pos;
  int total_neg;
  
  // Vector to store the bins
  std::vector<NumericalBin> bins;
  
  /**
   * @brief Validate algorithm parameters
   * 
   * Ensures that all parameters are within valid ranges and
   * adjusts them if necessary.
   */
  void validate_parameters() {
    if (min_bins < 2) {
      Rcpp::stop("min_bins must be at least 2.");
    }
    
    if (max_bins < min_bins) {
      Rcpp::stop("max_bins must be >= min_bins.");
    }
    
    if (bin_cutoff <= 0.0 || bin_cutoff >= 1.0) {
      Rcpp::stop("bin_cutoff must be between 0 and 1.");
    }
    
    if (max_n_prebins < min_bins) {
      Rcpp::warning("max_n_prebins adjusted to min_bins.");
      max_n_prebins = min_bins;
    }
    
    if (convergence_threshold <= 0.0) {
      Rcpp::stop("convergence_threshold must be positive.");
    }
    
    if (max_iterations <= 0) {
      Rcpp::stop("max_iterations must be positive.");
    }
    
    if (laplace_smoothing < 0.0) {
      Rcpp::stop("laplace_smoothing must be non-negative.");
    }
  }
  
  /**
   * @brief Validate input data
   * 
   * Checks that feature and target vectors are valid and contain
   * appropriate values.
   * 
   * @param feature_ Numeric vector of feature values
   * @param target_ Binary vector of target values (0/1)
   */
  void validate_input_data(const std::vector<double>& feature_, const std::vector<int>& target_) {
    if (feature_.size() != target_.size()) {
      Rcpp::stop("feature and target must have the same length.");
    }
    
    if (feature_.empty()) {
      Rcpp::stop("Feature vector is empty.");
    }
    
    // Verify target is binary (0 and 1)
    bool has_zero = false, has_one = false;
    for (int t : target_) {
      if (t == 0) has_zero = true;
      else if (t == 1) has_one = true;
      else Rcpp::stop("Target must contain only 0 and 1.");
      
      if (has_zero && has_one) break;
    }
    
    if (!has_zero || !has_one) {
      Rcpp::stop("Target must contain both classes (0 and 1).");
    }
  }
  
  /**
   * @brief Handle missing values in the feature vector
   * 
   * This function would typically remove or impute NaN/Inf values.
   * In this implementation, we check and report these values.
   */
  void handle_missing_values() {
    int nan_count = 0;
    int inf_count = 0;
    
    for (double val : feature) {
      if (std::isnan(val)) {
        nan_count++;
      } else if (std::isinf(val)) {
        inf_count++;
      }
    }
    
    if (nan_count > 0 || inf_count > 0) {
      Rcpp::warning("%d NaN and %d Inf values found. These will be handled separately.", 
                    nan_count, inf_count);
    }
  }
  
  /**
   * @brief Adjust bin parameters based on the number of unique values
   * 
   * Makes sure min_bins and max_bins are appropriate given the number
   * of unique values in the feature.
   * 
   * @param n_unique Number of unique values in the feature
   */
  void adjust_bin_parameters(int n_unique) {
    if (n_unique < min_bins) {
      min_bins = std::max(1, n_unique);
      if (max_bins < min_bins) {
        max_bins = min_bins;
      }
    }
    
    if (n_unique < max_bins) {
      max_bins = n_unique;
    }
  }
  
  /**
   * @brief Prepare sorted data for binning
   * 
   * Creates and sorts pairs of (feature, target) values, filtering out
   * NaN values in the process.
   * 
   * @return Vector of sorted (feature, target) pairs
   */
  std::vector<std::pair<double, int>> prepare_sorted_data() {
    std::vector<std::pair<double, int>> sorted_data;
    sorted_data.reserve(feature.size());
    
    // Create pairs and filter out NaNs
    for (size_t i = 0; i < feature.size(); i++) {
      if (!std::isnan(feature[i])) {
        sorted_data.emplace_back(feature[i], target[i]);
      }
    }
    
    // Sort by feature value
    std::sort(sorted_data.begin(), sorted_data.end(),
              [](const std::pair<double, int>& a, const std::pair<double, int>& b) {
                return a.first < b.first;
              });
    
    return sorted_data;
  }
  
  /**
   * @brief Handle special cases in the data
   * 
   * Deals with edge cases like few unique values, all values the same, etc.
   * 
   * @param sorted_data Sorted vector of (feature, target) pairs
   * @param n_unique Number of unique values in the feature
   * @return Boolean indicating whether a special case was handled
   */
  bool handle_special_cases(const std::vector<std::pair<double, int>>& sorted_data, int n_unique) {
    // Count actual unique values in sorted data
    int actual_unique = 1;
    for (size_t i = 1; i < sorted_data.size(); i++) {
      if (sorted_data[i].first != sorted_data[i-1].first) {
        actual_unique++;
      }
    }
    
    // Case 1: Very few unique values
    if (actual_unique <= 2) {
      handle_few_unique_values(sorted_data, actual_unique);
      converged = true;
      iterations = 0;
      return true;
    }
    
    // Case 2: All values are the same
    if (sorted_data.front().first == sorted_data.back().first) {
      NumericalBin bin;
      bin.lower_bound = -std::numeric_limits<double>::infinity();
      bin.upper_bound = std::numeric_limits<double>::infinity();
      bin.count = total_count;
      bin.count_pos = total_pos;
      bin.count_neg = total_neg;
      // bin.event_rate() assignment removed (calculated dynamically)
      
      bins.clear();
      bins.push_back(bin);
      calculate_woe_iv();
      converged = true;
      iterations = 0;
      return true;
    }
    
    return false;
  }
  
  /**
   * @brief Handle cases with very few unique values
   * 
   * Creates optimal bins when there are only 1 or 2 unique values.
   * 
   * @param sorted_data Sorted vector of (feature, target) pairs
   * @param unique_count Number of unique values
   */
  void handle_few_unique_values(const std::vector<std::pair<double, int>>& sorted_data, int unique_count) {
    bins.clear();
    
    // Case: Only one unique value
    if (unique_count == 1) {
      NumericalBin bin;
      bin.lower_bound = -std::numeric_limits<double>::infinity();
      bin.upper_bound = std::numeric_limits<double>::infinity();
      bin.count = total_count;
      bin.count_pos = total_pos;
      bin.count_neg = total_neg;
      // bin.event_rate() assignment removed (calculated dynamically)
      
      bins.push_back(bin);
      calculate_woe_iv();
    } 
    // Case: Two unique values - create two bins
    else if (unique_count == 2) {
      // Find the two unique values
      double val1 = sorted_data.front().first;
      double val2 = 0.0;
      
      for (size_t i = 1; i < sorted_data.size(); i++) {
        if (sorted_data[i].first != val1) {
          val2 = sorted_data[i].first;
          break;
        }
      }
      
      // Create two bins
      NumericalBin bin1, bin2;
      bin1.lower_bound = -std::numeric_limits<double>::infinity();
      bin1.upper_bound = (val1 + val2) / 2.0;  // Midpoint as boundary
      bin1.count = 0; 
      bin1.count_pos = 0; 
      bin1.count_neg = 0;
      
      bin2.lower_bound = bin1.upper_bound;
      bin2.upper_bound = std::numeric_limits<double>::infinity();
      bin2.count = 0; 
      bin2.count_pos = 0; 
      bin2.count_neg = 0;
      
      // Assign observations to bins
      for (const auto& p : sorted_data) {
        if (p.first == val1) {
          bin1.count++;
          if (p.second == 1) bin1.count_pos++; else bin1.count_neg++;
        } else {  // Must be val2
          bin2.count++;
          if (p.second == 1) bin2.count_pos++; else bin2.count_neg++;
        }
      }
      
      // Calculate event rates
      // bin1.event_rate() assignment removed (calculated dynamically)
      // bin2.event_rate() assignment removed (calculated dynamically)
      
      bins.push_back(bin1);
      bins.push_back(bin2);
      calculate_woe_iv();
    }
  }
  
  /**
   * @brief Create initial pre-bins
   * 
   * Creates the initial set of bins based on frequency distribution.
   * 
   * @param sorted_data Sorted vector of (feature, target) pairs
   * @param n_unique Number of unique values in the feature
   */
  void create_prebins(const std::vector<std::pair<double, int>>& sorted_data, int n_unique) {
    bins.clear();
    
    size_t n = sorted_data.size();
    size_t n_prebins = std::min(static_cast<size_t>(max_n_prebins), static_cast<size_t>(n_unique));
    n_prebins = std::max(static_cast<size_t>(min_bins), n_prebins);
    
    // Equal-frequency binning
    size_t bin_size = n / n_prebins;
    if (bin_size < 1) bin_size = 1;
    
    for (size_t i = 0; i < n; i += bin_size) {
      size_t end = std::min(i + bin_size, n);
      
      NumericalBin bin;
      
      // Set bin boundaries
      if (i == 0) {
        bin.lower_bound = -std::numeric_limits<double>::infinity();
      } else {
        // Ensure no gaps between bins
        if (!bins.empty()) {
          bin.lower_bound = bins.back().upper_bound;
        } else {
          bin.lower_bound = sorted_data[i].first;
        }
      }
      
      if (end == n) {
        bin.upper_bound = std::numeric_limits<double>::infinity();
      } else {
        bin.upper_bound = sorted_data[end].first;
      }
      
      // Collect bin statistics
      bin.count = static_cast<int>(end - i);
      bin.count_pos = 0;
      bin.count_neg = 0;
      
      for (size_t j = i; j < end; j++) {
        if (sorted_data[j].second == 1) {
          bin.count_pos++;
        } else {
          bin.count_neg++;
        }
      }
      
      // bin.event_rate() assignment removed (calculated dynamically)
      
      bins.push_back(bin);
    }
    
    // Ensure monotonicity of bin boundaries
    fix_bin_boundaries();
  }
  
  /**
   * @brief Fix bin boundaries
   * 
   * Ensures bin boundaries are consistent with no gaps or overlaps.
   */
  void fix_bin_boundaries() {
    if (bins.size() <= 1) return;
    
    // Ensure first bin starts at -Inf
    bins.front().lower_bound = -std::numeric_limits<double>::infinity();
    
    // Ensure last bin ends at +Inf
    bins.back().upper_bound = std::numeric_limits<double>::infinity();
    
    // Fix any gaps or overlaps between bins
    for (size_t i = 1; i < bins.size(); i++) {
      if (bins[i].lower_bound != bins[i-1].upper_bound) {
        bins[i].lower_bound = bins[i-1].upper_bound;
      }
    }
  }
  
  /**
   * @brief Optimize the bins
   * 
   * Main optimization routine that:
   * 1. Merges rare bins
   * 2. Enforces monotonicity
   * 3. Adjusts number of bins
   */
  void optimize_bins() {
    iterations = 0;
    converged = true;
    
    // Step 1: Merge rare bins
    merge_rare_bins();
    
    // Step 2: Calculate WoE/IV
    calculate_woe_iv();
    
    // Step 3: Enforce monotonicity
    if (!is_monotonic_woe()) {
      enforce_monotonicity();
    }
    
    // Step 4: Adjust number of bins if exceeding max_bins
    while (static_cast<int>(bins.size()) > max_bins && iterations < max_iterations) {
      size_t merge_idx = find_optimal_merge();
      if (merge_idx >= bins.size() - 1) break;
      
      merge_bins(merge_idx, merge_idx + 1);
      iterations++;
    }
    
    if (iterations >= max_iterations) {
      converged = false;
      Rcpp::warning("Algorithm did not converge within %d iterations", max_iterations);
    }
  }
  
  /**
   * @brief Merge bins with frequency below the threshold
   * 
   * Identifies and merges bins that have a proportion of records below bin_cutoff.
   */
  void merge_rare_bins() {
    double total = static_cast<double>(total_count);
    bool merged = true;
    
    while (merged && iterations < max_iterations && static_cast<int>(bins.size()) > min_bins) {
      merged = false;
      double min_freq = std::numeric_limits<double>::max();
      size_t min_freq_idx = bins.size();
      
      // Find bin with lowest frequency
      for (size_t i = 0; i < bins.size(); i++) {
        double freq = static_cast<double>(bins[i].count) / total;
        if (freq < min_freq) {
          min_freq = freq;
          min_freq_idx = i;
        }
      }
      
      // If smallest bin is below threshold, merge it
      if (min_freq < bin_cutoff && min_freq_idx < bins.size()) {
        // Determine optimal merge direction
        if (min_freq_idx == 0) {
          // Leftmost bin, can only merge right
          merge_bins(0, 1);
        } else if (min_freq_idx == bins.size() - 1) {
          // Rightmost bin, can only merge left
          merge_bins(bins.size() - 2, bins.size() - 1);
        } else {
          // Middle bin, determine best merge direction
          double iv_loss_left = bins[min_freq_idx - 1].iv + bins[min_freq_idx].iv;
          double iv_loss_right = bins[min_freq_idx].iv + bins[min_freq_idx + 1].iv;
          
          if (iv_loss_left <= iv_loss_right) {
            merge_bins(min_freq_idx - 1, min_freq_idx);
          } else {
            merge_bins(min_freq_idx, min_freq_idx + 1);
          }
        }
        
        merged = true;
      } else {
        break;  // No more rare bins to merge
      }
      
      iterations++;
    }
    
    if (iterations >= max_iterations && merged) {
      converged = false;
      Rcpp::warning("Rare bin merging did not complete within %d iterations", max_iterations);
    }
  }
  
  /**
   * @brief Check if WoE values are monotonic
   * 
   * @return Boolean indicating whether WoE values are monotonically increasing
   */
  bool is_monotonic_woe() const {
    if (bins.size() < 2) return true;
    
    // Determine direction (increasing or decreasing)
    bool increasing = bins[1].woe >= bins[0].woe;
    
    for (size_t i = 1; i < bins.size(); i++) {
      if (increasing && bins[i].woe < bins[i - 1].woe) {
        return false;
      }
      if (!increasing && bins[i].woe > bins[i - 1].woe) {
        return false;
      }
    }
    
    return true;
  }
  
  /**
   * @brief Enforce monotonicity of WoE values
   * 
   * Merges bins to ensure WoE values are monotonically increasing or decreasing.
   */
  void enforce_monotonicity() {
    if (bins.size() < 2) return;
    
    // Determine desired direction (increasing or decreasing)
    bool should_increase = bins[1].woe >= bins[0].woe;
    
    while (!is_monotonic_woe() && iterations < max_iterations && static_cast<int>(bins.size()) > min_bins) {
      bool merged = false;
      
      // Find first violation of monotonicity
      for (size_t i = 1; i < bins.size(); i++) {
        if ((should_increase && bins[i].woe < bins[i - 1].woe) ||
            (!should_increase && bins[i].woe > bins[i - 1].woe)) {
          
          // Determine which merge preserves more information
          
          // Try merging i-1 and i
          NumericalBin merged_bin = bins[i-1];
          merged_bin.upper_bound = bins[i].upper_bound;
          merged_bin.count += bins[i].count;
          merged_bin.count_pos += bins[i].count_pos;
          merged_bin.count_neg += bins[i].count_neg;
          // merged_bin.event_rate() assignment removed (calculated dynamically)
          
          // Calculate WoE for merged bin
          double merged_woe = calculate_bin_woe(merged_bin);
          
          // Check if merge would fix monotonicity
          bool merge_fixes = true;
          if (i > 1) {
            if ((should_increase && merged_woe < bins[i-2].woe) ||
                (!should_increase && merged_woe > bins[i-2].woe)) {
              merge_fixes = false;
            }
          }
          if (i < bins.size() - 1) {
            if ((should_increase && bins[i+1].woe < merged_woe) ||
                (!should_increase && bins[i+1].woe > merged_woe)) {
              merge_fixes = false;
            }
          }
          
          if (merge_fixes) {
            merge_bins(i-1, i);
            merged = true;
            break;
          } else if (i < bins.size() - 1) {
            // Try merging i and i+1 instead
            merge_bins(i, i+1);
            merged = true;
            break;
          } else {
            // Last resort: just merge i-1 and i
            merge_bins(i-1, i);
            merged = true;
            break;
          }
        }
      }
      
      if (!merged) break;
      iterations++;
      
      // Recalculate WoE/IV after merge
      calculate_woe_iv();
    }
    
    if (iterations >= max_iterations) {
      converged = false;
      Rcpp::warning("Monotonicity enforcement did not converge after %d iterations", max_iterations);
    }
  }
  
  /**
   * @brief Find optimal pair of bins to merge
   * 
   * @return Index of the left bin in the optimal merge pair
   */
  size_t find_optimal_merge() const {
    if (bins.size() < 2) return bins.size();
    
    double min_iv_loss = std::numeric_limits<double>::max();
    size_t merge_idx = bins.size();
    
    for (size_t i = 0; i < bins.size() - 1; i++) {
      // Calculate IV loss from merging
      double iv_before = bins[i].iv + bins[i+1].iv;
      
      // Create a temporary merged bin
      NumericalBin merged;
      merged.lower_bound = bins[i].lower_bound;
      merged.upper_bound = bins[i+1].upper_bound;
      merged.count = bins[i].count + bins[i+1].count;
      merged.count_pos = bins[i].count_pos + bins[i+1].count_pos;
      merged.count_neg = bins[i].count_neg + bins[i+1].count_neg;
      
      // Calculate WoE and IV for merged bin
      double woe = calculate_bin_woe(merged);
      double iv = calculate_bin_iv(merged, woe);
      
      // Calculate information loss
      double iv_loss = iv_before - iv;
      
      // Check if this loss is minimal
      if (iv_loss < min_iv_loss) {
        min_iv_loss = iv_loss;
        merge_idx = i;
      }
    }
    
    return merge_idx;
  }
  
  /**
   * @brief Merge two adjacent bins
   * 
   * @param i Index of the left bin
   * @param j Index of the right bin
   */
  void merge_bins(size_t i, size_t j) {
    if (i > j) std::swap(i, j);
    if (j >= bins.size()) return;
    
    // Update left bin with combined statistics
    bins[i].upper_bound = bins[j].upper_bound;
    bins[i].count += bins[j].count;
    bins[i].count_pos += bins[j].count_pos;
    bins[i].count_neg += bins[j].count_neg;
    // bins[i].event_rate() assignment removed (calculated dynamically)
    
    // Remove right bin
    bins.erase(bins.begin() + j);
    
    // Recalculate WoE/IV for the merged bin
    calculate_woe_iv();
  }
  
  /**
   * @brief Calculate WoE for a single bin
   * 
   * Applies Laplace smoothing for robust calculation.
   * 
   * @param bin The bin to calculate WoE for
   * @return Double WoE value
   */
  double calculate_bin_woe(const NumericalBin& bin) const {
    // Apply Laplace smoothing
    double smoothed_pos = bin.count_pos + laplace_smoothing;
    double smoothed_neg = bin.count_neg + laplace_smoothing;
    
    double total_smoothed_pos = total_pos + bins.size() * laplace_smoothing;
    double total_smoothed_neg = total_neg + bins.size() * laplace_smoothing;
    
    double dist_pos = smoothed_pos / total_smoothed_pos;
    double dist_neg = smoothed_neg / total_smoothed_neg;
    
    // Handle edge cases
    if (dist_pos <= 0.0 && dist_neg <= 0.0) {
      return 0.0;
    } else if (dist_pos <= 0.0) {
      return -20.0;  // Capped negative value for stability
    } else if (dist_neg <= 0.0) {
      return 20.0;   // Capped positive value for stability
    } else {
      return std::log(dist_pos / dist_neg);
    }
  }
  
  /**
   * @brief Calculate IV for a single bin
   * 
   * @param bin The bin to calculate IV for
   * @param woe Pre-calculated WoE value
   * @return Double IV value
   */
  double calculate_bin_iv(const NumericalBin& bin, double woe) const {
    // Apply Laplace smoothing
    double smoothed_pos = bin.count_pos + laplace_smoothing;
    double smoothed_neg = bin.count_neg + laplace_smoothing;
    
    double total_smoothed_pos = total_pos + bins.size() * laplace_smoothing;
    double total_smoothed_neg = total_neg + bins.size() * laplace_smoothing;
    
    double dist_pos = smoothed_pos / total_smoothed_pos;
    double dist_neg = smoothed_neg / total_smoothed_neg;
    
    // Calculate IV
    return (dist_pos - dist_neg) * woe;
  }
  
  /**
   * @brief Calculate WoE and IV for all bins
   * 
   * Updates the WoE and IV fields in the bins vector.
   */
  void calculate_woe_iv() {
    // Calculate total positives and negatives
    int pos_total = 0;
    int neg_total = 0;
    
    for (const auto& bin : bins) {
      pos_total += bin.count_pos;
      neg_total += bin.count_neg;
    }
    
    // Apply Laplace smoothing and calculate metrics
    for (auto& bin : bins) {
      // Apply smoothing
      double smoothed_pos = bin.count_pos + laplace_smoothing;
      double smoothed_neg = bin.count_neg + laplace_smoothing;
      
      double total_smoothed_pos = pos_total + bins.size() * laplace_smoothing;
      double total_smoothed_neg = neg_total + bins.size() * laplace_smoothing;
      
      double dist_pos = smoothed_pos / total_smoothed_pos;
      double dist_neg = smoothed_neg / total_smoothed_neg;
      
      // Calculate WoE with numeric stability
      if (dist_pos <= 0.0 && dist_neg <= 0.0) {
        bin.woe = 0.0;
      } else if (dist_pos <= 0.0) {
        bin.woe = -20.0;  // Cap for stability
      } else if (dist_neg <= 0.0) {
        bin.woe = 20.0;   // Cap for stability
      } else {
        bin.woe = std::log(dist_pos / dist_neg);
      }
      
      // Calculate IV
      if (std::isfinite(bin.woe)) {
        bin.iv = (dist_pos - dist_neg) * bin.woe;
      } else {
        bin.iv = 0.0;
      }
    }
  }
  
  /**
   * @brief Validate the final bin structure
   * 
   * Performs sanity checks on bins, ensuring they are well-formed.
   */
  void validate_bins() const {
    // Check if bins exist
    if (bins.empty()) {
      Rcpp::stop("No bins available after binning.");
    }
    
    // Check bin boundaries
    if (bins.front().lower_bound != -std::numeric_limits<double>::infinity()) {
      Rcpp::warning("First bin doesn't start at -Inf.");
    }
    
    if (bins.back().upper_bound != std::numeric_limits<double>::infinity()) {
      Rcpp::warning("Last bin doesn't end at +Inf.");
    }
    
    // Check for gaps between bins
    for (size_t i = 1; i < bins.size(); i++) {
      if (std::abs(bins[i].lower_bound - bins[i-1].upper_bound) > 1e-10) {
        Rcpp::warning("Gap detected between bins %d and %d", i-1, i);
      }
    }
    
    // Check for empty bins
    for (size_t i = 0; i < bins.size(); i++) {
      if (bins[i].count == 0) {
        Rcpp::warning("NumericalBin %d is empty", i);
      }
    }
    
    // Check monotonicity
    if (!is_monotonic_woe()) {
      Rcpp::warning("Final bins do not have monotonic WoE values");
    }
  }
};


// [[Rcpp::export]]
List optimal_binning_numerical_mob(IntegerVector target, 
                                  NumericVector feature,
                                  int min_bins = 3, 
                                  int max_bins = 5,
                                  double bin_cutoff = 0.05, 
                                  int max_n_prebins = 20,
                                  double convergence_threshold = 1e-6,
                                  int max_iterations = 1000,
                                  double laplace_smoothing = 0.5) {
 // Convert R types to C++ types
 std::vector<double> f = as<std::vector<double>>(feature);
 std::vector<int> t = as<std::vector<int>>(target);
 
 // Create and fit the MOB model
 OBN_MOB mob(min_bins, max_bins, bin_cutoff, max_n_prebins,
                                convergence_threshold, max_iterations, laplace_smoothing);
 mob.fit(f, t);
 
 // Get bin metrics
 std::vector<NumericalBin> bins = mob.get_bin_metrics();
 
 // Prepare output vectors
 std::vector<std::string> bin_labels;
 std::vector<double> woe_values;
 std::vector<double> iv_values;
 std::vector<int> counts;
 std::vector<int> counts_pos;
 std::vector<int> counts_neg;
 std::vector<double> event_rates;
 
 // Format output data
 for (const auto& b : bins) {
   // Create readable bin label
   std::ostringstream oss;
   oss << std::fixed << std::setprecision(6);
   
   if (b.lower_bound == -std::numeric_limits<double>::infinity()) {
     oss << "[-Inf";
   } else {
     oss << "[" << b.lower_bound;
   }
   oss << ";";
   if (b.upper_bound == std::numeric_limits<double>::infinity()) {
     oss << "+Inf)";
   } else {
     oss << b.upper_bound << ")";
   }
   
   bin_labels.push_back(oss.str());
   woe_values.push_back(b.woe);
   iv_values.push_back(b.iv);
   counts.push_back(b.count);
   counts_pos.push_back(b.count_pos);
   counts_neg.push_back(b.count_neg);
   event_rates.push_back(static_cast<double>(b.count_pos) / std::max(1, b.count));
 }
 
 // Create bin IDs (1-based indexing)
 Rcpp::NumericVector ids(bin_labels.size());
 for(size_t i = 0; i < bin_labels.size(); i++) {
   ids[i] = static_cast<double>(i + 1);
 }
 
 // Return results
 return Rcpp::List::create(
   Named("id") = ids,
   Named("bin") = bin_labels,
   Named("woe") = woe_values,
   Named("iv") = iv_values,
   Named("count") = counts,
   Named("count_pos") = counts_pos,
   Named("count_neg") = counts_neg,
   Named("event_rate") = event_rates,
   Named("cutpoints") = mob.get_cutpoints(),
   Named("total_iv") = mob.get_total_iv(),
   Named("converged") = mob.has_converged(),
   Named("iterations") = mob.get_iterations()
 );
}
