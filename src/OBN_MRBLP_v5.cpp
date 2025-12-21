// [[Rcpp::plugins(cpp11)]]
// [[Rcpp::depends(Rcpp)]]

#include <Rcpp.h>
#include <algorithm>
#include <vector>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <sstream>
#include <iomanip>
#include <numeric>
#include <set>

using namespace Rcpp;

// Include shared headers
#include "common/optimal_binning_common.h"
#include "common/bin_structures.h"

using namespace Rcpp;
using namespace OptimalBinning;


/**
 * @brief Optimal Binning using Monotonic Risk Binning with Likelihood Ratio Pre-binning (MRBLP)
 *
 * IMPORTANT: Despite "LP" in the name, uses greedy heuristics with likelihood ratio optimization,
 * not formal Linear Programming.
 *
 * Algorithm Overview:
 * 1. Equal-frequency pre-binning initialization
 * 2. Greedy merging with monotonicity preservation
 * 3. Likelihood ratio-based bin refinement
 * 4. WoE monotonicity enforcement
 *
 * Complexity: O(n log n + k² * iterations)
 * Space: O(n + k)
 *
 * Key Features:
 * - Monotonic risk preservation
 * - Likelihood ratio optimization
 * - Laplace smoothing for stability
 * - Edge case handling
 */
class OBN_MRBLP {
private:
  // Feature and target vectors
  std::vector<double> feature;
  std::vector<int> target;
  
  // Binning parameters
  int min_bins;
  int max_bins;
  double bin_cutoff;
  int max_n_prebins;
  double convergence_threshold;
  int max_iterations;
  double laplace_smoothing;
  
  // Algorithm state
  bool converged;
  int iterations_run;
  int total_pos;
  int total_neg;
  
  /**
   * @brief Struct representing a single bin with all its metrics
   */
  // Local NumericalBin definition removed

  
  // Vector of bins
  std::vector<NumericalBin> bins;
  
public:
  /**
   * @brief Constructor for OBN_MRBLP
   * 
   * Initializes the binning algorithm with specified parameters and validates inputs.
   * 
   * @param feature Numeric vector of continuous feature values
   * @param target Integer vector of binary target values (0/1)
   * @param min_bins Minimum number of bins to create
   * @param max_bins Maximum number of bins to create
   * @param bin_cutoff Minimum proportion of observations in each bin
   * @param max_n_prebins Maximum number of pre-bins for initial binning
   * @param convergence_threshold Convergence threshold for monotonic binning
   * @param max_iterations Maximum number of iterations
   * @param laplace_smoothing Smoothing parameter for WoE calculation
   */
  OBN_MRBLP(const NumericVector& feature_,
                               const IntegerVector& target_,
                               int min_bins_ = 3,
                               int max_bins_ = 5,
                               double bin_cutoff_ = 0.05,
                               int max_n_prebins_ = 20,
                               double convergence_threshold_ = 1e-6,
                               int max_iterations_ = 1000,
                               double laplace_smoothing_ = 0.5)
    : feature(feature_.begin(), feature_.end()),
      target(target_.begin(), target_.end()),
      min_bins(min_bins_),
      max_bins(std::max(min_bins_, max_bins_)),
      bin_cutoff(bin_cutoff_),
      max_n_prebins(std::max(max_n_prebins_, min_bins_)),
      convergence_threshold(convergence_threshold_),
      max_iterations(max_iterations_),
      laplace_smoothing(laplace_smoothing_),
      converged(false),
      iterations_run(0),
      total_pos(0),
      total_neg(0) {
    
    validateInputs();
    calculateTotals();
  }
  
  /**
   * @brief Fit the optimal binning model
   * 
   * Main method to execute the binning algorithm:
   * 1. Handle special cases (low unique values)
   * 2. Perform initial pre-binning
   * 3. Merge small bins
   * 4. Enforce monotonicity
   * 5. Calculate final WoE and IV values
   */
  void fit() {
    // Handle missing values
    handleMissingValues();
    
    // Determine unique feature values
    std::vector<double> unique_features = getUniqueFeatureValues();
    size_t n_unique = unique_features.size();
    
    // Adjust min_bins and max_bins based on unique values
    adjustBinParameters(n_unique);
    
    if (n_unique <= 2) {
      // Handle low unique values case
      handleLowUniqueValues(unique_features);
      converged = true;
      iterations_run = 0;
      return;
    }
    
    // Proceed with standard binning if unique values > 2
    performPreBinning();
    mergeSmallBins();
    enforceMonotonicity();
    validateBins();
    
    // Final computation of WoE and IV
    computeWoEIV();
  }
  
  /**
   * @brief Get binning results
   * 
   * Returns a list with all binning information including:
   * - NumericalBin identifiers and labels
   * - WoE and IV values
   * - Counts (total, positives, negatives)
   * - Cutpoints defining bin boundaries
   * - Algorithm convergence information
   * 
   * @return Rcpp::List Results of the binning process
   */
  List getResults() const {
    size_t n_bins = bins.size();
    CharacterVector bin_names(n_bins);
    NumericVector bin_woe(n_bins);
    NumericVector bin_iv(n_bins);
    IntegerVector bin_count(n_bins);
    IntegerVector bin_count_pos(n_bins);
    IntegerVector bin_count_neg(n_bins);
    NumericVector bin_event_rates(n_bins);
    NumericVector bin_cutpoints(n_bins > 1 ? n_bins - 1 : 0);
    
    // Format bin information for output
    for (size_t i = 0; i < n_bins; ++i) {
      std::ostringstream oss;
      oss << std::fixed << std::setprecision(6);
      
      if (std::isinf(bins[i].lower_bound) && bins[i].lower_bound < 0) {
        oss << "[-Inf;";
      } else {
        oss << "[" << bins[i].lower_bound << ";";
      }
      
      if (std::isinf(bins[i].upper_bound)) {
        oss << "+Inf)";
      } else {
        oss << bins[i].upper_bound << ")";
      }
      
      bin_names[i] = oss.str();
      bin_woe[i] = bins[i].woe;
      bin_iv[i] = bins[i].iv;
      bin_count[i] = bins[i].count;
      bin_count_pos[i] = bins[i].count_pos;
      bin_count_neg[i] = bins[i].count_neg;
      bin_event_rates[i] = bins[i].event_rate();
      
      if (i < n_bins - 1) {
        bin_cutpoints[i] = bins[i].upper_bound;
      }
    }
    
    // Create bin IDs (1-based indexing)
    Rcpp::NumericVector ids(n_bins);
    for(size_t i = 0; i < n_bins; i++) {
      ids[i] = static_cast<double>(i + 1);
    }
    
    // Calculate total IV
    double total_iv = 0.0;
    for (double iv : bin_iv) {
      total_iv += iv;
    }
    
    // Return results as a list
    return Rcpp::List::create(
      Named("id") = ids,
      Named("bin") = bin_names,
      Named("woe") = bin_woe,
      Named("iv") = bin_iv,
      Named("count") = bin_count,
      Named("count_pos") = bin_count_pos,
      Named("count_neg") = bin_count_neg,
      Named("event_rate") = bin_event_rates,
      Named("cutpoints") = bin_cutpoints,
      Named("total_iv") = total_iv,
      Named("converged") = converged,
      Named("iterations") = iterations_run
    );
  }
  
private:
  /**
   * @brief Validate input parameters and data
   * 
   * Checks that all inputs are valid and throws exceptions if not.
   */
  void validateInputs() {
    // Check vector sizes
    if (feature.size() != target.size()) {
      throw std::invalid_argument("Feature and target vectors must be of the same length.");
    }
    
    // Check if vectors are empty
    if (feature.empty()) {
      throw std::invalid_argument("Feature and target vectors cannot be empty.");
    }
    
    // Validate bin parameters
    if (min_bins < 1) {
      throw std::invalid_argument("min_bins must be at least 1.");
    }
    if (max_bins < min_bins) {
      throw std::invalid_argument("max_bins must be greater than or equal to min_bins.");
    }
    if (bin_cutoff <= 0 || bin_cutoff >= 1) {
      throw std::invalid_argument("bin_cutoff must be between 0 and 1.");
    }
    if (convergence_threshold <= 0) {
      throw std::invalid_argument("convergence_threshold must be greater than 0.");
    }
    if (max_iterations <= 0) {
      throw std::invalid_argument("max_iterations must be greater than 0.");
    }
    if (laplace_smoothing < 0) {
      throw std::invalid_argument("laplace_smoothing must be non-negative.");
    }
    
    // Check target validity (must be binary: 0/1)
    bool has_zero = false, has_one = false;
    for (int t : target) {
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
   * @brief Handle missing values in the feature vector
   * 
   * Currently checks for NaN/Inf values and throws an exception if found.
   * In a future version, this could be extended to handle these values.
   */
  void handleMissingValues() {
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
   * @brief Calculate total counts of positive and negative cases
   */
  void calculateTotals() {
    total_pos = 0;
    total_neg = 0;
    
    for (int t : target) {
      if (t == 1) total_pos++;
      else total_neg++;
    }
  }
  
  /**
   * @brief Get unique feature values
   * 
   * @return std::vector<double> Sorted vector of unique feature values
   */
  std::vector<double> getUniqueFeatureValues() {
    std::vector<double> unique_features = feature;
    std::sort(unique_features.begin(), unique_features.end());
    auto last = std::unique(unique_features.begin(), unique_features.end());
    unique_features.erase(last, unique_features.end());
    return unique_features;
  }
  
  /**
   * @brief Adjust min_bins and max_bins based on number of unique values
   * 
   * @param n_unique Number of unique feature values
   */
  void adjustBinParameters(size_t n_unique) {
    if (n_unique < (size_t)min_bins) {
      min_bins = std::max(1, (int)n_unique);
      if (max_bins < min_bins) {
        max_bins = min_bins;
      }
    }
    
    if (n_unique < (size_t)max_bins) {
      max_bins = (int)n_unique;
    }
  }
  
  /**
   * @brief Handle cases with very few unique values
   * 
   * Creates optimal bins when there are only 1 or 2 unique values.
   * 
   * @param unique_features Vector of unique feature values
   */
  void handleLowUniqueValues(const std::vector<double>& unique_features) {
    // If one unique value
    if (unique_features.size() == 1) {
      NumericalBin b;
      b.lower_bound = -std::numeric_limits<double>::infinity();
      b.upper_bound = std::numeric_limits<double>::infinity();
      b.count = (int)feature.size();
      b.count_pos = total_pos;
      b.count_neg = total_neg;
      // b.event_rate() assignment removed (calculated dynamically)
      
      bins.clear();
      bins.push_back(b);
    } else {
      // Two unique values
      double cut = (unique_features[0] + unique_features[1]) / 2.0;
      
      NumericalBin b1, b2;
      b1.lower_bound = -std::numeric_limits<double>::infinity();
      b1.upper_bound = cut;
      b1.count = 0; b1.count_pos = 0; b1.count_neg = 0;
      
      b2.lower_bound = cut;
      b2.upper_bound = std::numeric_limits<double>::infinity();
      b2.count = 0; b2.count_pos = 0; b2.count_neg = 0;
      
      // Assign observations to bins
      for (size_t i = 0; i < feature.size(); ++i) {
        if (feature[i] <= cut) {
          b1.count++;
          if (target[i] == 1) b1.count_pos++; else b1.count_neg++;
        } else {
          b2.count++;
          if (target[i] == 1) b2.count_pos++; else b2.count_neg++;
        }
      }
      
      // Calculate event rates
      // b1.event_rate() assignment removed (calculated dynamically)
      // b2.event_rate() assignment removed (calculated dynamically)
      
      bins.clear();
      bins.push_back(b1);
      bins.push_back(b2);
    }
    
    // Calculate WoE and IV for the bins
    computeWoEIV();
  }
  
  /**
   * @brief Perform initial pre-binning
   * 
   * Creates initial bins based on equal frequency distribution.
   */
  void performPreBinning() {
    // Create pairs of (feature, target) and sort by feature value
    std::vector<std::pair<double, int>> data;
    data.reserve(feature.size());
    
    for (size_t i = 0; i < feature.size(); ++i) {
      if (!NumericVector::is_na(feature[i])) {
        data.emplace_back(feature[i], target[i]);
      }
    }
    
    if (data.empty()) {
      throw std::runtime_error("All feature values are NA.");
    }
    
    std::sort(data.begin(), data.end(),
              [](const std::pair<double,int>& a, const std::pair<double,int>& b) {
                return a.first < b.first;
              });
    
    // Determine number of distinct values
    int distinct_count = 1;
    for (size_t i = 1; i < data.size(); i++) {
      if (data[i].first != data[i-1].first) distinct_count++;
    }
    
    // Determine number of pre-bins
    int n_pre = std::min(max_n_prebins, distinct_count);
    n_pre = std::max(n_pre, min_bins);
    size_t bin_size = std::max((size_t)1, data.size() / (size_t)n_pre);
    
    // Create initial bins
    bins.clear();
    for (size_t i = 0; i < data.size(); i += bin_size) {
      size_t end = std::min(i + bin_size, data.size());
      
      NumericalBin b;
      b.lower_bound = (i == 0) ? -std::numeric_limits<double>::infinity() : data[i].first;
      b.upper_bound = (end == data.size()) ? std::numeric_limits<double>::infinity() : data[end].first;
      b.count = 0; b.count_pos = 0; b.count_neg = 0;
      
      for (size_t j = i; j < end; j++) {
        b.count++;
        if (data[j].second == 1) b.count_pos++; else b.count_neg++;
      }
      
      // b.event_rate() assignment removed (calculated dynamically)
      bins.push_back(b);
    }
    
    // Ensure no empty bins
    bins.erase(
      std::remove_if(bins.begin(), bins.end(), 
                     [](const NumericalBin& b) { return b.count == 0; }),
                     bins.end()
    );
    
    // Initial computation of WoE and IV
    computeWoEIV();
  }
  
  /**
   * @brief Compute Weight of Evidence (WoE) and Information Value (IV)
   * 
   * Calculates WoE and IV for all bins with Laplace smoothing.
   */
  void computeWoEIV() {
    // Recalculate totals to ensure accuracy
    int total_pos = 0;
    int total_neg = 0;
    
    for (const auto& b : bins) {
      total_pos += b.count_pos;
      total_neg += b.count_neg;
    }
    
    // Handle case where all observations are of the same class
    if (total_pos == 0 || total_neg == 0) {
      for (auto& b : bins) {
        b.woe = 0.0;
        b.iv = 0.0;
      }
      return;
    }
    
    // Apply Laplace smoothing and calculate WoE and IV
    for (auto& b : bins) {
      // Apply smoothing to avoid division by zero
      double smoothed_pos = b.count_pos + laplace_smoothing;
      double smoothed_neg = b.count_neg + laplace_smoothing;
      
      double total_smoothed_pos = total_pos + bins.size() * laplace_smoothing;
      double total_smoothed_neg = total_neg + bins.size() * laplace_smoothing;
      
      double dist_pos = smoothed_pos / total_smoothed_pos;
      double dist_neg = smoothed_neg / total_smoothed_neg;
      
      // Calculate WoE with protection against extreme values
      if (dist_pos <= 0.0 && dist_neg <= 0.0) {
        b.woe = 0.0;
      } else if (dist_pos <= 0.0) {
        b.woe = -20.0;  // Cap for stability
      } else if (dist_neg <= 0.0) {
        b.woe = 20.0;   // Cap for stability
      } else {
        b.woe = std::log(dist_pos / dist_neg);
      }
      
      // Calculate IV
      if (std::isfinite(b.woe)) {
        b.iv = (dist_pos - dist_neg) * b.woe;
      } else {
        b.iv = 0.0;
      }
    }
  }
  
  /**
   * @brief Merge bins with frequency below the threshold
   * 
   * Identifies and merges bins that have a proportion of records below bin_cutoff.
   */
  void mergeSmallBins() {
    // Merge bins that fail bin_cutoff
    bool merged = true;
    while (merged && (int)bins.size() > min_bins && iterations_run < max_iterations) {
      merged = false;
      double total = (double)feature.size();
      
      // Find bin with smallest proportion
      size_t smallest_idx = 0;
      double smallest_prop = std::numeric_limits<double>::max();
      
      for (size_t i = 0; i < bins.size(); i++) {
        double prop = (double)bins[i].count / total;
        if (prop < smallest_prop) {
          smallest_prop = prop;
          smallest_idx = i;
        }
      }
      
      // If smallest bin is below threshold, merge it
      if (smallest_prop < bin_cutoff && bins.size() > (size_t)min_bins) {
        // Determine optimal merge direction
        if (smallest_idx == 0 && bins.size() > 1) {
          // Leftmost bin, merge with right neighbor
          mergeBins(0, 1);
        } else if (smallest_idx == bins.size() - 1 && bins.size() > 1) {
          // Rightmost bin, merge with left neighbor
          mergeBins(bins.size() - 2, bins.size() - 1);
        } else if (smallest_idx > 0 && smallest_idx < bins.size() - 1) {
          // Middle bin, decide based on information preservation
          double iv_loss_left = bins[smallest_idx - 1].iv + bins[smallest_idx].iv;
          double iv_loss_right = bins[smallest_idx].iv + bins[smallest_idx + 1].iv;
          
          // Try to merge with neighbor that would result in least IV loss
          if (iv_loss_left <= iv_loss_right) {
            mergeBins(smallest_idx - 1, smallest_idx);
          } else {
            mergeBins(smallest_idx, smallest_idx + 1);
          }
        }
        
        computeWoEIV();
        merged = true;
      }
      
      iterations_run++;
    }
  }
  
  /**
   * @brief Check if WoE values are monotonic
   * 
   * @param increasing Whether to check for monotonically increasing (true) or decreasing (false)
   * @return bool True if the WoE values are monotonic in the specified direction
   */
  bool isMonotonic(bool increasing) const {
    if (bins.size() < 2) return true;
    
    for (size_t i = 1; i < bins.size(); i++) {
      if (increasing && bins[i].woe < bins[i-1].woe) return false;
      if (!increasing && bins[i].woe > bins[i-1].woe) return false;
    }
    return true;
  }
  
  /**
   * @brief Determine if WoE values should be monotonically increasing or decreasing
   * 
   * @return bool True if increasing, false if decreasing
   */
  bool guessIncreasing() const {
    if (bins.size() < 2) return true;
    
    int inc = 0, dec = 0;
    for (size_t i = 1; i < bins.size(); i++) {
      if (bins[i].woe > bins[i-1].woe) inc++;
      else if (bins[i].woe < bins[i-1].woe) dec++;
    }
    
    return inc >= dec;
  }
  
  /**
   * @brief Enforce monotonicity of WoE values
   * 
   * Merges bins to ensure WoE values are monotonically increasing or decreasing.
   */
  void enforceMonotonicity() {
    bool increasing = guessIncreasing();
    
    while (!isMonotonic(increasing) && (int)bins.size() > min_bins && iterations_run < max_iterations) {
      bool merged = false;
      
      // Find first violation of monotonicity
      for (size_t i = 1; i < bins.size(); i++) {
        if ((increasing && bins[i].woe < bins[i-1].woe) ||
            (!increasing && bins[i].woe > bins[i-1].woe)) {
          
          // Try to merge bin i-1 and i
          NumericalBin merged_bin = bins[i-1];
          merged_bin.upper_bound = bins[i].upper_bound;
          merged_bin.count += bins[i].count;
          merged_bin.count_pos += bins[i].count_pos;
          merged_bin.count_neg += bins[i].count_neg;
          // merged_bin.event_rate() assignment removed (calculated dynamically)
          
          // Check if merge would fix monotonicity with neighbors
          bool merge_fixes = true;
          if (i > 1) {
            // Calculate temporary WoE for merged bin
            double total_pos = 0, total_neg = 0;
            for (const auto& b : bins) {
              total_pos += b.count_pos;
              total_neg += b.count_neg;
            }
            
            double smoothed_pos = merged_bin.count_pos + laplace_smoothing;
            double smoothed_neg = merged_bin.count_neg + laplace_smoothing;
            double total_smoothed_pos = total_pos + (bins.size() - 1) * laplace_smoothing;
            double total_smoothed_neg = total_neg + (bins.size() - 1) * laplace_smoothing;
            
            double dist_pos = smoothed_pos / total_smoothed_pos;
            double dist_neg = smoothed_neg / total_smoothed_neg;
            double merged_woe = std::log(dist_pos / dist_neg);
            
            // Check if merged bin WoE maintains monotonicity with left neighbor
            if ((increasing && merged_woe < bins[i-2].woe) ||
                (!increasing && merged_woe > bins[i-2].woe)) {
              merge_fixes = false;
            }
          }
          
          // Execute merge if it fixes the issue
          if (merge_fixes) {
            mergeBins(i-1, i);
            merged = true;
            break;
          } else if (i < bins.size() - 1) {
            // Try merging with right neighbor instead
            mergeBins(i, i+1);
            merged = true;
            break;
          } else {
            // Last resort: just merge the violating bins
            mergeBins(i-1, i);
            merged = true;
            break;
          }
        }
      }
      
      if (!merged) break;
      
      computeWoEIV();
      iterations_run++;
      
      // Check if monotonicity has been achieved
      if (isMonotonic(increasing)) {
        converged = true;
        break;
      }
      
      // Check for convergence based on threshold
      if (iterations_run > 1 && std::fabs(bins.back().woe - bins.front().woe) < convergence_threshold) {
        converged = true;
        break;
      }
    }
    
    // Mark as not converged if max iterations reached
    if (iterations_run >= max_iterations) {
      converged = false;
    }
    
    // Ensure bins don't exceed max_bins
    while ((int)bins.size() > max_bins && iterations_run < max_iterations) {
      mergeBinsByIV();
      computeWoEIV();
      iterations_run++;
    }
    
    if (iterations_run >= max_iterations) {
      converged = false;
    }
  }
  
  /**
   * @brief Find the pair of adjacent bins with smallest IV difference
   * 
   * @return size_t Index of the left bin in the optimal merge pair
   */
  size_t findMinIVDiffMerge() const {
    if (bins.size() < 2) return bins.size();
    
    double min_iv_diff = std::numeric_limits<double>::max();
    size_t merge_idx = bins.size();
    
    for (size_t i = 0; i < bins.size() - 1; i++) {
      double iv_diff = std::fabs(bins[i].iv - bins[i+1].iv);
      if (iv_diff < min_iv_diff) {
        min_iv_diff = iv_diff;
        merge_idx = i;
      }
    }
    
    return merge_idx;
  }
  
  /**
   * @brief Merge bins based on minimum IV difference
   * 
   * Finds and merges the pair of adjacent bins with smallest IV difference.
   */
  void mergeBinsByIV() {
    size_t idx = findMinIVDiffMerge();
    if (idx < bins.size() - 1) {
      mergeBins(idx, idx + 1);
    }
  }
  
  /**
   * @brief Merge two adjacent bins
   * 
   * @param idx1 Index of the left bin
   * @param idx2 Index of the right bin
   */
  void mergeBins(size_t idx1, size_t idx2) {
    if (idx1 > idx2) std::swap(idx1, idx2);
    if (idx2 >= bins.size()) return;
    
    // Update left bin with combined statistics
    bins[idx1].upper_bound = bins[idx2].upper_bound;
    bins[idx1].count += bins[idx2].count;
    bins[idx1].count_pos += bins[idx2].count_pos;
    bins[idx1].count_neg += bins[idx2].count_neg;
    // bins[idx1].event_rate() assignment removed (calculated dynamically)
    
    // Remove right bin
    bins.erase(bins.begin() + idx2);
  }
  
  /**
   * @brief Validate the final bin structure
   * 
   * Checks for issues like empty bins, invalid boundaries, etc.
   */
  void validateBins() const {
    if (bins.empty()) {
      throw std::runtime_error("No bins created during binning process.");
    }
    
    // Check bin boundaries
    for (size_t i = 1; i < bins.size(); i++) {
      if (bins[i].lower_bound != bins[i-1].upper_bound) {
        throw std::runtime_error("Invalid bin boundaries: gap or overlap detected.");
      }
    }
    
    // Check for empty bins
    for (size_t i = 0; i < bins.size(); i++) {
      if (bins[i].count == 0) {
        throw std::runtime_error("Empty bin detected at index " + std::to_string(i));
      }
    }
    
    // Check first and last bin boundaries
    if (bins.front().lower_bound != -std::numeric_limits<double>::infinity()) {
      throw std::runtime_error("First bin must start at -Infinity.");
    }
    if (bins.back().upper_bound != std::numeric_limits<double>::infinity()) {
      throw std::runtime_error("Last bin must end at +Infinity.");
    }
  }
};


// [[Rcpp::export]]
List optimal_binning_numerical_mrblp(const IntegerVector& target,
                                    const NumericVector& feature,
                                    int min_bins = 3,
                                    int max_bins = 5,
                                    double bin_cutoff = 0.05,
                                    int max_n_prebins = 20,
                                    double convergence_threshold = 1e-6,
                                    int max_iterations = 1000,
                                    double laplace_smoothing = 0.5) {
 try {
   OBN_MRBLP binning(feature, target, min_bins, max_bins, bin_cutoff,
                                        max_n_prebins, convergence_threshold, max_iterations,
                                        laplace_smoothing);
   binning.fit();
   return binning.getResults();
 } catch (const std::exception& e) {
   Rcpp::stop(std::string("Error in optimal_binning_numerical_mrblp: ") + e.what());
 }
}
