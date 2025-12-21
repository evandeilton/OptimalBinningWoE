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
 * @brief NumericalBin structure to store bin information
 * 
 * Contains all metrics associated with a single bin including
 * boundaries, counts, and statistical measures.
 */
// Local NumericalBin definition removed


/**
 * @brief Optimal Binning for Numerical Variables using Unsupervised Binning with Standard Deviation (UBSD)
 *
 * This class implements an advanced algorithm for optimal binning of numerical variables
 * using a hybrid approach that starts with unsupervised binning based on standard deviation
 * and refines the bins using Weight of Evidence (WoE) and Information Value (IV) criteria.
 *
 * Algorithm Overview:
 * 1. Initial bins created using statistical properties (mean ± k*std_dev)
 * 2. Supervised refinement using WoE/IV optimization
 * 3. Greedy merging of small or uninformative bins
 * 4. Monotonicity enforcement for WoE values
 * 5. Laplace smoothing for stable WoE in sparse bins
 *
 * Complexity: O(n log n + k²) where n = sample size, k = number of bins
 * Space: O(n + k)
 *
 * Key Features:
 * - Hybrid approach: Unsupervised initialization + supervised optimization
 * - Statistical foundation: Standard deviation-based initial splits
 * - Robust handling of edge cases and sparse bins
 * - Comprehensive bin validation
 *
 * References:
 * - Thomas, L.C. (2009). "Consumer Credit Models: Pricing, Profit, and Portfolios"
 * - Scott, D.W. (2015). "Multivariate Density Estimation: Theory, Practice, and Visualization"
 * - Good, I.J. (1952). "Rational Decisions", Journal of the Royal Statistical Society
 */
class OBN_UBSD {
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
  
  // NumericalBin structure and results
  std::vector<NumericalBin> bins;
  std::vector<double> cutpoints;
  double total_iv;
  bool converged;
  int iterations_run;
  
  // Constants
  // Constant removed (uses shared definition)
  
public:
  /**
   * @brief Constructor for OBN_UBSD
   * 
   * @param feat Vector of feature values to bin
   * @param targ Vector of binary target values (0/1)
   * @param min_b Minimum number of bins 
   * @param max_b Maximum number of bins
   * @param cutoff Minimum proportion of observations in a bin
   * @param max_prebins Maximum number of pre-bins
   * @param conv_threshold Convergence threshold for IV optimization
   * @param max_iter Maximum number of iterations
   * @param laplace_smooth Smoothing parameter for WoE calculation
   */
  OBN_UBSD(
    const std::vector<double>& feat, 
    const std::vector<double>& targ,
    int min_b = 3, 
    int max_b = 5, 
    double cutoff = 0.05, 
    int max_prebins = 20,
    double conv_threshold = 1e-6, 
    int max_iter = 1000,
    double laplace_smooth = 0.5
  ) : feature(feat), target(targ),
  min_bins(std::max(min_b, 2)), 
  max_bins(std::max(max_b, min_bins)),
  bin_cutoff(cutoff),
  max_n_prebins(std::max(max_prebins, min_bins)),
  convergence_threshold(conv_threshold),
  max_iterations(max_iter),
  laplace_smoothing(laplace_smooth),
  total_iv(0.0),
  converged(false),
  iterations_run(0) {
    // Validate inputs immediately upon construction
    validate_inputs();
  }
  
  /**
   * @brief Fit the optimal binning model
   * 
   * Main method to execute the binning algorithm and calculate all metrics
   */
  void fit() {
    // Handle missing or extreme values
    handle_missing_values();
    
    // Check unique values
    std::vector<double> unique_values = get_unique_values();
    
    if(unique_values.size() <= 2) {
      // Handle special case: very few unique values
      handle_low_unique_values(unique_values);
      converged = true;
      iterations_run = 0;
      return;
    }
    
    // Main algorithm steps
    create_initial_bins();
    assign_observations_to_bins();
    merge_small_bins();
    calculate_woe_iv();
    
    // Optimization loop
    double prev_iv = get_total_iv();
    for (int iter = 0; iter < max_iterations; ++iter) {
      enforce_monotonicity();
      adjust_bin_count();
      calculate_woe_iv();
      
      double current_iv = get_total_iv();
      if (std::fabs(current_iv - prev_iv) < convergence_threshold) {
        converged = true;
        iterations_run = iter + 1;
        break;
      }
      prev_iv = current_iv;
      iterations_run = iter + 1;
    }
    
    if (!converged) {
      iterations_run = max_iterations;
    }
    
    // Update cutpoints for reporting
    update_cutpoints();
    
    // Validate final binning
    validate_final_bins();
  }
  
  /**
   * @brief Create output list with all binning results
   * 
   * @return Rcpp::List Results of the binning process
   */
  Rcpp::List create_output() const {
    std::vector<std::string> bin_names = get_bin_names();
    std::vector<double> woe_vals = get_bin_woe();
    std::vector<double> iv_vals = get_bin_iv();
    std::vector<int> c_vals = get_bin_count();
    std::vector<int> cpos_vals = get_bin_count_pos();
    std::vector<int> cneg_vals = get_bin_count_neg();
    std::vector<double> event_rates = get_bin_event_rates();
    
    // Create bin IDs (1-based indexing)
    Rcpp::NumericVector ids(bin_names.size());
    for(size_t i = 0; i < bin_names.size(); i++) {
      ids[i] = static_cast<double>(i + 1);
    }
    
    return Rcpp::List::create(
      Named("id") = ids,
      Named("bin") = bin_names,
      Named("woe") = woe_vals,
      Named("iv") = iv_vals,
      Named("count") = c_vals,
      Named("count_pos") = cpos_vals,
      Named("count_neg") = cneg_vals,
      Named("event_rate") = event_rates,
      Named("cutpoints") = cutpoints,
      Named("total_iv") = total_iv,
      Named("converged") = converged,
      Named("iterations") = iterations_run
    );
  }
  
  // Accessors for testing and inspection - all marked as const
  std::vector<std::string> get_bin_names() const {
    std::vector<std::string> names;
    for (const auto &b : bins) {
      std::ostringstream oss;
      oss << std::fixed << std::setprecision(6);
      
      if (std::isinf(b.lower_bound) && b.lower_bound < 0) {
        oss << "[-Inf;";
      } else {
        oss << "[" << b.lower_bound << ";";
      }
      
      if (std::isinf(b.upper_bound)) {
        oss << "+Inf)";
      } else {
        oss << b.upper_bound << ")";
      }
      
      names.push_back(oss.str());
    }
    return names;
  }
  
  std::vector<double> get_bin_woe() const {
    std::vector<double> w;
    for(const auto &b : bins) w.push_back(b.woe);
    return w;
  }
  
  std::vector<double> get_bin_iv() const {
    std::vector<double> v;
    for(const auto &b : bins) v.push_back(b.iv);
    return v;
  }
  
  std::vector<int> get_bin_count() const {
    std::vector<int> c;
    for(const auto &b : bins) c.push_back(b.count);
    return c;
  }
  
  std::vector<int> get_bin_count_pos() const {
    std::vector<int> c;
    for(const auto &b : bins) c.push_back(b.count_pos);
    return c;
  }
  
  std::vector<int> get_bin_count_neg() const {
    std::vector<int> c;
    for(const auto &b : bins) c.push_back(b.count_neg);
    return c;
  }
  
  std::vector<double> get_bin_event_rates() const {
    std::vector<double> rates;
    for(const auto &b : bins) rates.push_back(b.event_rate());
    return rates;
  }
  
  std::vector<double> get_cutpoints() const {
    return cutpoints;
  }
  
  bool get_converged() const {
    return converged;
  }
  
  int get_iterations_run() const {
    return iterations_run;
  }
  
  double get_total_iv() const {
    double sum = 0.0;
    for (const auto &b : bins) {
      sum += b.iv;
    }
    return sum;
  }
  
private:
  /**
   * @brief Validate input parameters and data
   * 
   * Checks for valid inputs and throws exceptions if necessary
   */
  void validate_inputs() {
    if (feature.size() != target.size()) {
      Rcpp::stop("Feature and target must have the same length.");
    }
    
    if (feature.empty()) {
      Rcpp::stop("Feature and target vectors cannot be empty.");
    }
    
    if (min_bins < 2) {
      Rcpp::stop("min_bins must be at least 2.");
    }
    
    if (max_bins < min_bins) {
      Rcpp::stop("max_bins must be >= min_bins.");
    }
    
    if (bin_cutoff <= 0 || bin_cutoff >= 1) {
      Rcpp::stop("bin_cutoff must be between 0 and 1.");
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
    
    // Check that target is binary (0/1)
    bool has_zero = false, has_one = false;
    for (double t : target) {
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
   * @brief Handle missing or extreme values in the feature
   * 
   * Checks for NaN or Inf values and reports them
   */
  void handle_missing_values() {
    int nan_count = 0, inf_count = 0;
    
    for (double f : feature) {
      if (std::isnan(f)) {
        nan_count++;
      } else if (std::isinf(f)) {
        inf_count++;
      }
    }
    
    if (nan_count > 0 || inf_count > 0) {
      Rcpp::stop("Feature contains " + std::to_string(nan_count) + " NaN and " +
        std::to_string(inf_count) + " Inf values. Please handle these values before binning.");
    }
  }
  
  /**
   * @brief Get unique values in the feature vector
   * 
   * @return std::vector<double> Sorted vector of unique values
   */
  std::vector<double> get_unique_values() const {
    std::set<double> unique_set(feature.begin(), feature.end());
    return std::vector<double>(unique_set.begin(), unique_set.end());
  }
  
  /**
   * @brief Handle cases with very few unique values
   * 
   * Creates appropriate bins when there are only 1 or 2 unique values
   * 
   * @param unique_vals Vector of unique feature values
   */
  void handle_low_unique_values(const std::vector<double>& unique_vals) {
    bins.clear();
    
    if(unique_vals.size() == 1) {
      // All identical values - create a single bin
      bins.emplace_back(-std::numeric_limits<double>::infinity(),
                        std::numeric_limits<double>::infinity());
      
      for (size_t i = 0; i < feature.size(); i++) {
        bins[0].count++;
        if (target[i] == 1) bins[0].count_pos++; else bins[0].count_neg++;
      }
      
      // Calculate event rate
      // bins[0].event_rate() assignment removed (calculated dynamically)
      
      calculate_woe_iv();
    } else {
      // Two unique values - create two bins with a boundary in between
      double midpoint = (unique_vals[0] + unique_vals[1]) / 2.0;
      
      bins.emplace_back(-std::numeric_limits<double>::infinity(), midpoint);
      bins.emplace_back(midpoint, std::numeric_limits<double>::infinity());
      
      for (size_t i = 0; i < feature.size(); i++) {
        int bin_idx = (feature[i] <= midpoint) ? 0 : 1;
        
        bins[bin_idx].count++;
        if(target[i] == 1) bins[bin_idx].count_pos++; else bins[bin_idx].count_neg++;
      }
      
      // Event rates are calculated dynamically via event_rate() method
      // Removed empty loop - no action needed
      
      
      calculate_woe_iv();
    }
  }
  
  /**
   * @brief Calculate mean of a vector
   * 
   * @param v Vector of values
   * @return double Mean value
   */
  double mean(const std::vector<double>& v) const {
    if(v.empty()) return 0.0;
    return std::accumulate(v.begin(), v.end(), 0.0) / v.size();
  }
  
  /**
   * @brief Calculate standard deviation of a vector
   * 
   * @param v Vector of values
   * @return double Standard deviation
   */
  double stddev(const std::vector<double>& v) const {
    if(v.size() < 2) return 0.0;
    double m = mean(v);
    double accum = 0.0;
    for(double val : v) {
      accum += (val - m) * (val - m);
    }
    return std::sqrt(accum / (v.size() - 1));
  }
  
  /**
   * @brief Create initial bins using statistical properties
   * 
   * Creates bins based on mean and standard deviation or quantile-based approach
   */
  void create_initial_bins() {
    double m = mean(feature);
    double sd = stddev(feature);
    
    // Reset bins
    bins.clear();
    
    // Define edges starting with -Infinity
    std::vector<double> edges;
    edges.push_back(-std::numeric_limits<double>::infinity());
    
    if (sd == 0.0 || sd < EPSILON) {
      // Special case: All values are very close together
      // Use linear spacing between min and max
      double min_val = *std::min_element(feature.begin(), feature.end());
      double max_val = *std::max_element(feature.begin(), feature.end());
      
      // If min and max are effectively the same, create a single bin
      if (std::fabs(max_val - min_val) < EPSILON) {
        edges.push_back(std::numeric_limits<double>::infinity());
      } else {
        // Create equally spaced bins
        double step = (max_val - min_val) / min_bins;
        for (int i = 1; i < min_bins; i++) {
          edges.push_back(min_val + i * step);
        }
        edges.push_back(std::numeric_limits<double>::infinity());
      }
    } else {
      // Normal case: Use standard deviation-based binning
      
      // Determine number of pre-bins
      int n_pre = std::min(max_n_prebins, max_bins);
      n_pre = std::max(n_pre, min_bins);
      
      // Create bins around mean using standard deviation
      double min_val = *std::min_element(feature.begin(), feature.end());
      double max_val = *std::max_element(feature.begin(), feature.end());
      
      // Use a hybrid approach: combine SD-based and equal-width binning
      double range = max_val - min_val;
      
      // Try standard deviation-based bounds first
      std::set<double> edge_set;
      edge_set.insert(-std::numeric_limits<double>::infinity());
      
      // Add mean-based cutpoints
      edge_set.insert(m - 2.0 * sd);
      edge_set.insert(m - 1.0 * sd);
      edge_set.insert(m);
      edge_set.insert(m + 1.0 * sd);
      edge_set.insert(m + 2.0 * sd);
      
      // Add equally spaced points to fill in
      double step = range / n_pre;
      for (int i = 1; i < n_pre; i++) {
        edge_set.insert(min_val + i * step);
      }
      
      // Add +Infinity
      edge_set.insert(std::numeric_limits<double>::infinity());
      
      // Convert to vector and sort
      edges = std::vector<double>(edge_set.begin(), edge_set.end());
      std::sort(edges.begin(), edges.end());
      
      // Limit to max_n_prebins
      if (edges.size() > static_cast<size_t>(max_n_prebins + 1)) {
        std::vector<double> sampled_edges;
        sampled_edges.push_back(-std::numeric_limits<double>::infinity());
        
        double sampling_step = static_cast<double>(edges.size() - 2) / (max_n_prebins - 1);
        for (int i = 1; i < max_n_prebins; i++) {
          int idx = 1 + static_cast<int>(i * sampling_step);
          if (idx < static_cast<int>(edges.size()) - 1) {
            sampled_edges.push_back(edges[idx]);
          }
        }
        
        sampled_edges.push_back(std::numeric_limits<double>::infinity());
        edges = sampled_edges;
      }
    }
    
    // Create bins from edges
    for (size_t i = 0; i < edges.size() - 1; i++) {
      bins.emplace_back(edges[i], edges[i+1]);
    }
  }
  
  /**
   * @brief Assign observations to bins
   * 
   * Counts observations in each bin and calculates initial statistics
   */
  void assign_observations_to_bins() {
    // Reset counts
    for (auto &b : bins) {
      b.count = 0;
      b.count_pos = 0;
      b.count_neg = 0;
    }
    
    // Assign observations to bins
    for (size_t i = 0; i < feature.size(); i++) {
      double val = feature[i];
      int idx = find_bin(val);
      
      if (idx >= 0 && static_cast<size_t>(idx) < bins.size()) {
        bins[idx].count++;
        if (target[i] == 1) {
          bins[idx].count_pos++;
        } else {
          bins[idx].count_neg++;
        }
      }
    }
    
    // Event rates are calculated dynamically via event_rate() method
    // Removed empty loop - no action needed
  }
  
  /**
   * @brief Find which bin a value belongs to
   * 
   * @param val Value to find bin for
   * @return int Index of the bin
   */
  int find_bin(double val) const {
    for (size_t i = 0; i < bins.size(); i++) {
      if (val > bins[i].lower_bound && val <= bins[i].upper_bound) {
        return static_cast<int>(i);
      }
    }
    
    // Fallback: return last bin
    return static_cast<int>(bins.size()) - 1;
  }
  
  /**
   * @brief Merge bins with frequency below the threshold
   * 
   * Identifies and merges small bins to ensure statistical significance
   */
  void merge_small_bins() {
    bool merged = true;
    double total_count = static_cast<double>(feature.size());
    
    while (merged && static_cast<int>(bins.size()) > min_bins) {
      merged = false;
      
      // Find smallest bin
      size_t smallest_idx = 0;
      double smallest_prop = std::numeric_limits<double>::max();
      
      for (size_t i = 0; i < bins.size(); i++) {
        double prop = static_cast<double>(bins[i].count) / total_count;
        if (prop < smallest_prop) {
          smallest_prop = prop;
          smallest_idx = i;
        }
      }
      
      // If smallest bin is below cutoff, merge it
      if (smallest_prop < bin_cutoff && bins.size() > static_cast<size_t>(min_bins)) {
        // Determine optimal merge direction
        if (smallest_idx == 0) {
          // Leftmost bin, merge with right
          if (bins.size() < 2) break;
          
          merge_bins(0, 1);
        } else if (smallest_idx == bins.size() - 1) {
          // Rightmost bin, merge with left
          merge_bins(bins.size() - 2, smallest_idx);
        } else {
          // Middle bin, determine best merge direction based on info value
          double iv_left = bins[smallest_idx - 1].iv + bins[smallest_idx].iv;
          double iv_right = bins[smallest_idx].iv + bins[smallest_idx + 1].iv;
          
          if (iv_left <= iv_right) {
            // Merge with left bin
            merge_bins(smallest_idx - 1, smallest_idx);
          } else {
            // Merge with right bin
            merge_bins(smallest_idx, smallest_idx + 1);
          }
        }
        
        merged = true;
      } else {
        break; // No more bins to merge
      }
    }
  }
  
  /**
   * @brief Calculate Weight of Evidence (WoE) and Information Value (IV)
   * 
   * Computes WoE and IV for all bins with Laplace smoothing
   */
  void calculate_woe_iv() {
    // Calculate totals
    double total_pos = 0.0;
    double total_neg = 0.0;
    
    for (const auto &b : bins) {
      total_pos += b.count_pos;
      total_neg += b.count_neg;
    }
    
    // Reset total IV
    total_iv = 0.0;
    
    // Calculate WoE and IV for each bin with Laplace smoothing
    for (auto &b : bins) {
      // Apply smoothing
      double smoothed_pos = b.count_pos + laplace_smoothing;
      double smoothed_neg = b.count_neg + laplace_smoothing;
      
      double total_smoothed_pos = total_pos + bins.size() * laplace_smoothing;
      double total_smoothed_neg = total_neg + bins.size() * laplace_smoothing;
      
      double p = smoothed_pos / total_smoothed_pos;
      double q = smoothed_neg / total_smoothed_neg;
      
      // Calculate WoE with protection against extreme values
      if (p <= 0.0 && q <= 0.0) {
        b.woe = 0.0;
      } else if (p <= 0.0) {
        b.woe = -20.0;  // Cap for stability
      } else if (q <= 0.0) {
        b.woe = 20.0;   // Cap for stability
      } else {
        b.woe = std::log(p / q);
      }
      
      // Calculate IV
      if (std::isfinite(b.woe)) {
        b.iv = (p - q) * b.woe;
      } else {
        b.iv = 0.0;
      }
      
      // Add to total IV
      total_iv += b.iv;
    }
  }
  
  /**
   * @brief Enforce monotonicity of WoE values
   * 
   * Merges bins to ensure monotonically increasing or decreasing WoE
   */
  void enforce_monotonicity() {
    if (bins.size() <= 2) return;
    
    // Determine if WoE should be increasing or decreasing
    bool increasing = guess_trend();
    
    bool merged = true;
    while (merged && static_cast<int>(bins.size()) > min_bins && iterations_run < max_iterations) {
      merged = false;
      
      // Find first violation of monotonicity
      for (size_t i = 1; i < bins.size(); i++) {
        if ((increasing && bins[i].woe < bins[i-1].woe) ||
            (!increasing && bins[i].woe > bins[i-1].woe)) {
          
          // Try to determine which merge would preserve more information
          // (iv_before could be calculated as bins[i-1].iv + bins[i].iv if needed)
          
          // Estimate IV after merge
          NumericalBin merged_bin = bins[i-1];
          merged_bin.upper_bound = bins[i].upper_bound;
          merged_bin.count += bins[i].count;
          merged_bin.count_pos += bins[i].count_pos;
          merged_bin.count_neg += bins[i].count_neg;
          
          // Calculate temporary WoE and IV
          double total_pos = 0.0, total_neg = 0.0;
          for (const auto &b : bins) {
            total_pos += b.count_pos;
            total_neg += b.count_neg;
          }
          
          double smoothed_pos = merged_bin.count_pos + laplace_smoothing;
          double smoothed_neg = merged_bin.count_neg + laplace_smoothing;
          
          double total_smoothed_pos = total_pos + (bins.size() - 1) * laplace_smoothing;
          double total_smoothed_neg = total_neg + (bins.size() - 1) * laplace_smoothing;
          
          double p = smoothed_pos / total_smoothed_pos;
          double q = smoothed_neg / total_smoothed_neg;
          
          double merged_woe = std::log(p / q);
          // merged_iv could be calculated as: (p - q) * merged_woe if needed
          
          // Check if merge will create a new violation with previous bin
          bool new_violation = false;
          if (i > 1) {
            if ((increasing && merged_woe < bins[i-2].woe) ||
                (!increasing && merged_woe > bins[i-2].woe)) {
              new_violation = true;
            }
          }
          
          if (!new_violation) {
            // Standard case: merge i-1 and i
            merge_bins(i-1, i);
          } else if (i < bins.size() - 1) {
            // Try merging i and i+1 instead
            merge_bins(i, i+1);
          } else {
            // Last resort: force merge i-1 and i despite new violation
            merge_bins(i-1, i);
          }
          
          calculate_woe_iv();
          merged = true;
          break;
        }
      }
      
      iterations_run++;
    }
  }
  
  /**
   * @brief Adjust number of bins to respect max_bins
   * 
   * Merges bins if there are too many, prioritizing minimal information loss
   */
  void adjust_bin_count() {
    // If too many bins, merge by minimal IV difference
    while (static_cast<int>(bins.size()) > max_bins && iterations_run < max_iterations) {
      size_t idx = find_min_iv_merge();
      if (idx >= bins.size() - 1) break;
      
      merge_bins(idx, idx + 1);
      calculate_woe_iv();
      iterations_run++;
    }
  }
  
  /**
   * @brief Find the pair of adjacent bins with smallest IV sum
   * 
   * @return size_t Index of the left bin in the pair with smallest IV
   */
  size_t find_min_iv_merge() const {
    if (bins.size() < 2) return bins.size();
    
    double min_iv_sum = std::numeric_limits<double>::max();
    size_t idx = bins.size();
    
    for (size_t i = 0; i < bins.size() - 1; i++) {
      double iv_sum = bins[i].iv + bins[i+1].iv;
      if (iv_sum < min_iv_sum) {
        min_iv_sum = iv_sum;
        idx = i;
      }
    }
    
    return idx;
  }
  
  /**
   * @brief Determine if WoE values should be increasing or decreasing
   * 
   * @return bool True if WoE should be increasing, false if decreasing
   */
  bool guess_trend() const {
    int inc = 0;
    int dec = 0;
    
    for (size_t i = 1; i < bins.size(); i++) {
      if (bins[i].woe > bins[i-1].woe) inc++;
      else if (bins[i].woe < bins[i-1].woe) dec++;
    }
    
    return inc >= dec;
  }
  
  /**
   * @brief Merge two bins
   * 
   * Combines the statistics of two adjacent bins
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
    
    // Update event rate
    // bins[i].event_rate() assignment removed (calculated dynamically)
    
    // Remove right bin
    bins.erase(bins.begin() + j);
  }
  
  /**
   * @brief Update cutpoints for reporting
   * 
   * Extracts bin boundaries excluding -Inf and +Inf
   */
  void update_cutpoints() {
    cutpoints.clear();
    
    for (size_t i = 1; i < bins.size(); i++) {
      if(std::isfinite(bins[i].lower_bound)) {
        cutpoints.push_back(bins[i].lower_bound);
      }
    }
  }
  
  /**
   * @brief Format a bin edge as a string
   * 
   * Handles special values like -Inf and +Inf
   * 
   * @param val Numeric value to format
   * @return std::string Formatted string representation
   */
  std::string edge_to_str(double val) const {
    if (std::isinf(val)) {
      return val < 0 ? "-Inf" : "+Inf";
    } else {
      std::ostringstream oss;
      oss << std::fixed << std::setprecision(6) << val;
      return oss.str();
    }
  }
  
  /**
   * @brief Validate the final binning solution
   * 
   * Performs checks to ensure the binning is valid
   */
  void validate_final_bins() const {
    // Check if there are any bins
    if (bins.empty()) {
      Rcpp::warning("No bins were created.");
      return;
    }
    
    // Check bin boundaries
    if (bins.front().lower_bound != -std::numeric_limits<double>::infinity()) {
      Rcpp::warning("First bin doesn't start at -Inf.");
    }
    
    if (bins.back().upper_bound != std::numeric_limits<double>::infinity()) {
      Rcpp::warning("Last bin doesn't end at +Inf.");
    }
    
    // Check for empty bins
    for (size_t i = 0; i < bins.size(); i++) {
      if (bins[i].count == 0) {
        Rcpp::warning("NumericalBin %d is empty.", i+1);
      }
    }
    
    // Check for monotonicity
    bool increasing = guess_trend();
    bool is_monotonic = true;
    
    for (size_t i = 1; i < bins.size(); i++) {
      if ((increasing && bins[i].woe < bins[i-1].woe) ||
          (!increasing && bins[i].woe > bins[i-1].woe)) {
        is_monotonic = false;
        break;
      }
    }
    
    if (!is_monotonic) {
      Rcpp::warning("WoE values are not monotonic.");
    }
  }
};


// [[Rcpp::export]]
Rcpp::List optimal_binning_numerical_ubsd(
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
 
 if (min_bins < 2) {
   Rcpp::stop("min_bins must be at least 2.");
 }
 
 if (max_bins < min_bins) {
   Rcpp::stop("max_bins must be greater than or equal to min_bins.");
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
 std::vector<double> f = as<std::vector<double>>(feature);
 std::vector<double> t = as<std::vector<double>>(target);
 
 try {
   // Create and run the binning algorithm
   OBN_UBSD model(
       f, t, 
       min_bins, max_bins, 
       bin_cutoff, max_n_prebins, 
       convergence_threshold, max_iterations,
       laplace_smoothing
   );
   
   model.fit();
   return model.create_output();
 } catch (const std::exception &e) {
   Rcpp::stop(std::string("Error in optimal_binning_numerical_ubsd: ") + e.what());
 }
}

