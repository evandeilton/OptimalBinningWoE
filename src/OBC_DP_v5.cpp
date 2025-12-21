// [[Rcpp::depends(Rcpp)]]

#include <Rcpp.h>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <string>
#include <algorithm>
#include <cmath>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <memory>
#include <limits>
#include <chrono>

using namespace Rcpp;

// Include shared headers
#include "common/optimal_binning_common.h"
#include "common/bin_structures.h"

using namespace Rcpp;
using namespace OptimalBinning;


// Constants for optimization
// Local constant removed (uses shared definition)
constexpr double NEGATIVE_INFINITY = -std::numeric_limits<double>::infinity();

/**
 * @brief Structure to store information about a category
 * 
 * This structure maintains count statistics for each category,
 * including total observations, positive events, and event rate.
 */
// struct CategoryStats {
//   int count;          // Total count of observations
//   int pos_count;      // Count of positive events (target=1)
//   int neg_count;      // Count of negative events (target=0)
//   double event_rate;  // Ratio of positive events to total count
//   
//   // Default constructor
//   CategoryStats() : count(0), pos_count(0), neg_count(0), event_rate(0.0) {}
//   
//   /**
//    * @brief Update category statistics with a new observation
//    * 
//    * @param is_positive Boolean indicating if the observation is a positive event
//    */
//   void update(int is_positive) {
//     count++;
//     if (is_positive) {
//       pos_count++;
//     } else {
//       neg_count++;
//     }
//     event_rate = static_cast<double>(pos_count) / static_cast<double>(count);
//   }
// };

/**
 * @brief Structure to store information about a bin
 * 
 * Contains all the relevant metrics for a bin in the final binning solution,
 * including WoE (Weight of Evidence) and IV (Information Value).
 */
// Local CategoricalBin definition removed


/**
 * @brief Cache for bin counting operations
 * 
 * Optimizes repeated bin count calculations by caching results
 * of previous calculations to avoid redundant processing.
 */
class BinCountCache {
private:
  std::unordered_map<std::string, std::pair<int, int>> cache;
  std::string bin_separator;
  const std::unordered_map<std::string, CategoryStats>& category_info;
  
public:
  /**
   * @brief Constructor for the bin count cache
   * 
   * @param separator String used to separate category names
   * @param info Reference to the category information map
   */
  BinCountCache(const std::string& separator, const std::unordered_map<std::string, CategoryStats>& info)
    : bin_separator(separator), category_info(info) {
    cache.reserve(256); // Pre-allocate space to reduce reallocations
  }
  
  /**
   * @brief Get count statistics for a bin
   * 
   * Returns a pair with (total_count, positive_count) for the specified bin.
   * Uses caching to optimize repeated calls with the same bin.
   * 
   * @param bin String representation of the bin (concatenated categories)
   * @return std::pair<int, int> Pair of (total_count, positive_count)
   */
  std::pair<int, int> get_bin_counts(const std::string& bin) {
    // Check cache first
    auto it = cache.find(bin);
    if (it != cache.end()) {
      return it->second;
    }
    
    std::pair<int, int> counts = {0, 0}; // {total, pos}
    
    // Parse the bin string and sum category counts
    size_t start = 0, end = 0;
    while ((end = bin.find(bin_separator, start)) != std::string::npos) {
      const std::string& cat = bin.substr(start, end - start);
      auto cat_it = category_info.find(cat);
      if (cat_it != category_info.end()) {
        counts.first += cat_it->second.count;
        counts.second += cat_it->second.count_pos;
      }
      start = end + bin_separator.length();
    }
    
    // Process the last category
    const std::string& cat = bin.substr(start);
    auto cat_it = category_info.find(cat);
    if (cat_it != category_info.end()) {
      counts.first += cat_it->second.count;
      counts.second += cat_it->second.count_pos;
    }
    
    // Cache and return the result
    cache[bin] = counts;
    return counts;
  }
  
  /**
   * @brief Clear the cache
   */
  void clear() {
    cache.clear();
  }
};

/**
 * @brief Calculate Weight of Evidence (WoE) and Information Value (IV)
 * 
 * Optimized function to calculate WoE and IV for a bin, given its counts
 * and the total counts of events in the dataset.
 * 
 * @param count_pos Positive events in the bin
 * @param count_neg Negative events in the bin
 * @param total_pos Total positive events in the dataset
 * @param total_neg Total negative events in the dataset
 * @param woe Output parameter for Weight of Evidence
 * @param iv Output parameter for Information Value
 */
inline void compute_woe_iv(double count_pos, double count_neg, double total_pos, double total_neg,
                           double &woe, double &iv) {
  // Prevent division by zero and log of zero
  if (count_pos <= EPSILON || count_neg <= EPSILON) {
    woe = 0.0;
    iv = 0.0;
    return;
  }
  
  // Calculate distribution ratios
  const double dist_pos = count_pos / total_pos;
  const double dist_neg = count_neg / total_neg;
  
  // Calculate WoE and IV
  woe = std::log(dist_pos / dist_neg);
  iv = (dist_pos - dist_neg) * woe;
}

/**
 * @brief Split a string by a delimiter
 * 
 * Optimized function to split a string into tokens based on a delimiter.
 * 
 * @param s String to split
 * @param delimiter Delimiter string
 * @return std::vector<std::string> Vector of tokens
 */
inline std::vector<std::string> split_string(const std::string &s, const std::string &delimiter) {
  std::vector<std::string> tokens;
  tokens.reserve(8); // Estimate that most bins will have fewer than 8 categories
  
  size_t start = 0, end = 0;
  while ((end = s.find(delimiter, start)) != std::string::npos) {
    tokens.push_back(s.substr(start, end - start));
    start = end + delimiter.length();
  }
  tokens.push_back(s.substr(start));
  return tokens;
}

/**
 * @class OBC_DP
 * @brief Optimal binning for categorical variables using dynamic programming
 * 
 * This class implements an algorithm for optimal binning of categorical variables
 * using dynamic programming with linear constraints (e.g., monotonicity).
 * The algorithm aims to maximize the total Information Value (IV) while
 * respecting constraints on the number of bins and other requirements.
 * 
 * Based on the methodology described in:
 * - Navas-Palencia, G. (2022). OptBinning: Mathematical Optimization for Optimal Binning.
 * - Siddiqi, N. (2017). Intelligent Credit Scoring: Building and Implementing Better Credit Risk Scorecards.
 * - Thomas, L.C., Edelman, D.B., & Crook, J.N. (2017). Credit Scoring and Its Applications.
 */
class OBC_DP {
public:
  /**
   * @brief Constructor for OBC_DP
   * 
   * @param feature Vector of categorical feature values
   * @param target Vector of binary target values (0/1)
   * @param min_bins Minimum number of bins to create
   * @param max_bins Maximum number of bins to create
   * @param bin_cutoff Minimum proportion of observations for a bin
   * @param max_n_prebins Maximum number of pre-bins before final optimization
   * @param convergence_threshold Threshold for algorithm convergence
   * @param max_iterations Maximum number of iterations
   * @param bin_separator String separator for concatenating category names
   * @param monotonic_trend Force monotonic trend ('auto', 'ascending', 'descending', 'none')
   */
  OBC_DP(const std::vector<std::string> &feature,
                              const std::vector<int> &target,
                              int min_bins,
                              int max_bins,
                              double bin_cutoff,
                              int max_n_prebins,
                              double convergence_threshold,
                              int max_iterations,
                              const std::string &bin_separator,
                              const std::string &monotonic_trend = "auto") :
  feature(feature),
  target(target),
  min_bins(min_bins),
  max_bins(max_bins),
  bin_cutoff(bin_cutoff),
  max_n_prebins(max_n_prebins),
  convergence_threshold(convergence_threshold),
  max_iterations(max_iterations),
  bin_separator(bin_separator),
  monotonic_trend(monotonic_trend),
  converged(false),
  iterations_run(0),
  total_iv(0.0),
  execution_time_ms(0) {
    // Pre-allocate memory for vectors to reduce reallocations
    const size_t est_categories = std::min(feature.size() / 4, static_cast<size_t>(1024));
    category_info.reserve(est_categories);
    merged_categories.reserve(est_categories);
    sorted_categories.reserve(est_categories);
    bin_results.reserve(max_bins);
  }
  
  /**
   * @brief Perform the optimal binning algorithm
   * 
   * This is the main entry point that executes the complete binning process
   * and returns the results.
   * 
   * @return Rcpp::List List containing the binning results
   */
  Rcpp::List perform_binning() {
    try {
      // Start timing execution
      auto start_time = std::chrono::high_resolution_clock::now();
      
      // Step 1: Validate input parameters
      validate_input();
      
      // Step 2: Preprocess data (counts and statistics)
      preprocess_data();
      
      // Initialize cache after preprocessing
      bin_cache = std::make_unique<BinCountCache>(bin_separator, category_info);
      
      // Check if we already have fewer categories than max_bins
      size_t ncat = category_info.size();
      if (ncat <= static_cast<size_t>(max_bins)) {
        compute_bins_no_optimization();
      } else {
        // Step 3: Merge rare categories
        merge_rare_categories();
        
        // Step 4: Limit the number of pre-bins
        ensure_max_prebins();
        
        // Step 5: Calculate event rates and sort categories
        compute_and_sort_event_rates();
        
        // Step 6: Initialize DP structures
        initialize_dp_structures();
        
        // Step 7: Perform dynamic programming optimization
        perform_dynamic_programming();
        
        // Step 8: Backtrack to find optimal bins
        backtrack_optimal_bins();
      }
      
      // Calculate total IV across all bins
      calculate_total_iv();
      
      // Measure execution time
      auto end_time = std::chrono::high_resolution_clock::now();
      execution_time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        end_time - start_time).count();
      
      // Step 9: Return results
      return prepare_output();
      
    } catch (const std::exception &e) {
      Rcpp::stop("Error in optimal binning: " + std::string(e.what()));
    }
  }
  
private:
  // Input parameters
  const std::vector<std::string> &feature;
  const std::vector<int> &target;
  int min_bins;
  int max_bins;
  double bin_cutoff;
  int max_n_prebins;
  double convergence_threshold;
  int max_iterations;
  std::string bin_separator;
  std::string monotonic_trend;
  
  // Auxiliary variables
  std::unordered_map<std::string, CategoryStats> category_info;
  double total_count;
  double total_pos;
  double total_neg;
  
  // CategoricalBin counting cache
  std::unique_ptr<BinCountCache> bin_cache;
  
  // Intermediate vectors
  std::vector<std::string> merged_categories;
  std::unordered_map<std::string, std::string> category_mapping;
  std::vector<std::string> sorted_categories;
  
  // DP optimized structures
  std::vector<std::vector<double>> dp;
  std::vector<std::vector<int>> prev_bin;
  std::vector<double> cum_count_pos;
  std::vector<double> cum_count_neg;
  
  // Final results
  std::vector<CategoricalBin> bin_results;
  
  // Execution results
  bool converged;
  int iterations_run;
  double total_iv;
  long execution_time_ms;
  
  /**
   * @brief Validate input parameters
   * 
   * Checks that all input parameters are valid and consistent.
   * Throws exceptions with descriptive error messages if validation fails.
   */
  void validate_input() {
    if (min_bins < 2) {
      throw std::invalid_argument("min_bins must be >= 2.");
    }
    if (max_bins < min_bins) {
      throw std::invalid_argument("max_bins must be >= min_bins.");
    }
    if (feature.size() != target.size()) {
      throw std::invalid_argument("feature and target must have the same size.");
    }
    if (feature.empty()) {
      throw std::invalid_argument("Input vectors cannot be empty.");
    }
    if (bin_cutoff <= 0 || bin_cutoff >= 1) {
      throw std::invalid_argument("bin_cutoff must be between 0 and 1 (exclusive).");
    }
    if (convergence_threshold <= 0) {
      throw std::invalid_argument("convergence_threshold must be positive.");
    }
    if (max_iterations <= 0) {
      throw std::invalid_argument("max_iterations must be positive.");
    }
    
    // Validate monotonic_trend parameter
    if (monotonic_trend != "auto" && monotonic_trend != "ascending" && 
        monotonic_trend != "descending" && monotonic_trend != "none") {
      throw std::invalid_argument("monotonic_trend must be one of: 'auto', 'ascending', 'descending', 'none'.");
    }
    
    // Check if target is binary with a single pass
    bool has_zero = false;
    bool has_one = false;
    for (int t : target) {
      if (t == 0) has_zero = true;
      else if (t == 1) has_one = true;
      else throw std::invalid_argument("Target must contain only values 0 and 1.");
      
      if (has_zero && has_one) break; // Optimization: stop checking once confirmed
    }
    
    if (!has_zero || !has_one) {
      throw std::invalid_argument("Target must contain both values 0 and 1.");
    }
  }
  
  /**
   * @brief Preprocess data
   * 
   * Count occurrences and calculate statistics in a single pass through the data.
   */
  void preprocess_data() {
    total_count = static_cast<double>(feature.size());
    total_pos = 0.0;
    
    // Count in a single pass
    for (size_t i = 0; i < feature.size(); ++i) {
      const int is_positive = target[i];
      const std::string& cat = feature[i];
      
      category_info[cat].update(is_positive);
      if (is_positive) total_pos += 1.0;
    }
    
    total_neg = total_count - total_pos;
  }
  
  /**
   * @brief Compute bins without optimization
   * 
   * Used when the number of categories is already less than or equal to max_bins.
   */
  void compute_bins_no_optimization() {
    for (const auto& pair : category_info) {
      const std::string& cat = pair.first;
      const CategoryStats& info = pair.second;
      
      double woe, iv;
      compute_woe_iv(info.count_pos, info.count_neg, total_pos, total_neg, woe, iv);
      
      bin_results.emplace_back(cat, woe, iv, info.count, info.count_pos, info.count_neg);
    }
    
    // Sort bins by event rate if monotonicity is required
    if (monotonic_trend != "none") {
      std::sort(bin_results.begin(), bin_results.end(),
                [](const CategoricalBin& a, const CategoricalBin& b) {
                  return a.event_rate() < b.event_rate();
                });
    }
  }
  
  /**
   * @brief Merge rare categories
   * 
   * Groups categories with fewer observations than the bin_cutoff threshold.
   */
  void merge_rare_categories() {
    const double cutoff_count = bin_cutoff * total_count;
    
    // Sort categories by count
    std::vector<std::pair<std::string, const CategoryStats*>> sorted_cats;
    sorted_cats.reserve(category_info.size());
    
    for (const auto& pair : category_info) {
      sorted_cats.emplace_back(pair.first, &pair.second);
    }
    
    std::sort(sorted_cats.begin(), sorted_cats.end(),
              [](const auto& a, const auto& b) {
                return a.second->count < b.second->count;
              });
    
    // Process categories in ascending order of count
    std::string current_merged;
    int current_count = 0;
    int current_pos_count = 0;
    std::vector<std::string> current_cats;
    
    for (const auto& [cat, info] : sorted_cats) {
      if (info->count < cutoff_count) {
        // Rare category, add to current group
        if (current_merged.empty()) {
          current_merged = cat;
        } else {
          current_merged += bin_separator + cat;
        }
        current_cats.push_back(cat);
        current_count += info->count;
        current_pos_count += info->count_pos;
        
        // If the group exceeds the threshold, add as bin
        if (current_count >= cutoff_count) {
          merged_categories.push_back(current_merged);
          for (const auto& merged_cat : current_cats) {
            category_mapping[merged_cat] = current_merged;
          }
          current_merged.clear();
          current_count = 0;
          current_pos_count = 0;
          current_cats.clear();
        }
      } else {
        // Non-rare category, add current group if not empty
        if (!current_merged.empty()) {
          merged_categories.push_back(current_merged);
          for (const auto& merged_cat : current_cats) {
            category_mapping[merged_cat] = current_merged;
          }
          current_merged.clear();
          current_count = 0;
          current_pos_count = 0;
          current_cats.clear();
        }
        
        // Add non-rare category as its own bin
        merged_categories.push_back(cat);
        category_mapping[cat] = cat;
      }
    }
    
    // Add final group if not empty
    if (!current_merged.empty()) {
      merged_categories.push_back(current_merged);
      for (const auto& merged_cat : current_cats) {
        category_mapping[merged_cat] = current_merged;
      }
    }
    
    // Clear cache after modifying categories
    bin_cache->clear();
  }
  
  /**
   * @brief Ensure maximum number of pre-bins
   * 
   * Reduces the number of bins to max_n_prebins if needed.
   */
  void ensure_max_prebins() {
    if (merged_categories.size() <= static_cast<size_t>(max_n_prebins)) {
      return;
    }
    
    // Calculate counts for all bins once
    std::vector<std::pair<std::string, std::pair<int, int>>> bins_with_counts;
    bins_with_counts.reserve(merged_categories.size());
    
    for (const auto& bin : merged_categories) {
      auto counts = bin_cache->get_bin_counts(bin);
      bins_with_counts.emplace_back(bin, counts);
    }
    
    // Sort by total count
    std::sort(bins_with_counts.begin(), bins_with_counts.end(),
              [](const auto& a, const auto& b) {
                return a.second.first < b.second.first;
              });
    
    // Merge categories until reaching max_n_prebins
    while (bins_with_counts.size() > static_cast<size_t>(max_n_prebins)) {
      // Merge the two smallest bins
      std::string merged_bin = bins_with_counts[0].first + bin_separator + bins_with_counts[1].first;
      std::pair<int, int> merged_counts = {
        bins_with_counts[0].second.first + bins_with_counts[1].second.first,
        bins_with_counts[0].second.second + bins_with_counts[1].second.second
      };
      
      // Update category mapping with the merge
      auto categories1 = split_string(bins_with_counts[0].first, bin_separator);
      auto categories2 = split_string(bins_with_counts[1].first, bin_separator);
      
      for (const auto& cat : categories1) {
        category_mapping[cat] = merged_bin;
      }
      for (const auto& cat : categories2) {
        category_mapping[cat] = merged_bin;
      }
      
      // Remove the two merged bins and add the new one
      bins_with_counts.erase(bins_with_counts.begin(), bins_with_counts.begin() + 2);
      bins_with_counts.emplace_back(merged_bin, merged_counts);
      
      // Re-sort by count
      std::sort(bins_with_counts.begin(), bins_with_counts.end(),
                [](const auto& a, const auto& b) {
                  return a.second.first < b.second.first;
                });
    }
    
    // Update merged_categories with final result
    merged_categories.clear();
    for (const auto& bin_info : bins_with_counts) {
      merged_categories.push_back(bin_info.first);
    }
    
    // Clear cache after modifying categories
    bin_cache->clear();
  }
  
  /**
   * @brief Compute and sort event rates
   * 
   * Calculates event rates for all merged categories and sorts them.
   */
  void compute_and_sort_event_rates() {
    // Calculate event rates for all merged categories
    std::vector<std::pair<std::string, double>> categories_with_rates;
    categories_with_rates.reserve(merged_categories.size());
    
    for (const auto& bin : merged_categories) {
      auto counts = bin_cache->get_bin_counts(bin);
      double event_rate = counts.second / static_cast<double>(counts.first);
      categories_with_rates.emplace_back(bin, event_rate);
    }
    
    // Sort by event rate
    std::sort(categories_with_rates.begin(), categories_with_rates.end(),
              [](const auto& a, const auto& b) {
                return a.second < b.second;
              });
    
    // Update sorted categories
    sorted_categories.clear();
    for (const auto& [bin, rate] : categories_with_rates) {
      sorted_categories.push_back(bin);
    }
    
    // If monotonic_trend is "descending", reverse the order
    if (monotonic_trend == "descending") {
      std::reverse(sorted_categories.begin(), sorted_categories.end());
    }
    // For "auto", determine trend based on correlation with target
    else if (monotonic_trend == "auto") {
      // This would be implemented here - for now we default to ascending
    }
  }
  
  /**
   * @brief Initialize DP structures
   * 
   * Sets up the dynamic programming tables and arrays.
   */
  void initialize_dp_structures() {
    const size_t n = sorted_categories.size();
    
    // Use resize instead of assign to avoid unnecessary initialization
    dp.resize(n + 1);
    prev_bin.resize(n + 1);
    
    for (size_t i = 0; i <= n; ++i) {
      dp[i].resize(max_bins + 1, NEGATIVE_INFINITY);
      prev_bin[i].resize(max_bins + 1, -1);
    }
    
    dp[0][0] = 0.0; // Base case
    
    // Calculate cumulative counts optimally
    cum_count_pos.resize(n + 1, 0.0);
    cum_count_neg.resize(n + 1, 0.0);
    
    for (size_t i = 0; i < n; ++i) {
      auto counts = bin_cache->get_bin_counts(sorted_categories[i]);
      cum_count_pos[i + 1] = cum_count_pos[i] + counts.second;
      cum_count_neg[i + 1] = cum_count_neg[i] + (counts.first - counts.second);
    }
  }
  
  /**
   * @brief Perform dynamic programming
   * 
   * Executes the main dynamic programming algorithm to find optimal bins.
   */
  void perform_dynamic_programming() {
    const size_t n = sorted_categories.size();
    std::vector<double> last_dp_row(max_bins + 1, NEGATIVE_INFINITY);
    last_dp_row[0] = 0.0;
    
    converged = false;
    iterations_run = 0;
    
    bool enforce_monotonicity = (monotonic_trend != "none");
    
    for (iterations_run = 1; iterations_run <= max_iterations; ++iterations_run) {
      bool any_update = false;
      
      // Optimization: precompute values for inner loop
      std::vector<std::vector<std::pair<double, double>>> bin_woe_iv(n + 1);
      for (size_t i = 1; i <= n; ++i) {
        bin_woe_iv[i].resize(i);
        for (size_t j = 0; j < i; ++j) {
          double count_pos_bin = cum_count_pos[i] - cum_count_pos[j];
          double count_neg_bin = cum_count_neg[i] - cum_count_neg[j];
          
          double woe, iv;
          compute_woe_iv(count_pos_bin, count_neg_bin, total_pos, total_neg, woe, iv);
          
          bin_woe_iv[i][j] = {woe, iv};
        }
      }
      
      // Main DP loop
      for (size_t i = 1; i <= n; ++i) {
        for (int k = 1; k <= max_bins && k <= static_cast<int>(i); ++k) {
          for (size_t j = (k - 1 > 0 ? k - 1 : 0); j < i; ++j) {
            // Apply monotonicity constraint if required
            if (enforce_monotonicity && k > min_bins && j > 0) {
              double prev_woe = bin_woe_iv[j][j-1].first;
              double curr_woe = bin_woe_iv[i][j].first;
              
              // Check monotonicity based on trend
              bool monotonicity_violated = false;
              if (monotonic_trend == "ascending" || monotonic_trend == "auto") {
                monotonicity_violated = (prev_woe > curr_woe);
              } else if (monotonic_trend == "descending") {
                monotonicity_violated = (prev_woe < curr_woe);
              }
              
              if (monotonicity_violated) {
                continue;
              }
            }
            
            double iv_bin = bin_woe_iv[i][j].second;
            double total_iv = dp[j][k - 1] + iv_bin;
            
            if (total_iv > dp[i][k]) {
              dp[i][k] = total_iv;
              prev_bin[i][k] = static_cast<int>(j);
              any_update = true;
            }
          }
        }
      }
      
      // Check convergence
      double max_diff = 0.0;
      for (int k = 0; k <= max_bins; ++k) {
        double diff = std::fabs(dp[n][k] - last_dp_row[k]);
        max_diff = std::max(max_diff, diff);
        last_dp_row[k] = dp[n][k];
      }
      
      converged = (max_diff < convergence_threshold);
      if (converged || !any_update) {
        break;
      }
    }
    
    if (!converged && iterations_run >= max_iterations) {
      Rcpp::warning("Convergence not reached in max_iterations. Using best solution found.");
    }
  }
  
  /**
   * @brief Backtrack to find optimal bins
   * 
   * Traces back through the DP table to construct the optimal binning solution.
   */
  void backtrack_optimal_bins() {
    const size_t n = sorted_categories.size();
    double max_total_iv = NEGATIVE_INFINITY;
    int best_k = -1;
    
    // Find the best number of bins
    for (int k = min_bins; k <= max_bins; ++k) {
      if (dp[n][k] > max_total_iv) {
        max_total_iv = dp[n][k];
        best_k = k;
      }
    }
    
    if (best_k == -1) {
      throw std::runtime_error("Failed to find optimal binning with the given constraints.");
    }
    
    // Determine bin edges using backtracking
    std::vector<size_t> bin_edges;
    bin_edges.reserve(best_k);
    
    size_t idx = n;
    int k = best_k;
    
    while (k > 0) {
      int prev_j = prev_bin[idx][k];
      bin_edges.push_back(static_cast<size_t>(prev_j));
      idx = static_cast<size_t>(prev_j);
      k -= 1;
    }
    
    std::reverse(bin_edges.begin(), bin_edges.end());
    
    // Build final bins
    bin_results.clear();
    bin_results.reserve(best_k);
    
    size_t start = 0;
    for (size_t edge_idx = 0; edge_idx <= bin_edges.size(); ++edge_idx) {
      size_t end = (edge_idx < bin_edges.size()) ? bin_edges[edge_idx] : n;
      
      if (start >= end) continue;
      
      int bin_count = 0;
      int bin_count_pos = 0;
      std::string bin_name;
      bin_name.reserve(256); // Estimate for long names
      
      for (size_t i = start; i < end; ++i) {
        if (i > start) bin_name += bin_separator;
        bin_name += sorted_categories[i];
        
        auto counts = bin_cache->get_bin_counts(sorted_categories[i]);
        bin_count += counts.first;
        bin_count_pos += counts.second;
      }
      
      int bin_count_neg = bin_count - bin_count_pos;
      
      if (bin_count > 0) {
        double woe, iv;
        compute_woe_iv(bin_count_pos, bin_count_neg, total_pos, total_neg, woe, iv);
        
        bin_results.emplace_back(bin_name, woe, iv, bin_count, bin_count_pos, bin_count_neg);
      }
      
      start = end;
    }
  }
  
  /**
   * @brief Calculate total IV
   * 
   * Sums up the IV values across all final bins.
   */
  void calculate_total_iv() {
    total_iv = 0.0;
    for (const auto& bin : bin_results) {
      total_iv += bin.iv;
    }
  }
  
  /**
   * @brief Prepare final output
   * 
   * Organizes the binning results into an R list for return.
   * 
   * @return Rcpp::List List containing binning results
   */
  Rcpp::List prepare_output() const {
    const size_t n_bins = bin_results.size();
    
    Rcpp::NumericVector ids(n_bins);
    Rcpp::CharacterVector bin_names(n_bins);
    Rcpp::NumericVector woe_values(n_bins);
    Rcpp::NumericVector iv_values(n_bins);
    Rcpp::IntegerVector count_values(n_bins);
    Rcpp::IntegerVector pos_count_values(n_bins);
    Rcpp::IntegerVector neg_count_values(n_bins);
    Rcpp::NumericVector event_rate_values(n_bins);
    
    for (size_t i = 0; i < n_bins; ++i) {
      ids[i] = i + 1;
      bin_names[i] = bin_results[i].name();
      woe_values[i] = bin_results[i].woe;
      iv_values[i] = bin_results[i].iv;
      count_values[i] = bin_results[i].count;
      pos_count_values[i] = bin_results[i].count_pos;
      neg_count_values[i] = bin_results[i].count_neg;
      event_rate_values[i] = bin_results[i].event_rate();
    }
    
    return Rcpp::List::create(
      Rcpp::Named("id") = ids,
      Rcpp::Named("bin") = bin_names,
      Rcpp::Named("woe") = woe_values,
      Rcpp::Named("iv") = iv_values,
      Rcpp::Named("count") = count_values,
      Rcpp::Named("count_pos") = pos_count_values,
      Rcpp::Named("count_neg") = neg_count_values,
      Rcpp::Named("event_rate") = event_rate_values,
      Rcpp::Named("total_iv") = total_iv,
      Rcpp::Named("converged") = converged,
      Rcpp::Named("iterations") = iterations_run,
      Rcpp::Named("execution_time_ms") = execution_time_ms
    );
  }
};


// [[Rcpp::export]]
Rcpp::List optimal_binning_categorical_dp(
   Rcpp::IntegerVector target,
   Rcpp::CharacterVector feature,
   int min_bins = 3,
   int max_bins = 5,
   double bin_cutoff = 0.05,
   int max_n_prebins = 20,
   double convergence_threshold = 1e-6,
   int max_iterations = 1000,
   std::string bin_separator = "%;%",
   std::string monotonic_trend = "auto"
) {
 // Preliminary check for empty inputs (optimization)
 if (feature.size() == 0 || target.size() == 0) {
   Rcpp::stop("Input vectors cannot be empty.");
 }
 
 if (feature.size() != target.size()) {
   Rcpp::stop("feature and target must have the same size.");
 }
 
 // Convert R vectors to C++
 std::vector<std::string> feature_vec;
 std::vector<int> target_vec;
 
 feature_vec.reserve(feature.size());
 target_vec.reserve(target.size());
 
 for (R_xlen_t i = 0; i < feature.size(); ++i) {
   if (feature[i] == NA_STRING) {
     feature_vec.push_back("NA");
   } else {
     feature_vec.push_back(Rcpp::as<std::string>(feature[i]));
   }
   
   if (IntegerVector::is_na(target[i])) {
     Rcpp::stop("Target cannot contain missing values.");
   } else {
     target_vec.push_back(target[i]);
   }
 }
 
 // Execute algorithm
 OBC_DP binning(
     feature_vec, target_vec, min_bins, max_bins, bin_cutoff, max_n_prebins,
     convergence_threshold, max_iterations, bin_separator, monotonic_trend
 );
 
 return binning.perform_binning();
}