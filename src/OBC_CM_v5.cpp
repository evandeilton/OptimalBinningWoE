// [[Rcpp::depends(Rcpp)]]
#include <Rcpp.h>
#include <vector>
#include <string>
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <numeric>
#include <memory>
#include <sstream>
#include <iomanip>
#include <chrono>

using namespace Rcpp;

// Include shared headers
#include "common/optimal_binning_common.h"
#include "common/bin_structures.h"

using namespace Rcpp;
using namespace OptimalBinning;


// =============================================================================
// NUMERICAL STABILITY UTILITIES
// =============================================================================

/**
 * @brief Safe logarithm with underflow protection
 * @param x Input value
 * @param epsilon Minimum value before log (default: 1e-12)
 * @return log(max(x, epsilon))
 */
inline double safe_log(double x, double epsilon = 1e-12) {
  return std::log(std::max(x, epsilon));
}

/**
 * @brief Safe division with zero denominator protection
 * @param num Numerator
 * @param denom Denominator
 * @param epsilon Minimum denominator magnitude (default: 1e-12)
 * @return num/denom with protection against division by zero
 */
inline double safe_divide(double num, double denom, double epsilon = 1e-12) {
  if (std::abs(denom) < epsilon) {
    return 0.0; // Or could return sign(num) * large_value
  }
  return num / denom;
}

/**
 * @brief Clamp value to range [min_val, max_val]
 * @param value Value to clamp
 * @param min_val Minimum allowed value
 * @param max_val Maximum allowed value
 * @return Clamped value
 */
inline double clamp(double value, double min_val, double max_val) {
  return std::max(min_val, std::min(value, max_val));
}

// =============================================================================
// BIN STRUCTURE
// =============================================================================

/**
 * @brief Structure representing a bin of categorical values with full statistics
 */
// Local CategoricalBin definition removed


// =============================================================================
// CHI-SQUARE CACHE CLASS (ENHANCED)
// =============================================================================

/**
 * @brief Efficient cache for chi-square calculations with validation
 */
class ChiSquareCache {
private:
  std::vector<double> cache;
  size_t num_bins;
  static constexpr double INVALID_VALUE = -1.0;
  
public:
  ChiSquareCache() : num_bins(0) {}
  
  /**
   * @brief Resize cache for n bins (stores n-1 adjacent pairs)
   * @param n Number of bins
   */
  void resize(size_t n) {
    num_bins = n;
    size_t cache_size = (n > 1) ? (n - 1) : 0;
    cache.assign(cache_size, INVALID_VALUE);
  }
  
  /**
   * @brief Get cached chi-square value for adjacent bins i and i+1
   * @param i First bin index
   * @return Chi-square value or INVALID_VALUE if not cached
   */
  double get(size_t i) const {
    if (i >= num_bins - 1 || i >= cache.size()) {
      return INVALID_VALUE;
    }
    return cache[i];
  }
  
  /**
   * @brief Store chi-square value for adjacent bins i and i+1
   * @param i First bin index
   * @param value Chi-square value (must be >= 0)
   */
  void set(size_t i, double value) {
    if (i < cache.size() && value >= 0) {
      cache[i] = value;
    }
  }
  
  /**
   * @brief Invalidate cache entries after merge at index
   * @param merge_index Index where merge occurred
   */
  void invalidate_after_merge(size_t merge_index) {
    // Invalidate the pair before the merge point
    if (merge_index > 0 && merge_index - 1 < cache.size()) {
      cache[merge_index - 1] = INVALID_VALUE;
    }
    // Invalidate the pair at the merge point
    if (merge_index < cache.size()) {
      cache[merge_index] = INVALID_VALUE;
    }
  }
  
  /**
   * @brief Completely invalidate the cache
   */
  void clear() {
    std::fill(cache.begin(), cache.end(), INVALID_VALUE);
  }
  
  /**
   * @brief Check if value is valid
   */
  static bool is_valid(double value) {
    return value >= 0;
  }
};

// =============================================================================
// OPTIMAL BINNING CATEGORICAL CLASS (ENHANCED)
// =============================================================================

/**
 * @brief Chi-Merge optimal binning for categorical variables
 * 
 * Implements the ChiMerge algorithm (Kerber, 1992) with optional Chi2 
 * enhancement (Liu & Setiono, 1995) for categorical variable discretization.
 * 
 * Time Complexity: O(n log n + k²m) where n = observations, k = categories, m = iterations
 * Space Complexity: O(k) for bins and cache
 */
class OBC_ {
private:
  // Input parameters (const references)
  const std::vector<std::string>& feature;
  const std::vector<int>& target;
  
  // Configuration parameters
  int min_bins;
  int max_bins;
  const double bin_cutoff;
  const int max_n_prebins;
  const std::string bin_separator;
  const double convergence_threshold;
  const int max_iterations;
  double chi_merge_threshold;
  const bool use_chi2_algorithm;
  
  // Internal state
  std::vector<CategoricalBin> bins;
  int total_pos;
  int total_neg;
  std::unordered_map<std::string, int> count_pos_map;
  std::unordered_map<std::string, int> count_neg_map;
  std::unordered_map<std::string, int> total_count_map;
  int unique_categories;
  bool is_increasing;
  bool converged;
  int iterations_run;
  std::vector<std::string> warnings;
  
  // Chi-square cache
  std::unique_ptr<ChiSquareCache> chi_cache;
  
  // Constants
  // Constant removed (uses shared definition)
  // Local constant removed (uses shared definition)  // Cap for numerical stability
  // Local constant removed (uses shared definition) // Cap for numerical stability
  
  // Corrected chi-square critical values for DF=1
  const std::unordered_map<double, double> CHI_SQUARE_CRITICAL_VALUES = {
    {0.995, 7.879},   // 99.5% confidence
    {0.99, 6.635},    // 99% confidence
    {0.975, 5.024},   // 97.5% confidence
    {0.95, 3.841},    // 95% confidence
    {0.90, 2.706},    // 90% confidence
    {0.80, 1.642},    // 80% confidence
    {0.70, 1.074},    // 70% confidence
    {0.50, 0.455},    // 50% confidence
    {0.30, 0.148},    // 30% confidence
    {0.20, 0.064},    // 20% confidence
    {0.10, 0.016},    // 10% confidence
    {0.05, 0.004},    // 5% confidence
    {0.01, 0.0002},   // 1% confidence
    {0.001, 0.00001}  // 0.1% confidence
  };
  
public:
  /**
   * @brief Constructor with comprehensive parameter validation
   */
  OBC_(
    const std::vector<std::string>& feature_,
    const std::vector<int>& target_,
    int min_bins_,
    int max_bins_,
    double bin_cutoff_,
    int max_n_prebins_,
    const std::string& bin_separator_,
    double convergence_threshold_,
    int max_iterations_,
    double chi_merge_threshold_ = 0.05,
    bool use_chi2_algorithm_ = false
  ) : feature(feature_),
  target(target_),
  min_bins(min_bins_),
  max_bins(max_bins_),
  bin_cutoff(bin_cutoff_),
  max_n_prebins(max_n_prebins_),
  bin_separator(bin_separator_),
  convergence_threshold(convergence_threshold_),
  max_iterations(max_iterations_),
  chi_merge_threshold(chi_merge_threshold_),
  use_chi2_algorithm(use_chi2_algorithm_),
  total_pos(0),
  total_neg(0),
  unique_categories(0),
  is_increasing(true),
  converged(false),
  iterations_run(0),
  chi_cache(std::make_unique<ChiSquareCache>())
  {
    // Reserve memory based on heuristics
    size_t estimated_categories = std::min(
      static_cast<size_t>(feature.size() / 10), // Assume ~10 obs per category
      static_cast<size_t>(1000)                  // Cap at 1000 for memory
    );
    count_pos_map.reserve(estimated_categories);
    count_neg_map.reserve(estimated_categories);
    total_count_map.reserve(estimated_categories);
    warnings.reserve(10);
  }
  
  /**
   * @brief Main method to perform optimal binning
   * @return Rcpp::List with binning results and diagnostics
   */
  Rcpp::List perform_binning() {
    auto start_time = std::chrono::high_resolution_clock::now();
    
    try {
      // 1. Validate inputs
      validate_inputs();
      
      // 2. Initialize bins
      initialize_bins();
      chi_cache->resize(bins.size());
      
      // 3. Apply selected algorithm
      if (use_chi2_algorithm) {
        perform_chi2_binning();
      } else {
        perform_chimerge_binning();
      }
      
      // 4. Calculate final metrics
      calculate_woe_iv();
      
      // 5. Validate final bins
      validate_final_bins();
      
      // 6. Prepare output
      auto end_time = std::chrono::high_resolution_clock::now();
      auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
        end_time - start_time).count();
      
      return prepare_output(duration);
      
    } catch (const std::exception& e) {
      Rcpp::stop("OBC_ error: " + std::string(e.what()));
    }
  }
  
private:
  /**
   * @brief Comprehensive input validation with informative errors
   */
  void validate_inputs() {
    // Size validation
    if (feature.empty() || target.empty()) {
      Rcpp::stop("OBC_: Feature and target cannot be empty");
    }
    
    if (feature.size() != target.size()) {
      Rcpp::stop("OBC_: Feature size (" + 
        std::to_string(feature.size()) + ") != target size (" + 
        std::to_string(target.size()) + ")");
    }
    
    // Parameter validation
    if (min_bins < 2) {
      Rcpp::stop("OBC_: min_bins must be >= 2, got " + 
        std::to_string(min_bins));
    }
    
    if (max_bins < min_bins) {
      Rcpp::stop("OBC_: max_bins (" + 
        std::to_string(max_bins) + ") < min_bins (" + 
        std::to_string(min_bins) + ")");
    }
    
    if (bin_cutoff <= 0 || bin_cutoff >= 1) {
      Rcpp::stop("OBC_: bin_cutoff must be in (0,1), got " + 
        std::to_string(bin_cutoff));
    }
    
    if (max_n_prebins < 2) {
      Rcpp::stop("OBC_: max_n_prebins must be >= 2, got " + 
        std::to_string(max_n_prebins));
    }
    
    if (convergence_threshold <= 0) {
      Rcpp::stop("OBC_: convergence_threshold must be > 0, got " + 
        std::to_string(convergence_threshold));
    }
    
    if (max_iterations <= 0) {
      Rcpp::stop("OBC_: max_iterations must be > 0, got " + 
        std::to_string(max_iterations));
    }
    
    if (chi_merge_threshold <= 0 || chi_merge_threshold >= 1) {
      Rcpp::stop("OBC_: chi_merge_threshold must be in (0,1), got " + 
        std::to_string(chi_merge_threshold));
    }
    
    // Process data and validate target values
    process_input_data();
    
    // Adjust constraints based on data
    adjust_constraints();
  }
  
  /**
   * @brief Process input data and build frequency maps
   */
  void process_input_data() {
    total_pos = 0;
    total_neg = 0;
    count_pos_map.clear();
    count_neg_map.clear();
    total_count_map.clear();
    
    for (size_t i = 0; i < target.size(); ++i) {
      // Validate target
      if (target[i] != 0 && target[i] != 1) {
        Rcpp::stop("OBC_: Target must be binary (0/1), found " + 
          std::to_string(target[i]) + " at position " + std::to_string(i));
      }
      
      const std::string& cat = feature[i];
      
      if (target[i] == 1) {
        count_pos_map[cat]++;
        total_pos++;
      } else {
        count_neg_map[cat]++;
        total_neg++;
      }
      total_count_map[cat]++;
    }
    
    // Validate we have both classes
    if (total_pos == 0) {
      Rcpp::stop("OBC_: No positive cases (target=1) found");
    }
    if (total_neg == 0) {
      Rcpp::stop("OBC_: No negative cases (target=0) found");
    }
    
    unique_categories = static_cast<int>(total_count_map.size());
    
    if (unique_categories < 2) {
      Rcpp::stop("OBC_: Need at least 2 unique categories, found " + 
        std::to_string(unique_categories));
    }
  }
  
  /**
   * @brief Adjust constraints based on actual data
   */
  void adjust_constraints() {
    // Ensure min_bins doesn't exceed unique categories
    if (min_bins > unique_categories) {
      warnings.push_back("Adjusted min_bins from " + std::to_string(min_bins) + 
        " to " + std::to_string(unique_categories) + 
        " (number of unique categories)");
      min_bins = unique_categories;
    }
    
    // Ensure max_bins doesn't exceed unique categories
    if (max_bins > unique_categories) {
      warnings.push_back("Adjusted max_bins from " + std::to_string(max_bins) + 
        " to " + std::to_string(unique_categories) + 
        " (number of unique categories)");
      max_bins = unique_categories;
    }
    
    // Final sanity check
    min_bins = std::max(2, min_bins);
    max_bins = std::max(min_bins, max_bins);
  }
  
  /**
   * @brief Initialize bins with one category per bin
   */
  void initialize_bins() {
    bins.clear();
    bins.reserve(unique_categories);
    
    for (const auto& [cat, count] : total_count_map) {
      CategoricalBin bin;
      bin.categories.push_back(cat);
      bin.count_pos = count_pos_map[cat];
      bin.count_neg = count_neg_map[cat];
      bin.update_count();
      bins.push_back(std::move(bin));
    }
    
    // Initial sort by WoE
    sort_bins_by_woe();
  }
  
  /**
   * @brief Sort bins by Weight of Evidence
   */
  void sort_bins_by_woe() {
    // Calculate WoE for each bin
    for (auto& bin : bins) {
      bin.woe = calculate_woe(bin.count_pos, bin.count_neg);
    }
    
    // Stable sort to maintain relative order of equal WoE bins
    std::stable_sort(bins.begin(), bins.end(),
                     [](const CategoricalBin& a, const CategoricalBin& b) { 
                       return a.woe < b.woe; 
                     });
  }
  
  /**
   * @brief Perform standard ChiMerge binning
   */
  void perform_chimerge_binning() {
    // Step 1: Handle rare categories
    handle_rare_categories();
    
    // Step 2: Limit to max_n_prebins
    limit_prebins();
    
    // Step 3: Main ChiMerge loop
    merge_bins_chimerge();
    
    // Step 4: Ensure min_bins
    ensure_min_bins();
    
    // Step 5: Enforce monotonicity
    enforce_monotonicity();
  }
  
  /**
   * @brief Perform Chi2 algorithm with multiple passes
   */
  void perform_chi2_binning() {
    const std::vector<double> significance_levels = {
      0.5, 0.25, 0.1, 0.05, 0.01, 0.005, 0.001
    };
    
    for (double sig_level : significance_levels) {
      chi_merge_threshold = sig_level;
      merge_bins_chimerge();
      
      if (bins.size() <= static_cast<size_t>(max_bins)) {
        break;
      }
      
      // Check inconsistency rate (optional early stopping)
      double inconsistency = calculate_inconsistency_rate();
      if (inconsistency < 0.05) { // 5% threshold
        warnings.push_back("Chi2: Stopped at significance " + 
          std::to_string(sig_level) + 
          " due to low inconsistency rate");
        break;
      }
    }
    
    ensure_min_bins();
    enforce_monotonicity();
  }
  
  /**
   * @brief Handle rare categories by merging with most similar neighbor
   */
  void handle_rare_categories() {
    if (bins.size() <= static_cast<size_t>(min_bins)) return;
    
    bool merged = true;
    while (merged && bins.size() > static_cast<size_t>(min_bins)) {
      merged = false;
      
      for (size_t i = 0; i < bins.size(); ++i) {
        double freq = static_cast<double>(bins[i].total()) / 
          (total_pos + total_neg);
        
        if (freq < bin_cutoff) {
          // Find best neighbor to merge with
          size_t merge_with = find_best_merge_neighbor(i);
          if (merge_with != i) {
            merge_bins_at_indices(i, merge_with);
            merged = true;
            break; // Restart scan
          }
        }
      }
    }
    
    if (merged) {
      chi_cache->clear();
      chi_cache->resize(bins.size());
    }
  }
  
  /**
   * @brief Find best neighbor for merging based on chi-square
   */
  size_t find_best_merge_neighbor(size_t idx) {
    double min_chi = std::numeric_limits<double>::max();
    size_t best_neighbor = idx;
    
    // Check left neighbor
    if (idx > 0) {
      double chi = compute_chi_square(bins[idx - 1], bins[idx]);
      if (chi < min_chi) {
        min_chi = chi;
        best_neighbor = idx - 1;
      }
    }
    
    // Check right neighbor
    if (idx < bins.size() - 1) {
      double chi = compute_chi_square(bins[idx], bins[idx + 1]);
      if (chi < min_chi) {
        best_neighbor = idx + 1;
      }
    }
    
    return best_neighbor;
  }
  
  /**
   * @brief Merge two bins at given indices
   */
  void merge_bins_at_indices(size_t idx1, size_t idx2) {
    if (idx1 == idx2) return;
    
    // Ensure idx1 < idx2
    if (idx1 > idx2) std::swap(idx1, idx2);
    
    // Merge idx2 into idx1
    bins[idx1].categories.insert(bins[idx1].categories.end(),
                                 bins[idx2].categories.begin(),
                                 bins[idx2].categories.end());
    bins[idx1].count_pos += bins[idx2].count_pos;
    bins[idx1].count_neg += bins[idx2].count_neg;
    bins[idx1].update_count();
    
    // Remove idx2
    bins.erase(bins.begin() + idx2);
  }
  
  /**
   * @brief Limit number of prebins
   */
  void limit_prebins() {
    while (bins.size() > static_cast<size_t>(max_n_prebins) && 
           bins.size() > static_cast<size_t>(min_bins)) {
      auto [min_chi, idx] = find_min_chi_square_pair();
      
      if (min_chi > get_chi_square_critical_value()) {
        break; // Stop if all pairs are significantly different
      }
      
      merge_adjacent_bins(idx);
      update_cache_after_merge(idx);
    }
  }
  
  /**
   * @brief Main ChiMerge merging loop
   */
  void merge_bins_chimerge() {
    iterations_run = 0;
    converged = false;
    double critical_value = get_chi_square_critical_value();
    
    // First ensure we're within max_bins
    while (bins.size() > static_cast<size_t>(max_bins)) {
      auto [min_chi, idx] = find_min_chi_square_pair();
      merge_adjacent_bins(idx);
      update_cache_after_merge(idx);
      iterations_run++;
    }
    
    // Then merge statistically similar bins
    double prev_min_chi = -1.0;
    while (can_merge_further() && iterations_run < max_iterations) {
      auto [min_chi, idx] = find_min_chi_square_pair();
      
      // Stop if bins are significantly different
      if (min_chi > critical_value) {
        converged = true;
        break;
      }
      
      // Check convergence
      if (prev_min_chi >= 0 && 
          std::abs(min_chi - prev_min_chi) < convergence_threshold) {
        converged = true;
        break;
      }
      
      prev_min_chi = min_chi;
      merge_adjacent_bins(idx);
      update_cache_after_merge(idx);
      iterations_run++;
    }
    
    if (iterations_run >= max_iterations) {
      warnings.push_back("ChiMerge reached max_iterations (" + 
        std::to_string(max_iterations) + ")");
    }
  }
  
  /**
   * @brief Ensure minimum number of bins by splitting
   */
  void ensure_min_bins() {
    while (bins.size() < static_cast<size_t>(min_bins)) {
      // Find bin with most categories
      size_t best_idx = 0;
      size_t max_cats = 0;
      
      for (size_t i = 0; i < bins.size(); ++i) {
        if (bins[i].categories.size() > max_cats) {
          max_cats = bins[i].categories.size();
          best_idx = i;
        }
      }
      
      if (max_cats <= 1) {
        warnings.push_back("Cannot split bins further to reach min_bins=" + 
          std::to_string(min_bins));
        break;
      }
      
      split_bin(best_idx);
    }
  }
  
  /**
   * @brief Split a bin into two parts
   */
  void split_bin(size_t idx) {
    CategoricalBin& bin = bins[idx];
    if (bin.categories.size() <= 1) return;
    
    // Sort categories by individual WoE
    std::vector<std::pair<std::string, double>> cat_woes;
    for (const auto& cat : bin.categories) {
      double woe = calculate_woe(count_pos_map[cat], count_neg_map[cat]);
      cat_woes.emplace_back(cat, woe);
    }
    
    std::sort(cat_woes.begin(), cat_woes.end(),
              [](const auto& a, const auto& b) { return a.second < b.second; });
    
    // Split at midpoint
    size_t split_point = cat_woes.size() / 2;
    
    CategoricalBin new_bin1, new_bin2;
    for (size_t i = 0; i < cat_woes.size(); ++i) {
      const std::string& cat = cat_woes[i].first;
      CategoricalBin& target_bin = (i < split_point) ? new_bin1 : new_bin2;
      
      target_bin.categories.push_back(cat);
      target_bin.count_pos += count_pos_map[cat];
      target_bin.count_neg += count_neg_map[cat];
    }
    
    new_bin1.update_count();
    new_bin2.update_count();
    
    // Replace original bin with two new bins
    bins[idx] = std::move(new_bin1);
    bins.insert(bins.begin() + idx + 1, std::move(new_bin2));
    
    // Update cache
    chi_cache->clear();
    chi_cache->resize(bins.size());
  }
  
  /**
   * @brief Enforce monotonicity of WoE
   */
  void enforce_monotonicity() {
    if (bins.size() <= 1) return;
    
    determine_monotonicity_direction();
    
    bool changed = true;
    int mono_iterations = 0;
    const int max_mono_iterations = static_cast<int>(bins.size()) * 2;
    
    while (changed && mono_iterations < max_mono_iterations) {
      changed = false;
      
      for (size_t i = 0; i < bins.size() - 1; ++i) {
        bool violation = is_increasing ? 
        (bins[i].woe > bins[i + 1].woe + EPSILON) :
        (bins[i].woe < bins[i + 1].woe - EPSILON);
        
        if (violation && bins.size() > static_cast<size_t>(min_bins)) {
          merge_adjacent_bins(i);
          bins[i].woe = calculate_woe(bins[i].count_pos, bins[i].count_neg);
          changed = true;
          break; // Restart
        }
      }
      
      mono_iterations++;
    }
    
    if (mono_iterations >= max_mono_iterations) {
      warnings.push_back("Monotonicity enforcement reached iteration limit");
    }
  }
  
  /**
   * @brief Determine monotonicity direction using regression
   */
  void determine_monotonicity_direction() {
    if (bins.size() < 2) {
      is_increasing = true;
      return;
    }
    
    // Use simple linear regression
    double sum_x = 0, sum_y = 0, sum_xy = 0, sum_x2 = 0;
    for (size_t i = 0; i < bins.size(); ++i) {
      double x = static_cast<double>(i);
      double y = bins[i].woe;
      sum_x += x;
      sum_y += y;
      sum_xy += x * y;
      sum_x2 += x * x;
    }
    
    double n = static_cast<double>(bins.size());
    double denom = n * sum_x2 - sum_x * sum_x;
    
    if (std::abs(denom) < EPSILON) {
      is_increasing = (bins.back().woe >= bins.front().woe);
    } else {
      double slope = (n * sum_xy - sum_x * sum_y) / denom;
      is_increasing = (slope >= 0);
    }
  }
  
  /**
   * @brief Calculate WoE and IV for all bins
   */
  void calculate_woe_iv() {
    double total_iv = 0.0;
    
    for (auto& bin : bins) {
      bin.woe = calculate_woe(bin.count_pos, bin.count_neg);
      
      double dist_pos = safe_divide(bin.count_pos, total_pos, EPSILON);
      double dist_neg = safe_divide(bin.count_neg, total_neg, EPSILON);
      
      bin.iv = (dist_pos - dist_neg) * bin.woe;
      total_iv += bin.iv;
    }
    
    // Add total IV to metadata
    if (total_iv < 0.02) {
      warnings.push_back("Low total IV (" + std::to_string(total_iv) + 
        ") - feature may have low predictive power");
    } else if (total_iv > 0.5) {
      warnings.push_back("High total IV (" + std::to_string(total_iv) + 
        ") - check for overfitting");
    }
  }
  
  /**
   * @brief Validate final bins
   */
  void validate_final_bins() {
    for (size_t i = 0; i < bins.size(); ++i) {
      if (bins[i].total() < 0) {
        Rcpp::stop("Invalid bin detected at position " + std::to_string(i));
      }
      
      if (bins[i].categories.empty()) {
        Rcpp::stop("Empty bin detected at position " + std::to_string(i));
      }
    }
    
    // Check monotonicity
    bool mono_violated = false;
    for (size_t i = 0; i < bins.size() - 1; ++i) {
      if (is_increasing && bins[i].woe > bins[i + 1].woe + EPSILON) {
        mono_violated = true;
      } else if (!is_increasing && bins[i].woe < bins[i + 1].woe - EPSILON) {
        mono_violated = true;
      }
    }
    
    if (mono_violated) {
      warnings.push_back("Monotonicity not perfectly achieved");
    }
  }
  
  /**
   * @brief Calculate Weight of Evidence with numerical stability
   */
  double calculate_woe(int count_pos, int count_neg) const {
    double dist_pos = safe_divide(count_pos + 0.5, total_pos + 1.0, EPSILON);
    double dist_neg = safe_divide(count_neg + 0.5, total_neg + 1.0, EPSILON);
    
    double woe = safe_log(dist_pos / dist_neg);
    return clamp(woe, MIN_WOE, MAX_WOE);
  }
  
  /**
   * @brief Compute chi-square statistic between two bins
   */
  double compute_chi_square(const CategoricalBin& bin1, const CategoricalBin& bin2) const {
    int o11 = bin1.count_pos, o12 = bin1.count_neg;
    int o21 = bin2.count_pos, o22 = bin2.count_neg;
    
    int r1 = o11 + o12, r2 = o21 + o22;
    int c1 = o11 + o21, c2 = o12 + o22;
    int n = r1 + r2;
    
    if (n == 0 || r1 == 0 || r2 == 0 || c1 == 0 || c2 == 0) {
      return 0.0;
    }
    
    // Expected frequencies
    double e11 = static_cast<double>(r1 * c1) / n;
    double e12 = static_cast<double>(r1 * c2) / n;
    double e21 = static_cast<double>(r2 * c1) / n;
    double e22 = static_cast<double>(r2 * c2) / n;
    
    // Chi-square with continuity correction
    double chi2 = 0.0;
    if (e11 > EPSILON) chi2 += std::pow(std::abs(o11 - e11) - 0.5, 2) / e11;
    if (e12 > EPSILON) chi2 += std::pow(std::abs(o12 - e12) - 0.5, 2) / e12;
    if (e21 > EPSILON) chi2 += std::pow(std::abs(o21 - e21) - 0.5, 2) / e21;
    if (e22 > EPSILON) chi2 += std::pow(std::abs(o22 - e22) - 0.5, 2) / e22;
    
    return chi2;
  }
  
  /**
   * @brief Find pair of adjacent bins with minimum chi-square
   */
  std::pair<double, size_t> find_min_chi_square_pair() {
    double min_chi = std::numeric_limits<double>::max();
    size_t min_idx = 0;
    
    for (size_t i = 0; i < bins.size() - 1; ++i) {
      double chi = chi_cache->get(i);
      
      if (!ChiSquareCache::is_valid(chi)) {
        chi = compute_chi_square(bins[i], bins[i + 1]);
        chi_cache->set(i, chi);
      }
      
      if (chi < min_chi) {
        min_chi = chi;
        min_idx = i;
      }
    }
    
    return {min_chi, min_idx};
  }
  
  /**
   * @brief Merge adjacent bins at index
   */
  void merge_adjacent_bins(size_t idx) {
    if (idx >= bins.size() - 1) return;
    
    bins[idx].categories.insert(bins[idx].categories.end(),
                                bins[idx + 1].categories.begin(),
                                bins[idx + 1].categories.end());
    bins[idx].count_pos += bins[idx + 1].count_pos;
    bins[idx].count_neg += bins[idx + 1].count_neg;
    bins[idx].update_count();
    
    bins.erase(bins.begin() + idx + 1);
    chi_cache->resize(bins.size());
  }
  
  /**
   * @brief Update cache after merge
   */
  void update_cache_after_merge(size_t idx) {
    chi_cache->invalidate_after_merge(idx);
    
    if (idx > 0) {
      double chi = compute_chi_square(bins[idx - 1], bins[idx]);
      chi_cache->set(idx - 1, chi);
    }
    
    if (idx < bins.size() - 1) {
      double chi = compute_chi_square(bins[idx], bins[idx + 1]);
      chi_cache->set(idx, chi);
    }
  }
  
  /**
   * @brief Get chi-square critical value
   */
  double get_chi_square_critical_value() const {
    // Find in lookup table
    auto it = CHI_SQUARE_CRITICAL_VALUES.find(chi_merge_threshold);
    if (it != CHI_SQUARE_CRITICAL_VALUES.end()) {
      return it->second;
    }
    
    // Find closest value
    double closest_key = 0.05;
    double min_diff = std::abs(chi_merge_threshold - 0.05);
    
    for (const auto& [key, value] : CHI_SQUARE_CRITICAL_VALUES) {
      double diff = std::abs(chi_merge_threshold - key);
      if (diff < min_diff) {
        min_diff = diff;
        closest_key = key;
      }
    }
    
    return CHI_SQUARE_CRITICAL_VALUES.at(closest_key);
  }
  
  /**
   * @brief Check if further merging is possible
   */
  bool can_merge_further() const {
    return bins.size() > static_cast<size_t>(min_bins) && bins.size() > 1;
  }
  
  /**
   * @brief Calculate inconsistency rate
   */
  double calculate_inconsistency_rate() const {
    double inconsistent_count = 0;
    
    for (const auto& bin : bins) {
      int minority = std::min(bin.count_pos, bin.count_neg);
      inconsistent_count += minority;
    }
    
    return safe_divide(inconsistent_count, total_pos + total_neg);
  }
  
  /**
   * @brief Join categories with separator
   */
  std::string join_categories(const std::vector<std::string>& cats) const {
    if (cats.empty()) return "";
    if (cats.size() == 1) return cats[0];
    
    std::ostringstream oss;
    oss << cats[0];
    for (size_t i = 1; i < cats.size(); ++i) {
      oss << bin_separator << cats[i];
    }
    return oss.str();
  }
  
  /**
   * @brief Prepare final output
   */
  Rcpp::List prepare_output(long duration_ms) const {
    size_t n_bins = bins.size();
    
    Rcpp::IntegerVector ids(n_bins);
    Rcpp::StringVector bin_names(n_bins);
    Rcpp::NumericVector woe_values(n_bins);
    Rcpp::NumericVector iv_values(n_bins);
    Rcpp::IntegerVector counts(n_bins);
    Rcpp::IntegerVector counts_pos(n_bins);
    Rcpp::IntegerVector counts_neg(n_bins);
    
    double total_iv = 0.0;
    for (size_t i = 0; i < n_bins; ++i) {
      ids[i] = static_cast<int>(i + 1);
      bin_names[i] = join_categories(bins[i].categories);
      woe_values[i] = bins[i].woe;
      iv_values[i] = bins[i].iv;
      counts[i] = bins[i].total();
      counts_pos[i] = bins[i].count_pos;
      counts_neg[i] = bins[i].count_neg;
      total_iv += bins[i].iv;
    }
    
    return Rcpp::List::create(
      Rcpp::Named("id") = ids,
      Rcpp::Named("bin") = bin_names,
      Rcpp::Named("woe") = woe_values,
      Rcpp::Named("iv") = iv_values,
      Rcpp::Named("count") = counts,
      Rcpp::Named("count_pos") = counts_pos,
      Rcpp::Named("count_neg") = counts_neg,
      Rcpp::Named("converged") = converged,
      Rcpp::Named("iterations") = iterations_run,
      Rcpp::Named("algorithm") = use_chi2_algorithm ? "Chi2" : "ChiMerge",
      Rcpp::Named("warnings") = warnings,
      Rcpp::Named("metadata") = Rcpp::List::create(
        Rcpp::Named("version") = "2.0.0",
        Rcpp::Named("total_iv") = total_iv,
        Rcpp::Named("n_bins") = static_cast<int>(n_bins),
        Rcpp::Named("unique_categories") = unique_categories,
        Rcpp::Named("total_obs") = total_pos + total_neg,
        Rcpp::Named("execution_time_ms") = duration_ms,
        Rcpp::Named("monotonic") = is_increasing ? "increasing" : "decreasing"
      )
    );
  }
};

// [[Rcpp::export]]
Rcpp::List optimal_binning_categorical_cm(
   Rcpp::IntegerVector target,
   Rcpp::CharacterVector feature,
   int min_bins = 3,
   int max_bins = 5,
   double bin_cutoff = 0.05,
   int max_n_prebins = 20,
   std::string bin_separator = "%;%",
   double convergence_threshold = 1e-6,
   int max_iterations = 1000,
   double chi_merge_threshold = 0.05,
   bool use_chi2_algorithm = false
) {
 // Input validation and conversion
 if (target.size() != feature.size()) {
   Rcpp::stop("OBC_: target and feature must have same length");
 }
 
 // Convert to C++ vectors with NA handling
 std::vector<std::string> feature_vec;
 std::vector<int> target_vec;
 
 feature_vec.reserve(feature.size());
 target_vec.reserve(target.size());
 
 for (R_xlen_t i = 0; i < feature.size(); ++i) {
   // Handle NA in feature
   if (CharacterVector::is_na(feature[i])) {
     feature_vec.push_back("NA");
   } else {
     feature_vec.push_back(Rcpp::as<std::string>(feature[i]));
   }
   
   // Check for NA in target
   if (IntegerVector::is_na(target[i])) {
     Rcpp::stop("OBC_: target cannot contain NA values");
   }
   target_vec.push_back(target[i]);
 }
 
 // Create and run the binning algorithm
 try {
   OBC_ binner(
       feature_vec, target_vec,
       min_bins, max_bins,
       bin_cutoff, max_n_prebins,
       bin_separator,
       convergence_threshold, max_iterations,
       chi_merge_threshold, use_chi2_algorithm
   );
   
   return binner.perform_binning();
   
 } catch (const std::exception& e) {
   Rcpp::stop(std::string("OBC_ error: ") + e.what());
 }
}


