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

using namespace Rcpp;

/**
 * Structure representing a bin of categorical values
 */
struct Bin {
  std::vector<std::string> categories;  // Categories in this bin
  int count_pos;                        // Count of positive class
  int count_neg;                        // Count of negative class
  double woe;                           // Weight of Evidence
  double iv;                            // Information Value
  int total_count;                      // Total observations
  
  // Constructor with sensible initial capacity
  Bin() : count_pos(0), count_neg(0), woe(0.0), iv(0.0), total_count(0) {
    categories.reserve(8);  // Reasonable initial capacity for small datasets
  }
};

/**
 * Efficient cache for chi-square calculations to avoid redundant computations
 */
class ChiSquareCache {
private:
  // Use a flat array for faster access compared to unordered_map
  std::vector<double> cache;
  size_t num_bins;
  
  // Compute index in the triangular matrix (only storing upper triangle)
  inline size_t compute_index(size_t i, size_t j) const {
    // Ensure i <= j
    if (i > j) std::swap(i, j);
    // Triangular number formula: i*(2n-i-1)/2 + (j-i)
    return (i * (2 * num_bins - i - 1)) / 2 + (j - i);
  }
  
public:
  /**
   * Initialize the cache with a given number of bins
   * @param n Number of bins
   */
  explicit ChiSquareCache(size_t n) : num_bins(n) {
    // Only need to store upper triangular matrix
    size_t size = (n * (n - 1)) / 2;
    cache.resize(size, -1.0);  // Initialize with -1 to indicate uncached
  }
  
  /**
   * Resize the cache when the number of bins changes
   * @param new_size New number of bins
   */
  void resize(size_t new_size) {
    num_bins = new_size;
    size_t new_cache_size = (num_bins * (num_bins - 1)) / 2;
    cache.resize(new_cache_size, -1.0);
  }
  
  /**
   * Get cached chi-square value
   * @param i First bin index
   * @param j Second bin index
   * @return Chi-square value or -1 if not cached
   */
  double get(size_t i, size_t j) {
    if (i >= num_bins || j >= num_bins) return -1.0;
    if (i == j) return 0.0;  // Same bin has chi-square of 0
    
    size_t idx = compute_index(i, j);
    return (idx < cache.size()) ? cache[idx] : -1.0;
  }
  
  /**
   * Store chi-square value in cache
   * @param i First bin index
   * @param j Second bin index
   * @param value Chi-square value
   */
  void set(size_t i, size_t j, double value) {
    if (i >= num_bins || j >= num_bins) return;
    if (i == j) return;  // Don't store diagonal elements
    
    size_t idx = compute_index(i, j);
    if (idx < cache.size()) {
      cache[idx] = value;
    }
  }
  
  /**
   * Invalidate cache entries related to a specific bin
   * @param index Bin index to invalidate
   */
  void invalidate_bin(size_t index) {
    if (index >= num_bins) return;
    
    // For each potential pair with this index
    for (size_t i = 0; i < num_bins; ++i) {
      if (i == index) continue;
      set(i, index, -1.0);
    }
  }
  
  /**
   * Completely invalidate the cache
   */
  void invalidate() {
    std::fill(cache.begin(), cache.end(), -1.0);
  }
};

/**
 * Chi-Merge and Chi2 optimal binning implementation for categorical variables
 * Based on Kerber (1992) and Liu & Setiono (1995)
 */
class OptimalBinningCategorical {
private:
  // Input parameters
  const std::vector<std::string>& feature;
  const std::vector<int>& target;
  int min_bins;
  int max_bins;
  double bin_cutoff;
  int max_n_prebins;
  std::string bin_separator;
  double convergence_threshold;
  int max_iterations;
  double chi_merge_threshold;  // Chi-square threshold for merging
  bool use_chi2_algorithm;     // Whether to use Chi2 extensions
  
  // Internal state
  std::vector<Bin> bins;
  int total_pos;
  int total_neg;
  std::unordered_map<std::string, int> count_pos_map;
  std::unordered_map<std::string, int> count_neg_map;
  std::unordered_map<std::string, int> total_count_map;
  int unique_categories;
  bool is_increasing;
  bool converged;
  int iterations_run;
  
  // Chi-square statistics cache
  std::unique_ptr<ChiSquareCache> chi_cache;
  
  // Constants
  static constexpr double EPSILON = 1e-10;
  
  // Chi-square critical values for common significance levels
  // Degrees of freedom = 1 for binary classification
  const std::unordered_map<double, double> CHI_SQUARE_CRITICAL_VALUES = {
    {0.995, 0.000393}, {0.99, 0.000157}, {0.975, 0.000982},
    {0.95, 0.00393}, {0.9, 0.0158}, {0.5, 0.455},
    {0.1, 2.71}, {0.05, 3.84}, {0.025, 5.02},
    {0.01, 6.63}, {0.005, 7.88}, {0.001, 10.8}
  };
  
public:
  /**
   * Constructor for OptimalBinningCategorical
   * 
   * @param feature_ Feature vector of categorical values
   * @param target_ Binary target vector (0/1)
   * @param min_bins_ Minimum number of bins to create
   * @param max_bins_ Maximum number of bins to create
   * @param bin_cutoff_ Minimum frequency for a category to avoid "rare" handling
   * @param max_n_prebins_ Maximum number of pre-bins before merging
   * @param bin_separator_ String separator for bin names
   * @param convergence_threshold_ Threshold for convergence
   * @param max_iterations_ Maximum iterations for merging
   * @param chi_merge_threshold_ Significance level for chi-merge (0.05 = 95% confidence)
   * @param use_chi2_algorithm_ Whether to use Chi2 extensions from Liu & Setiono (1995)
   */
  OptimalBinningCategorical(
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
  iterations_run(0) {
    
    // Estimate better initial allocations based on dataset size
    int estimated_categories = std::min(
      static_cast<int>(feature.size() / 4),  // Heuristic: avg 4 samples per category
      2048  // Cap for very large datasets
    );
    
    bins.reserve(estimated_categories);
    count_pos_map.reserve(estimated_categories);
    count_neg_map.reserve(estimated_categories);
    total_count_map.reserve(estimated_categories);
  }
  
  /**
   * Main method to perform optimal binning
   * @return Rcpp::List with binning results
   */
  Rcpp::List perform_binning() {
    try {
      validate_inputs();
      initialize_bins();
      
      // Initialize chi-square cache after we know bin count
      chi_cache = std::make_unique<ChiSquareCache>(bins.size());
      
      if (use_chi2_algorithm) {
        perform_chi2_binning();
      } else {
        handle_rare_categories();
        limit_prebins();
        ensure_min_bins();
        merge_bins_using_chimerge();
        enforce_monotonicity();
      }
      
      calculate_woe_iv_bins();
      return prepare_output();
    } catch (const std::exception& e) {
      Rcpp::stop("Error in optimal binning: " + std::string(e.what()));
    }
  }
  
private:
  /**
   * Validate input parameters and preprocess data
   * Throws exceptions for invalid inputs
   */
  void validate_inputs() {
    // Basic validation
    if (feature.empty() || target.empty()) {
      throw std::invalid_argument("Feature and target cannot be empty.");
    }
    if (feature.size() != target.size()) {
      throw std::invalid_argument("Feature and target must have the same length.");
    }
    if (min_bins <= 0 || max_bins <= 0 || min_bins > max_bins) {
      throw std::invalid_argument("Invalid values for min_bins or max_bins.");
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
    if (chi_merge_threshold <= 0 || chi_merge_threshold >= 1) {
      throw std::invalid_argument("chi_merge_threshold must be between 0 and 1.");
    }
    
    // Efficiently process data in a single pass
    int total_count = target.size();
    std::unordered_map<std::string, std::pair<int, int>> counts;
    counts.reserve(std::min(total_count, 10000));  // Reasonable upper limit
    
    // Count positives and negatives for each category
    for (int i = 0; i < total_count; ++i) {
      const int t = target[i];
      if (t != 0 && t != 1) {
        throw std::invalid_argument("Target must be binary (0 or 1).");
      }
      
      const std::string& cat = feature[i];
      auto& count_pair = counts[cat];
      
      if (t == 1) {
        count_pair.first++;
        total_pos++;
      } else {
        count_pair.second++;
        total_neg++;
      }
    }
    
    if (total_pos == 0 || total_neg == 0) {
      throw std::invalid_argument("Target must contain both 0 and 1 values.");
    }
    
    // Transfer counts to final maps
    for (const auto& item : counts) {
      const std::string& cat = item.first;
      const auto& count_pair = item.second;
      
      count_pos_map[cat] = count_pair.first;
      count_neg_map[cat] = count_pair.second;
      total_count_map[cat] = count_pair.first + count_pair.second;
    }
    
    unique_categories = static_cast<int>(counts.size());
    
    // Adjust bin constraints based on available unique categories
    min_bins = std::max(2, std::min(min_bins, unique_categories));
    max_bins = std::min(max_bins, unique_categories);
    if (min_bins > max_bins) {
      min_bins = max_bins;
    }
  }
  
  /**
   * Initialize bins with one category per bin
   */
  void initialize_bins() {
    bins.clear();
    bins.reserve(unique_categories);
    
    for (const auto& item : total_count_map) {
      Bin bin;
      bin.categories.push_back(item.first);
      bin.count_pos = count_pos_map[item.first];
      bin.count_neg = count_neg_map[item.first];
      bin.total_count = item.second;
      bins.push_back(std::move(bin));
    }
    
    // Pre-sort bins by WoE for better initial state
    sort_bins_by_woe();
  }
  
  /**
   * Sort bins by Weight of Evidence
   */
  void sort_bins_by_woe() {
    for (auto& bin : bins) {
      bin.woe = compute_woe(bin.count_pos, bin.count_neg);
    }
    
    std::sort(bins.begin(), bins.end(),
              [](const Bin& a, const Bin& b) { return a.woe < b.woe; });
  }
  
  /**
   * Handle rare categories by merging them with similar bins
   * As suggested in the Chi-Merge paper (Kerber, 1992)
   */
  void handle_rare_categories() {
    if (unique_categories <= 2) return;
    
    int total_count = total_pos + total_neg;
    std::vector<Bin> updated_bins;
    updated_bins.reserve(bins.size());
    std::vector<Bin> rare_bins;
    rare_bins.reserve(bins.size() / 3);  // Estimate based on bin_cutoff
    
    // Separate rare and non-rare bins
    for (auto& bin : bins) {
      double freq = static_cast<double>(bin.total_count) / static_cast<double>(total_count);
      if (freq < bin_cutoff) {
        rare_bins.push_back(std::move(bin));
      } else {
        updated_bins.push_back(std::move(bin));
      }
    }
    
    // If no rare bins or not enough non-rare bins, return
    if (rare_bins.empty() || updated_bins.size() < 2) {
      if (!rare_bins.empty()) {
        // If all bins are rare, keep them all
        updated_bins.insert(updated_bins.end(), 
                            std::make_move_iterator(rare_bins.begin()),
                            std::make_move_iterator(rare_bins.end()));
      }
      bins = std::move(updated_bins);
      return;
    }
    
    // Per Kerber's paper: merge rare categories with most similar bins
    for (auto& rare_bin : rare_bins) {
      size_t best_merge_index = 0;
      double min_chi_square = std::numeric_limits<double>::max();
      
      // Find most similar bin (lowest chi-square)
      for (size_t i = 0; i < updated_bins.size(); ++i) {
        double chi_sq = compute_chi_square_between_bins(rare_bin, updated_bins[i]);
        if (chi_sq < min_chi_square) {
          min_chi_square = chi_sq;
          best_merge_index = i;
        }
      }
      
      // Merge with most similar bin
      merge_two_bins(updated_bins[best_merge_index], rare_bin);
    }
    
    bins = std::move(updated_bins);
    chi_cache->invalidate();  // Reset cache after bin structure changes
  }
  
  /**
   * Limit the number of pre-bins to max_n_prebins
   */
  void limit_prebins() {
    if (bins.size() <= static_cast<size_t>(max_n_prebins) || !can_merge_further()) {
      return;
    }
    
    // Compute chi-square critical value based on threshold
    double critical_value = get_chi_square_critical_value();
    
    // Continue merging until we reach max_n_prebins or can't merge further
    while (bins.size() > static_cast<size_t>(max_n_prebins) && can_merge_further()) {
      // Find pair with minimum chi-square
      std::pair<double, size_t> min_chi_pair = find_min_chi_square_pair();
      double min_chi = min_chi_pair.first;
      size_t min_index = min_chi_pair.second;
      
      // Stop if chi-square exceeds threshold (bins are significantly different)
      if (min_chi > critical_value) {
        break;
      }
      
      // Merge bins with lowest chi-square
      merge_adjacent_bins(min_index);
      
      // Update cache for affected region
      update_chi_cache_after_merge(min_index);
    }
  }
  
  /**
   * Ensure minimum number of bins by splitting largest bins
   */
  void ensure_min_bins() {
    if (bins.size() >= static_cast<size_t>(min_bins)) {
      return;
    }
    
    // Continue splitting bins until we reach min_bins
    while (bins.size() < static_cast<size_t>(min_bins)) {
      // Find bin with most observations
      auto max_it = std::max_element(
        bins.begin(), bins.end(),
        [](const Bin& a, const Bin& b) { return a.total_count < b.total_count; }
      );
      
      // Can't split a bin with only one category
      if (max_it->categories.size() <= 1) {
        break;
      }
      
      // Split bin in two, preserving WoE order
      split_bin(*max_it);
      
      // Remove original bin and add the two new ones
      size_t split_index = std::distance(bins.begin(), max_it);
      bins.erase(max_it);
      
      // Update chi-square cache
      chi_cache->invalidate();  // Reset cache (simplest approach for split)
      chi_cache->resize(bins.size());
    }
  }
  
  /**
   * Split a bin into two approximately equal parts
   * @param bin The bin to split
   */
  void split_bin(const Bin& bin) {
    Bin bin1, bin2;
    bin1.categories.reserve(bin.categories.size() / 2 + 1);
    bin2.categories.reserve(bin.categories.size() / 2 + 1);
    
    // Sort categories by WoE for better splitting
    std::vector<std::string> sorted_categories = bin.categories;
    std::sort(sorted_categories.begin(), sorted_categories.end(),
              [this](const std::string& a, const std::string& b) {
                double woe_a = compute_woe(count_pos_map.at(a), count_neg_map.at(a));
                double woe_b = compute_woe(count_pos_map.at(b), count_neg_map.at(b));
                return woe_a < woe_b;
              });
    
    // Find optimal split point based on total count
    size_t total_so_far = 0;
    size_t target_total = bin.total_count / 2;
    size_t split_index = 0;
    
    for (size_t i = 0; i < sorted_categories.size(); ++i) {
      const std::string& cat = sorted_categories[i];
      total_so_far += total_count_map.at(cat);
      if (total_so_far >= target_total) {
        split_index = i + 1;
        break;
      }
    }
    
    // Ensure at least one category in each bin
    if (split_index == 0 || split_index >= sorted_categories.size()) {
      split_index = sorted_categories.size() / 2;
    }
    
    // Divide categories between bins
    bin1.categories.insert(bin1.categories.end(),
                           sorted_categories.begin(),
                           sorted_categories.begin() + split_index);
    bin2.categories.insert(bin2.categories.end(),
                           sorted_categories.begin() + split_index,
                           sorted_categories.end());
    
    // Calculate counts for new bins
    for (const auto& cat : bin1.categories) {
      bin1.count_pos += count_pos_map.at(cat);
      bin1.count_neg += count_neg_map.at(cat);
    }
    bin1.total_count = bin1.count_pos + bin1.count_neg;
    
    for (const auto& cat : bin2.categories) {
      bin2.count_pos += count_pos_map.at(cat);
      bin2.count_neg += count_neg_map.at(cat);
    }
    bin2.total_count = bin2.count_pos + bin2.count_neg;
    
    // Add new bins
    bins.push_back(std::move(bin1));
    bins.push_back(std::move(bin2));
  }
  
  /**
   * Merge bins using the Chi-Merge algorithm (Kerber, 1992)
   * Uses chi-square threshold based on significance level
   */
  void merge_bins_using_chimerge() {
    iterations_run = 0;
    bool keep_merging = true;
    double critical_value = get_chi_square_critical_value();
    
    // Sort bins by WoE to maintain monotonicity
    sort_bins_by_woe();
    
    // Initialize chi-square cache
    chi_cache->invalidate();
    chi_cache->resize(bins.size());
    
    while (can_merge_further() && keep_merging && iterations_run < max_iterations) {
      // Find pair with minimum chi-square
      std::pair<double, size_t> min_chi_pair = find_min_chi_square_pair();
      double min_chi = min_chi_pair.first;
      size_t min_index = min_chi_pair.second;
      
      // Check if we should stop merging based on:
      // 1. Statistical significance (chi-square > critical value)
      // 2. Reached max_bins
      if (min_chi > critical_value && bins.size() <= static_cast<size_t>(max_bins)) {
        converged = true;
        break;
      }
      
      // Store old minimum chi-square for convergence check
      double old_min_chi = min_chi;
      
      // Merge bins with lowest chi-square
      merge_adjacent_bins(min_index);
      
      // Update chi-square cache for affected region
      update_chi_cache_after_merge(min_index);
      
      // Check if we've converged or reached max_bins
      if (bins.size() <= static_cast<size_t>(max_bins)) {
        keep_merging = false;
        converged = true;
        break;
      }
      
      // Recalculate minimum chi-square to check convergence
      min_chi_pair = find_min_chi_square_pair();
      double new_min_chi = min_chi_pair.first;
      
      // Check convergence based on change in minimum chi-square
      keep_merging = std::fabs(new_min_chi - old_min_chi) > convergence_threshold;
      iterations_run++;
    }
    
    converged = (iterations_run < max_iterations) || !can_merge_further();
  }
  
  /**
   * Perform the Chi2 algorithm from Liu & Setiono (1995)
   * This extends Chi-Merge with automated significance levels
   */
  void perform_chi2_binning() {
    // Chi2 algorithm uses decreasing significance levels
    const std::vector<double> significance_levels = {0.5, 0.1, 0.05, 0.01, 0.005, 0.001};
    
    // Initial equal-frequency discretization
    initialize_equal_frequency_bins();
    
    // Main Chi2 algorithm phases
    for (double significance : significance_levels) {
      chi_merge_threshold = significance;
      
      // Reset cache for new phase
      chi_cache->invalidate();
      chi_cache->resize(bins.size());
      
      // Apply chi-merge with current significance level
      merge_bins_using_chimerge();
      
      // Check if we've reached target bin count
      if (bins.size() <= static_cast<size_t>(max_bins)) {
        break;
      }
      
      // Check inconsistency rate for feature selection (Chi2 specific)
      if (calculate_inconsistency_rate() < 0.05) {
        break;  // Feature is sufficiently discriminative
      }
    }
    
    // Final adjustment for min_bins
    ensure_min_bins();
    
    // Enforce monotonicity
    enforce_monotonicity();
  }
  
  /**
   * Initialize bins using equal frequency discretization
   * Used as starting point for Chi2 algorithm
   */
  void initialize_equal_frequency_bins() {
    // Sort categories by total count
    std::vector<std::pair<std::string, int>> sorted_categories;
    sorted_categories.reserve(total_count_map.size());
    
    for (const auto& entry : total_count_map) {
      sorted_categories.emplace_back(entry.first, entry.second);
    }
    
    std::sort(sorted_categories.begin(), sorted_categories.end(),
              [](const auto& a, const auto& b) { return a.second < b.second; });
    
    // Determine initial bin count for equal frequency
    int initial_bins = std::min(max_n_prebins, 
                                std::max(min_bins, 
                                         static_cast<int>(sqrt(sorted_categories.size()))));
    
    // Create equal frequency bins
    bins.clear();
    bins.resize(initial_bins);
    
    int total_observations = total_pos + total_neg;
    int target_bin_size = total_observations / initial_bins;
    int current_bin = 0;
    int current_bin_size = 0;
    
    for (const auto& cat_pair : sorted_categories) {
      const std::string& category = cat_pair.first;
      int cat_count = cat_pair.second;
      
      // Add category to current bin
      bins[current_bin].categories.push_back(category);
      bins[current_bin].count_pos += count_pos_map[category];
      bins[current_bin].count_neg += count_neg_map[category];
      bins[current_bin].total_count += cat_count;
      
      current_bin_size += cat_count;
      
      // Move to next bin if this one is full (except for last bin)
      if (current_bin_size >= target_bin_size && current_bin < initial_bins - 1) {
        current_bin++;
        current_bin_size = 0;
      }
    }
    
    // Remove any empty bins
    bins.erase(std::remove_if(bins.begin(), bins.end(),
                              [](const Bin& b) { return b.total_count == 0; }),
                              bins.end());
    
    // Sort bins by WoE
    sort_bins_by_woe();
  }
  
  /**
   * Calculate inconsistency rate for Chi2 algorithm
   * @return Inconsistency rate (0-1)
   */
  double calculate_inconsistency_rate() {
    // Map each category to its bin index
    std::unordered_map<std::string, size_t> category_to_bin;
    for (size_t i = 0; i < bins.size(); ++i) {
      for (const auto& category : bins[i].categories) {
        category_to_bin[category] = i;
      }
    }
    
    // Count inconsistent instances
    int inconsistent_count = 0;
    int total_instances = feature.size();
    
    // Use map to count class distribution per bin
    std::unordered_map<size_t, std::pair<int, int>> bin_class_counts;
    
    // First pass: count class distribution
    for (size_t i = 0; i < feature.size(); ++i) {
      const std::string& cat = feature[i];
      size_t bin_idx = category_to_bin[cat];
      
      if (target[i] == 1) {
        bin_class_counts[bin_idx].first++;
      } else {
        bin_class_counts[bin_idx].second++;
      }
    }
    
    // Second pass: count inconsistencies
    for (size_t i = 0; i < feature.size(); ++i) {
      const std::string& cat = feature[i];
      size_t bin_idx = category_to_bin[cat];
      
      // Determine majority class in this bin
      bool majority_positive = bin_class_counts[bin_idx].first > bin_class_counts[bin_idx].second;
      
      // Check if instance matches majority class
      if ((majority_positive && target[i] == 0) || 
          (!majority_positive && target[i] == 1)) {
        inconsistent_count++;
      }
    }
    
    return static_cast<double>(inconsistent_count) / total_instances;
  }
  
  /**
   * Determine whether WoE should be monotonically increasing or decreasing
   */
  void determine_monotonicity_robust() {
    if (bins.size() < 3) {
      is_increasing = true;
      return;
    }
    
    // Calculate trend using linear regression
    double sum_x = 0, sum_y = 0, sum_xy = 0, sum_x2 = 0;
    int n = bins.size();
    
    for (int i = 0; i < n; ++i) {
      double x = static_cast<double>(i);
      double y = bins[i].woe;
      sum_x += x;
      sum_y += y;
      sum_xy += x * y;
      sum_x2 += x * x;
    }
    
    double slope_numerator = n * sum_xy - sum_x * sum_y;
    double slope_denominator = n * sum_x2 - sum_x * sum_x;
    
    if (std::fabs(slope_denominator) < EPSILON) {
      is_increasing = true;  // Default to increasing if trend is flat
    } else {
      double slope = slope_numerator / slope_denominator;
      is_increasing = (slope >= 0);
    }
  }
  
  /**
   * Enforce monotonicity of Weight of Evidence
   */
  void enforce_monotonicity() {
    if (bins.size() <= 2) {
      return;
    }
    
    // Calculate WoE for each bin
    for (auto& bin : bins) {
      bin.woe = compute_woe(bin.count_pos, bin.count_neg);
    }
    
    // Determine direction of monotonicity
    determine_monotonicity_robust();
    
    bool monotonic = false;
    int iterations = 0;
    const int max_mono_iter = 100;  // Avoid infinite loops
    
    while (!monotonic && can_merge_further() && iterations < max_mono_iter) {
      monotonic = true;
      
      for (size_t i = 0; i < bins.size() - 1; ++i) {
        bool violation = is_increasing ?
        (bins[i].woe > bins[i + 1].woe + EPSILON) :
        (bins[i].woe < bins[i + 1].woe - EPSILON);
        
        if (violation) {
          // Merge bins that violate monotonicity
          double chi_sq = compute_chi_square_between_bins(bins[i], bins[i + 1]);
          chi_cache->set(i, i + 1, chi_sq);
          
          merge_adjacent_bins(i);
          update_chi_cache_after_merge(i);
          
          monotonic = false;
          break;
        }
      }
      
      iterations++;
    }
    
    // In extreme cases, resort bins to enforce monotonicity
    if (!monotonic) {
      sort_bins_by_woe();
      if (!is_increasing) {
        std::reverse(bins.begin(), bins.end());
      }
    }
  }
  
  /**
   * Calculate Weight of Evidence and Information Value for each bin
   */
  void calculate_woe_iv_bins() {
    double total_iv = 0.0;
    
    for (auto& bin : bins) {
      bin.woe = compute_woe(bin.count_pos, bin.count_neg);
      
      // Information Value calculation
      double dist_pos = static_cast<double>(bin.count_pos) / static_cast<double>(total_pos);
      double dist_neg = static_cast<double>(bin.count_neg) / static_cast<double>(total_neg);
      dist_pos = std::max(dist_pos, EPSILON);
      dist_neg = std::max(dist_neg, EPSILON);
      
      bin.iv = (dist_pos - dist_neg) * bin.woe;
      total_iv += bin.iv;
    }
  }
  
  /**
   * Prepare output List for R
   * @return Rcpp::List with binning results
   */
  Rcpp::List prepare_output() const {
    // Pre-allocate vectors for efficiency
    std::vector<std::string> bin_names;
    std::vector<double> bin_woe, bin_iv;
    std::vector<int> bin_count, bin_count_pos, bin_count_neg;
    
    size_t bin_count_size = bins.size();
    bin_names.reserve(bin_count_size);
    bin_woe.reserve(bin_count_size);
    bin_iv.reserve(bin_count_size);
    bin_count.reserve(bin_count_size);
    bin_count_pos.reserve(bin_count_size);
    bin_count_neg.reserve(bin_count_size);
    
    for (const auto& bin : bins) {
      bin_names.push_back(join_categories(bin.categories));
      bin_woe.push_back(bin.woe);
      bin_iv.push_back(bin.iv);
      bin_count.push_back(bin.total_count);
      bin_count_pos.push_back(bin.count_pos);
      bin_count_neg.push_back(bin.count_neg);
    }
    
    // Create IDs (1-based indexing for R)
    Rcpp::NumericVector ids(bin_count_size);
    for (size_t i = 0; i < bin_count_size; i++) {
      ids[i] = i + 1;
    }
    
    return Rcpp::List::create(
      Rcpp::Named("id") = ids,
      Rcpp::Named("bin") = bin_names,
      Rcpp::Named("woe") = bin_woe,
      Rcpp::Named("iv") = bin_iv,
      Rcpp::Named("count") = bin_count,
      Rcpp::Named("count_pos") = bin_count_pos,
      Rcpp::Named("count_neg") = bin_count_neg,
      Rcpp::Named("converged") = converged,
      Rcpp::Named("iterations") = iterations_run,
      Rcpp::Named("algorithm") = use_chi2_algorithm ? "Chi2" : "ChiMerge"
    );
  }
  
  /**
   * Find the pair of adjacent bins with lowest chi-square
   * @return pair<chi-square value, index of first bin>
   */
  std::pair<double, size_t> find_min_chi_square_pair() {
    double min_chi_square = std::numeric_limits<double>::max();
    size_t min_index = 0;
    
    for (size_t i = 0; i < bins.size() - 1; ++i) {
      // Check cache first
      double chi_square = chi_cache->get(i, i + 1);
      
      // Compute if not in cache
      if (chi_square < 0) {
        chi_square = compute_chi_square_between_bins(bins[i], bins[i + 1]);
        chi_cache->set(i, i + 1, chi_square);
      }
      
      if (chi_square < min_chi_square) {
        min_chi_square = chi_square;
        min_index = i;
      }
    }
    
    return {min_chi_square, min_index};
  }
  
  /**
   * Update chi-square cache after merging bins
   * @param merge_index Index of the first bin in the merge
   */
  void update_chi_cache_after_merge(size_t merge_index) {
    // Invalidate entries involving the merged bins
    chi_cache->invalidate_bin(merge_index);
    if (merge_index + 1 < bins.size()) {
      chi_cache->invalidate_bin(merge_index + 1);
    }
    
    // Update chi-square values for adjacent bins
    if (merge_index > 0) {
      double chi = compute_chi_square_between_bins(bins[merge_index - 1], bins[merge_index]);
      chi_cache->set(merge_index - 1, merge_index, chi);
    }
    
    if (merge_index + 1 < bins.size()) {
      double chi = compute_chi_square_between_bins(bins[merge_index], bins[merge_index + 1]);
      chi_cache->set(merge_index, merge_index + 1, chi);
    }
  }
  
  /**
   * Get chi-square critical value based on threshold
   * @return Critical chi-square value
   */
  double get_chi_square_critical_value() {
    // Find closest significance level in the table
    auto it = CHI_SQUARE_CRITICAL_VALUES.find(chi_merge_threshold);
    if (it != CHI_SQUARE_CRITICAL_VALUES.end()) {
      return it->second;
    }
    
    // Find nearest significance level if exact match not found
    double closest_threshold = 0.05;  // Default
    double min_diff = std::abs(chi_merge_threshold - closest_threshold);
    
    for (const auto& entry : CHI_SQUARE_CRITICAL_VALUES) {
      double diff = std::abs(chi_merge_threshold - entry.first);
      if (diff < min_diff) {
        min_diff = diff;
        closest_threshold = entry.first;
      }
    }
    
    return CHI_SQUARE_CRITICAL_VALUES.at(closest_threshold);
  }
  
  /**
   * Calculate chi-square statistic between two bins
   * Implementation follows Kerber (1992)
   * 
   * @param bin1 First bin
   * @param bin2 Second bin
   * @return Chi-square statistic
   */
  inline double compute_chi_square_between_bins(const Bin& bin1, const Bin& bin2) const {
    int total = bin1.total_count + bin2.total_count;
    if (total == 0) return 0.0;
    
    int total_pos_bins = bin1.count_pos + bin2.count_pos;
    int total_neg_bins = bin1.count_neg + bin2.count_neg;
    
    // Special cases
    if (total_pos_bins == 0 || total_neg_bins == 0) return 0.0;
    
    // Expected frequencies
    double expected1_pos = static_cast<double>(bin1.total_count * total_pos_bins) / total;
    double expected1_neg = static_cast<double>(bin1.total_count * total_neg_bins) / total;
    double expected2_pos = static_cast<double>(bin2.total_count * total_pos_bins) / total;
    double expected2_neg = static_cast<double>(bin2.total_count * total_neg_bins) / total;
    
    // Prevent division by zero
    expected1_pos = std::max(expected1_pos, EPSILON);
    expected1_neg = std::max(expected1_neg, EPSILON);
    expected2_pos = std::max(expected2_pos, EPSILON);
    expected2_neg = std::max(expected2_neg, EPSILON);
    
    // Chi-square computation
    // χ² = Σ (observed - expected)² / expected
    double chi_square = 0.0;
    chi_square += std::pow(bin1.count_pos - expected1_pos, 2) / expected1_pos;
    chi_square += std::pow(bin1.count_neg - expected1_neg, 2) / expected1_neg;
    chi_square += std::pow(bin2.count_pos - expected2_pos, 2) / expected2_pos;
    chi_square += std::pow(bin2.count_neg - expected2_neg, 2) / expected2_neg;
    
    return chi_square;
  }
  
  /**
   * Merge two adjacent bins at the given index
   * @param index Index of first bin to merge
   */
  inline void merge_adjacent_bins(size_t index) {
    if (index >= bins.size() - 1) return;
    
    // Merge categories and counts
    merge_two_bins(bins[index], bins[index + 1]);
    
    // Remove the second bin
    bins.erase(bins.begin() + index + 1);
    
    // Update chi-square cache size
    chi_cache->resize(bins.size());
  }
  
  /**
   * Merge two bins (add second to first)
   * @param bin1 Destination bin (modified)
   * @param bin2 Source bin
   */
  inline void merge_two_bins(Bin& bin1, const Bin& bin2) {
    // Reserve space for merged categories
    bin1.categories.reserve(bin1.categories.size() + bin2.categories.size());
    
    // Append categories from bin2 to bin1
    bin1.categories.insert(bin1.categories.end(), 
                           bin2.categories.begin(), 
                           bin2.categories.end());
    
    // Update counts
    bin1.count_pos += bin2.count_pos;
    bin1.count_neg += bin2.count_neg;
    bin1.total_count += bin2.total_count;
  }
  
  /**
   * Join category names with separator
   * @param categories Vector of category names
   * @return Joined string
   */
  inline std::string join_categories(const std::vector<std::string>& categories) const {
    if (categories.empty()) return "";
    if (categories.size() == 1) return categories[0];
    
    // Estimate final string length for pre-allocation
    size_t total_length = 0;
    for (const auto& cat : categories) {
      total_length += cat.length();
    }
    total_length += bin_separator.length() * (categories.size() - 1);
    
    // Build the joined string efficiently
    std::string result;
    result.reserve(total_length);
    
    result = categories[0];
    for (size_t i = 1; i < categories.size(); ++i) {
      result += bin_separator;
      result += categories[i];
    }
    
    return result;
  }
  
  /**
   * Check if bins can be merged further
   * @return true if more merging is possible
   */
  inline bool can_merge_further() const {
    return bins.size() > static_cast<size_t>(std::max(min_bins, 2));
  }
  
  /**
   * Calculate Weight of Evidence
   * @param c_pos Positive count
   * @param c_neg Negative count
   * @return WoE value
   */
  inline double compute_woe(int c_pos, int c_neg) const {
    double dist_pos = static_cast<double>(c_pos) / static_cast<double>(total_pos);
    double dist_neg = static_cast<double>(c_neg) / static_cast<double>(total_neg);
    
    // Prevent division by zero and log(0)
    dist_pos = std::max(dist_pos, EPSILON);
    dist_neg = std::max(dist_neg, EPSILON);
    
    return std::log(dist_pos / dist_neg);
  }
};

//' @title Optimal Binning for Categorical Variables using ChiMerge
//'
//' @description
//' Implements optimal binning for categorical variables using the ChiMerge algorithm
//' (Kerber, 1992) and Chi2 algorithm (Liu & Setiono, 1995), calculating Weight of 
//' Evidence (WoE) and Information Value (IV) for the resulting bins.
//'
//' @param target Integer vector of binary target values (0 or 1)
//' @param feature Character vector of categorical feature values
//' @param min_bins Minimum number of bins (default: 3)
//' @param max_bins Maximum number of bins (default: 5)
//' @param bin_cutoff Minimum frequency for a separate bin (default: 0.05)
//' @param max_n_prebins Maximum number of pre-bins before merging (default: 20)
//' @param bin_separator Separator for concatenating category names in bins (default: "%;%")
//' @param convergence_threshold Threshold for convergence in Chi-square difference (default: 1e-6)
//' @param max_iterations Maximum number of iterations for bin merging (default: 1000)
//' @param chi_merge_threshold Significance level threshold for chi-square test (default: 0.05)
//' @param use_chi2_algorithm Whether to use the enhanced Chi2 algorithm (default: FALSE)
//'
//' @return A list containing:
//' \itemize{
//'   \item id: Vector of numeric IDs for each bin
//'   \item bin: Vector of bin names (concatenated categories)
//'   \item woe: Vector of Weight of Evidence values for each bin
//'   \item iv: Vector of Information Value for each bin
//'   \item count: Vector of total counts for each bin
//'   \item count_pos: Vector of positive class counts for each bin
//'   \item count_neg: Vector of negative class counts for each bin
//'   \item converged: Boolean indicating whether the algorithm converged
//'   \item iterations: Number of iterations run
//'   \item algorithm: Which algorithm was used (ChiMerge or Chi2)
//' }
//'
//' @details
//' The ChiMerge algorithm (Kerber, 1992) uses chi-square statistics to determine when to 
//' merge adjacent bins. The chi-square statistic is calculated as:
//'
//' \deqn{\chi^2 = \sum_{i=1}^{2}\sum_{j=1}^{2} \frac{(O_{ij} - E_{ij})^2}{E_{ij}}}
//'
//' where \eqn{O_{ij}} is the observed frequency and \eqn{E_{ij}} is the expected frequency
//' for bin i and class j.
//'
//' The Chi2 algorithm (Liu & Setiono, 1995) extends ChiMerge with automated threshold 
//' determination and feature selection capabilities.
//'
//' Weight of Evidence (WoE) is calculated as:
//'
//' \deqn{WoE = \ln(\frac{P(X|Y=1)}{P(X|Y=0)})}
//'
//' Information Value (IV) for each bin is calculated as:
//'
//' \deqn{IV = (P(X|Y=1) - P(X|Y=0)) * WoE}
//'
//' The algorithm works by:
//' 1. Initializing each category as a separate bin
//' 2. Merging rare categories based on bin_cutoff
//' 3. Limiting the number of pre-bins to max_n_prebins
//' 4. Iteratively merging bins with the lowest chi-square until max_bins is reached,
//'    or no further merging is possible based on the chi-square threshold
//' 5. Ensuring monotonicity of WoE across bins
//'
//' The chi_merge_threshold parameter controls the statistical significance level for 
//' merging. A value of 0.05 corresponds to a 95% confidence level.
//'
//' References:
//' \itemize{
//'   \item Kerber, R. (1992). ChiMerge: Discretization of Numeric Attributes. 
//'         In Proceedings of the Tenth National Conference on Artificial Intelligence, 
//'         AAAI'92, pages 123-128.
//'   \item Liu, H. & Setiono, R. (1995). Chi2: Feature Selection and Discretization 
//'         of Numeric Attributes. In Proceedings of the 7th IEEE International Conference 
//'         on Tools with Artificial Intelligence, pages 388-391.
//' }
//'
//' @examples
//' \dontrun{
//' # Example data
//' target <- c(1, 0, 1, 1, 0, 1, 0, 0, 1, 1)
//' feature <- c("A", "B", "A", "C", "B", "D", "C", "A", "D", "B")
//'
//' # Run optimal binning with ChiMerge
//' result <- optimal_binning_categorical_cm(target, feature, min_bins = 2, max_bins = 4)
//'
//' # Use the Chi2 algorithm instead
//' result_chi2 <- optimal_binning_categorical_cm(target, feature, min_bins = 2, 
//'                                              max_bins = 4, use_chi2_algorithm = TRUE)
//'
//' # View results
//' print(result)
//' }
//'
//' @export
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
 // Convert R vectors to C++
 std::vector<std::string> feature_vec(feature.size());
 std::vector<int> target_vec(target.size());
 
 for (R_xlen_t i = 0; i < feature.size(); ++i) {
   if (feature[i] == NA_STRING) {
     feature_vec[i] = "NA";  // Handle missing values
   } else {
     feature_vec[i] = Rcpp::as<std::string>(feature[i]);
   }
 }
 
 for (R_xlen_t i = 0; i < target.size(); ++i) {
   if (IntegerVector::is_na(target[i])) {
     Rcpp::stop("Target cannot contain missing values.");
   } else {
     target_vec[i] = target[i];
   }
 }
 
 // Create object and perform binning
 OptimalBinningCategorical obcm(
     feature_vec, target_vec, min_bins, max_bins, bin_cutoff, max_n_prebins,
     bin_separator, convergence_threshold, max_iterations, chi_merge_threshold,
     use_chi2_algorithm
 );
 
 return obcm.perform_binning();
}




// // [[Rcpp::depends(Rcpp)]]
// 
// #include <Rcpp.h>
// #include <vector>
// #include <string>
// #include <algorithm>
// #include <cmath>
// #include <limits>
// #include <stdexcept>
// #include <unordered_map>
// #include <unordered_set>
// 
// using namespace Rcpp;
// 
// // Estrutura do bin
// struct Bin {
//   std::vector<std::string> categories;
//   int count_pos;
//   int count_neg;
//   double woe;
//   double iv;
//   int total_count;
// 
//   Bin() : count_pos(0), count_neg(0), woe(0.0), iv(0.0), total_count(0) {
//     categories.reserve(8);
//   }
// };
// 
// // Cache para cálculos de Chi-quadrado
// class ChiSquareCache {
// private:
//   std::unordered_map<size_t, double> cache;
//   size_t size;
// 
//   // Função para calcular um hash único para cada par de índices
//   inline size_t hash_pair(size_t i, size_t j) const {
//     if (i > j) std::swap(i, j);
//     return (i * size) + j;
//   }
// 
// public:
//   explicit ChiSquareCache(size_t n) : size(n) {
//     // Reservar espaço para potencialmente n*(n-1)/2 pares
//     cache.reserve(n * (n - 1) / 2);
//   }
// 
//   double get(size_t i, size_t j) {
//     size_t key = hash_pair(i, j);
//     auto it = cache.find(key);
//     return (it != cache.end()) ? it->second : -1.0;
//   }
// 
//   void set(size_t i, size_t j, double value) {
//     cache[hash_pair(i, j)] = value;
//   }
// 
//   void invalidate_row(size_t i) {
//     // Invalidar todas as entradas envolvendo o índice i
//     std::vector<size_t> to_erase;
//     for (const auto& entry : cache) {
//       size_t idx1 = entry.first / size;
//       size_t idx2 = entry.first % size;
//       if (idx1 == i || idx2 == i) {
//         to_erase.push_back(entry.first);
//       }
//     }
// 
//     for (size_t key : to_erase) {
//       cache.erase(key);
//     }
//   }
// 
//   void invalidate() {
//     cache.clear();
//   }
// };
// 
// // Classe principal para binagem ótima - otimizada
// class OptimalBinningCategorical {
// private:
//   // Entradas
//   const std::vector<std::string>& feature;
//   const std::vector<int>& target;
//   int min_bins;
//   int max_bins;
//   double bin_cutoff;
//   int max_n_prebins;
//   std::string bin_separator;
//   double convergence_threshold;
//   int max_iterations;
// 
//   // Variáveis internas
//   std::vector<Bin> bins;
//   int total_pos;
//   int total_neg;
//   std::unordered_map<std::string,int> count_pos_map;
//   std::unordered_map<std::string,int> count_neg_map;
//   std::unordered_map<std::string,int> total_count_map;
//   int unique_categories;
//   bool is_increasing;
//   bool converged;
//   int iterations_run;
// 
//   // Cache para cálculos de Chi-quadrado
//   std::unique_ptr<ChiSquareCache> chi_cache;
// 
//   // Constante para evitar divisão por zero
//   static constexpr double EPSILON = 1e-10;
// 
// public:
//   // Construtor
//   OptimalBinningCategorical(
//     const std::vector<std::string>& feature_,
//     const std::vector<int>& target_,
//     int min_bins_,
//     int max_bins_,
//     double bin_cutoff_,
//     int max_n_prebins_,
//     const std::string& bin_separator_,
//     double convergence_threshold_,
//     int max_iterations_
//   ) : feature(feature_),
//   target(target_),
//   min_bins(min_bins_),
//   max_bins(max_bins_),
//   bin_cutoff(bin_cutoff_),
//   max_n_prebins(max_n_prebins_),
//   bin_separator(bin_separator_),
//   convergence_threshold(convergence_threshold_),
//   max_iterations(max_iterations_),
//   total_pos(0),
//   total_neg(0),
//   unique_categories(0),
//   is_increasing(true),
//   converged(false),
//   iterations_run(0) {
//     // Estimativa melhor para alocação inicial
//     int estimated_categories = std::min(
//       static_cast<int>(feature.size() / 2),
//       1024
//     );
//     bins.reserve(estimated_categories);
//     count_pos_map.reserve(estimated_categories);
//     count_neg_map.reserve(estimated_categories);
//     total_count_map.reserve(estimated_categories);
//   }
// 
//   // Função principal
//   Rcpp::List perform_binning() {
//     try {
//       validate_inputs();
//       initialize_bins();
// 
//       // Inicializar cache após conhecer o número de bins
//       chi_cache = std::make_unique<ChiSquareCache>(bins.size());
// 
//       handle_rare_categories();
//       limit_prebins();
//       ensure_min_bins();
//       merge_bins_optimized();
//       enforce_monotonicity();
//       calculate_woe_iv_bins();
//       return prepare_output();
//     } catch (const std::exception& e) {
//       Rcpp::stop("Error in optimal binning: " + std::string(e.what()));
//     }
//   }
// 
// private:
//   // Valida parâmetros e prepara contagens
//   void validate_inputs() {
//     if (feature.empty() || target.empty()) {
//       throw std::invalid_argument("Feature e target não podem ser vazios.");
//     }
//     if (feature.size() != target.size()) {
//       throw std::invalid_argument("Feature e target devem ter o mesmo tamanho.");
//     }
//     if (min_bins <= 0 || max_bins <= 0 || min_bins > max_bins) {
//       throw std::invalid_argument("Valores inválidos para min_bins ou max_bins.");
//     }
//     if (bin_cutoff <= 0 || bin_cutoff >= 1) {
//       throw std::invalid_argument("bin_cutoff deve estar entre 0 e 1.");
//     }
//     if (convergence_threshold <= 0) {
//       throw std::invalid_argument("convergence_threshold deve ser positivo.");
//     }
//     if (max_iterations <= 0) {
//       throw std::invalid_argument("max_iterations deve ser positivo.");
//     }
// 
//     // Otimização: percorre o dataset uma única vez para contar
//     int total_count = target.size();
//     std::unordered_map<std::string, std::pair<int, int>> counts;
//     counts.reserve(total_count / 4);  // Heurística: estima número de categorias únicas
// 
//     for (int i = 0; i < total_count; ++i) {
//       const int t = target[i];
//       if (t != 0 && t != 1) {
//         throw std::invalid_argument("Target deve ser binário (0 ou 1).");
//       }
// 
//       const std::string& cat = feature[i];
//       auto& count_pair = counts[cat];
// 
//       if (t == 1) {
//         count_pair.first++;
//         total_pos++;
//       } else {
//         count_pair.second++;
//         total_neg++;
//       }
//     }
// 
//     if (total_pos == 0 || total_neg == 0) {
//       throw std::invalid_argument("O target deve conter tanto 0 quanto 1.");
//     }
// 
//     // Transfere para os mapas finais
//     for (const auto& item : counts) {
//       const std::string& cat = item.first;
//       const auto& count_pair = item.second;
// 
//       count_pos_map[cat] = count_pair.first;
//       count_neg_map[cat] = count_pair.second;
//       total_count_map[cat] = count_pair.first + count_pair.second;
//     }
// 
//     unique_categories = static_cast<int>(counts.size());
//     min_bins = std::max(2, std::min(min_bins, unique_categories));
//     max_bins = std::min(max_bins, unique_categories);
//     if (min_bins > max_bins) {
//       min_bins = max_bins;
//     }
// 
//     bins.reserve(unique_categories);
//   }
// 
//   // Inicializa cada categoria em um bin separado
//   void initialize_bins() {
//     bins.clear();
//     bins.reserve(unique_categories);
// 
//     for (const auto& item : total_count_map) {
//       Bin bin;
//       bin.categories.push_back(item.first);
//       bin.count_pos = count_pos_map[item.first];
//       bin.count_neg = count_neg_map[item.first];
//       bin.total_count = item.second;
//       bins.push_back(std::move(bin));
//     }
// 
//     // Pré-ordenar bins pelo WoE para facilitar monotonia
//     sort_bins_by_woe();
//   }
// 
//   // Ordenar bins pelo WoE
//   void sort_bins_by_woe() {
//     for (auto& bin : bins) {
//       double dist_pos = static_cast<double>(bin.count_pos) / static_cast<double>(total_pos);
//       double dist_neg = static_cast<double>(bin.count_neg) / static_cast<double>(total_neg);
//       dist_pos = std::max(dist_pos, EPSILON);
//       dist_neg = std::max(dist_neg, EPSILON);
//       bin.woe = std::log(dist_pos / dist_neg);
//     }
// 
//     std::sort(bins.begin(), bins.end(),
//               [](const Bin& a, const Bin& b) { return a.woe < b.woe; });
//   }
// 
//   // Trata categorias raras (frequência baixa) - Otimizado
//   void handle_rare_categories() {
//     if (unique_categories <= 2) return;
// 
//     int total_count = total_pos + total_neg;
//     std::vector<Bin> updated_bins;
//     updated_bins.reserve(bins.size());
//     std::vector<Bin> rare_bins;
//     rare_bins.reserve(bins.size() / 4); // Estimativa
// 
//     // Separar bins raros mais eficientemente
//     for (auto& bin : bins) {
//       double freq = static_cast<double>(bin.total_count) / static_cast<double>(total_count);
//       if (freq < bin_cutoff) {
//         rare_bins.push_back(std::move(bin));
//       } else {
//         updated_bins.push_back(std::move(bin));
//       }
//     }
// 
//     // Se não houver bins raros ou não sobrariam bins não-raros suficientes, retornar
//     if (rare_bins.empty() || updated_bins.size() < 2) {
//       if (rare_bins.empty()) {
//         bins = std::move(updated_bins);
//       }
//       return;
//     }
// 
//     // Usar matriz de similaridade para mesclar bins raros
//     for (auto& rare_bin : rare_bins) {
//       size_t best_merge_index = 0;
//       double min_chi_square = std::numeric_limits<double>::max();
// 
//       // Encontrar o bin mais similar para mesclar
//       for (size_t i = 0; i < updated_bins.size(); ++i) {
//         double chi_sq = compute_chi_square_between_bins(rare_bin, updated_bins[i]);
//         if (chi_sq < min_chi_square) {
//           min_chi_square = chi_sq;
//           best_merge_index = i;
//         }
//       }
// 
//       merge_two_bins(updated_bins[best_merge_index], rare_bin);
//     }
// 
//     bins = std::move(updated_bins);
//     chi_cache->invalidate(); // Resetar cache após reestruturação
//   }
// 
//   // Limita pré-bins ao máximo definido - Otimizado
//   void limit_prebins() {
//     if (bins.size() <= static_cast<size_t>(max_n_prebins) || !can_merge_further()) {
//       return;
//     }
// 
//     // Manter o número de bins até max_n_prebins
//     while (bins.size() > static_cast<size_t>(max_n_prebins) && can_merge_further()) {
//       // Calcular similaridade entre bins adjacentes
//       std::vector<std::pair<double, size_t>> chi_scores;
//       chi_scores.reserve(bins.size() - 1);
// 
//       for (size_t i = 0; i < bins.size() - 1; ++i) {
//         double chi = compute_chi_square_between_bins(bins[i], bins[i + 1]);
//         chi_scores.emplace_back(chi, i);
//       }
// 
//       // Ordenar por similaridade (menor chi-quadrado)
//       std::sort(chi_scores.begin(), chi_scores.end());
// 
//       // Mesclar os bins mais similares
//       size_t merge_index = chi_scores[0].second;
//       merge_adjacent_bins(merge_index);
//     }
// 
//     chi_cache->invalidate(); // Resetar cache após reestruturação
//   }
// 
//   // Garante que min_bins seja respeitado, dividindo bins grandes - Otimizado
//   void ensure_min_bins() {
//     if (bins.size() >= static_cast<size_t>(min_bins)) {
//       return;
//     }
// 
//     // Continuar dividindo bins até atingir min_bins
//     while (bins.size() < static_cast<size_t>(min_bins)) {
//       // Encontrar o bin com mais observações
//       auto max_it = std::max_element(
//         bins.begin(), bins.end(),
//         [](const Bin& a, const Bin& b) { return a.total_count < b.total_count; }
//       );
// 
//       // Se o bin tiver apenas uma categoria, não pode ser dividido
//       if (max_it->categories.size() <= 1) {
//         break;
//       }
// 
//       // Dividir o bin em dois, balanceando o número de observações
//       Bin bin1, bin2;
//       bin1.categories.reserve(max_it->categories.size() / 2);
//       bin2.categories.reserve(max_it->categories.size() / 2);
// 
//       // Ordenar categorias por WoE para preservar monotonicidade
//       std::sort(max_it->categories.begin(), max_it->categories.end(),
//                 [this](const std::string& a, const std::string& b) {
//                   double woe_a = compute_woe(count_pos_map.at(a), count_neg_map.at(a));
//                   double woe_b = compute_woe(count_pos_map.at(b), count_neg_map.at(b));
//                   return woe_a < woe_b;
//                 });
// 
//       size_t total_so_far = 0;
//       size_t target_total = max_it->total_count / 2;
//       size_t split_index = 0;
// 
//       // Encontrar o ponto de divisão mais balanceado
//       for (size_t i = 0; i < max_it->categories.size(); ++i) {
//         const std::string& cat = max_it->categories[i];
//         total_so_far += total_count_map.at(cat);
//         if (total_so_far >= target_total) {
//           split_index = i + 1;
//           break;
//         }
//       }
// 
//       if (split_index == 0 || split_index >= max_it->categories.size()) {
//         split_index = max_it->categories.size() / 2;
//       }
// 
//       // Dividir categorias em dois bins
//       bin1.categories.insert(bin1.categories.end(),
//                              max_it->categories.begin(),
//                              max_it->categories.begin() + split_index);
//       bin2.categories.insert(bin2.categories.end(),
//                              max_it->categories.begin() + split_index,
//                              max_it->categories.end());
// 
//       // Calcular contagens para os novos bins
//       for (const auto& cat : bin1.categories) {
//         bin1.count_pos += count_pos_map.at(cat);
//         bin1.count_neg += count_neg_map.at(cat);
//       }
//       bin1.total_count = bin1.count_pos + bin1.count_neg;
// 
//       for (const auto& cat : bin2.categories) {
//         bin2.count_pos += count_pos_map.at(cat);
//         bin2.count_neg += count_neg_map.at(cat);
//       }
//       bin2.total_count = bin2.count_pos + bin2.count_neg;
// 
//       // Substituir o bin original pelos dois novos
//       *max_it = std::move(bin1);
//       bins.push_back(std::move(bin2));
//     }
// 
//     chi_cache->invalidate(); // Resetar cache após reestruturação
//   }
// 
//   // Mescla bins baseando-se no Chi-quadrado até atingir max_bins ou convergência - Otimizado
//   void merge_bins_optimized() {
//     iterations_run = 0;
//     bool keep_merging = true;
// 
//     while (can_merge_further() && keep_merging && iterations_run < max_iterations) {
//       // Calcular chi-quadrado para todos os pares adjacentes
//       std::vector<double> chi_squares(bins.size() - 1);
// 
// #pragma omp parallel for if(bins.size() > 100)
//       for (size_t i = 0; i < bins.size() - 1; ++i) {
//         double cached = chi_cache->get(i, i+1);
//         if (cached >= 0) {
//           chi_squares[i] = cached;
//         } else {
//           chi_squares[i] = compute_chi_square_between_bins(bins[i], bins[i+1]);
//           chi_cache->set(i, i+1, chi_squares[i]);
//         }
//       }
// 
//       // Encontrar o menor chi-quadrado
//       auto min_it = std::min_element(chi_squares.begin(), chi_squares.end());
//       size_t min_index = std::distance(chi_squares.begin(), min_it);
//       double old_chi_square = *min_it;
// 
//       // Mesclar bins com menor chi-quadrado
//       merge_adjacent_bins(min_index);
// 
//       // Invalidar cache para os bins afetados
//       chi_cache->invalidate_row(min_index);
//       if (min_index + 1 < bins.size()) {
//         chi_cache->invalidate_row(min_index + 1);
//       }
// 
//       if (!can_merge_further()) {
//         break;
//       }
// 
//       // Calcular novo chi-quadrado mínimo
//       std::vector<double> new_chi_squares;
//       new_chi_squares.reserve(2);
// 
//       if (min_index > 0) {
//         double chi = compute_chi_square_between_bins(bins[min_index-1], bins[min_index]);
//         chi_cache->set(min_index-1, min_index, chi);
//         new_chi_squares.push_back(chi);
//       }
// 
//       if (min_index < bins.size() - 1) {
//         double chi = compute_chi_square_between_bins(bins[min_index], bins[min_index+1]);
//         chi_cache->set(min_index, min_index+1, chi);
//         new_chi_squares.push_back(chi);
//       }
// 
//       // Verificar convergência
//       double new_min_chi = *std::min_element(new_chi_squares.begin(), new_chi_squares.end());
//       keep_merging = std::fabs(new_min_chi - old_chi_square) > convergence_threshold;
//       iterations_run++;
//     }
// 
//     converged = (iterations_run < max_iterations) || !can_merge_further();
//   }
// 
//   // Método melhorado para determinar monotonicidade
//   void determine_monotonicity_robust() {
//     if (bins.size() < 3) {
//       is_increasing = true;
//       return;
//     }
// 
//     double sum_x = 0, sum_y = 0, sum_xy = 0, sum_x2 = 0;
//     int n = bins.size();
// 
//     // Calcular coeficiente de correlação linear
//     for (int i = 0; i < n; ++i) {
//       double x = i;
//       double y = bins[i].woe;
//       sum_x += x;
//       sum_y += y;
//       sum_xy += x * y;
//       sum_x2 += x * x;
//     }
// 
//     double slope_numerator = n * sum_xy - sum_x * sum_y;
//     double slope_denominator = n * sum_x2 - sum_x * sum_x;
// 
//     if (std::fabs(slope_denominator) < EPSILON) {
//       is_increasing = true; // Padrão para crescente se a tendência for plana
//     } else {
//       double slope = slope_numerator / slope_denominator;
//       is_increasing = (slope >= 0);
//     }
//   }
// 
//   // Impõe monotonicidade do WoE - Otimizado
//   void enforce_monotonicity() {
//     if (unique_categories <= 2) {
//       return;
//     }
// 
//     // Calcular WoE para cada bin para verificação de monotonicidade
//     for (auto& bin : bins) {
//       double dist_pos = static_cast<double>(bin.count_pos) / static_cast<double>(total_pos);
//       double dist_neg = static_cast<double>(bin.count_neg) / static_cast<double>(total_neg);
//       dist_pos = std::max(dist_pos, EPSILON);
//       dist_neg = std::max(dist_neg, EPSILON);
//       bin.woe = std::log(dist_pos / dist_neg);
//     }
// 
//     determine_monotonicity_robust();
// 
//     bool monotonic = false;
//     int iterations = 0;
//     const int max_mono_iter = 100; // Limite para evitar loops infinitos
// 
//     while (!monotonic && can_merge_further() && iterations < max_mono_iter) {
//       monotonic = true;
// 
//       for (size_t i = 0; i < bins.size() - 1; ++i) {
//         bool violation = is_increasing ?
//         (bins[i].woe > bins[i + 1].woe + EPSILON) :
//         (bins[i].woe < bins[i + 1].woe - EPSILON);
// 
//         if (violation) {
//           // Mesclar bins que violam a monotonicidade
//           merge_adjacent_bins(i);
//           monotonic = false;
// 
//           // Recalcular WoE após mesclagem
//           double dist_pos = static_cast<double>(bins[i].count_pos) / static_cast<double>(total_pos);
//           double dist_neg = static_cast<double>(bins[i].count_neg) / static_cast<double>(total_neg);
//           dist_pos = std::max(dist_pos, EPSILON);
//           dist_neg = std::max(dist_neg, EPSILON);
//           bins[i].woe = std::log(dist_pos / dist_neg);
// 
//           break;
//         }
//       }
// 
//       iterations++;
//     }
//   }
// 
//   // Calcula WoE e IV para cada bin
//   void calculate_woe_iv_bins() {
//     double total_iv = 0.0;
// 
//     for (auto& bin : bins) {
//       double dist_pos = static_cast<double>(bin.count_pos) / static_cast<double>(total_pos);
//       double dist_neg = static_cast<double>(bin.count_neg) / static_cast<double>(total_neg);
//       dist_pos = std::max(dist_pos, EPSILON);
//       dist_neg = std::max(dist_neg, EPSILON);
//       bin.woe = std::log(dist_pos / dist_neg);
//       bin.iv = (dist_pos - dist_neg) * bin.woe;
//       total_iv += bin.iv;
//     }
//   }
// 
//   // Prepara a saída R
//   Rcpp::List prepare_output() const {
//     std::vector<std::string> bin_names;
//     bin_names.reserve(bins.size());
//     std::vector<double> bin_woe, bin_iv;
//     bin_woe.reserve(bins.size());
//     bin_iv.reserve(bins.size());
//     std::vector<int> bin_count, bin_count_pos, bin_count_neg;
//     bin_count.reserve(bins.size());
//     bin_count_pos.reserve(bins.size());
//     bin_count_neg.reserve(bins.size());
// 
//     for (const auto& bin : bins) {
//       bin_names.push_back(join_categories(bin.categories));
//       bin_woe.push_back(bin.woe);
//       bin_iv.push_back(bin.iv);
//       bin_count.push_back(bin.total_count);
//       bin_count_pos.push_back(bin.count_pos);
//       bin_count_neg.push_back(bin.count_neg);
//     }
// 
//     // Criar vetor de IDs com o mesmo tamanho de bin_names
//     Rcpp::NumericVector ids(bin_names.size());
//     for(int i = 0; i < bin_names.size(); i++) {
//       ids[i] = i + 1;  // Começa em 1 até size(bin_names)
//     }
// 
//     return Rcpp::List::create(
//       Rcpp::Named("id") = ids,
//       Rcpp::Named("bin") = bin_names,
//       Rcpp::Named("woe") = bin_woe,
//       Rcpp::Named("iv") = bin_iv,
//       Rcpp::Named("count") = bin_count,
//       Rcpp::Named("count_pos") = bin_count_pos,
//       Rcpp::Named("count_neg") = bin_count_neg,
//       Rcpp::Named("converged") = converged,
//       Rcpp::Named("iterations") = iterations_run
//     );
//   }
// 
//   // Calcula Chi-quadrado entre dois bins - Otimizado
//   inline double compute_chi_square_between_bins(const Bin& bin1, const Bin& bin2) const {
//     int total = bin1.total_count + bin2.total_count;
//     if (total == 0) return 0.0;
// 
//     int total_pos_bins = bin1.count_pos + bin2.count_pos;
//     int total_neg_bins = bin1.count_neg + bin2.count_neg;
// 
//     // Cálculo rápido para casos especiais
//     if (total_pos_bins == 0 || total_neg_bins == 0) return 0.0;
// 
//     // Cálculo otimizado do chi-quadrado
//     double expected1_pos = static_cast<double>(bin1.total_count * total_pos_bins) / static_cast<double>(total);
//     double expected1_neg = static_cast<double>(bin1.total_count * total_neg_bins) / static_cast<double>(total);
// 
//     // Prevenir divisão por zero
//     expected1_pos = std::max(expected1_pos, EPSILON);
//     expected1_neg = std::max(expected1_neg, EPSILON);
// 
//     // Cálculo do chi-quadrado usando apenas os valores do primeiro bin
//     // (o segundo bin é determinado pela diferença, reduzindo cálculos)
//     double chi_square = 0.0;
//     chi_square += std::pow(bin1.count_pos - expected1_pos, 2.0) / expected1_pos;
//     chi_square += std::pow(bin1.count_neg - expected1_neg, 2.0) / expected1_neg;
// 
//     double expected2_pos = static_cast<double>(bin2.total_count * total_pos_bins) / static_cast<double>(total);
//     double expected2_neg = static_cast<double>(bin2.total_count * total_neg_bins) / static_cast<double>(total);
// 
//     expected2_pos = std::max(expected2_pos, EPSILON);
//     expected2_neg = std::max(expected2_neg, EPSILON);
// 
//     chi_square += std::pow(bin2.count_pos - expected2_pos, 2.0) / expected2_pos;
//     chi_square += std::pow(bin2.count_neg - expected2_neg, 2.0) / expected2_neg;
// 
//     return chi_square;
//   }
// 
//   // Mescla bins adjacentes no índice dado
//   inline void merge_adjacent_bins(size_t index) {
//     if (index >= bins.size() - 1) return;
//     merge_two_bins(bins[index], bins[index + 1]);
//     bins.erase(bins.begin() + index + 1);
//   }
// 
//   // Mescla dois bins - Otimizado
//   inline void merge_two_bins(Bin& bin1, const Bin& bin2) {
//     // Reservar espaço para as categorias antes de inserir
//     bin1.categories.reserve(bin1.categories.size() + bin2.categories.size());
//     bin1.categories.insert(bin1.categories.end(), bin2.categories.begin(), bin2.categories.end());
//     bin1.count_pos += bin2.count_pos;
//     bin1.count_neg += bin2.count_neg;
//     bin1.total_count += bin2.total_count;
//   }
// 
//   // Junta nomes de categorias - Otimizado
//   inline std::string join_categories(const std::vector<std::string>& categories) const {
//     if (categories.empty()) return "";
//     if (categories.size() == 1) return categories[0];
// 
//     // Estimar o tamanho da string final para pré-alocação
//     size_t total_length = 0;
//     for (const auto& cat : categories) {
//       total_length += cat.length();
//     }
//     total_length += bin_separator.length() * (categories.size() - 1);
// 
//     // Construir a string de forma eficiente
//     std::string result;
//     result.reserve(total_length);
// 
//     result = categories[0];
//     for (size_t i = 1; i < categories.size(); ++i) {
//       result += bin_separator;
//       result += categories[i];
//     }
// 
//     return result;
//   }
// 
//   // Verifica se pode mesclar mais
//   inline bool can_merge_further() const {
//     return bins.size() > static_cast<size_t>(std::max(min_bins, 2));
//   }
// 
//   // Calcula WoE para uma categoria
//   inline double compute_woe(int c_pos, int c_neg) const {
//     double dist_pos = static_cast<double>(c_pos) / static_cast<double>(total_pos);
//     double dist_neg = static_cast<double>(c_neg) / static_cast<double>(total_neg);
//     dist_pos = std::max(dist_pos, EPSILON);
//     dist_neg = std::max(dist_neg, EPSILON);
//     return std::log(dist_pos / dist_neg);
//   }
// };
// 
// //' @title Optimal Binning for Categorical Variables by Chi-Merge
// //'
// //' @description
// //' Implements optimal binning for categorical variables using the Chi-Merge algorithm,
// //' calculating Weight of Evidence (WoE) and Information Value (IV) for resulting bins.
// //'
// //' @param target Integer vector of binary target values (0 or 1).
// //' @param feature Character vector of categorical feature values.
// //' @param min_bins Minimum number of bins (default: 3).
// //' @param max_bins Maximum number of bins (default: 5).
// //' @param bin_cutoff Minimum frequency for a separate bin (default: 0.05).
// //' @param max_n_prebins Maximum number of pre-bins before merging (default: 20).
// //' @param bin_separator Separator for concatenating category names in bins (default: "%;%").
// //' @param convergence_threshold Threshold for convergence in Chi-square difference (default: 1e-6).
// //' @param max_iterations Maximum number of iterations for bin merging (default: 1000).
// //'
// //' @return A list containing:
// //' \itemize{
// //'   \item id: Vector of numeric IDs for each bin.
// //'   \item bin: Vector of bin names (concatenated categories).
// //'   \item woe: Vector of Weight of Evidence values for each bin.
// //'   \item iv: Vector of Information Value for each bin.
// //'   \item count: Vector of total counts for each bin.
// //'   \item count_pos: Vector of positive class counts for each bin.
// //'   \item count_neg: Vector of negative class counts for each bin.
// //'   \item converged: Boolean indicating whether the algorithm converged.
// //'   \item iterations: Number of iterations run.
// //' }
// //'
// //' @details
// //' The algorithm uses Chi-square statistics to merge adjacent bins:
// //'
// //' \deqn{\chi^2 = \sum_{i=1}^{2}\sum_{j=1}^{2} \frac{(O_{ij} - E_{ij})^2}{E_{ij}}}
// //'
// //' where \eqn{O_{ij}} is the observed frequency and \eqn{E_{ij}} is the expected frequency
// //' for bin i and class j.
// //'
// //' Weight of Evidence (WoE) for each bin:
// //'
// //' \deqn{WoE = \ln(\frac{P(X|Y=1)}{P(X|Y=0)})}
// //'
// //' Information Value (IV) for each bin:
// //'
// //' \deqn{IV = (P(X|Y=1) - P(X|Y=0)) * WoE}
// //'
// //' The algorithm initializes bins for each category, merges rare categories based on
// //' bin_cutoff, and then iteratively merges bins with the lowest chi-square
// //' until max_bins is reached or no further merging is possible. It determines the
// //' direction of monotonicity based on the initial trend and enforces it, allowing
// //' deviations if min_bins constraints are triggered.
// //'
// //' @examples
// //' \dontrun{
// //' # Example data
// //' target <- c(1, 0, 1, 1, 0, 1, 0, 0, 1, 1)
// //' feature <- c("A", "B", "A", "C", "B", "D", "C", "A", "D", "B")
// //'
// //' # Run optimal binning
// //' result <- optimal_binning_categorical_cm(target, feature, min_bins = 2, max_bins = 4)
// //'
// //' # View results
// //' print(result)
// //' }
// //'
// //' @export
// // [[Rcpp::export]]
// Rcpp::List optimal_binning_categorical_cm(
//    Rcpp::IntegerVector target,
//    Rcpp::CharacterVector feature,
//    int min_bins = 3,
//    int max_bins = 5,
//    double bin_cutoff = 0.05,
//    int max_n_prebins = 20,
//    std::string bin_separator = "%;%",
//    double convergence_threshold = 1e-6,
//    int max_iterations = 1000
// ) {
//  // Converte vetores R para C++
//  std::vector<std::string> feature_vec(feature.size());
//  std::vector<int> target_vec(target.size());
// 
//  for (size_t i = 0; i < feature.size(); ++i) {
//    if (feature[i] == NA_STRING) {
//      feature_vec[i] = "NA";
//    } else {
//      feature_vec[i] = Rcpp::as<std::string>(feature[i]);
//    }
//  }
// 
//  for (size_t i = 0; i < target.size(); ++i) {
//    if (IntegerVector::is_na(target[i])) {
//      Rcpp::stop("Target não pode conter valores ausentes.");
//    } else {
//      target_vec[i] = target[i];
//    }
//  }
// 
//  // Cria objeto e executa binagem
//  OptimalBinningCategorical obcm(
//      feature_vec, target_vec, min_bins, max_bins, bin_cutoff, max_n_prebins,
//      bin_separator, convergence_threshold, max_iterations
//  );
// 
//  return obcm.perform_binning();
// }





// // [[Rcpp::depends(Rcpp, RcppParallel)]]
// 
// #include <Rcpp.h>
// #include <RcppParallel.h>
// #include <vector>
// #include <string>
// #include <algorithm>
// #include <cmath>
// #include <limits>
// #include <stdexcept>
// #include <unordered_map>
// #include <unordered_set>
// #include <memory>
// #include <numeric>
// #include <atomic>
// #include <thread>
// 
// using namespace Rcpp;
// using namespace RcppParallel;
// 
// // Estrutura do bin com melhorias de armazenamento e eficiência
// struct Bin {
//   std::vector<std::string> categories;
//   int count_pos;
//   int count_neg;
//   double woe;
//   double iv;
//   int total_count;
//   double event_rate;  // Nova métrica: taxa de eventos (count_pos/total_count)
//   
//   Bin() : count_pos(0), count_neg(0), woe(0.0), iv(0.0), total_count(0), event_rate(0.0) {
//     categories.reserve(8);
//   }
//   
//   // Adicionado construtor de movimentação para melhorar performance
//   Bin(Bin&& other) noexcept : 
//     categories(std::move(other.categories)),
//     count_pos(other.count_pos),
//     count_neg(other.count_neg),
//     woe(other.woe),
//     iv(other.iv),
//     total_count(other.total_count),
//     event_rate(other.event_rate) {
//   }
//   
//   // Operador de atribuição por movimentação
//   Bin& operator=(Bin&& other) noexcept {
//     if (this != &other) {
//       categories = std::move(other.categories);
//       count_pos = other.count_pos;
//       count_neg = other.count_neg;
//       woe = other.woe;
//       iv = other.iv;
//       total_count = other.total_count;
//       event_rate = other.event_rate;
//     }
//     return *this;
//   }
//   
//   // Atualiza a taxa de eventos
//   void update_event_rate() {
//     event_rate = total_count > 0 ? 
//     static_cast<double>(count_pos) / static_cast<double>(total_count) : 0.0;
//   }
// };
// 
// // Estrutura para armazenar estatísticas sobre os bins
// struct BinStats {
//   double total_iv;
//   bool monotonic;
//   std::string direction;
//   double min_woe;
//   double max_woe;
//   double avg_bin_size;
//   int largest_bin;
//   int smallest_bin;
//   
//   BinStats() : 
//     total_iv(0.0), 
//     monotonic(false), 
//     direction("unknown"), 
//     min_woe(std::numeric_limits<double>::max()), 
//     max_woe(std::numeric_limits<double>::lowest()),
//     avg_bin_size(0.0),
//     largest_bin(0),
//     smallest_bin(std::numeric_limits<int>::max()) {
//   }
// };
// 
// // Worker para computação paralela do chi-quadrado
// struct ChiSquareWorker : public Worker {
//   // Dados de entrada
//   const std::vector<Bin>* bins;
//   const int total_pos;
//   const int total_neg;
//   const double epsilon;
//   
//   // Saída
//   std::vector<double>* chi_squares;
//   
//   // Construtor
//   ChiSquareWorker(
//     const std::vector<Bin>* bins_,
//     int total_pos_,
//     int total_neg_,
//     double epsilon_,
//     std::vector<double>* chi_squares_
//   ) : bins(bins_), total_pos(total_pos_), total_neg(total_neg_), 
//   epsilon(epsilon_), chi_squares(chi_squares_) {}
//   
//   // Função de cálculo de chi-quadrado otimizada
//   inline double compute_chi_square(const Bin& bin1, const Bin& bin2) const {
//     int total = bin1.total_count + bin2.total_count;
//     if (total == 0) return 0.0;
//     
//     int total_pos_bins = bin1.count_pos + bin2.count_pos;
//     int total_neg_bins = bin1.count_neg + bin2.count_neg;
//     
//     // Cálculo rápido para casos especiais
//     if (total_pos_bins == 0 || total_neg_bins == 0) return 0.0;
//     
//     // Cálculo otimizado do chi-quadrado
//     double expected1_pos = static_cast<double>(bin1.total_count * total_pos_bins) / static_cast<double>(total);
//     double expected1_neg = static_cast<double>(bin1.total_count * total_neg_bins) / static_cast<double>(total);
//     
//     // Prevenir divisão por zero
//     expected1_pos = std::max(expected1_pos, epsilon);
//     expected1_neg = std::max(expected1_neg, epsilon);
//     
//     // Cálculo do chi-quadrado usando apenas os valores do primeiro bin
//     double chi_square = 0.0;
//     chi_square += std::pow(bin1.count_pos - expected1_pos, 2.0) / expected1_pos;
//     chi_square += std::pow(bin1.count_neg - expected1_neg, 2.0) / expected1_neg;
//     
//     double expected2_pos = static_cast<double>(bin2.total_count * total_pos_bins) / static_cast<double>(total);
//     double expected2_neg = static_cast<double>(bin2.total_count * total_neg_bins) / static_cast<double>(total);
//     
//     expected2_pos = std::max(expected2_pos, epsilon);
//     expected2_neg = std::max(expected2_neg, epsilon);
//     
//     chi_square += std::pow(bin2.count_pos - expected2_pos, 2.0) / expected2_pos;
//     chi_square += std::pow(bin2.count_neg - expected2_neg, 2.0) / expected2_neg;
//     
//     return chi_square;
//   }
//   
//   // Função de processamento paralelo
//   void operator()(std::size_t begin, std::size_t end) {
//     for (std::size_t i = begin; i < end; i++) {
//       if (i < bins->size() - 1) {
//         (*chi_squares)[i] = compute_chi_square((*bins)[i], (*bins)[i+1]);
//       }
//     }
//   }
// };
// 
// // Cache para cálculos de Chi-quadrado melhorado com lock-free
// class ChiSquareCache {
// private:
//   std::unordered_map<size_t, double> cache;
//   size_t size;
//   std::atomic<size_t> hits{0};
//   std::atomic<size_t> misses{0};
//   
//   // Função para calcular um hash único para cada par de índices
//   inline size_t hash_pair(size_t i, size_t j) const {
//     if (i > j) std::swap(i, j);
//     return (i * size) + j;
//   }
//   
// public:
//   explicit ChiSquareCache(size_t n) : size(n) {
//     // Reservar espaço para potencialmente n*(n-1)/2 pares
//     cache.reserve(n * (n - 1) / 2);
//   }
//   
//   double get(size_t i, size_t j) {
//     size_t key = hash_pair(i, j);
//     auto it = cache.find(key);
//     if (it != cache.end()) {
//       hits++;
//       return it->second;
//     }
//     misses++;
//     return -1.0;
//   }
//   
//   void set(size_t i, size_t j, double value) {
//     cache[hash_pair(i, j)] = value;
//   }
//   
//   void invalidate_row(size_t i) {
//     // Invalidar todas as entradas envolvendo o índice i
//     std::vector<size_t> to_erase;
//     to_erase.reserve(cache.size() / size);  // Estimativa melhorada
//     
//     for (const auto& entry : cache) {
//       size_t idx1 = entry.first / size;
//       size_t idx2 = entry.first % size;
//       if (idx1 == i || idx2 == i) {
//         to_erase.push_back(entry.first);
//       }
//     }
//     
//     for (size_t key : to_erase) {
//       cache.erase(key);
//     }
//   }
//   
//   void invalidate() {
//     cache.clear();
//     hits = 0;
//     misses = 0;
//   }
//   
//   // Função para obter estatísticas do cache
//   std::pair<size_t, size_t> get_stats() const {
//     size_t h = hits.load();
//     size_t m = misses.load();
//     return std::make_pair(h, m);
//   }
//   
//   // Taxa de acertos do cache
//   double hit_rate() const {
//     size_t h = hits.load();
//     size_t m = misses.load();
//     return (h + m > 0) ? static_cast<double>(h) / static_cast<double>(h + m) : 0.0;
//   }
// };
// 
// // Classe principal para binagem ótima - otimizada e estendida
// class OptimalBinningCategorical {
// private:
//   // Entradas
//   const std::vector<std::string>& feature;
//   const std::vector<int>& target;
//   int min_bins;
//   int max_bins;
//   double bin_cutoff;
//   int max_n_prebins;
//   std::string bin_separator;
//   double convergence_threshold;
//   int max_iterations;
//   bool force_monotonic;  // Nova opção para forçar monotonicidade
//   int n_jobs;           // Número de threads para paralelização
//   double iv_regularization; // Regularização para IV (evitar overfitting)
//   
//   // Variáveis internas
//   std::vector<Bin> bins;
//   int total_pos;
//   int total_neg;
//   std::unordered_map<std::string,int> count_pos_map;
//   std::unordered_map<std::string,int> count_neg_map;
//   std::unordered_map<std::string,int> total_count_map;
//   int unique_categories;
//   bool is_increasing;
//   bool converged;
//   int iterations_run;
//   BinStats stats;  // Novas estatísticas para análise
//   
//   // Cache para cálculos de Chi-quadrado
//   std::unique_ptr<ChiSquareCache> chi_cache;
//   
//   // Constante para evitar divisão por zero
//   static constexpr double EPSILON = 1e-10;
//   
// public:
//   // Construtor melhorado com novos parâmetros
//   OptimalBinningCategorical(
//     const std::vector<std::string>& feature_,
//     const std::vector<int>& target_,
//     int min_bins_,
//     int max_bins_,
//     double bin_cutoff_,
//     int max_n_prebins_,
//     const std::string& bin_separator_,
//     double convergence_threshold_,
//     int max_iterations_,
//     bool force_monotonic_ = true,
//     int n_jobs_ = -1,
//     double iv_regularization_ = 0.0
//   ) : feature(feature_),
//   target(target_),
//   min_bins(min_bins_),
//   max_bins(max_bins_),
//   bin_cutoff(bin_cutoff_),
//   max_n_prebins(max_n_prebins_),
//   bin_separator(bin_separator_),
//   convergence_threshold(convergence_threshold_),
//   max_iterations(max_iterations_),
//   force_monotonic(force_monotonic_),
//   n_jobs(n_jobs_),
//   iv_regularization(iv_regularization_),
//   total_pos(0),
//   total_neg(0),
//   unique_categories(0),
//   is_increasing(true),
//   converged(false),
//   iterations_run(0) {
//     // Estimativa melhor para alocação inicial
//     int estimated_categories = std::min(
//       static_cast<int>(feature.size() / 2),
//       1024
//     );
//     bins.reserve(estimated_categories);
//     count_pos_map.reserve(estimated_categories);
//     count_neg_map.reserve(estimated_categories);
//     total_count_map.reserve(estimated_categories);
//     
//     // Configurar número de threads
//     if (n_jobs <= 0) {
//       n_jobs = std::thread::hardware_concurrency();
//     }
//   }
//   
//   // Função principal
//   Rcpp::List perform_binning() {
//     try {
//       validate_inputs();
//       initialize_bins();
//       
//       // Inicializar cache após conhecer o número de bins
//       chi_cache = std::make_unique<ChiSquareCache>(bins.size());
//       
//       handle_rare_categories();
//       limit_prebins();
//       ensure_min_bins();
//       merge_bins_optimized();
//       
//       if (force_monotonic) {
//         enforce_monotonicity();
//       }
//       
//       calculate_woe_iv_bins();
//       compute_bin_statistics();
//       return prepare_output();
//     } catch (const std::exception& e) {
//       Rcpp::stop("Error in optimal binning: " + std::string(e.what()));
//     }
//   }
//   
// private:
//   // Valida parâmetros e prepara contagens
//   void validate_inputs() {
//     if (feature.empty() || target.empty()) {
//       throw std::invalid_argument("Feature e target não podem ser vazios.");
//     }
//     if (feature.size() != target.size()) {
//       throw std::invalid_argument("Feature e target devem ter o mesmo tamanho.");
//     }
//     if (min_bins <= 0 || max_bins <= 0 || min_bins > max_bins) {
//       throw std::invalid_argument("Valores inválidos para min_bins ou max_bins.");
//     }
//     if (bin_cutoff <= 0 || bin_cutoff >= 1) {
//       throw std::invalid_argument("bin_cutoff deve estar entre 0 e 1.");
//     }
//     if (convergence_threshold <= 0) {
//       throw std::invalid_argument("convergence_threshold deve ser positivo.");
//     }
//     if (max_iterations <= 0) {
//       throw std::invalid_argument("max_iterations deve ser positivo.");
//     }
//     if (iv_regularization < 0) {
//       throw std::invalid_argument("iv_regularization não pode ser negativo.");
//     }
//     
//     // Otimização: percorre o dataset uma única vez para contar
//     int total_count = target.size();
//     
//     // Reservar espaço para mapas
//     const int est_unique = std::min(total_count, 10000);  // Estimativa razoável
//     std::unordered_map<std::string, std::pair<int, int>> counts;
//     counts.reserve(est_unique);
//     
//     // Percurso único pelo dataset
//     for (int i = 0; i < total_count; ++i) {
//       const int t = target[i];
//       if (t != 0 && t != 1) {
//         throw std::invalid_argument("Target deve ser binário (0 ou 1).");
//       }
//       
//       const std::string& cat = feature[i];
//       auto& count_pair = counts[cat];
//       
//       if (t == 1) {
//         count_pair.first++;
//         total_pos++;
//       } else {
//         count_pair.second++;
//         total_neg++;
//       }
//     }
//     
//     if (total_pos == 0 || total_neg == 0) {
//       throw std::invalid_argument("O target deve conter tanto 0 quanto 1.");
//     }
//     
//     // Verificar se o dataset está extremamente desbalanceado (aviso)
//     double imbalance_ratio = std::max(
//       static_cast<double>(total_pos) / static_cast<double>(total_neg),
//       static_cast<double>(total_neg) / static_cast<double>(total_pos)
//     );
//     
//     if (imbalance_ratio > 10.0) {
//       Rcpp::warning("Dataset extremamente desbalanceado (razão: %f). Considere técnicas de balanceamento.", imbalance_ratio);
//     }
//     
//     // Transfere para os mapas finais
//     for (const auto& item : counts) {
//       const std::string& cat = item.first;
//       const auto& count_pair = item.second;
//       
//       count_pos_map[cat] = count_pair.first;
//       count_neg_map[cat] = count_pair.second;
//       total_count_map[cat] = count_pair.first + count_pair.second;
//     }
//     
//     unique_categories = static_cast<int>(counts.size());
//     min_bins = std::max(2, std::min(min_bins, unique_categories));
//     max_bins = std::min(max_bins, unique_categories);
//     if (min_bins > max_bins) {
//       min_bins = max_bins;
//     }
//     
//     // Verificar se há muitas categorias únicas
//     if (unique_categories > 1000) {
//       Rcpp::warning("Muitas categorias únicas (%d). Considere pré-processar ou aumentar max_n_prebins.", unique_categories);
//     }
//     
//     bins.reserve(unique_categories);
//   }
//   
//   // Inicializa cada categoria em um bin separado
//   void initialize_bins() {
//     bins.clear();
//     bins.reserve(unique_categories);
//     
//     for (const auto& item : total_count_map) {
//       Bin bin;
//       bin.categories.push_back(item.first);
//       bin.count_pos = count_pos_map[item.first];
//       bin.count_neg = count_neg_map[item.first];
//       bin.total_count = item.second;
//       bin.update_event_rate();
//       bins.push_back(std::move(bin));
//     }
//     
//     // Pré-ordenar bins pelo WoE para facilitar monotonia
//     sort_bins_by_woe();
//   }
//   
//   // Ordenar bins pelo WoE
//   void sort_bins_by_woe() {
//     for (auto& bin : bins) {
//       double dist_pos = static_cast<double>(bin.count_pos) / static_cast<double>(total_pos);
//       double dist_neg = static_cast<double>(bin.count_neg) / static_cast<double>(total_neg);
//       dist_pos = std::max(dist_pos, EPSILON);
//       dist_neg = std::max(dist_neg, EPSILON);
//       bin.woe = std::log(dist_pos / dist_neg);
//     }
//     
//     std::sort(bins.begin(), bins.end(), 
//               [](const Bin& a, const Bin& b) { return a.woe < b.woe; });
//   }
//   
//   // Trata categorias raras (frequência baixa) - Otimizado
//   void handle_rare_categories() {
//     if (unique_categories <= 2) return;
//     
//     int total_count = total_pos + total_neg;
//     std::vector<Bin> updated_bins;
//     updated_bins.reserve(bins.size());
//     std::vector<Bin> rare_bins;
//     rare_bins.reserve(bins.size() / 4); // Estimativa
//     
//     // Separar bins raros mais eficientemente
//     for (auto& bin : bins) {
//       double freq = static_cast<double>(bin.total_count) / static_cast<double>(total_count);
//       if (freq < bin_cutoff) {
//         rare_bins.push_back(std::move(bin));
//       } else {
//         updated_bins.push_back(std::move(bin));
//       }
//     }
//     
//     // Se não houver bins raros ou não sobrariam bins não-raros suficientes, retornar
//     if (rare_bins.empty() || updated_bins.size() < 2) {
//       if (rare_bins.empty()) {
//         bins = std::move(updated_bins);
//       }
//       return;
//     }
//     
//     // Usar matriz de similaridade para mesclar bins raros de forma mais inteligente
//     for (auto& rare_bin : rare_bins) {
//       size_t best_merge_index = 0;
//       double best_metric = std::numeric_limits<double>::max();
//       
//       // Nova abordagem: considerar tanto chi-quadrado quanto similaridade de taxas de evento
//       for (size_t i = 0; i < updated_bins.size(); ++i) {
//         double chi_sq = compute_chi_square_between_bins(rare_bin, updated_bins[i]);
//         
//         // Penalizar bins com taxas de evento muito diferentes
//         double rate_diff = std::abs(rare_bin.event_rate - updated_bins[i].event_rate);
//         double combined_metric = chi_sq + (rate_diff * 100.0);  // Ponderação da diferença de taxas
//         
//         if (combined_metric < best_metric) {
//           best_metric = combined_metric;
//           best_merge_index = i;
//         }
//       }
//       
//       merge_two_bins(updated_bins[best_merge_index], rare_bin);
//     }
//     
//     bins = std::move(updated_bins);
//     chi_cache->invalidate(); // Resetar cache após reestruturação
//   }
//   
//   // Limita pré-bins ao máximo definido - Otimizado
//   void limit_prebins() {
//     if (bins.size() <= static_cast<size_t>(max_n_prebins) || !can_merge_further()) {
//       return;
//     }
//     
//     // Manter o número de bins até max_n_prebins
//     while (bins.size() > static_cast<size_t>(max_n_prebins) && can_merge_further()) {
//       // Calcular similaridade entre bins adjacentes de forma paralela
//       std::vector<double> chi_scores(bins.size() - 1, 0.0);
//       
//       if (n_jobs > 1 && bins.size() > 100) {
//         // Paralelizar cálculo usando RcppParallel
//         ChiSquareWorker worker(&bins, total_pos, total_neg, EPSILON, &chi_scores);
//         parallelFor(0, bins.size(), worker);
//       } else {
//         // Versão sequencial
//         for (size_t i = 0; i < bins.size() - 1; ++i) {
//           chi_scores[i] = compute_chi_square_between_bins(bins[i], bins[i + 1]);
//         }
//       }
//       
//       // Encontrar o par com menor chi-quadrado
//       auto min_it = std::min_element(chi_scores.begin(), chi_scores.end());
//       size_t merge_index = std::distance(chi_scores.begin(), min_it);
//       
//       // Mesclar os bins mais similares
//       merge_adjacent_bins(merge_index);
//     }
//     
//     chi_cache->invalidate(); // Resetar cache após reestruturação
//   }
//   
//   // Garante que min_bins seja respeitado, dividindo bins grandes - Otimizado
//   void ensure_min_bins() {
//     if (bins.size() >= static_cast<size_t>(min_bins)) {
//       return;
//     }
//     
//     // Nova abordagem: dividir bins de forma mais inteligente, considerando informação
//     while (bins.size() < static_cast<size_t>(min_bins)) {
//       // Critério aprimorado: dividir o bin com maior variância interna
//       size_t best_bin_index = 0;
//       double max_variance = -1.0;
//       
//       for (size_t i = 0; i < bins.size(); ++i) {
//         const Bin& current_bin = bins[i];
//         if (current_bin.categories.size() <= 1) continue;
//         
//         // Calcular variância de taxas de evento dentro do bin
//         std::vector<double> category_rates;
//         category_rates.reserve(current_bin.categories.size());
//         
//         for (const auto& cat : current_bin.categories) {
//           int cat_pos = count_pos_map.at(cat);
//           int cat_total = total_count_map.at(cat);
//           double cat_rate = cat_total > 0 ? 
//           static_cast<double>(cat_pos) / static_cast<double>(cat_total) : 0.0;
//           category_rates.push_back(cat_rate);
//         }
//         
//         // Calcular variância
//         double sum = std::accumulate(category_rates.begin(), category_rates.end(), 0.0);
//         double mean = sum / category_rates.size();
//         double sq_sum = std::inner_product(
//           category_rates.begin(), category_rates.end(), 
//           category_rates.begin(), 0.0,
//           std::plus<>(), [mean](double x, double y) { return std::pow(x - mean, 2); }
//         );
//         double variance = sq_sum / category_rates.size();
//         
//         if (variance > max_variance) {
//           max_variance = variance;
//           best_bin_index = i;
//         }
//       }
//       
//       // Se não encontramos um bin com mais de uma categoria, sair do loop
//       if (max_variance < 0 || bins[best_bin_index].categories.size() <= 1) {
//         break;
//       }
//       
//       // Dividir o bin selecionado em dois, melhor balanceado
//       Bin bin1, bin2;
//       bin1.categories.reserve(bins[best_bin_index].categories.size() / 2);
//       bin2.categories.reserve(bins[best_bin_index].categories.size() / 2);
//       
//       // Ordenar categorias por taxa de evento para melhor divisão
//       auto& categories = bins[best_bin_index].categories;
//       std::sort(categories.begin(), categories.end(),
//                 [this](const std::string& a, const std::string& b) {
//                   double rate_a = count_pos_map.at(a) / static_cast<double>(total_count_map.at(a));
//                   double rate_b = count_pos_map.at(b) / static_cast<double>(total_count_map.at(b));
//                   return rate_a < rate_b;
//                 });
//       
//       // Divisão mais balanceada
//       size_t split_index = categories.size() / 2;
//       
//       // Dividir categorias em dois bins
//       bin1.categories.insert(bin1.categories.end(), 
//                              categories.begin(),
//                              categories.begin() + split_index);
//       bin2.categories.insert(bin2.categories.end(), 
//                              categories.begin() + split_index,
//                              categories.end());
//       
//       // Calcular contagens para os novos bins
//       for (const auto& cat : bin1.categories) {
//         bin1.count_pos += count_pos_map.at(cat);
//         bin1.count_neg += count_neg_map.at(cat);
//       }
//       bin1.total_count = bin1.count_pos + bin1.count_neg;
//       bin1.update_event_rate();
//       
//       for (const auto& cat : bin2.categories) {
//         bin2.count_pos += count_pos_map.at(cat);
//         bin2.count_neg += count_neg_map.at(cat);
//       }
//       bin2.total_count = bin2.count_pos + bin2.count_neg;
//       bin2.update_event_rate();
//       
//       // Substituir o bin original pelos dois novos
//       bins[best_bin_index] = std::move(bin1);
//       bins.push_back(std::move(bin2));
//     }
//     
//     chi_cache->invalidate(); // Resetar cache após reestruturação
//   }
//   
//   // Mescla bins baseando-se no Chi-quadrado até atingir max_bins ou convergência - Otimizado
//   void merge_bins_optimized() {
//     iterations_run = 0;
//     bool keep_merging = true;
//     
//     // Estrutura para rastrear a história de convergência
//     std::vector<double> convergence_history;
//     convergence_history.reserve(max_iterations);
//     
//     while (can_merge_further() && keep_merging && iterations_run < max_iterations) {
//       // Calcular chi-quadrado para todos os pares adjacentes de forma paralela
//       std::vector<double> chi_squares(bins.size() - 1, 0.0);
//       
//       // Usar paralelização apenas quando útil
//       if (n_jobs > 1 && bins.size() > 50) {
//         ChiSquareWorker worker(&bins, total_pos, total_neg, EPSILON, &chi_squares);
//         parallelFor(0, bins.size(), worker);
//       } else {
//         // Versão sequencial com cache
//         for (size_t i = 0; i < bins.size() - 1; ++i) {
//           double cached = chi_cache->get(i, i+1);
//           if (cached >= 0) {
//             chi_squares[i] = cached;
//           } else {
//             chi_squares[i] = compute_chi_square_between_bins(bins[i], bins[i+1]);
//             chi_cache->set(i, i+1, chi_squares[i]);
//           }
//         }
//       }
//       
//       // Encontrar o menor chi-quadrado
//       auto min_it = std::min_element(chi_squares.begin(), chi_squares.end());
//       size_t min_index = std::distance(chi_squares.begin(), min_it);
//       double old_chi_square = *min_it;
//       
//       // Mesclar bins com menor chi-quadrado
//       merge_adjacent_bins(min_index);
//       
//       // Invalidar cache para os bins afetados
//       chi_cache->invalidate_row(min_index);
//       if (min_index + 1 < bins.size()) {
//         chi_cache->invalidate_row(min_index + 1);
//       }
//       
//       if (!can_merge_further()) {
//         break;
//       }
//       
//       // Calcular novo chi-quadrado mínimo
//       std::vector<double> new_chi_squares;
//       new_chi_squares.reserve(2);
//       
//       if (min_index > 0) {
//         double chi = compute_chi_square_between_bins(bins[min_index-1], bins[min_index]);
//         chi_cache->set(min_index-1, min_index, chi);
//         new_chi_squares.push_back(chi);
//       }
//       
//       if (min_index < bins.size() - 1) {
//         double chi = compute_chi_square_between_bins(bins[min_index], bins[min_index+1]);
//         chi_cache->set(min_index, min_index+1, chi);
//         new_chi_squares.push_back(chi);
//       }
//       
//       // Verificar convergência
//       double new_min_chi = !new_chi_squares.empty() ? 
//       *std::min_element(new_chi_squares.begin(), new_chi_squares.end()) : 0.0;
//       
//       double delta = std::fabs(new_min_chi - old_chi_square);
//       convergence_history.push_back(delta);
//       
//       // Critério de convergência melhorado: média das últimas 3 iterações
//       if (convergence_history.size() >= 3) {
//         double avg_delta = (convergence_history.end()[-1] + 
//                             convergence_history.end()[-2] + 
//                             convergence_history.end()[-3]) / 3.0;
//         keep_merging = avg_delta > convergence_threshold;
//       } else {
//         keep_merging = delta > convergence_threshold;
//       }
//       
//       iterations_run++;
//     }
//     
//     converged = (iterations_run < max_iterations) || !can_merge_further();
//   }
//   
//   // Método melhorado para determinar monotonicidade com regressão robusta
//   void determine_monotonicity_robust() {
//     if (bins.size() < 3) {
//       is_increasing = true;
//       return;
//     }
//     
//     // Usar Theil-Sen estimator para mais robustez a outliers
//     std::vector<double> slopes;
//     int n = bins.size();
//     slopes.reserve((n * (n - 1)) / 2);
//     
//     for (int i = 0; i < n; ++i) {
//       for (int j = i + 1; j < n; ++j) {
//         double slope = (bins[j].woe - bins[i].woe) / (j - i);
//         slopes.push_back(slope);
//       }
//     }
//     
//     // Ordenar slopes e pegar a mediana
//     std::sort(slopes.begin(), slopes.end());
//     double median_slope = slopes.size() % 2 == 0 ?
//     (slopes[slopes.size()/2 - 1] + slopes[slopes.size()/2]) / 2.0 :
//       slopes[slopes.size()/2];
//     
//     is_increasing = (median_slope >= 0);
//     stats.direction = is_increasing ? "increasing" : "decreasing";
//   }
//   
//   // Impõe monotonicidade do WoE - Otimizado
//   void enforce_monotonicity() {
//     if (unique_categories <= 2) {
//       stats.monotonic = true;
//       return;
//     }
//     
//     // Calcular WoE para cada bin para verificação de monotonicidade
//     for (auto& bin : bins) {
//       double dist_pos = static_cast<double>(bin.count_pos) / static_cast<double>(total_pos);
//       double dist_neg = static_cast<double>(bin.count_neg) / static_cast<double>(total_neg);
//       dist_pos = std::max(dist_pos, EPSILON);
//       dist_neg = std::max(dist_neg, EPSILON);
//       bin.woe = std::log(dist_pos / dist_neg);
//     }
//     
//     determine_monotonicity_robust();
//     
//     bool monotonic = false;
//     int iterations = 0;
//     const int max_mono_iter = 100; // Limite para evitar loops infinitos
//     
//     while (!monotonic && can_merge_further() && iterations < max_mono_iter) {
//       monotonic = true;
//       
//       // Armazenar pares que violam monotonia
//       std::vector<std::pair<size_t, double>> violations;
//       violations.reserve(bins.size());
//       
//       for (size_t i = 0; i < bins.size() - 1; ++i) {
//         bool violation = is_increasing ? 
//         (bins[i].woe > bins[i + 1].woe + EPSILON) : 
//         (bins[i].woe < bins[i + 1].woe - EPSILON);
//         
//         if (violation) {
//           // Calcular magnitude da violação
//           double magnitude = std::fabs(bins[i].woe - bins[i + 1].woe);
//           violations.emplace_back(i, magnitude);
//           monotonic = false;
//         }
//       }
//       
//       if (!violations.empty()) {
//         // Ordenar por magnitude de violação (maior primeiro)
//         std::sort(violations.begin(), violations.end(), 
//                   [](const auto& a, const auto& b) { return a.second > b.second; });
//         
//         // Mesclar o par com maior violação
//         merge_adjacent_bins(violations[0].first);
//         
//         // Recalcular WoE após mesclagem
//         double dist_pos = static_cast<double>(bins[violations[0].first].count_pos) / static_cast<double>(total_pos);
//         double dist_neg = static_cast<double>(bins[violations[0].first].count_neg) / static_cast<double>(total_neg);
//         dist_pos = std::max(dist_pos, EPSILON);
//         dist_neg = std::max(dist_neg, EPSILON);
//         bins[violations[0].first].woe = std::log(dist_pos / dist_neg);
//       }
//       
//       iterations++;
//     }
//     
//     stats.monotonic = monotonic;
//   }
//   
//   // Calcula WoE e IV para cada bin com regularização opcional
//   void calculate_woe_iv_bins() {
//     double total_iv = 0.0;
//     
//     for (auto& bin : bins) {
//       double dist_pos = static_cast<double>(bin.count_pos) / static_cast<double>(total_pos);
//       double dist_neg = static_cast<double>(bin.count_neg) / static_cast<double>(total_neg);
//       
//       // Adicionar regularização (similar a Laplace smoothing)
//       if (iv_regularization > 0) {
//         dist_pos = (bin.count_pos + iv_regularization) / (total_pos + iv_regularization * bins.size());
//         dist_neg = (bin.count_neg + iv_regularization) / (total_neg + iv_regularization * bins.size());
//       }
//       
//       dist_pos = std::max(dist_pos, EPSILON);
//       dist_neg = std::max(dist_neg, EPSILON);
//       bin.woe = std::log(dist_pos / dist_neg);
//       bin.iv = (dist_pos - dist_neg) * bin.woe;
//       total_iv += bin.iv;
//       
//       // Atualizar taxa de eventos
//       bin.update_event_rate();
//       
//       // Atualizar estatísticas min/max
//       stats.min_woe = std::min(stats.min_woe, bin.woe);
//       stats.max_woe = std::max(stats.max_woe, bin.woe);
//     }
//     
//     stats.total_iv = total_iv;
//   }
//   
//   // Calcula estatísticas adicionais sobre os bins
//   void compute_bin_statistics() {
//     int total_size = std::accumulate(bins.begin(), bins.end(), 0,
//                                      [](int sum, const Bin& bin) { return sum + bin.total_count; });
//     
//     stats.avg_bin_size = static_cast<double>(total_size) / bins.size();
//     
//     // Encontrar o menor e maior bin
//     stats.largest_bin = 0;
//     stats.smallest_bin = std::numeric_limits<int>::max();
//     
//     for (const auto& bin : bins) {
//       if (bin.total_count > stats.largest_bin) {
//         stats.largest_bin = bin.total_count;
//       }
//       if (bin.total_count < stats.smallest_bin) {
//         stats.smallest_bin = bin.total_count;
//       }
//     }
//   }
//   
//   // Prepara a saída R
//   Rcpp::List prepare_output() const {
//     std::vector<std::string> bin_names;
//     bin_names.reserve(bins.size());
//     std::vector<double> bin_woe, bin_iv, bin_event_rate;
//     bin_woe.reserve(bins.size());
//     bin_iv.reserve(bins.size());
//     bin_event_rate.reserve(bins.size());
//     std::vector<int> bin_count, bin_count_pos, bin_count_neg;
//     bin_count.reserve(bins.size());
//     bin_count_pos.reserve(bins.size());
//     bin_count_neg.reserve(bins.size());
//     
//     for (const auto& bin : bins) {
//       bin_names.push_back(join_categories(bin.categories));
//       bin_woe.push_back(bin.woe);
//       bin_iv.push_back(bin.iv);
//       bin_event_rate.push_back(bin.event_rate);
//       bin_count.push_back(bin.total_count);
//       bin_count_pos.push_back(bin.count_pos);
//       bin_count_neg.push_back(bin.count_neg);
//     }
//     
//     // Criar vetor de IDs com o mesmo tamanho de bin_names
//     Rcpp::NumericVector ids(bin_names.size());
//     for (int i = 0; i < static_cast<int>(bin_names.size()); i++) {
//       ids[i] = i + 1;  // Começa em 1 até size(bin_names)
//     }
//     
//     // Estatísticas do cache
//     auto cache_stats = chi_cache->get_stats();
//     
//     return Rcpp::List::create(
//       Rcpp::Named("id") = ids,
//       Rcpp::Named("bin") = bin_names,
//       Rcpp::Named("woe") = bin_woe,
//       Rcpp::Named("iv") = bin_iv,
//       Rcpp::Named("event_rate") = bin_event_rate,
//       Rcpp::Named("count") = bin_count,
//       Rcpp::Named("count_pos") = bin_count_pos,
//       Rcpp::Named("count_neg") = bin_count_neg,
//       Rcpp::Named("converged") = converged,
//       Rcpp::Named("iterations") = iterations_run,
//       Rcpp::Named("stats") = Rcpp::List::create(
//         Rcpp::Named("total_iv") = stats.total_iv,
//         Rcpp::Named("monotonic") = stats.monotonic,
//         Rcpp::Named("direction") = stats.direction,
//         Rcpp::Named("min_woe") = stats.min_woe,
//         Rcpp::Named("max_woe") = stats.max_woe,
//         Rcpp::Named("avg_bin_size") = stats.avg_bin_size,
//         Rcpp::Named("largest_bin") = stats.largest_bin,
//         Rcpp::Named("smallest_bin") = stats.smallest_bin,
//         Rcpp::Named("cache_hits") = static_cast<int>(cache_stats.first),
//         Rcpp::Named("cache_misses") = static_cast<int>(cache_stats.second),
//         Rcpp::Named("cache_hit_rate") = chi_cache->hit_rate()
//       )
//     );
//   }
//   
//   // Calcula Chi-quadrado entre dois bins - Otimizado
//   inline double compute_chi_square_between_bins(const Bin& bin1, const Bin& bin2) const {
//     int total = bin1.total_count + bin2.total_count;
//     if (total == 0) return 0.0;
//     
//     int total_pos_bins = bin1.count_pos + bin2.count_pos;
//     int total_neg_bins = bin1.count_neg + bin2.count_neg;
//     
//     // Cálculo rápido para casos especiais
//     if (total_pos_bins == 0 || total_neg_bins == 0) return 0.0;
//     
//     // Cálculo otimizado do chi-quadrado
//     double expected1_pos = static_cast<double>(bin1.total_count * total_pos_bins) / static_cast<double>(total);
//     double expected1_neg = static_cast<double>(bin1.total_count * total_neg_bins) / static_cast<double>(total);
//     
//     // Prevenir divisão por zero
//     expected1_pos = std::max(expected1_pos, EPSILON);
//     expected1_neg = std::max(expected1_neg, EPSILON);
//     
//     // Cálculo do chi-quadrado usando apenas os valores do primeiro bin
//     // (o segundo bin é determinado pela diferença, reduzindo cálculos)
//     double chi_square = 0.0;
//     chi_square += std::pow(bin1.count_pos - expected1_pos, 2.0) / expected1_pos;
//     chi_square += std::pow(bin1.count_neg - expected1_neg, 2.0) / expected1_neg;
//     
//     double expected2_pos = static_cast<double>(bin2.total_count * total_pos_bins) / static_cast<double>(total);
//     double expected2_neg = static_cast<double>(bin2.total_count * total_neg_bins) / static_cast<double>(total);
//     
//     expected2_pos = std::max(expected2_pos, EPSILON);
//     expected2_neg = std::max(expected2_neg, EPSILON);
//     
//     chi_square += std::pow(bin2.count_pos - expected2_pos, 2.0) / expected2_pos;
//     chi_square += std::pow(bin2.count_neg - expected2_neg, 2.0) / expected2_neg;
//     
//     return chi_square;
//   }
//   
//   // Mescla bins adjacentes no índice dado
//   inline void merge_adjacent_bins(size_t index) {
//     if (index >= bins.size() - 1) return;
//     merge_two_bins(bins[index], bins[index + 1]);
//     bins.erase(bins.begin() + index + 1);
//   }
//   
//   // Mescla dois bins - Otimizado
//   inline void merge_two_bins(Bin& bin1, const Bin& bin2) {
//     // Reservar espaço para as categorias antes de inserir
//     bin1.categories.reserve(bin1.categories.size() + bin2.categories.size());
//     bin1.categories.insert(bin1.categories.end(), bin2.categories.begin(), bin2.categories.end());
//     bin1.count_pos += bin2.count_pos;
//     bin1.count_neg += bin2.count_neg;
//     bin1.total_count += bin2.total_count;
//     bin1.update_event_rate();
//   }
//   
//   // Junta nomes de categorias - Otimizado
//   inline std::string join_categories(const std::vector<std::string>& categories) const {
//     if (categories.empty()) return "";
//     if (categories.size() == 1) return categories[0];
//     
//     // Estimar o tamanho da string final para pré-alocação
//     size_t total_length = 0;
//     for (const auto& cat : categories) {
//       total_length += cat.length();
//     }
//     total_length += bin_separator.length() * (categories.size() - 1);
//     
//     // Construir a string de forma eficiente
//     std::string result;
//     result.reserve(total_length);
//     
//     result = categories[0];
//     for (size_t i = 1; i < categories.size(); ++i) {
//       result += bin_separator;
//       result += categories[i];
//     }
//     
//     return result;
//   }
//   
//   // Verifica se pode mesclar mais
//   inline bool can_merge_further() const {
//     return bins.size() > static_cast<size_t>(std::max(min_bins, 2));
//   }
//   
//   // Calcula WoE para uma categoria
//   inline double compute_woe(int c_pos, int c_neg) const {
//     double dist_pos = static_cast<double>(c_pos) / static_cast<double>(total_pos);
//     double dist_neg = static_cast<double>(c_neg) / static_cast<double>(total_neg);
//     dist_pos = std::max(dist_pos, EPSILON);
//     dist_neg = std::max(dist_neg, EPSILON);
//     return std::log(dist_pos / dist_neg);
//   }
// };
// 
// 
// //' @title Optimal Binning for Categorical Variables by Chi-Merge
// //'
// //' @description
// //' Implementa binagem ótima para variáveis categóricas usando o algoritmo Chi-Merge,
// //' calculando Weight of Evidence (WoE) e Information Value (IV) para os bins resultantes.
// //' Versão melhorada com paralelização, estatísticas adicionais e monotonicidade robusta.
// //'
// //' @param target Vetor de inteiros com valores binários do target (0 ou 1).
// //' @param feature Vetor de caracteres com valores categóricos do atributo.
// //' @param min_bins Número mínimo de bins (padrão: 3).
// //' @param max_bins Número máximo de bins (padrão: 5).
// //' @param bin_cutoff Frequência mínima para um bin separado (padrão: 0.05).
// //' @param max_n_prebins Número máximo de pré-bins antes da mesclagem (padrão: 20).
// //' @param bin_separator Separador para concatenar nomes de categoria em bins (padrão: "%;%").
// //' @param convergence_threshold Limiar para convergência na diferença Chi-square (padrão: 1e-6).
// //' @param max_iterations Número máximo de iterações para mesclagem de bins (padrão: 1000).
// //' @param force_monotonic Forçar monotonicidade dos bins por WoE (padrão: TRUE).
// //' @param n_jobs Número de threads para paralelização. -1 para auto (padrão: -1).
// //' @param iv_regularization Parâmetro de regularização para IV, similar ao smoothing de Laplace (padrão: 0).
// //'
// //' @return Uma lista contendo:
// //' \itemize{
// //'   \item id: Vetor de IDs numéricos para cada bin.
// //'   \item bin: Vetor de nomes de bin (categorias concatenadas).
// //'   \item woe: Vetor de valores Weight of Evidence para cada bin.
// //'   \item iv: Vetor de Information Value para cada bin.
// //'   \item event_rate: Taxa de eventos (count_pos/count) para cada bin.
// //'   \item count: Vetor de contagens totais para cada bin.
// //'   \item count_pos: Vetor de contagens da classe positiva para cada bin.
// //'   \item count_neg: Vetor de contagens da classe negativa para cada bin.
// //'   \item converged: Booleano indicando se o algoritmo convergiu.
// //'   \item iterations: Número de iterações executadas.
// //'   \item stats: Estatísticas adicionais sobre os bins e o processo de binagem.
// //' }
// //'
// //' @details
// //' O algoritmo usa estatísticas Chi-quadrado para mesclar bins adjacentes:
// //'
// //' \deqn{\chi^2 = \sum_{i=1}^{2}\sum_{j=1}^{2} \frac{(O_{ij} - E_{ij})^2}{E_{ij}}}
// //'
// //' onde \eqn{O_{ij}} é a frequência observada e \eqn{E_{ij}} é a frequência esperada
// //' para o bin i e classe j.
// //'
// //' Weight of Evidence (WoE) para cada bin:
// //'
// //' \deqn{WoE = \ln(\frac{P(X|Y=1)}{P(X|Y=0)})}
// //'
// //' Information Value (IV) para cada bin:
// //'
// //' \deqn{IV = (P(X|Y=1) - P(X|Y=0)) * WoE}
// //'
// //' O algoritmo inicializa bins para cada categoria, mescla categorias raras com base em
// //' bin_cutoff, e então iterativamente mescla bins com o menor chi-quadrado
// //' até que max_bins seja atingido ou não seja possível fazer mais mesclagens. Determina a
// //' direção da monotonicidade com base na tendência inicial e a impõe, permitindo
// //' desvios se as restrições de min_bins forem acionadas.
// //'
// //' Esta implementação melhorada inclui:
// //' - Paralelização para datasets grandes
// //' - Estratégia de cache otimizada
// //' - Método robusto para determinar monotonicidade (estimador Theil-Sen)
// //' - Regularização opcional para cálculo de IV
// //' - Métricas adicionais como taxa de eventos
// //' - Estatísticas detalhadas sobre os bins
// //' - Suporte para multithreading
// //'
// //' @examples
// //' \dontrun{
// //' # Dados de exemplo
// //' target <- c(1, 0, 1, 1, 0, 1, 0, 0, 1, 1)
// //' feature <- c("A", "B", "A", "C", "B", "D", "C", "A", "D", "B")
// //'
// //' # Executar binagem ótima com configurações padrão
// //' result <- optimal_binning_categorical_cm(target, feature)
// //'
// //' # Executar com opções personalizadas
// //' result <- optimal_binning_categorical_cm(
// //'   target, feature, 
// //'   min_bins = 2, 
// //'   max_bins = 4,
// //'   force_monotonic = TRUE,
// //'   n_jobs = 2
// //' )
// //'
// //' # Ver resultados
// //' print(result)
// //'
// //' # Acessar estatísticas adicionais
// //' print(result$stats)
// //' }
// //'
// //' @export
// // [[Rcpp::export]]
// Rcpp::List optimal_binning_categorical_cm(
//    Rcpp::IntegerVector target,
//    Rcpp::CharacterVector feature,
//    int min_bins = 3,
//    int max_bins = 5,
//    double bin_cutoff = 0.05,
//    int max_n_prebins = 20,
//    std::string bin_separator = "%;%",
//    double convergence_threshold = 1e-6,
//    int max_iterations = 1000,
//    bool force_monotonic = true,
//    int n_jobs = -1,
//    double iv_regularization = 0.0
// ) {
//  // Converte vetores R para C++
//  std::vector<std::string> feature_vec(feature.size());
//  std::vector<int> target_vec(target.size());
//  
//  for (size_t i = 0; i < feature.size(); ++i) {
//    if (feature[i] == NA_STRING) {
//      feature_vec[i] = "NA";
//    } else {
//      feature_vec[i] = Rcpp::as<std::string>(feature[i]);
//    }
//  }
//  
//  for (size_t i = 0; i < target.size(); ++i) {
//    if (IntegerVector::is_na(target[i])) {
//      Rcpp::stop("Target não pode conter valores ausentes.");
//    } else {
//      target_vec[i] = target[i];
//    }
//  }
//  
//  // Cria objeto e executa binagem
//  OptimalBinningCategorical obcm(
//      feature_vec, target_vec, min_bins, max_bins, bin_cutoff, max_n_prebins,
//      bin_separator, convergence_threshold, max_iterations, force_monotonic,
//      n_jobs, iv_regularization
//  );
//  
//  return obcm.perform_binning();
// }
