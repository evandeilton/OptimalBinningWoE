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

using namespace Rcpp;

// -----------------------------------------------------------------------------
// Bin Structure
// -----------------------------------------------------------------------------
/**
 * Structure representing a bin of categorical values
 */
struct Bin {
  std::vector<std::string> categories; // Categories in this bin
  int count_pos;                       // Count of positive class
  int count_neg;                       // Count of negative class
  double woe;                          // Weight of Evidence
  double iv;                           // Information Value
  int total_count;                     // Total observations
  
  // Constructor
  Bin() : count_pos(0), count_neg(0), woe(0.0), iv(0.0), total_count(0) {
    // Initial reservation can be done dynamically if needed
  }
  
  // Helper function to get total count
  int get_total_count() const { return count_pos + count_neg; }
};

// -----------------------------------------------------------------------------
// ChiSquareCache Class
// -----------------------------------------------------------------------------
/**
 * Efficient cache for chi-square calculations to avoid redundant computations
 */
class ChiSquareCache {
private:
  // Use a flat array for faster access compared to unordered_map
  std::vector<double> cache;
  size_t num_bins = 0; // Initialize with 0
  
  // Compute index in the triangular matrix (only storing upper triangle for adjacent pairs)
  // Cache only stores chi-square between adjacent bins (i, i+1)
  inline size_t compute_index(size_t i) const {
    // We only cache adjacent pairs (i, i+1)
    return i;
  }
  
public:
  /**
   * Initialize or resize the cache with a given number of bins
   * @param n Number of bins
   */
  void resize(size_t n) {
    num_bins = n;
    // Cache size is n-1 for adjacent pairs
    size_t cache_size = (n > 0) ? (n - 1) : 0;
    cache.assign(cache_size, -1.0); // Initialize with -1 to indicate uncached
  }
  
  /**
   * Get cached chi-square value for adjacent bins i and i+1
   * @param i First bin index
   * @return Chi-square value or -1 if not cached or invalid index
   */
  double get(size_t i) {
    if (i + 1 >= num_bins) return -1.0; // Check bounds for pair (i, i+1)
    
    size_t idx = compute_index(i);
    return (idx < cache.size()) ? cache[idx] : -1.0;
  }
  
  /**
   * Get cached chi-square value for adjacent bins i and i+1 (const version)
   * @param i First bin index
   * @return Chi-square value or -1 if not cached or invalid index
   */
  double get(size_t i) const {
    if (i + 1 >= num_bins) return -1.0;
    
    size_t idx = compute_index(i);
    return (idx < cache.size()) ? cache[idx] : -1.0;
  }
  
  /**
   * Store chi-square value for adjacent bins i and i+1
   * @param i First bin index
   * @param value Chi-square value
   */
  void set(size_t i, double value) {
    if (i + 1 >= num_bins) return; // Check bounds for pair (i, i+1)
    
    size_t idx = compute_index(i);
    if (idx < cache.size()) {
      cache[idx] = value;
    }
  }
  
  /**
   * Invalidate cache entries related to merges at index `merge_index`
   * This means invalidating pairs (merge_index-1, merge_index) and (merge_index, merge_index+1)
   * after the merge happens and bin vector/cache are resized.
   * @param merge_index Index where the merge occurred (index of the first merged bin)
   */
  void invalidate_after_merge(size_t merge_index) {
    // After merging bins[merge_index] and bins[merge_index+1], the new merged
    // bin is at merge_index. We need to invalidate cache entries for the
    // new pairs involving this merged bin: (merge_index-1, merge_index) and
    // (merge_index, merge_index+1) in the *new* cache structure.
    
    // Invalidate cache for the pair before the merge point
    if (merge_index > 0) {
      set(merge_index - 1, -1.0);
    }
    // Invalidate cache for the pair at the merge point
    if (merge_index < num_bins -1) { // Check if there's a pair starting at merge_index
      set(merge_index, -1.0);
    }
  }
  
  /**
   * Completely invalidate the cache
   */
  void invalidate() {
    std::fill(cache.begin(), cache.end(), -1.0);
  }
};

// -----------------------------------------------------------------------------
// OptimalBinningCategorical Class
// -----------------------------------------------------------------------------
/**
 * Chi-Merge and Chi2 optimal binning implementation for categorical variables (V4)
 * Based on Kerber (1992) and Liu & Setiono (1995)
 */
class OptimalBinningCategorical {
private:
  // Input parameters (const where appropriate)
  const std::vector<std::string>& feature;
  const std::vector<int>& target;
  int min_bins;
  int max_bins;
  const double bin_cutoff;
  const int max_n_prebins;
  const std::string bin_separator;
  const double convergence_threshold;
  const int max_iterations;
  double chi_merge_threshold; // Can be modified by Chi2 algorithm
  const bool use_chi2_algorithm;
  
  // Internal state
  std::vector<Bin> bins;
  int total_pos = 0;
  int total_neg = 0;
  std::unordered_map<std::string, int> count_pos_map;
  std::unordered_map<std::string, int> count_neg_map;
  std::unordered_map<std::string, int> total_count_map;
  int unique_categories = 0;
  bool is_increasing = true; // Default monotonicity
  bool converged = false;
  int iterations_run = 0;
  
  // Chi-square statistics cache
  std::unique_ptr<ChiSquareCache> chi_cache;
  
  // Constants
  static constexpr double EPSILON = 1e-10;
  
  // Chi-square critical values for common significance levels (DF=1)
  const std::unordered_map<double, double> CHI_SQUARE_CRITICAL_VALUES = {
    {0.995, 0.0000393}, {0.99, 0.000157}, {0.975, 0.000982}, // Corrected values
    {0.95, 0.00393},   {0.9, 0.0158},    {0.5, 0.455},
    {0.1, 2.71},       {0.05, 3.84},     {0.025, 5.02},
    {0.01, 6.63},      {0.005, 7.88},    {0.001, 10.8}
  };
  
  
public:
  /**
   * Constructor for OptimalBinningCategorical
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
  // Initialize chi_cache with default constructor (no parameters)
  chi_cache(std::make_unique<ChiSquareCache>())
  {
    // Initialize chi_cache with size 0
    chi_cache->resize(0);
    
    // Memory reservations
    int estimated_categories = std::min(
      static_cast<int>(feature.size() / 4), // Heuristic
      2048                                  // Cap
    );
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
      // 1. Validate and preprocess
      validate_inputs();
      // 2. Initialize bins (1 category per bin)
      initialize_bins();
      
      // Resize cache after knowing the initial number of bins
      chi_cache->resize(bins.size());
      
      // 3. Apply algorithm (ChiMerge or Chi2)
      if (use_chi2_algorithm) {
        perform_chi2_binning();
      } else {
        // ChiMerge Steps:
        // 3a. Group rare categories
        handle_rare_categories();
        // 3b. Limit number of pre-bins (optional, may be redundant with ChiMerge)
        limit_prebins(); // This function also uses ChiMerge internally
        // 3c. Ensure min_bins (may need to split)
        ensure_min_bins(); // This function can split bins
        // 3d. Merge using ChiMerge until max_bins or threshold
        merge_bins_using_chimerge();
        // 3e. Force monotonicity if necessary
        enforce_monotonicity();
      }
      
      // 4. Calculate final metrics (WoE, IV)
      calculate_woe_iv_bins();
      // 5. Prepare and return output
      return prepare_output();
      
    } catch (const std::exception& e) {
      Rcpp::stop("Error in optimal binning v4: " + std::string(e.what()));
    } catch (...) {
      Rcpp::stop("Unknown error occurred during optimal binning v4.");
    }
  }
  
  private: // Private helper methods
    
    /**
     * Validate input parameters and preprocess data
     */
    void validate_inputs() {
      if (feature.empty() || target.empty()) {
        throw std::invalid_argument("Feature and target cannot be empty.");
      }
      if (feature.size() != target.size()) {
        throw std::invalid_argument("Feature and target must have the same length.");
      }
      if (min_bins <= 0 || max_bins <= 0 || min_bins > max_bins) {
        throw std::invalid_argument("Invalid values for min_bins or max_bins (must be > 0 and min_bins <= max_bins).");
      }
      if (bin_cutoff <= 0 || bin_cutoff >= 1) {
        throw std::invalid_argument("bin_cutoff must be between 0 and 1 (exclusive).");
      }
      if (max_n_prebins < 2) {
        throw std::invalid_argument("max_n_prebins must be at least 2.");
      }
      if (convergence_threshold <= 0) {
        throw std::invalid_argument("convergence_threshold must be positive.");
      }
      if (max_iterations <= 0) {
        throw std::invalid_argument("max_iterations must be positive.");
      }
      if (chi_merge_threshold <= 0 || chi_merge_threshold >= 1) {
        throw std::invalid_argument("chi_merge_threshold (significance level) must be between 0 and 1.");
      }
      
      // Efficiently process data in a single pass
      int total_count = target.size();
      std::unordered_map<std::string, std::pair<int, int>> counts;
      
      total_pos = 0; // Reset counts
      total_neg = 0;
      
      for (int i = 0; i < total_count; ++i) {
        const int t = target[i];
        if (t != 0 && t != 1) {
          throw std::invalid_argument("Target must be binary (0 or 1).");
        }
        
        const std::string& cat = feature[i]; // Assume "NA" handling in wrapper
        auto& count_pair = counts[cat]; // Creates if not exists
        
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
      
      // Clear previous maps and transfer counts
      count_pos_map.clear();
      count_neg_map.clear();
      total_count_map.clear();
      for (const auto& item : counts) {
        const std::string& cat = item.first;
        const auto& count_pair = item.second;
        count_pos_map[cat] = count_pair.first;
        count_neg_map[cat] = count_pair.second;
        total_count_map[cat] = count_pair.first + count_pair.second;
      }
      
      unique_categories = static_cast<int>(counts.size());
      
      if (unique_categories < 2) {
        throw std::invalid_argument("Feature must have at least 2 unique categories.");
      }
      
      // Adjust bin constraints based on available unique categories
      // Ensure constraints are valid *after* potential merging/splitting
      min_bins = std::max(2, std::min(min_bins, unique_categories));
      max_bins = std::min(max_bins, unique_categories);
      if (min_bins > max_bins) {
        min_bins = max_bins; // Ensure min <= max
      }
      // max_n_prebins doesn't need adjustment here, it's handled in limit_prebins
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
        bin.total_count = item.second; // Use value directly
        bins.push_back(std::move(bin));
      }
      
      // Sort bins by WoE initially (important for ChiMerge/monotonicity)
      sort_bins_by_woe();
    }
    
    /**
     * Sort bins by Weight of Evidence (ascending)
     */
    void sort_bins_by_woe() {
      // First compute WoE for all current bins
      for (auto& bin : bins) {
        bin.woe = compute_woe(bin.count_pos, bin.count_neg);
      }
      // Then sort
      std::sort(bins.begin(), bins.end(),
                [](const Bin& a, const Bin& b) { return a.woe < b.woe; });
    }
    
    /**
     * Handle rare categories by merging them with the most similar adjacent bin (lowest chi-square).
     * This step is often done *before* the main ChiMerge loop.
     */
    void handle_rare_categories() {
      if (bins.size() <= 2) return; // Nothing to merge
      
      int total_count = total_pos + total_neg;
      bool merged_in_pass;
      
      do {
        merged_in_pass = false;
        std::vector<Bin> next_bins;
        next_bins.reserve(bins.size());
        bool merged_current = false;
        
        for (size_t i = 0; i < bins.size(); ++i) {
          if (merged_current) {
            merged_current = false; // Skip the next bin as it was merged into the previous
            continue;
          }
          
          double freq = static_cast<double>(bins[i].get_total_count()) / static_cast<double>(total_count);
          
          if (freq < bin_cutoff && bins.size() > static_cast<size_t>(min_bins)) { // Check if rare and if we can merge
            // Find best adjacent bin to merge with (lowest chi-square)
            double chi_left = (i > 0) ? compute_chi_square_between_bins(bins[i-1], bins[i]) : std::numeric_limits<double>::max();
            double chi_right = (i < bins.size() - 1) ? compute_chi_square_between_bins(bins[i], bins[i+1]) : std::numeric_limits<double>::max();
            
            if (chi_left <= chi_right && i > 0) {
              // Merge with left bin (which is already in next_bins)
              if (!next_bins.empty()) {
                merge_two_bins(next_bins.back(), bins[i]);
                merged_in_pass = true;
                // Don't set merged_current=true here, as we modified the *previous* bin
              } else {
                next_bins.push_back(std::move(bins[i])); // Cannot merge left
              }
            } else if (chi_right < chi_left && i < bins.size() - 1) {
              // Merge with right bin
              Bin merged_bin = bins[i]; // Copy current
              merge_two_bins(merged_bin, bins[i+1]); // Merge right into copy
              next_bins.push_back(std::move(merged_bin));
              merged_current = true; // Skip the next bin in the original vector
              merged_in_pass = true;
            } else {
              // Cannot merge (e.g., first/last rare bin or chi-square is max for both)
              next_bins.push_back(std::move(bins[i]));
            }
          } else {
            // Not rare or cannot merge further, keep the bin
            next_bins.push_back(std::move(bins[i]));
          }
        }
        bins = std::move(next_bins);
      } while (merged_in_pass && bins.size() > static_cast<size_t>(min_bins)); // Repeat if merges happened and > min_bins
      
      
      // Reset cache after bin structure potentially changes significantly
      if (merged_in_pass) {
        chi_cache->invalidate();
        chi_cache->resize(bins.size());
      }
    }
    
    /**
     * Limit the number of bins to max_n_prebins using ChiMerge logic.
     */
    void limit_prebins() {
      if (bins.size() <= static_cast<size_t>(max_n_prebins) || !can_merge_further()) {
        return;
      }
      
      // Use a slightly relaxed threshold for pre-binning, or the main one? Use main one.
      double critical_value = get_chi_square_critical_value();
      int prebin_iterations = 0; // Use separate counter if needed
      
      // Initial population of cache for adjacent pairs
      populate_chi_cache();
      
      while (bins.size() > static_cast<size_t>(max_n_prebins) &&
             can_merge_further() &&
             prebin_iterations < max_iterations) // Prevent infinite loops
      {
        
        std::pair<double, size_t> min_chi_pair = find_min_chi_square_pair();
        double min_chi = min_chi_pair.first;
        size_t min_index = min_chi_pair.second;
        
        // Even if below max_n_prebins, stop if statistically significant difference found
        if (min_chi > critical_value) {
          break;
        }
        
        // Merge the pair with the lowest chi-square
        merge_adjacent_bins(min_index);
        
        // Update the cache around the merge point
        update_chi_cache_after_merge(min_index);
        prebin_iterations++;
      }
    }
    
    
    /**
     * Ensure minimum number of bins by splitting bins with the most categories.
     */
    void ensure_min_bins() {
      if (bins.size() >= static_cast<size_t>(min_bins)) {
        return;
      }
      Rcpp::Rcout << "Info: Current bins (" << bins.size() << ") < min_bins (" << min_bins
                  << "). Attempting to split bins." << std::endl;
      
      
      while (bins.size() < static_cast<size_t>(min_bins)) {
        // Find bin with most categories that can be split
        int best_split_idx = -1;
        size_t max_cats = 1; // Must have > 1 category to split
        
        for(size_t i = 0; i < bins.size(); ++i) {
          if (bins[i].categories.size() > max_cats) {
            max_cats = bins[i].categories.size();
            best_split_idx = static_cast<int>(i);
          }
        }
        
        // If no bin can be split (all have 1 category), stop
        if (best_split_idx == -1) {
          Rcpp::Rcout << "Warning: Cannot split further to reach min_bins. Final number of bins: " << bins.size() << std::endl;
          break;
        }
        
        // --- Perform the split ---
        // 1. Store the bin to be split
        Bin bin_to_split = std::move(bins[best_split_idx]);
        
        // 2. Remove original bin from the vector
        bins.erase(bins.begin() + best_split_idx);
        
        // 3. Create and add the two new bins resulting from the split
        split_bin(bin_to_split); // Adds new bins to the end
        
        // 4. Resort bins by WoE as splitting changes order
        sort_bins_by_woe();
        
        // 5. Invalidate and resize cache as structure changed significantly
        chi_cache->invalidate();
        chi_cache->resize(bins.size());
      }
    }
    
    /**
     * Split a bin into two approximately equal parts based on WoE order.
     * Adds the two new bins to the *end* of the main `bins` vector.
     * @param bin_to_split The bin object containing categories to be split.
     */
    void split_bin(const Bin& bin_to_split) {
      const size_t n_cats = bin_to_split.categories.size();
      if (n_cats <= 1) return; // Cannot split
      
      // Create copies of categories to sort
      std::vector<std::string> sorted_categories = bin_to_split.categories;
      
      // Sort categories by individual WoE
      std::sort(sorted_categories.begin(), sorted_categories.end(),
                [this](const std::string& a, const std::string& b) {
                  // Use .at() for safety, keys must exist
                  double woe_a = compute_woe(count_pos_map.at(a), count_neg_map.at(a));
                  double woe_b = compute_woe(count_pos_map.at(b), count_neg_map.at(b));
                  return woe_a < woe_b;
                });
      
      // Find split point aiming for roughly equal total counts
      size_t total_so_far = 0;
      size_t target_total = bin_to_split.get_total_count() / 2;
      size_t split_idx = 0; // Index *after* which to split
      
      for (size_t i = 0; i < sorted_categories.size() - 1; ++i) { // Iterate up to second-to-last
        const std::string& cat = sorted_categories[i];
        total_so_far += total_count_map.at(cat);
        // Split if current count exceeds target OR if adding next forces it way over?
        // Simpler: split after the element that gets us closest to/past half count
        if (total_so_far >= target_total) {
          split_idx = i + 1;
          break;
        }
      }
      // Ensure split point is valid (at least one category in each)
      if (split_idx == 0) {
        split_idx = 1; // Force at least one in first bin if target wasn't met
      } else if (split_idx >= sorted_categories.size()) {
        split_idx = sorted_categories.size() - 1; // Force at least one in second bin
      }
      
      
      // Create the two new bins
      Bin bin1, bin2;
      bin1.categories.reserve(split_idx);
      bin2.categories.reserve(n_cats - split_idx);
      
      // Assign categories and calculate counts
      for (size_t i = 0; i < n_cats; ++i) {
        const std::string& cat = sorted_categories[i];
        int pos = count_pos_map.at(cat);
        int neg = count_neg_map.at(cat);
        int total = total_count_map.at(cat);
        
        if (i < split_idx) {
          bin1.categories.push_back(cat);
          bin1.count_pos += pos;
          bin1.count_neg += neg;
          bin1.total_count += total;
        } else {
          bin2.categories.push_back(cat);
          bin2.count_pos += pos;
          bin2.count_neg += neg;
          bin2.total_count += total;
        }
      }
      
      // Add new bins to the end of the main vector
      bins.push_back(std::move(bin1));
      bins.push_back(std::move(bin2));
    }
    
    
    /**
     * Merge bins using the Chi-Merge algorithm.
     */
    void merge_bins_using_chimerge() {
      iterations_run = 0;
      converged = false; // Reset convergence flag
      double critical_value = get_chi_square_critical_value();
      
      // Ensure bins are sorted by WoE before starting ChiMerge
      sort_bins_by_woe();
      
      // Initialize/populate cache for adjacent pairs
      populate_chi_cache();
      
      // STEP 1: First, ensure we have at most max_bins by merging lowest chi-square pairs
      // regardless of statistical significance
      while (bins.size() > static_cast<size_t>(max_bins) && bins.size() > 1) {
        std::pair<double, size_t> min_chi_pair = find_min_chi_square_pair();
        double min_chi = min_chi_pair.first;
        size_t min_index = min_chi_pair.second;
        
        // Perform the merge
        merge_adjacent_bins(min_index);
        
        // Update the cache
        update_chi_cache_after_merge(min_index);
        
        iterations_run++;
      }
      
      // STEP 2: Now continue with original ChiMerge algorithm to merge statistically similar bins
      // but only if we're above min_bins
      double prev_min_chi = -1.0; // Initialize for convergence check
      
      while (can_merge_further() && iterations_run < max_iterations) {
        std::pair<double, size_t> min_chi_pair = find_min_chi_square_pair();
        double min_chi = min_chi_pair.first;
        size_t min_index = min_chi_pair.second;
        
        // Check stopping conditions:
        // Stop if minimum chi-square is above critical value (statistically significant difference)
        if (min_chi > critical_value) {
          converged = true;
          break;
        }
        
        // Check convergence based on small change in min_chi
        if (prev_min_chi >= 0 && std::fabs(min_chi - prev_min_chi) < convergence_threshold) {
          converged = true;
          break;
        }
        prev_min_chi = min_chi;
        
        // Perform the merge
        merge_adjacent_bins(min_index);
        
        // Update the cache
        update_chi_cache_after_merge(min_index);
        
        iterations_run++;
      }
      
      // Update final convergence status
      if (!converged && iterations_run == max_iterations) {
        Rcpp::Rcout << "Warning: ChiMerge reached max_iterations (" << max_iterations << ")." << std::endl;
      } else {
        converged = true; // If loop finished normally or broke early
      }
      // No need to invalidate cache here unless structure changes elsewhere
    }
    
    /**
     * Perform the Chi2 algorithm from Liu & Setiono (1995).
     */
    void perform_chi2_binning() {
      const std::vector<double> significance_levels = {0.5, 0.1, 0.05, 0.01, 0.005, 0.001};
      
      // 1. Initial discretization (e.g., one bin per category or equal frequency)
      // Using initialize_bins (one per category) and then sorting by WoE
      // initialize_bins(); // Already called in perform_binning
      // chi_cache->resize(bins.size()); // Already resized
      
      // 2. Iterative ChiMerge phases with decreasing significance
      for (double significance : significance_levels) {
        chi_merge_threshold = significance; // Set threshold for this phase
        
        // Apply ChiMerge logic for this phase
        merge_bins_using_chimerge(); // This merges until threshold is met or convergence
        
        // Check if target bin count reached
        if (bins.size() <= static_cast<size_t>(max_bins)) {
          break; // Stop if we reached max_bins or fewer
        }
        
        // Optional: Check inconsistency rate (not fully implemented here, complex)
        // if (calculate_inconsistency_rate() < some_threshold) {
        //     break;
        // }
      }
      
      // 3. Final adjustments
      ensure_min_bins(); // Ensure minimum bin count
      enforce_monotonicity(); // Ensure WoE monotonicity
    }
    
    
    /**
     * Initialize bins using equal frequency discretization (Alternative init).
     * Used only if called explicitly (e.g., for Chi2 if preferred).
     */
    void initialize_equal_frequency_bins() {
      // Sort categories by total count for frequency binning
      std::vector<std::pair<std::string, int>> sorted_categories;
      sorted_categories.reserve(total_count_map.size());
      for (const auto& entry : total_count_map) {
        sorted_categories.emplace_back(entry.first, entry.second);
      }
      std::sort(sorted_categories.begin(), sorted_categories.end(),
                [](const auto& a, const auto& b) { return a.second < b.second; });
      
      // Determine initial bin count
      int initial_bins = std::min(max_n_prebins,
                                  std::max(min_bins,
                                           static_cast<int>(sqrt(sorted_categories.size()))));
      initial_bins = std::max(2, initial_bins); // Ensure at least 2
      
      bins.clear();
      bins.resize(initial_bins);
      
      int total_observations = total_pos + total_neg;
      double target_obs_per_bin = static_cast<double>(total_observations) / initial_bins;
      int current_bin_idx = 0;
      int current_bin_obs = 0;
      
      for (const auto& cat_pair : sorted_categories) {
        const std::string& category = cat_pair.first;
        int cat_count = cat_pair.second;
        
        // If adding this category significantly overfills the current bin AND
        // it's not the last bin, move to the next bin.
        if (current_bin_idx < initial_bins - 1 &&
            current_bin_obs > 0 && // Don't advance if current bin is empty
            current_bin_obs + cat_count > target_obs_per_bin * 1.5) // Heuristic to prevent large overflow
        {
          // Check if moving improves balance
          double current_imbalance = std::abs(current_bin_obs - target_obs_per_bin);
          double next_imbalance = std::abs(cat_count - target_obs_per_bin); // If moved to next
          if (next_imbalance < current_imbalance) { // Or some other condition
            current_bin_idx++;
            current_bin_obs = 0; // Reset for new bin
          }
        }
        
        
        // Add category to current bin
        bins[current_bin_idx].categories.push_back(category);
        bins[current_bin_idx].count_pos += count_pos_map.at(category);
        bins[current_bin_idx].count_neg += count_neg_map.at(category);
        bins[current_bin_idx].total_count += cat_count;
        current_bin_obs += cat_count;
      }
      
      // Remove any potentially empty bins (if initial_bins > unique_categories)
      bins.erase(std::remove_if(bins.begin(), bins.end(),
                                [](const Bin& b) { return b.categories.empty(); }),
                                bins.end());
      
      // Sort final bins by WoE
      sort_bins_by_woe();
      // Resize cache
      chi_cache->invalidate();
      chi_cache->resize(bins.size());
    }
    
    /**
     * Calculate inconsistency rate (Simplified version).
     * A measure used in the Chi2 algorithm context.
     * @return Inconsistency rate (0-1).
     */
    double calculate_inconsistency_rate() const {
      // Map each category to its current bin index
      std::unordered_map<std::string, size_t> category_to_bin;
      for (size_t i = 0; i < bins.size(); ++i) {
        for (const auto& category : bins[i].categories) {
          category_to_bin[category] = i;
        }
      }
      
      double inconsistent_sum = 0;
      for(const auto& bin : bins) {
        int bin_total = bin.get_total_count();
        if (bin_total > 0) {
          // Majority class count in the bin
          int majority_count = std::max(bin.count_pos, bin.count_neg);
          // Minority class count is the number of inconsistent items
          inconsistent_sum += (bin_total - majority_count);
        }
      }
      
      int total_instances = total_pos + total_neg;
      return (total_instances > 0) ? (inconsistent_sum / total_instances) : 0.0;
    }
    
    
    /**
     * Determine preferred WoE monotonicity (increasing or decreasing).
     */
    void determine_monotonicity_robust() {
      if (bins.size() < 2) {
        is_increasing = true; // Default for 0 or 1 bin
        return;
      }
      if (bins.size() == 2) {
        is_increasing = (bins[0].woe <= bins[1].woe); // Based on the two bins
        return;
      }
      
      // Use linear regression slope for > 2 bins
      double sum_x = 0, sum_y = 0, sum_xy = 0, sum_x2 = 0;
      int n = bins.size();
      
      for (int i = 0; i < n; ++i) {
        double x = static_cast<double>(i);
        double y = bins[i].woe; // Assumes WoE is pre-calculated
        sum_x += x;
        sum_y += y;
        sum_xy += x * y;
        sum_x2 += x * x;
      }
      
      double slope_denominator = n * sum_x2 - sum_x * sum_x;
      
      if (std::fabs(slope_denominator) < EPSILON) {
        // Flat trend, default to increasing or check endpoints?
        is_increasing = (bins.back().woe >= bins.front().woe);
      } else {
        double slope = (n * sum_xy - sum_x * sum_y) / slope_denominator;
        is_increasing = (slope >= 0);
      }
    }
    
    /**
     * Enforce monotonicity of Weight of Evidence by merging adjacent violators.
     */
    void enforce_monotonicity() {
      if (bins.size() <= 1) return; // Nothing to enforce
      
      // Ensure WoE is calculated and bins are sorted
      sort_bins_by_woe(); // This calculates WoE and sorts
      
      // Determine desired direction
      determine_monotonicity_robust();
      
      bool violation_found;
      int iterations = 0;
      const int max_mono_iter = bins.size() * 2; // Limit iterations
      
      do {
        violation_found = false;
        size_t i = 0;
        while (i < bins.size() - 1) {
          bool violation = is_increasing ?
          (bins[i].woe > bins[i + 1].woe + EPSILON) : // Check increasing
          (bins[i].woe < bins[i + 1].woe - EPSILON); // Check decreasing
          
          if (violation) {
            // Merge the violating adjacent pair
            merge_adjacent_bins(i);
            // Re-calculate WoE for the merged bin
            bins[i].woe = compute_woe(bins[i].count_pos, bins[i].count_neg);
            // Reset cache and resize
            chi_cache->invalidate();
            chi_cache->resize(bins.size());
            // Invalidate cache entries around merge point more specifically
            // update_chi_cache_after_merge(i); // Better if only cache matters
            
            violation_found = true;
            // Restart scan from the beginning after a merge
            // as it can affect previous pairs
            i = 0;
          } else {
            i++; // Move to next pair only if no violation
          }
        }
        iterations++;
      } while (violation_found && can_merge_further() && iterations < max_mono_iter);
      
      if (iterations == max_mono_iter) {
        Rcpp::Rcout << "Warning: Monotonicity enforcement reached max iterations." << std::endl;
        // Final sort might be needed if loop terminated early
        sort_bins_by_woe();
        if (!is_increasing) {
          std::reverse(bins.begin(), bins.end());
        }
      }
    }
    
    /**
     * Calculate final WoE and IV for each bin.
     */
    void calculate_woe_iv_bins() {
      for (auto& bin : bins) {
        bin.woe = compute_woe(bin.count_pos, bin.count_neg);
        
        // Calculate Information Value contribution for this bin
        double dist_pos = (total_pos > 0) ? static_cast<double>(bin.count_pos) / total_pos : 0.0;
        double dist_neg = (total_neg > 0) ? static_cast<double>(bin.count_neg) / total_neg : 0.0;
        
        // Add epsilon *before* subtraction if we expect tiny differences
        // double iv_term = (std::max(dist_pos, EPSILON) - std::max(dist_neg, EPSILON));
        double iv_term = dist_pos - dist_neg; // Standard definition
        
        // WoE is already calculated and handles 0 counts via compute_woe's EPSILON
        bin.iv = iv_term * bin.woe;
      }
    }
    
    /**
     * Prepare output List for R.
     */
    Rcpp::List prepare_output() const {
      const size_t n_bins = bins.size();
      Rcpp::StringVector bin_names(n_bins);
      Rcpp::NumericVector woe_values(n_bins);
      Rcpp::NumericVector iv_values(n_bins);
      Rcpp::IntegerVector bin_counts(n_bins);
      Rcpp::IntegerVector counts_pos(n_bins);
      Rcpp::IntegerVector counts_neg(n_bins);
      Rcpp::IntegerVector ids(n_bins);
      
      for (size_t i = 0; i < n_bins; ++i) {
        ids[i] = i + 1; // 1-based index for R
        bin_names[i] = join_categories(bins[i].categories);
        woe_values[i] = bins[i].woe;
        iv_values[i] = bins[i].iv;
        bin_counts[i] = bins[i].get_total_count();
        counts_pos[i] = bins[i].count_pos;
        counts_neg[i] = bins[i].count_neg;
      }
      
      return Rcpp::List::create(
        Rcpp::Named("id") = ids,
        Rcpp::Named("bin") = bin_names,
        Rcpp::Named("woe") = woe_values,
        Rcpp::Named("iv") = iv_values,
        Rcpp::Named("count") = bin_counts,
        Rcpp::Named("count_pos") = counts_pos,
        Rcpp::Named("count_neg") = counts_neg,
        Rcpp::Named("converged") = converged,
        Rcpp::Named("iterations") = iterations_run,
        Rcpp::Named("algorithm") = use_chi2_algorithm ? "Chi2" : "ChiMerge"
      );
    }
    
    /**
     * Find the adjacent pair of bins with the minimum chi-square value.
     * Uses the cache if possible.
     * @return pair<min_chi_square, index_of_first_bin>. Returns max double if no pair exists.
     */
    std::pair<double, size_t> find_min_chi_square_pair() {
      double min_chi_square = std::numeric_limits<double>::max();
      size_t min_index = 0; // Default to first possible pair index
      
      if (bins.size() < 2) {
        return {min_chi_square, 0}; // Cannot find a pair
      }
      
      for (size_t i = 0; i < bins.size() - 1; ++i) {
        double chi_square = chi_cache->get(i); // Get chi-square for pair (i, i+1)
        
        if (chi_square < 0) { // Not in cache or invalid
          chi_square = compute_chi_square_between_bins(bins[i], bins[i + 1]);
          chi_cache->set(i, chi_square); // Store computed value
        }
        
        if (chi_square < min_chi_square) {
          min_chi_square = chi_square;
          min_index = i;
        }
      }
      return {min_chi_square, min_index};
    }
    
    /**
     * Populate the chi-square cache for all adjacent bins.
     */
    void populate_chi_cache() {
      chi_cache->invalidate(); // Clear any old values
      chi_cache->resize(bins.size()); // Ensure correct size
      if (bins.size() < 2) return;
      
      for (size_t i = 0; i < bins.size() - 1; ++i) {
        double chi_square = compute_chi_square_between_bins(bins[i], bins[i + 1]);
        chi_cache->set(i, chi_square);
      }
    }
    
    
    /**
     * Update chi-square cache after merging adjacent bins at `merge_index`.
     * @param merge_index Index of the first bin in the pair that was merged.
     */
    void update_chi_cache_after_merge(size_t merge_index) {
      // Cache needs resizing first because a bin was removed
      // chi_cache->resize(bins.size()); // Done inside merge_adjacent_bins
      
      // Invalidate entries around the merge point in the *new* structure
      chi_cache->invalidate_after_merge(merge_index);
      
      
      // Recalculate and cache chi-square for the pair before the merged bin (if exists)
      if (merge_index > 0) {
        double chi = compute_chi_square_between_bins(bins[merge_index - 1], bins[merge_index]);
        chi_cache->set(merge_index - 1, chi);
      }
      
      // Recalculate and cache chi-square for the pair starting at the merged bin (if exists)
      if (merge_index < bins.size() - 1) {
        double chi = compute_chi_square_between_bins(bins[merge_index], bins[merge_index + 1]);
        chi_cache->set(merge_index, chi);
      }
    }
    
    
    /**
     * Get chi-square critical value for DF=1 based on the significance threshold.
     * @return Critical chi-square value.
     */
    double get_chi_square_critical_value() const {
      // Find closest significance level in the lookup table
      auto it = CHI_SQUARE_CRITICAL_VALUES.find(chi_merge_threshold);
      if (it != CHI_SQUARE_CRITICAL_VALUES.end()) {
        return it->second;
      }
      
      // If exact match not found, find the nearest threshold in the table
      double closest_threshold = 0.05; // Default if lookup fails completely
      double min_diff = std::abs(chi_merge_threshold - 0.05); // Initialize difference
      
      for (const auto& entry : CHI_SQUARE_CRITICAL_VALUES) {
        double diff = std::abs(chi_merge_threshold - entry.first);
        if (diff < min_diff) {
          min_diff = diff;
          closest_threshold = entry.first;
        }
      }
      // Use .at() which throws if key not found (shouldn't happen after finding closest)
      return CHI_SQUARE_CRITICAL_VALUES.at(closest_threshold);
    }
    
    /**
     * Calculate chi-square statistic between two bins (DF=1).
     * @param bin1 First bin.
     * @param bin2 Second bin.
     * @return Chi-square statistic (>= 0).
     */
    inline double compute_chi_square_between_bins(const Bin& bin1, const Bin& bin2) const {
      int obs11 = bin1.count_pos; int obs12 = bin1.count_neg;
      int obs21 = bin2.count_pos; int obs22 = bin2.count_neg;
      
      int row1_total = bin1.get_total_count();
      int row2_total = bin2.get_total_count();
      int col1_total = obs11 + obs21; // Total Positives in these two bins
      int col2_total = obs12 + obs22; // Total Negatives in these two bins
      int grand_total = row1_total + row2_total;
      
      if (grand_total == 0 || row1_total == 0 || row2_total == 0 || col1_total == 0 || col2_total == 0) {
        // If any marginal total is zero, chi-square is 0 (no variation possible)
        return 0.0;
      }
      
      double chi_square = 0.0;
      
      // Expected values
      double exp11 = static_cast<double>(row1_total) * col1_total / grand_total;
      double exp12 = static_cast<double>(row1_total) * col2_total / grand_total;
      double exp21 = static_cast<double>(row2_total) * col1_total / grand_total;
      double exp22 = static_cast<double>(row2_total) * col2_total / grand_total;
      
      // Sum of (Obs - Exp)^2 / Exp
      if (exp11 > EPSILON) chi_square += std::pow(obs11 - exp11, 2) / exp11;
      if (exp12 > EPSILON) chi_square += std::pow(obs12 - exp12, 2) / exp12;
      if (exp21 > EPSILON) chi_square += std::pow(obs21 - exp21, 2) / exp21;
      if (exp22 > EPSILON) chi_square += std::pow(obs22 - exp22, 2) / exp22;
      
      return chi_square;
    }
    
    
    /**
     * Merge two adjacent bins at the given index.
     * @param index Index of the first bin in the adjacent pair to merge.
     */
    inline void merge_adjacent_bins(size_t index) {
      if (index >= bins.size() - 1) return; // Cannot merge last bin with non-existent next
      
      // Merge categories and counts into bins[index]
      merge_two_bins(bins[index], bins[index + 1]);
      
      // Remove the second bin (at index + 1)
      bins.erase(bins.begin() + index + 1);
      
      // Update chi-square cache size AFTER removing the bin
      chi_cache->resize(bins.size());
    }
    
    /**
     * Merge contents of source bin (`src_bin`) into destination bin (`dest_bin`).
     * @param dest_bin Destination bin (modified in place).
     * @param src_bin Source bin.
     */
    inline void merge_two_bins(Bin& dest_bin, const Bin& src_bin) {
      // Reserve capacity for merged categories (optional optimization)
      // dest_bin.categories.reserve(dest_bin.categories.size() + src_bin.categories.size());
      
      // Append categories
      dest_bin.categories.insert(dest_bin.categories.end(),
                                 src_bin.categories.begin(),
                                 src_bin.categories.end());
      
      // Update counts
      dest_bin.count_pos += src_bin.count_pos;
      dest_bin.count_neg += src_bin.count_neg;
      dest_bin.total_count += src_bin.get_total_count();
      
      // WoE and IV will be recalculated later for the merged bin
      dest_bin.woe = 0.0;
      dest_bin.iv = 0.0;
    }
    
    /**
     * Join category names with separator.
     * @param categories Vector of category names.
     * @return Joined string.
     */
    inline std::string join_categories(const std::vector<std::string>& categories) const {
      if (categories.empty()) return "EMPTY_BIN";
      if (categories.size() == 1) return categories[0];
      
      // Estimate length
      size_t total_length = 0;
      for (const auto& cat : categories) total_length += cat.length();
      total_length += bin_separator.length() * (categories.size() - 1);
      
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
     * Check if further merging is possible based on min_bins.
     * @return True if bins.size() > min_bins (and > 1).
     */
    inline bool can_merge_further() const {
      // Need at least 2 bins to merge, and must be above min_bins target
      return bins.size() > static_cast<size_t>(std::max(min_bins, 1)) && bins.size() > 1;
    }
    
    /**
     * Calculate Weight of Evidence (WoE) for given counts.
     * Handles zero counts using EPSILON smoothing.
     * @param c_pos Positive count.
     * @param c_neg Negative count.
     * @return WoE value.
     */
    inline double compute_woe(int c_pos, int c_neg) const {
      // Use overall totals for distribution calculation
      double dist_pos = (total_pos > 0) ? static_cast<double>(c_pos) / total_pos : 0.0;
      double dist_neg = (total_neg > 0) ? static_cast<double>(c_neg) / total_neg : 0.0;
      
      // Apply smoothing before log/division
      dist_pos = std::max(dist_pos, EPSILON);
      dist_neg = std::max(dist_neg, EPSILON);
      
      return std::log(dist_pos / dist_neg);
    }
    
}; // End class OptimalBinningCategorical


//' @title Optimal Binning for Categorical Variables using ChiMerge
//'
//' @description
//' Implements optimal binning for categorical variables using the ChiMerge algorithm
//' (Kerber, 1992) and optionally the Chi2 algorithm (Liu & Setiono, 1995),
//' calculating Weight of Evidence (WoE) and Information Value (IV) for the
//' resulting bins. This is Version 4 with corrections based on previous code review.
//'
//' @param target Integer vector of binary target values (0 or 1). Cannot contain NAs.
//' @param feature Character vector of categorical feature values. `NA` values will be treated as a distinct category "NA".
//' @param min_bins Minimum number of bins (default: 3, must be >= 2).
//' @param max_bins Maximum number of bins (default: 5).
//' @param bin_cutoff Minimum frequency fraction for a category to potentially avoid being merged in the initial `handle_rare_categories` step (default: 0.05). Note: The main merging uses chi-square statistics.
//' @param max_n_prebins Maximum number of bins allowed after the initial pre-binning/rare handling step, before the main ChiMerge/Chi2 loop (default: 20). Merging stops if this limit is reached and statistical thresholds aren't met.
//' @param bin_separator Separator string for concatenating category names in bins (default: "%;%").
//' @param convergence_threshold Threshold for convergence based on the absolute difference in minimum chi-square between iterations during bin merging (default: 1e-6).
//' @param max_iterations Maximum number of iterations allowed for the bin merging loop (default: 1000).
//' @param chi_merge_threshold Significance level threshold for the chi-square test used in merging decisions (default: 0.05, corresponds to 95 pct confidence). Lower values lead to fewer merges.
//' @param use_chi2_algorithm Boolean indicating whether to use the enhanced Chi2 algorithm which involves multiple ChiMerge phases with decreasing significance levels (default: FALSE).
//'
//' @return A list containing:
//' \item{id}{Vector of numeric IDs (1-based) for each final bin.}
//' \item{bin}{Vector of character strings representing the final bins (concatenated category names).}
//' \item{woe}{Vector of numeric Weight of Evidence (WoE) values for each bin.}
//' \item{iv}{Vector of numeric Information Value (IV) contributions for each bin.}
//' \item{count}{Vector of integer total counts (observations) for each bin.}
//' \item{count_pos}{Vector of integer positive class counts for each bin.}
//' \item{count_neg}{Vector of integer negative class counts for each bin.}
//' \item{converged}{Boolean indicating whether the merging algorithm converged (either reached target bins, statistical threshold, or convergence threshold).}
//' \item{iterations}{Integer number of merging iterations performed.}
//' \item{algorithm}{Character string indicating the algorithm used ("ChiMerge" or "Chi2").}
//'
//' @details
//' This function implements categorical variable binning based on chi-square statistics.
//' The core logic follows the ChiMerge approach, iteratively merging adjacent bins (sorted by WoE)
//' that have the lowest chi-square statistic below a specified critical value (derived from `chi_merge_threshold`).
//' The optional Chi2 algorithm applies multiple rounds of ChiMerge with varying significance levels.
//' Monotonicity of WoE across the final bins is enforced by merging adjacent bins that violate the trend.
//'
//' Weight of Evidence (WoE) is calculated as: \eqn{WoE_i = \ln(\frac{p_{pos,i}}{p_{neg,i}})}
//' Information Value (IV) is calculated as: \eqn{IV = \sum_{i} (p_{pos,i} - p_{neg,i}) \times WoE_i}
//' where \eqn{p_{pos,i}} and \eqn{p_{neg,i}} are the proportions of positive and negative observations in bin i relative to the total positive and negative observations, respectively.
//'
//' V4 includes fixes for stability and corrects the initialization and usage of the internal chi-square cache.
//'
//' @references
//' \itemize{
//'   \item Kerber, R. (1992). ChiMerge: Discretization of Numeric Attributes. In AAAI'92.
//'   \item Liu, H. & Setiono, R. (1995). Chi2: Feature Selection and Discretization of Numeric Attributes. In TAI'95.
//'   \item Siddiqi, N. (2006). Credit Risk Scorecards: Developing and Implementing Intelligent Credit Scoring. John Wiley & Sons.
//' }
//'
//' @examples
//' \dontrun{
//' # Example data
//' set.seed(123)
//' target <- sample(0:1, 500, replace = TRUE, prob = c(0.7, 0.3))
//' feature <- sample(LETTERS[1:8], 500, replace = TRUE)
//' feature[sample(1:500, 20)] <- NA # Add some NAs
//'
//' # Run optimal binning with ChiMerge (V4)
//' result_v4 <- optimal_binning_categorical_cm_v4(target, feature,
//'                                            min_bins = 3, max_bins = 6,
//'                                            chi_merge_threshold = 0.05)
//' print(result_v4)
//'
//' # Check total IV
//' print(sum(result_v4$iv))
//'
//' # Run using the Chi2 algorithm variant
//' result_chi2_v4 <- optimal_binning_categorical_cm_v4(target, feature,
//'                                                min_bins = 3, max_bins = 6,
//'                                                use_chi2_algorithm = TRUE)
//' print(result_chi2_v4)
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
 try {
   // Convert R vectors to C++ std::vectors and handle NAs
   R_xlen_t n = feature.size();
   if (n != target.size()) {
     Rcpp::stop("V4 Error: Feature and target must have the same length.");
   }
   std::vector<std::string> feature_vec(n);
   std::vector<int> target_vec(n);
   
   for (R_xlen_t i = 0; i < n; ++i) {
     // Handle NAs in feature vector -> Treat as category "NA"
     if (CharacterVector::is_na(feature[i])) {
       feature_vec[i] = "NA";
     } else {
       feature_vec[i] = Rcpp::as<std::string>(feature[i]);
     }
     
     // Check NAs in target vector -> Error
     if (IntegerVector::is_na(target[i])) {
       Rcpp::stop("V4 Error: Target variable cannot contain missing values (NA).");
     }
     target_vec[i] = target[i];
   }
   
   // Create the OptimalBinningCategorical object (V4 logic)
   OptimalBinningCategorical obcm_v4(
       feature_vec, target_vec, min_bins, max_bins, bin_cutoff, max_n_prebins,
       bin_separator, convergence_threshold, max_iterations, chi_merge_threshold,
       use_chi2_algorithm
   );
   
   // Execute the binning process
   return obcm_v4.perform_binning();
   
 } catch (const std::exception& e) {
   Rcpp::stop("Error in optimal binning v4 wrapper: " + std::string(e.what()));
 } catch (...) {
   Rcpp::stop("Unknown error occurred during optimal binning v4 wrapper.");
 }
}
